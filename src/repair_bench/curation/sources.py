"""Explicit dataset adapters. Never execute dataset code or infer speaker labels."""

import hashlib
import io
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import numpy as np
import soundfile as sf
import soxr

from .models import Turn

MAX_UPLOAD = 512 * 1024 * 1024
MAX_DOWNLOAD = 6 * 1024**3
MAX_EXPANDED = 2 * 1024**3
REQUIREMENT = "conversations where there is an interruption"


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or p.is_absolute()
        or ".." in p.parts
        or ":" in name
        or "\x00" in name
    ):
        raise ValueError("Dataset file paths must stay inside the dataset folder")
    return p


def unpack(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        if len(entries) > 10000 or sum(e.file_size for e in entries) > MAX_EXPANDED:
            raise ValueError("ZIP exceeds 10,000 files or 2 GB uncompressed")
        seen = set()
        for e in entries:
            name = safe_relative(e.filename)
            if (
                str(name) in seen
                or stat.S_ISLNK(e.external_attr >> 16)
                or e.flag_bits & 1
            ):
                raise ValueError("ZIP contains duplicate, linked, or encrypted files")
            seen.add(str(name))
            target = dest / name
            if e.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(e) as src, target.open("xb") as out:
                    shutil.copyfileobj(src, out, 1024 * 1024)
    if not (dest / "dataset.jsonl").is_file():
        raise ValueError(
            "ZIP must contain dataset.jsonl at its root, plus the referenced audio files"
        )
    return dest


def repo_id(url):
    p = urlsplit(url.strip())
    if p.scheme != "https" or p.netloc != "huggingface.co" or p.query or p.fragment:
        raise ValueError("Use https://huggingface.co/datasets/owner/name")
    match = re.fullmatch(r"/datasets/([\w.-]+/[\w.-]+)/?", p.path)
    if not match:
        raise ValueError(
            "Use the Hugging Face dataset page URL, without a file or revision suffix"
        )
    return match[1]


def download_dataset(url, cache, progress):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    repo = repo_id(url)
    try:
        info = HfApi().dataset_info(repo, files_metadata=True)
        files = {s.rfilename: s for s in info.siblings}
        dest = cache / hashlib.sha256((repo + info.sha).encode()).hexdigest()[:20]

        def fetch(name):
            return Path(
                hf_hub_download(
                    repo, name, repo_type="dataset", revision=info.sha, local_dir=dest
                )
            )

        license_names = [
            n
            for n in files
            if "/" not in n and (n.upper().startswith("LICENSE") or n == "README.md")
        ]
        if "dataset.jsonl" in files:
            if (
                files["dataset.jsonl"].size is None
                or files["dataset.jsonl"].size > 20_000_000
            ):
                raise ValueError("dataset.jsonl must be at most 20 MB")
            manifest = fetch("dataset.jsonl")
            records = read_manifest(manifest)
            names = sorted({str(safe_relative(r["audio"])) for r in records}) + [
                "dataset.jsonl"
            ]
            adapter = "normalized-jsonl"
        elif repo == "mundo-ai/turn-benchmark-dev":
            names = sorted(
                n for n in files if n.startswith("data/") and n.endswith(".parquet")
            )
            adapter = "turnbench"
        else:
            raise ValueError(
                "Unsupported dataset schema. Provide a dataset.jsonl with audio paths and optional timed turns, or use mundo-ai/turn-benchmark-dev."
            )
        names = sorted(set(names + license_names))
        if not names or any(n not in files or files[n].size is None for n in names):
            raise ValueError(
                "Dataset references missing files or files with unknown size"
            )
        size = sum(files[n].size for n in names)
        if size > MAX_DOWNLOAD:
            raise ValueError("Selected dataset files exceed the 6 GB download limit")
        downloaded = []
        for i, name in enumerate(names):
            progress(
                message=f"Downloading dataset files {i + 1}/{len(names)}",
                download_bytes=size,
            )
            downloaded.append(fetch(name))
        return (
            dest,
            {
                "kind": "huggingface",
                "url": url,
                "repo": repo,
                "revision": info.sha,
                "adapter": adapter,
                "download_bytes": size,
                "files": names,
            },
            downloaded,
        )
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise ValueError(
            "Hugging Face access failed. Accept the dataset terms and log in with the local Hugging Face CLI, then submit again."
        ) from exc


def read_manifest(path):
    if path.stat().st_size > 20_000_000:
        raise ValueError("dataset.jsonl must be at most 20 MB")
    records = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    if not records or len(records) > 500:
        raise ValueError("Provide 1–500 conversations per run")
    return records


def validate_record(row):
    if (
        not isinstance(row, dict)
        or not isinstance(row.get("id"), str)
        or not row["id"]
        or len(row["id"]) > 200
    ):
        raise ValueError(
            "Each conversation needs a string id of at most 200 characters"
        )
    turns = row.get("turns")
    if turns is None or turns == []:
        return {**row, "turns": [], "timing_source": "none"}
    if not isinstance(turns, list) or not 2 <= len(turns) <= 10000:
        raise ValueError("Each conversation needs 2–10,000 timed speaker turns")
    turns = [Turn.model_validate(t).model_dump() for t in turns]
    if any(t["start_s"] is None for t in turns):
        raise ValueError(
            "Every turn needs start_s and end_s; automatic transcription is not supported yet"
        )
    if {t["role"] for t in turns} != {"user", "assistant"}:
        raise ValueError("Conversations need both user and assistant speaker roles")
    return {
        **row,
        "turns": turns,
        "timing_source": row.get("timing_source", "transcript_segments"),
    }


def normalize_audio(source, dest, last, max_bytes=MAX_DOWNLOAD):
    info = sf.info(source)
    if info.channels not in (1, 2) or not 0 < info.duration <= 3600:
        raise ValueError(
            "Audio must be mono/stereo and at most one hour per conversation"
        )
    if last > info.duration + 0.05:
        raise ValueError("Conversation timestamps exceed the audio duration")
    if info.duration * 16000 * info.channels * 2 > max_bytes:
        raise ValueError("Normalized audio exceeds the 6 GB per-run limit")
    data, rate = sf.read(source, dtype="float32", always_2d=True)
    sf.write(dest, soxr.resample(data, rate, 16000), 16000, subtype="PCM_16")


def prepare_records(source, dest, adapter, progress):
    """Normalize every record before spending on analysis; keep source labels out of inputs."""
    dest.mkdir(exist_ok=True)
    records = []
    used_bytes = 0
    if adapter == "turnbench":
        import pyarrow.parquet as pq

        def raw():
            for file in sorted((source / "data").glob("*.parquet")):
                columns = ["conversation_id"] + [
                    f"speaker_{s}_{k}"
                    for s in (1, 2)
                    for k in ["audio", "annotation_a"]
                ]
                for batch in pq.ParquetFile(file).iter_batches(
                    batch_size=1, columns=columns
                ):
                    yield batch.to_pylist()[0]

        for row in raw():
            i = len(records)
            channels = []
            for speaker in (1, 2):
                audio = io.BytesIO(row[f"speaker_{speaker}_audio"]["bytes"])
                info = sf.info(audio)
                if info.channels != 1 or not 0 < info.duration <= 3600:
                    raise ValueError(
                        "Invalid TurnBench audio duration or channel count"
                    )
                audio.seek(0)
                data, rate = sf.read(audio, dtype="float32")
                channels.append(soxr.resample(data, rate, 16000))
            if abs(len(channels[0]) - len(channels[1])) > 1:
                raise ValueError("TurnBench channels are not time-aligned")
            wav = dest / f"{i:05d}.wav"
            if used_bytes + min(map(len, channels)) * 4 > MAX_DOWNLOAD:
                raise ValueError("Normalized audio exceeds the 6 GB per-run limit")
            used_bytes += min(map(len, channels)) * 4
            sf.write(
                wav,
                np.stack([c[: min(map(len, channels))] for c in channels], axis=1),
                16000,
                subtype="PCM_16",
            )
            turns = sorted(
                [
                    {
                        "role": "assistant" if s == 1 else "user",
                        "text": e["text"],
                        "start_s": float(e["start_s"]),
                        "end_s": float(e["end_s"]),
                    }
                    for s in (1, 2)
                    for e in row[f"speaker_{s}_annotation_a"]
                    if e["text"].strip() and e["end_s"] > e["start_s"]
                ],
                key=lambda t: (t["start_s"], t["end_s"], t["role"]),
            )
            record = validate_record(
                {
                    "id": str(row["conversation_id"]),
                    "turns": turns,
                    "timing_source": "manual_annotations",
                    "source_group": str(row["conversation_id"]),
                    "split": "dev",
                    "provenance": "TurnBench human-human conversation. Annotator A timings/text; event labels excluded.",
                    "both_directions": True,
                }
            )
            if max(t["end_s"] for t in turns) > sf.info(wav).duration + 0.05:
                raise ValueError("Conversation timestamps exceed the audio duration")
            records.append({**record, "file": wav.name})
            progress(message=f"Preparing conversations: {len(records)}")
    else:
        for i, row in enumerate(read_manifest(source / "dataset.jsonl")):
            record = validate_record(row)
            if not isinstance(row.get("audio"), str):
                raise ValueError(  # noqa: TRY004 - user-facing input validation
                    "Each conversation needs an audio path inside the dataset"
                )
            path = source / safe_relative(row["audio"])
            if not path.is_file() or path.is_symlink():
                raise ValueError("Referenced audio file is missing or linked")
            wav = dest / f"{i:05d}.wav"
            normalize_audio(
                path,
                wav,
                max((t["end_s"] for t in record["turns"]), default=0),
                MAX_DOWNLOAD - used_bytes,
            )
            used_bytes += wav.stat().st_size
            records.append({**record, "file": wav.name, "both_directions": False})
            progress(message=f"Preparing conversations: {len(records)}")
    if (
        not records
        or len(records) > 500
        or len({r["id"] for r in records}) != len(records)
    ):
        raise ValueError("Provide 1–500 conversations with unique ids")
    return records
