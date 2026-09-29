"""Fresh SID clips with constructed duplex timelines; not natural dialogue."""

import hashlib
import json
import random
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from evaluate_interruption_laya import ROOT, load_rows

OUT = Path("artifacts/large-workflow-eval")
OUT.mkdir(exist_ok=True)
old = {r["id"] for r in json.loads((ROOT / "audio-manifest.json").read_text())}
old |= {
    r["id"].removesuffix("-overlap")
    for r in json.loads(Path("evidence/curation-workflow/results.json").read_text())[
        "predictions"
    ]
}
rows = [r for r in load_rows() if r["id"] not in old]
rng = random.Random(20261002)
selected = rng.sample([r for r in rows if r["positive"]], 100) + rng.sample(
    [r for r in rows if not r["positive"]], 200
)
rng.shuffle(selected)
shutil.copy2("artifacts/workflow-eval/agent.wav", OUT / "agent.wav")
agent, sr = sf.read(OUT / "agent.wav")
agent = soxr.resample(agent, sr, 16000)
agent_text = "Let me explain the available options. There are several things to consider before we decide what to do next. We can go through each option together and choose the one that works best for you."
manifest = []


def prepare(item):
    i, r = item
    local = []
    p = OUT / (r["id"] + ".wav")
    if not p.exists():
        url = (
            "https://huggingface.co/datasets/kxxia/SID-bench/resolve/6eb13b573ad588646ce0de513d4255e55caf858b/test_wavs/en/"
            + r["source_audio"]
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            p.write_bytes(response.read())
    user, sr = sf.read(p)
    user = user.mean(axis=1) if user.ndim == 2 else user
    user = soxr.resample(user, sr, 16000)
    for control in (False, True) if i < 100 else (False,):
        onset = len(agent) / 16000 + 0.3 if control else 1.5 + i * 0.005
        end = onset + len(user) / 16000
        mixed = np.zeros(
            (int((max(end, len(agent) / 16000) + 1) * 16000) + 1, 2), dtype=np.float32
        )
        mixed[: len(agent), 0] = agent
        start = round(onset * 16000)
        mixed[start : start + len(user), 1] = user
        cid = r["id"] + ("-handoff" if control else "-overlap")
        path = OUT / (cid + ".wav")
        sf.write(path, mixed, 16000)
        conversation = {
            "id": cid,
            "source_group": r["id"],
            "split": "test",
            "provenance": "Constructed duplex fixture: Kokoro agent speech and held-out SID user clip. Timings and agent context invented, not a naturally observed interaction.",
            "timing_source": "manual_annotations",
            "turns": [
                {
                    "role": "assistant",
                    "text": agent_text,
                    "start_s": 0.0,
                    "end_s": len(agent) / 16000,
                },
                {"role": "user", "text": r["text"], "start_s": onset, "end_s": end},
            ],
        }
        local.append(
            {
                "conversation": conversation,
                "audio": str(path),
                "positive": r["positive"] and not control,
                "intent_positive": r["positive"],
                "control": control,
            }
        )
    return local


with ThreadPoolExecutor(max_workers=6) as pool:
    for result in pool.map(prepare, enumerate(selected)):
        manifest.extend(result)
        if len(manifest) % 50 == 0:
            print("prepared", len(manifest), flush=True)
(OUT / "protocol.json").write_text(
    json.dumps(
        {
            "seed": 20261002,
            "source_ids": [r["id"] for r in selected],
            "excluded_previous_ids": sorted(old),
            "input_conversations": 400,
            "intent_positive_overlaps": 100,
            "intent_negative_overlaps": 200,
            "handoff_controls": 100,
            "agent_sha256": hashlib.sha256(
                (OUT / "agent.wav").read_bytes()
            ).hexdigest(),
            "scope": "Constructed agent context with SID source intent labels as proxies; not natural conversation ground truth.",
        },
        indent=2,
    )
)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("Prepared", len(manifest), "conversations")
