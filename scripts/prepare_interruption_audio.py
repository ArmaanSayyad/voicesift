"""Download a fixed 100-clip audit subset; no label-dependent model input."""

import hashlib
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from evaluate_interruption_laya import ROOT, SEED, load_rows

REV = "6eb13b573ad588646ce0de513d4255e55caf858b"
rows = load_rows()
rng = random.Random(SEED + 3)
selected = rng.sample([r for r in rows if r["positive"]], 50) + rng.sample(
    [r for r in rows if not r["positive"]], 50
)
rng.shuffle(selected)
(ROOT / "audio").mkdir(exist_ok=True)


def fetch(r):
    path = ROOT / "audio" / f"{r['id']}.wav"
    url = (
        "https://huggingface.co/datasets/kxxia/SID-bench/resolve/"
        + REV
        + "/test_wavs/en/"
        + r["source_audio"]
    )
    if not path.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        path.write_bytes(data)
    return {
        "id": r["id"],
        "source_audio": r["source_audio"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "positive": r["positive"],
        "revision": REV,
    }


with ThreadPoolExecutor(max_workers=4) as pool:
    manifest = list(pool.map(fetch, selected))
(ROOT / "audio-manifest.json").write_text(json.dumps(manifest, indent=2))
print("downloaded", len(manifest), "clips", sum(x["bytes"] for x in manifest), "bytes")
