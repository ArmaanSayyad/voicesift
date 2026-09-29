"""Real audio -> local Parakeet; reference transcripts and labels are not inputs."""

import json
import time
from pathlib import Path

import mlx.core as mx
import soundfile as sf
import soxr
from mlx_audio.stt.utils import load

ROOT = Path("artifacts/interruption-eval")
models = json.loads(Path("artifacts/models.json").read_text())
asr = load(models["parakeet"]["path"], model_type="parakeet")
manifest = json.loads((ROOT / "audio-manifest.json").read_text())
with (ROOT / "asr.jsonl").open("w") as f:
    for i, item in enumerate(manifest):
        a, sr = sf.read(ROOT / "audio" / f"{item['id']}.wav", dtype="float32")
        if a.ndim == 2:
            a = a.mean(axis=1)
        duration = len(a) / sr
        a = soxr.resample(a, sr, 16000)
        start = time.perf_counter()
        out = asr.generate(mx.array(a))
        mx.synchronize()
        f.write(
            json.dumps(
                {
                    "id": item["id"],
                    "text": out.text,
                    "seconds": time.perf_counter() - start,
                    "duration_s": duration,
                    "model_revision": models["parakeet"]["revision"],
                }
            )
            + "\n"
        )
        f.flush()
        if i % 20 == 0:
            print(i, flush=True)
