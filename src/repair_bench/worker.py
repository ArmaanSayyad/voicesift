"""Resident MLX worker. stdout is reserved for one-line protocol replies."""

import contextlib
import json
import sys
import traceback
from pathlib import Path

# All libraries may write logs; keep stdout clean even during imports/loading.
with contextlib.redirect_stdout(sys.stderr):
    import numpy as np
    import soundfile as sf
    import soxr
    import mlx.core as mx
    from mlx_audio.stt.utils import load
    from mlx_audio.tts.utils import load_model

asr = tts = voice = None


def execute(request):
    global asr, tts, voice
    op = request["op"]
    if op == "prepare":
        paths = json.loads(Path(request["models"]).read_text())
        if asr is None:
            asr = load(paths["parakeet"]["path"], model_type="parakeet")
            tts = load_model(paths["kokoro"]["path"], model_type="kokoro")
            voice = str(Path(paths["kokoro"]["path"]) / "voices/af_heart.safetensors")
            asr.generate(mx.zeros(16000))
            for chunk in tts.generate(
                text="The local speech system is ready.", voice=voice, lang_code="a"
            ):
                np.array(chunk.audio)
            mx.synchronize()
        return {"ready": True, "voice": "af_heart"}
    if asr is None:
        raise RuntimeError("worker not prepared")
    if op == "asr":
        audio, sr = sf.read(request["path"], dtype="float32")
        if audio.ndim != 1 or not 0 < len(audio) <= sr * 30:
            raise ValueError("input must be mono and at most 30 seconds")
        audio = soxr.resample(audio, sr, 16000)
        result = asr.generate(mx.array(audio))
        mx.synchronize()
        return {"text": result.text, "input_duration_s": len(audio) / 16000}
    if op == "tts":
        chunks = []
        for chunk in tts.generate(text=request["text"], voice=voice, lang_code="a"):
            chunks.append(np.array(chunk.audio))
            sr = chunk.sample_rate
        mx.synchronize()
        audio = np.concatenate(chunks)
        sf.write(request["path"], audio, sr, subtype="PCM_16")
        return {"sample_rate": sr, "samples": len(audio), "duration_s": len(audio) / sr}
    raise ValueError("unsupported operation")


for line in sys.stdin:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = execute(json.loads(line))
        response = {"ok": True, "result": result}
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        response = {"ok": False, "error": str(exc)}
    print(json.dumps(response, allow_nan=False), flush=True)
