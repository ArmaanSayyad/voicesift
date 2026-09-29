"""Synthetic development-only component probes; not a human-speech benchmark."""

import gc, json, time, resource, traceback
from pathlib import Path
import numpy as np
import soundfile as sf
import soxr
import mlx.core as mx

ROOT = Path("artifacts")
ROOT.mkdir(exist_ok=True)
paths = json.loads((ROOT / "models.json").read_text())
report = {
    "scope": "synthetic development fixtures only",
    "tts": [],
    "asr": [],
    "prefix": [],
    "vad": [],
}


def save():
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["mlx_peak_bytes"] = mx.get_peak_memory()
    (ROOT / "audio-probe.json").write_text(json.dumps(report, indent=2))


phrases = [
    "Plan dinner Friday at seven for two people.",
    "Actually Saturday. Keep the time and party size.",
    "Do not book Friday. I said Saturday at eight.",
    "Cancel the dinner plan. Do not make a reservation.",
    "Um, let me think. Seven, no, actually eight.",
    "I would like a quiet restaurant on Saturday at seven for two people. Please keep the budget below forty dollars per person and make sure there are vegetarian options.",
]
try:
    from mlx_audio.tts.utils import load_model

    t = time.perf_counter()
    tts = load_model(paths["kokoro"]["path"], model_type="kokoro")
    report["tts_load_s"] = time.perf_counter() - t
    voice = str(Path(paths["kokoro"]["path"]) / "voices/af_heart.safetensors")
    for i, text in enumerate(phrases):
        start = time.perf_counter()
        chunks = []
        arrivals = []
        for r in tts.generate(text=text, voice=voice, lang_code="a"):
            a = np.array(r.audio)
            mx.synchronize()
            chunks.append(a)
            arrivals.append(time.perf_counter() - start)
            sr = r.sample_rate
        wav = np.concatenate(chunks)
        sf.write(ROOT / f"phrase-{i}.wav", wav, sr)
        elapsed = time.perf_counter() - start
        report["tts"].append(
            {
                "text": text,
                "seconds": elapsed,
                "duration": len(wav) / sr,
                "rtf": elapsed / (len(wav) / sr),
                "segment_arrivals_s": arrivals,
                "sample_rate": sr,
                "first_call": i == 0,
            }
        )
        save()
        print("TTS", i, elapsed, flush=True)
    del tts
    gc.collect()
    mx.clear_cache()
    from mlx_audio.stt.utils import load

    t = time.perf_counter()
    asr = load(paths["parakeet"]["path"], model_type="parakeet")
    report["asr_load_s"] = time.perf_counter() - t
    for i, text in enumerate(phrases):
        a, sr = sf.read(ROOT / f"phrase-{i}.wav", dtype="float32")
        a = soxr.resample(a, sr, 16000)
        t = time.perf_counter()
        r = asr.generate(mx.array(a))
        mx.synchronize()
        elapsed = time.perf_counter() - t
        report["asr"].append(
            {
                "expected_script": text,
                "text": r.text,
                "seconds": elapsed,
                "duration": len(a) / 16000,
                "first_call": i == 0,
            }
        )
        save()
        print("ASR", i, r.text, elapsed, flush=True)
    a, sr = sf.read(ROOT / "phrase-2.wav", dtype="float32")
    a = soxr.resample(a, sr, 16000)
    for seconds in [0.5, 1.0, 1.5, 2.0, 3.0, len(a) / 16000]:
        prefix = a[: int(seconds * 16000)]
        t = time.perf_counter()
        r = asr.generate(mx.array(prefix))
        mx.synchronize()
        report["prefix"].append(
            {
                "available_audio_s": len(prefix) / 16000,
                "compute_s": time.perf_counter() - t,
                "text": r.text,
            }
        )
        save()
    t = time.perf_counter()
    r = asr.generate(mx.zeros(32000))
    mx.synchronize()
    report["silence_asr"] = {"text": r.text, "seconds": time.perf_counter() - t}
    save()
    del asr
    gc.collect()
    mx.clear_cache()
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    vad = load_silero_vad(onnx=True)
    for i in [1, 2]:
        a, sr = sf.read(ROOT / f"phrase-{i}.wav", dtype="float32")
        a = soxr.resample(a, sr, 16000)
        a = np.pad(a, (16000, 16000))
        t = time.perf_counter()
        spans = get_speech_timestamps(
            torch.from_numpy(a), vad, sampling_rate=16000, return_seconds=True
        )
        report["vad"].append(
            {
                "fixture": i,
                "spans": spans,
                "compute_s": time.perf_counter() - t,
                "duration": len(a) / 16000,
                "padding_s": 1,
            }
        )
        save()
    report["silence_vad"] = get_speech_timestamps(
        torch.zeros(32000), vad, sampling_rate=16000, return_seconds=True
    )
    report["status"] = "completed"
    save()
except Exception:
    report["status"] = "failed"
    report["error"] = traceback.format_exc()
    save()
    raise
