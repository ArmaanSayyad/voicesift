"""Resident ASR + local Ollama + TTS smoke probe. No endpointing/playback claim."""

import json, time, resource, traceback, urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf
import soxr
import mlx.core as mx
from mlx_audio.stt.utils import load
from mlx_audio.tts.utils import load_model

root = Path("artifacts")
paths = json.loads((root / "models.json").read_text())
report = {
    "scope": "sequential resident component-chain; synthetic completed utterances; no endpointing or speaker playback",
    "runs": [],
}


def save():
    report["peak_python_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["mlx_peak_bytes"] = mx.get_peak_memory()
    (root / "cascade-probe.json").write_text(json.dumps(report, indent=2))


def chat(text):
    schema = {
        "type": "object",
        "properties": {
            "day": {"type": "string"},
            "time": {"type": "string"},
            "party_size": {"type": "integer"},
        },
        "required": ["day", "time", "party_size"],
        "additionalProperties": False,
    }
    payload = {
        "model": "qwen3.5:9b",
        "think": False,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0, "num_predict": 128},
        "messages": [
            {
                "role": "system",
                "content": "Update the reservation from the user correction. Current state: Friday at 19:00 for two people. Preserve unmentioned fields. Return only JSON with day, time, party_size.",
            },
            {"role": "user", "content": text},
        ],
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        r = json.load(response)
    return json.loads(r["message"]["content"])


try:
    asr = load(paths["parakeet"]["path"], model_type="parakeet")
    tts = load_model(paths["kokoro"]["path"], model_type="kokoro")
    voice = str(Path(paths["kokoro"]["path"]) / "voices/af_heart.safetensors")
    for rep in range(3):
        a, sr = sf.read(root / "phrase-1.wav", dtype="float32")
        a = soxr.resample(a, sr, 16000)
        start = time.perf_counter()
        transcript = asr.generate(mx.array(a)).text
        mx.synchronize()
        t1 = time.perf_counter()
        state = chat(transcript)
        t2 = time.perf_counter()
        text = f"Dinner is now on {state['day']} at {state['time']} for {state['party_size']} people."
        chunks = []
        first = None
        for result in tts.generate(text=text, voice=voice, lang_code="a"):
            chunks.append(np.array(result.audio))
            mx.synchronize()
            if first is None:
                first = time.perf_counter() - start
        end = time.perf_counter()
        sf.write(
            root / f"cascade-{rep}.wav", np.concatenate(chunks), result.sample_rate
        )
        report["runs"].append(
            {
                "replicate": rep,
                "first_call": rep == 0,
                "transcript": transcript,
                "state": state,
                "correct": state
                == {"day": "Saturday", "time": "19:00", "party_size": 2},
                "asr_s": t1 - start,
                "llm_s": t2 - t1,
                "tts_s": end - t2,
                "first_audio_after_input_available_s": first,
                "total_s": end - start,
            }
        )
        save()
        print(report["runs"][-1], flush=True)
    report["status"] = "completed"
    save()
except Exception:
    report["status"] = "failed"
    report["error"] = traceback.format_exc()
    save()
    raise
