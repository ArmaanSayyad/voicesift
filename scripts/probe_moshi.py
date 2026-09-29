"""Offline native speech throughput, not physical playback or live turn-taking."""

import json, time, resource, traceback
from pathlib import Path
import numpy as np
import soundfile as sf
import mlx.core as mx
import mlx.nn as nn
import rustymimi, sentencepiece
from moshi_mlx import models, utils

root = Path("artifacts")
meta = json.loads((root / "models.json").read_text())["moshi"]
path = Path(meta["path"])
report = {
    "scope": "offline synthetic-input throughput; no microphone or speaker test",
    "revision": meta["revision"],
    "runs": [],
}


def save():
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["mlx_peak_bytes"] = mx.get_peak_memory()
    (root / "moshi-probe.json").write_text(json.dumps(report, indent=2))


try:
    mx.random.seed(42)
    cfg = models.config_v0_1()
    t = time.perf_counter()
    model = models.Lm(cfg)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, bits=4, group_size=32)
    model.load_weights(str(path / "model.q4.safetensors"), strict=True)
    model.warmup()
    mx.synchronize()
    report["load_and_warmup_s"] = time.perf_counter() - t
    save()
    tokenizer = sentencepiece.SentencePieceProcessor(str(next(path.glob("*.model"))))
    mimi_path = next(
        p for p in path.glob("*.safetensors") if p.name != "model.q4.safetensors"
    )
    phrase, sr = sf.read(root / "phrase-0.wav", dtype="float32")
    assert sr == 24000
    for seconds in [30, 60]:
        steps = int(seconds / 0.08)
        pcm = np.zeros(steps * 1920, dtype=np.float32)
        pcm[24000 : 24000 + len(phrase)] = phrase
        mimi = rustymimi.Tokenizer(str(mimi_path), num_codebooks=8)
        gen = models.LmGen(
            model=model,
            max_steps=steps,
            text_sampler=utils.Sampler(top_k=25, temp=0.8),
            audio_sampler=utils.Sampler(top_k=250, temp=0.8),
            check=False,
        )
        frames = []
        timings = []
        tokens = []
        arrivals = []
        t = time.perf_counter()
        for i in range(steps):
            ts = time.perf_counter()
            enc = mimi.encode_step(pcm[i * 1920 : (i + 1) * 1920].reshape(1, 1, 1920))
            enc = mx.array(enc).transpose(0, 2, 1)[:, :, : cfg.other_codebooks]
            text = gen.step(enc[0])
            token = text[0].item()
            if token not in (0, 3):
                tokens.append(tokenizer.id_to_piece(token).replace("▁", " "))
            audio = gen.last_audio_tokens()
            if audio is not None:
                frames.append(
                    mimi.decode_step(
                        np.array(audio[:, :, None]).astype(np.uint32)
                    ).reshape(-1)
                )
                arrivals.append(
                    {"input_frame": i, "wall_elapsed_s": time.perf_counter() - t}
                )
            mx.synchronize()
            timings.append(time.perf_counter() - ts)
        elapsed = time.perf_counter() - t
        out = np.concatenate(frames)
        sf.write(root / f"moshi-{seconds}s.wav", out, 24000)
        report["runs"].append(
            {
                "input_duration_s": seconds,
                "wall_s": elapsed,
                "rtf": elapsed / seconds,
                "frame_p50_s": float(np.median(timings)),
                "frame_p95_s": float(np.percentile(timings, 95)),
                "frames_over_80ms": sum(x > 0.08 for x in timings),
                "frames": steps,
                "output_duration_s": len(out) / 24000,
                "text": "".join(tokens),
                "first_audio_s": arrivals[0]["wall_elapsed_s"],
                "frame_times_s": timings,
                "audio_arrivals": arrivals,
            }
        )
        save()
        print(seconds, elapsed, elapsed / seconds, flush=True)
    report["status"] = "completed"
    save()
except Exception:
    report["status"] = "failed"
    report["error"] = traceback.format_exc()
    save()
    raise
