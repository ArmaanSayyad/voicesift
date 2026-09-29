import os

os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json, time, traceback, resource
from pathlib import Path
import torch, laya

torch.set_num_threads(4)
result = {
    "scope": "six constructed development cases; not held-out validation",
    "device": "cpu",
    "cases": [],
}


def save():
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path("artifacts/laya-probe.json").write_text(json.dumps(result, indent=2))


try:
    Path("artifacts").mkdir(exist_ok=True)
    revision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
    result["revision"] = revision
    t = time.perf_counter()
    model = laya.load("convaiinnovations/laya", device="cpu", revision=revision)
    result["load_seconds"] = time.perf_counter() - t
    schema = {
        "repair": {
            "type": "choice",
            "instructions": "Judge whether the assistant response incorporates the user correction. Classify the response, not the original request.",
            "criteria": {
                "incorporated": "The final plan is Saturday, as corrected.",
                "contradicted": "The final plan remains Friday, contradicting the correction.",
                "not_established": "The assistant acknowledges or asks a question without establishing the final day.",
            },
        }
    }
    cases = [
        ("Saturday at seven for two people.", "incorporated"),
        ("Understood. Friday at seven for two people.", "contradicted"),
        ("Got it.", "not_established"),
        ("Not Friday anymore. Saturday at seven.", "incorporated"),
        ("Friday, as requested.", "contradicted"),
        ("Would you like Saturday or Sunday?", "not_established"),
    ]
    for response, expected in cases:
        state = {
            "original": "Dinner Friday at seven for two people.",
            "user_correction": "Actually Saturday. Keep everything else.",
            "assistant_response": response,
        }
        t = time.perf_counter()
        out = model.predict(state, schema)
        elapsed = time.perf_counter() - t
        result["cases"].append(
            {
                "response": response,
                "expected": expected,
                "seconds": elapsed,
                "answer": out["answers"]["repair"],
            }
        )
        save()
        print(result["cases"][-1], flush=True)
    result["status"] = "completed"
    save()
except Exception:
    result["status"] = "failed"
    result["error"] = traceback.format_exc()
    save()
    raise
