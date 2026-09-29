import json, time, urllib.request
from pathlib import Path

cases = [
    (
        "Friday at 19:00 for two. Change only the day to Saturday.",
        {"day": "Saturday", "time": "19:00", "party_size": 2},
    ),
    (
        "Saturday at 18:00 for four. Change only the time to 20:00.",
        {"day": "Saturday", "time": "20:00", "party_size": 4},
    ),
    (
        "Sunday at 12:00 for three. Change only the party size to five.",
        {"day": "Sunday", "time": "12:00", "party_size": 5},
    ),
]
report = {
    "model": "qwen3.5:9b",
    "scope": "constructed development cases, not benchmark",
    "cases": [],
}
for replicate in range(2):
    for text, expected in cases:
        body = {
            "model": report["model"],
            "messages": [
                {
                    "role": "user",
                    "content": "Return the final plan as JSON with day, time, party_size. "
                    + text,
                }
            ],
            "stream": True,
            "think": False,
            "format": {
                "type": "object",
                "properties": {
                    "day": {"type": "string"},
                    "time": {"type": "string"},
                    "party_size": {"type": "integer"},
                },
                "required": ["day", "time", "party_size"],
            },
            "options": {"temperature": 0, "num_predict": 128},
        }
        start = time.perf_counter()
        first = None
        chunks = []
        last = {}
        with urllib.request.urlopen(
            urllib.request.Request(
                "http://127.0.0.1:11434/api/chat",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=120,
        ) as r:
            for line in r:
                last = json.loads(line)
                s = last.get("message", {}).get("content", "")
                if s:
                    if first is None:
                        first = time.perf_counter() - start
                    chunks.append(s)
        actual = json.loads("".join(chunks))
        report["cases"].append(
            {
                "replicate": replicate,
                "input": text,
                "expected": expected,
                "actual": actual,
                "correct": actual == expected,
                "ttfc_s": first,
                "total_s": time.perf_counter() - start,
                "load_ns": last.get("load_duration"),
                "eval_count": last.get("eval_count"),
            }
        )
        Path("artifacts/ollama-probe.json").write_text(json.dumps(report, indent=2))
        print(report["cases"][-1], flush=True)
