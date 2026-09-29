"""Fetch the pinned evaluation annotations; verify against the evaluated bytes."""

import hashlib
import urllib.request

from evaluate_interruption_laya import ROOT

REV = "6eb13b573ad588646ce0de513d4255e55caf858b"
EXPECTED = "ceb18d41d2e014e225b3d729bd3a380514cb1fec911c9747325094138a104544"
url = f"https://huggingface.co/datasets/kxxia/SID-bench/resolve/{REV}/test_wavs/en_test_lines.jsonl"
with urllib.request.urlopen(url, timeout=90) as response:
    data = response.read()
assert hashlib.sha256(data).hexdigest() == EXPECTED, "Unexpected annotation content"
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "en_test_lines.jsonl").write_bytes(data)
print("Verified pinned annotations")
