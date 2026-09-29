import json
from pathlib import Path
from huggingface_hub import snapshot_download

models = [
    (
        "moshi",
        "kyutai/moshika-mlx-q4",
        "ce562a51ae571c1a7b70e8230b9af53ecc0bee5b",
        ["*.json", "*.safetensors", "*.model", "README.md"],
    ),
    (
        "parakeet",
        "mlx-community/parakeet-tdt-0.6b-v3",
        "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
        ["*.json", "*.safetensors", "*.model", "README.md"],
    ),
    (
        "kokoro",
        "mlx-community/Kokoro-82M-bf16",
        "a71e4d38b236d968966a2002c4c895dbd12b1c3c",
        ["*.json", "*.safetensors", "voices/*", "*.txt", "README.md"],
    ),
]
Path("artifacts").mkdir(exist_ok=True)
paths = {}
for name, repo, rev, patterns in models:
    print("Downloading", name, flush=True)
    path = snapshot_download(repo, revision=rev, allow_patterns=patterns, max_workers=3)
    paths[name] = {"repo": repo, "revision": rev, "path": path}
    Path("artifacts/models.json").write_text(json.dumps(paths, indent=2))
    print("Ready", name, flush=True)
