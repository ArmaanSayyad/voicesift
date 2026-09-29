"""Generate the fixed artificial agent track using the existing pinned Kokoro setup."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

path = json.loads(Path("artifacts/models.json").read_text())["kokoro"]["path"]
model = load_model(path, model_type="kokoro")
text = "Let me explain the available options. There are several things to consider before we decide what to do next. We can go through each option together and choose the one that works best for you."
chunks = []
for result in model.generate(
    text=text, voice=str(Path(path) / "voices/af_heart.safetensors"), lang_code="a"
):
    chunks.append(np.array(result.audio))
    rate = result.sample_rate
out = Path("artifacts/workflow-eval")
out.mkdir(exist_ok=True)
sf.write(out / "agent.wav", np.concatenate(chunks), rate)
