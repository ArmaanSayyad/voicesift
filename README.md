# Conversation Repair Workbench — feasibility harness

Local feasibility probes and a deterministic **simulator**, preceding the live workbench implementation. No paid API, remote GPU, microphone capture, or physical speaker playback is used here.

## Reproduce the deterministic harness

Requires `uv` and Python 3.12 (uv can obtain Python). From this directory:

```sh
sh scripts/setup.sh core
.venv/bin/python -m pytest -q
.venv/bin/repair-bench doctor
.venv/bin/repair-bench fixture artifacts/my-run
.venv/bin/repair-bench verify-bundle artifacts/my-run
```

Output directories must not already exist. The two-second fixture uses **tones, not speech**. It writes input/generated/simulated-rendered WAVs, a hash-linked JSONL event log and artifact manifest. Interruption occurs at sample 7,200 (300 ms at 24 kHz); an obsolete chunk is rejected at 7,680; repair begins at 12,000. Two identical runs produce identical artifact bytes on the tested environment. Inference is measured separately and is not claimed deterministic.

## Optional model probes

These download multiple gigabytes of free model weights. Tested only on Apple silicon macOS. Separate environments are intentional: Moshi uses a different MLX version from mlx-audio. Run probes sequentially for less contention. `requirements/` contains the exact installed snapshots, including dependencies and pinned Git source for Laya; model revisions are pinned by the download script. These snapshots are not cross-platform lock guarantees.

```sh
sh scripts/setup.sh audio
sh scripts/setup.sh moshi
sh scripts/setup.sh laya
.venv/bin/python scripts/download_models.py
.venv-audio/bin/python scripts/probe_audio.py
.venv/bin/python scripts/probe_ollama.py
.venv-laya/bin/python scripts/probe_laya.py
.venv-moshi/bin/python scripts/probe_moshi.py
.venv-audio/bin/python scripts/probe_cascade.py
```

Ollama probes require a running local Ollama server with `qwen3.5:9b` already downloaded; no server is started automatically. Audio probe must precede Moshi/cascade because it creates synthetic input WAVs. Probes overwrite their own ignored `artifacts/*-probe.json` files; copy prior results to preserve comparisons. No models, voices, personal audio, secrets, or environments are committed. `evidence/` stores small measured results from the original run.

## Boundaries that matter

- Virtual media samples are not host, AudioContext, microphone or DAC timestamps.
- Renderer calls consume their complete requested interval. The caller must split blocks at scheduled interventions; retrospective cancellation cannot retract already rendered samples.
- Generated audio, queued audio and simulated consumed audio have distinct ledgers. Hardware buffering and acoustic echo remain untested.
- Transcript revisions are visible only at their availability sample. Prefix re-transcription is not a streaming decoder.
- State patches check version/epoch and preserve unmentioned fields, but this skeleton has no application-specific semantic validator.
- `public_scenario` is an orchestrator view that removes top-level evaluator fields. It includes intervention scheduling and must **not** be given wholesale to a conversational model. Full controller context isolation remains application work.
- Hashes detect corruption relative to the manifest; they are not signatures against an adversary rewriting the entire bundle.
- The harness has no frontend, microphone workflow, C0/C1 production adapter, durable streaming event writer, or real-world behavioral benchmark yet.

Read [feasibility findings](docs/FEASIBILITY.md) for the implementation decision and its limits.

## Initial application slice

A local React evidence viewer and serialized C0 development runner now wrap the tested models. This slice accepts **two completed synthetic utterances**, saves generated speech, proposed plans, host-clock events and hashes, and exposes replay controls. It does not exercise live interruption, perform real bookings, or establish a benchmark result.

```sh
sh scripts/setup.sh core
# From web/: pnpm install --frozen-lockfile && pnpm build
.venv/bin/repair-bench-serve
# Open http://127.0.0.1:8765
```

The earlier model/fixture setup is required to start a real run. Run records live under ignored `artifacts/runs/`; they survive restarts, and unfinished attempts are marked interrupted. The API accepts one active run. Replay never silently starts microphone capture or audio playback. Stop the server with Ctrl-C after active runs complete. Local write endpoints require a per-process token and matching browser origin; this is not a remotely deployed service.

**Direction checkpoint:** before extending this into a general runtime, assess reuse of Pipecat Evals and Full-Duplex-Bench. Existing tools overlap strongly. The highest-value custom work is the repair-specific dataset, diagnosis and controlled experiment, not duplicating generic voice infrastructure. Laya remains unvalidated; an offline repair-event miner is a candidate to evaluate, not an implemented or proven capability. See `docs/PRODUCT_DIRECTION.md`.
