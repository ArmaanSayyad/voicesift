# Setup and troubleshooting

Start with the [README quickstart](../README.md#get-started). Commands assume a macOS/POSIX shell and the repository root. macOS is the tested platform; Windows-native instructions and Linux installation have not been verified. Dependencies require network access for installation. The frontend was checked with Node 24.21.0, pnpm 12.6.0 and its committed lockfile; the Python core environment uses pinned requirements.

## First use

The app requires real conversation audio and speaker-labeled transcripts with start/end timestamps in seconds. It does not transcribe or diarize arbitrary recordings. See the [dataset format](DATASET_RUNS.md#supported-sources) before preparing a ZIP. The downloadable example is intentionally silent and demonstrates structure only.

The built-in Hugging Face adapter accepts this source:

```text
https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev
```

This is a multi-gigabyte dataset, not a small demo. Checking it downloads/caches source files but does not call Gemini. A small ZIP of your own recordings is a lighter first run. Use data you are permitted to process with your configured provider.

For a gated Hugging Face source, accept its access terms on its dataset page, then authenticate locally before checking the source:

```sh
.venv/bin/hf auth login
```

The app uses local Hub authentication. Signing into the browser alone does not authenticate the Python downloader.

## Starting, stopping and updating

Launch from the repository root. Use the README's hidden key prompt, or set `GEMINI_API_KEY` / `GOOGLE_API_KEY` in the launching process and run:

```sh
.venv/bin/curation-serve
```

The app does not automatically load a `.env` file. No key is needed to check sources or inspect history, but a configured key with model access is required to curate. The model is currently fixed to `gemini-3.8-flash`; there is no UI model switch or automatic fallback.

Closing a browser tab leaves the worker running. **Pause** stops new work; in-flight requests may finish. **Ctrl+C** stops the server and drains in-flight work. Restart and use **Resume** for unfinished runs. Avoid starting a second server against the same artifacts directory.

To update, stop the server, back up `artifacts/interruption-curation/dataset-runs/`, then:

```sh
git pull --ff-only
uv pip sync --python .venv/bin/python requirements/core.txt
uv pip install --python .venv/bin/python --no-deps -e .
(cd curation-web && pnpm install --frozen-lockfile && pnpm build)
.venv/bin/curation-serve
```

The final command assumes a key is set in the current environment; otherwise use the README's hidden key prompt. Refresh the browser after updating. Do not delete `artifacts/` during upgrades. This beta has no automatic update or migration guarantee; older history has explicitly limited omission previews.

## Common problems

| Symptom | What to do |
|---|---|
| GitHub says repository not found | The repo is private. Obtain collaborator access and authenticate GitHub; clone the branch shown in the README. |
| `uv`, `node` or `pnpm` not found | Install the missing prerequisite using the README links, then open a new terminal. |
| Frontend build fails with a Node engine error | Use Node 24 and pnpm 12.6.0; reinstall with the committed lockfile. |
| The root page is missing or stale | Run `pnpm build` inside `curation-web`, start the server from the repo root, and refresh the page. |
| Address already in use | A server is already listening on port 8766. Use it or stop its terminal process before restarting. |
| Gemini is not configured | Enter the key through the startup prompt or set the environment variable before launching; `.env` is not read automatically. |
| Provider access error or all items fail | Confirm the key's project can use the fixed model. A configured key is not proof of model access. No automatic fallback is attempted. |
| Paused after rate limiting / HTTP 429 | Wait until quota is available, then Resume. Completed judgments are reused. Repeated clicking does not restore quota. |
| Unsupported schema / missing audio or timestamps | Use the documented manifest format or TurnBench adapter. Renaming an arbitrary dataset does not convert its schema. |
| Hugging Face access fails | Accept source access terms and run `.venv/bin/hf auth login`. |
| No examples selected | This can be a valid result. Inspect coverage, rejected/unresolved items and unproposed turns. Zero selections does not prove no interruptions exist. |
| Reviewed ZIP is empty | Only explicitly kept events are included. Notes, Unsure and unreviewed examples do not count. |
| Reviewed download is marked older | Prepare a new reviewed ZIP after changing decisions. The previous ZIP remains immutable. |
| Reviews changed in another window | Refresh run details, then apply your decision against the latest revision. |

For a reproducible bug, [open an issue](https://github.com/ArmaanSayyad/voicesift/issues) with your OS, Node/Python versions, commit (`git rev-parse --short HEAD`), the action, and the visible error. Include a small non-sensitive reproducer when possible; omit credentials and private recordings.

## Checks

```sh
.venv/bin/python -m pytest -q
(cd curation-web && pnpm build)
```

The workflow verification records 109 passing tests and browser checks. These test application behavior, not classifier accuracy. See [verification details](../evidence/curation-workflow/README.md) and the [selected experimental results](README.md).
