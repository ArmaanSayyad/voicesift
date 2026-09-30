# VoiceSift

VoiceSift uses **large audio-language models (LALMs)** to help curate voice datasets. The goal is to extract relevant examples from larger collections using **freeform text objectives**: describe what you want, review the matches, and export a curated dataset.

The current implementation is a minimal localhost app; no GPU, Docker or database setup is required.

<p>
  <img src="docs/images/voicesift-start.jpg" width="49%" alt="VoiceSift dataset upload, Hugging Face link and interruption requirement" />
  <img src="docs/images/voicesift-review.jpg" width="49%" alt="VoiceSift audio playback and Keep, Exclude or Unsure review controls" />
</p>

*Choose a source, then listen and review. Screenshots show a development run, not benchmark results.*

**Current beta:** the web app implements one fixed objective—conversations containing a successful interruption. Arbitrary freeform objectives are not yet supported in the UI or a general-purpose CLI. Other objectives have been tested through preset audio-only evaluation scripts. Gemini is the current model provider. Tested on macOS; other platforms are not yet verified. Review model suggestions before using them as training labels.

## Get started

You need Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), [Node.js](https://nodejs.org/en/download) 24 and pnpm, plus a [Gemini API key](https://ai.google.dev/gemini-api/docs/api-key) with access to `gemini-3.8-flash` and available quota. Python 3.12 is installed by uv if needed.

```sh
# If pnpm is not already installed:
npm install -g pnpm@12.6.0

git clone https://github.com/ArmaanSayyad/voicesift.git
cd voicesift
sh scripts/setup.sh core
(cd curation-web && pnpm install --frozen-lockfile && pnpm build)
```

Start the server; enter your key at the hidden prompt:

```sh
.venv/bin/python -c 'import getpass, os; os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key: "); from repair_bench.curation.api import main; main()'
```

Open **http://127.0.0.1:8766/**. Keep the terminal running. If `GEMINI_API_KEY` is already set in your environment, simply run `.venv/bin/curation-serve`. Stop with **Ctrl+C**; rerun either startup command to restart. No reinstall is needed.

## Try the interruption workflow

1. Upload a **ZIP containing `dataset.jsonl` and audio**, or paste a supported Hugging Face dataset URL. [Format and limits](docs/DATASET_RUNS.md#supported-sources). The app’s Example ZIP is a silent format template, not real speech.
2. Click **Check dataset** to download/validate it without model calls, then **Curate dataset** to start analysis.
3. Open its history entry. Listen, inspect transcripts, and mark examples **Keep / Exclude / Unsure**. Filters let you check rejected, unresolved and unproposed examples too.
4. **Prepare reviewed ZIP**, then download it. Only explicitly kept events receive positive annotations. **Model ZIP** downloads the original unverified selections.

Pause/resume, retry failures, undo reviews, search, archive/restore and rerun are available in history. Earlier export versions are preserved.

## Know before running

- **Current web inputs:** timed `assistant`/`user` transcripts and audio are required. Supports our manifest format and `mundo-ai/turn-benchmark-dev`; arbitrary Hub schemas and raw-audio transcription are unsupported. ZIP uploads: 512 MB; selected Hub files: up to 6 GB.
- **Privacy and cost:** curation sends audio excerpts and nearby transcripts to Gemini. Checking can download the full supported source. Paid analysis uses your API quota; there is no dollar spending cap.
- **Persistence:** history, audio, reviews and exports stay under `artifacts/interruption-curation/dataset-runs/`. Back up the whole directory. Deleting a run retains shared caches. Run one server process locally; this is not a hosted multi-user service.
- **Quality:** the interruption workflow uses overlap to propose candidates; overlap alone does not prove interruption. Other [experimental goals](docs/README.md) judge audio directly. Their benchmark results do not establish accuracy for arbitrary objectives.

[Setup help, gated datasets, updates and troubleshooting](docs/GETTING_STARTED.md) · [Full workflow and ZIP contents](docs/DATASET_RUNS.md) · [Verification](evidence/curation-workflow/README.md)

**License:** [MIT](LICENSE), copyright 2026 Armaan Sayyad. The license covers this project’s code; source datasets, recordings and third-party dependencies retain their own licenses.
