# Conversation dataset curator

A local tool for finding useful voice conversations inside a larger dataset. Upload a dataset ZIP or provide a supported Hugging Face link, run curation, inspect the selected examples, and download a curated ZIP.

The current app curates **conversations containing a successful interruption**. Its minimal black-and-white interface has a source input, a fixed requirement, a submit button, and persistent run history. Click a history entry to listen to selected event clips, read transcript context and selection reasons, or save feedback.

Selections are model judgments, not human-confirmed labels. The selected experiment pages document their measured results and evaluation scope.

## Run locally

Prerequisites: `uv`, Node.js, `pnpm`, and a Gemini API key. The core setup uses Python 3.12; `uv` can obtain it. The application requires no GPU, Docker, or PostgreSQL.

Run these commands from the repository root:

```sh
sh scripts/setup.sh core
(cd curation-web && pnpm install --frozen-lockfile && pnpm build)
# Set GEMINI_API_KEY in your process environment, outside project files.
.venv/bin/curation-serve
```

Open **http://127.0.0.1:8766/**. Source audio and nearby transcript context are sent to Gemini during interruption classification. Downloaded source datasets, model caches, run artifacts and feedback stay on the local machine.

## Use the app

1. Upload a ZIP containing `dataset.jsonl` and its referenced audio, or paste a supported Hugging Face dataset URL.
2. Submit the read-only requirement: `conversations where there is an interruption`.
3. Wait for the background run. Closing the browser does not stop it.
4. Open the history entry to review selected event clips and transcript context, ten events per page. Save run-level feedback if needed.
5. Download the ZIP containing full selected conversations, audio, event clips, decisions and source notices.

Feedback is saved separately from the export. It does not change selections, retrain the model, or mark examples as verified. Empty selections still produce a ZIP with the run manifest.

### Supported datasets

- **Uploaded ZIP:** a root `dataset.jsonl` with unique conversation IDs, relative audio paths, and timed `assistant`/`user` turns, plus the referenced audio files.
- **Hugging Face:** repositories using that same manifest/audio format, and a dedicated adapter for `mundo-ai/turn-benchmark-dev`. Gated datasets use existing local Hugging Face authentication and require accepted access terms.

Arbitrary Hub schemas, raw audio without timed transcripts, and editable natural-language requirements are not supported yet. The current upload limit is 512 MB. See [input examples, limits and export format](docs/DATASET_RUNS.md).

## How selection works

Timed speech overlap proposes candidates. Gemini judges their audio and nearby transcript using the `successful-interruption-v1` policy. Only clear successful-interruption suggestions with no detected evidence truncation are shortlisted. A conversation is exported if it contains at least one shortlisted event.

Overlap alone does not establish an interruption. The classifier can confuse turn-taking, continuation and backchannels, and the pipeline can miss events absent from the supplied timing annotations. Exported labels are explicitly `model_selected_not_human_verified`.

The runtime is one FastAPI process serving the React UI, with one dataset run active at a time and up to four concurrent model requests. History, cached results, feedback and ZIPs persist under `artifacts/interruption-curation/dataset-runs/`; earlier manual-review records remain in SQLite. A server restart marks unfinished runs interrupted. Resubmitting can reuse exact-request caches. Large datasets can incur substantial API usage; the processing limits are not a spending cap.

**Laya is not used in the current selection pipeline.** Current classification uses Gemini; local model downloads are unnecessary for the curator.

## Selected experiment results

The [experiment docs](docs/README.md) describe selected audio-only curation goals and their final benchmark results. These are available through the evaluation CLI; the web app currently supports interruption curation only.

| Experiment | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| [Balance inquiries](docs/experiments/balance-inquiries.md) | 99.6% | 95.3% | 100% |
| [Card-freeze requests](docs/experiments/card-freeze-requests.md) | 100% | 100% | 100% |
| [Laughter](docs/experiments/laughter.md) | 97.7% | 93.9% | 92.0% |
| [Coughing](docs/experiments/coughing.md) | 94.0% | 79.6% | 86.0% |

Results measure agreement with reference labels on the documented datasets. They are selected examples of supported experimental goals, not accuracy claims for every curation requirement or the interruption UI. Review model-selected data before using it as training ground truth.

## Validation

```sh
.venv/bin/python -m pytest -q
(cd curation-web && pnpm build)
```

The current suite has **95 passing tests**. It covers source validation, safe ZIP handling, run persistence, model caching, export integrity, history pagination, audio access, feedback persistence and authorization, and evaluation label isolation. Browser smoke tests exercised upload/download, actual audio playback, transcript previews, saved feedback after reload, and empty selections. Workflow tests do not measure classifier accuracy.

## Documentation

- [Documentation and experiment index](docs/README.md)
- [Dataset workflow, persistence and format](docs/DATASET_RUNS.md)

Local datasets, credentials, model weights and run artifacts are not committed. Source dataset licenses continue to apply to downloaded and curated material.
