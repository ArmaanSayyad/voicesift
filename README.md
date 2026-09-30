# VoiceSift

A local tool for finding useful voice conversations inside a larger dataset. Upload a dataset ZIP or provide a supported Hugging Face link, run curation, inspect the selected examples, and download a curated ZIP.

The current app curates **conversations containing a successful interruption**. Its minimal black-and-white interface has a source input, a fixed requirement, a submit button, and persistent run history. Click a history entry to listen to selected event clips, read transcript context and selection reasons, or save feedback.

Selections are model judgments, not human-confirmed labels. The evaluation results below describe both strengths and limitations.

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

**Laya is not used in the current selection pipeline.** Its earlier probes and comparisons remain available as research evidence. Current classification uses Gemini; local model downloads are unnecessary for the curator.

## Measured results

These are development evaluations, not guarantees about an arbitrary uploaded dataset.

### Interruption curation: audio and transcript context

On 374 scored events in the TurnBench development evaluation:

| Outcome | Count |
|---|---:|
| Successful interruptions in the reference | 37 |
| Found | 22 |
| Missed | 15 |
| False positives | 17 |

That is **56.4% precision and 59.5% recall** on scored events. Unmatched events and model errors are documented in the [precision evaluation](docs/PRECISION_CURATION.md). The current curator should be used to generate review candidates, not assumed to produce a clean training dataset automatically.

### Correction/cancellation: separate text experiment

This goal is experimental and **not enabled in the app**. A frozen Gemini prompt evaluated 600 PRESTO examples using only dialogue text and withholding dataset labels from the model.

| PRESTO category | Tested | Selected | Not selected |
|---|---:|---:|---:|
| Within-turn correction | 100 | 98 | 2 |
| Argument correction | 100 | 88 | 12 |
| Action correction | 100 | 60 | 40 |
| Cancellation | 100 | 97 | 3 |
| Tagged-positive total | 400 | 343 | 57 |

The model also selected 19 of 200 comparison examples. Those examples lack reliable negative labels, so **precision and overall accuracy cannot be established from this run**. The 85.75% positive-tag recovery measures final-turn text detection, not audio understanding or dataset purity. See the [full results and error analysis](docs/PRESTO_EVALUATION.md) and [dataset suitability audit](docs/CORRECTION_DATASETS.md).

The next evaluation step is to define the correction/cancellation boundary precisely and independently adjudicate selected and rejected examples before claiming curation precision.

## Validation

```sh
.venv/bin/python -m pytest -q
(cd curation-web && pnpm build)
```

The current suite has **90 passing tests**. It covers source validation, safe ZIP handling, run persistence, model caching, export integrity, history pagination, audio access, feedback persistence and authorization, and evaluation label isolation. Browser smoke tests exercised upload/download, actual audio playback, transcript previews, saved feedback after reload, and empty selections. Workflow tests do not measure classifier accuracy.

To reproduce the PRESTO experiment after obtaining the official English test member:

```sh
.venv/bin/python scripts/evaluate_presto.py --source /path/to/test.jsonl --run
```

The script accepts the API key through the environment or an unechoed prompt. It verifies the source hash, freezes its sample and prompt, and caches responses. See [reproduction details and frozen evidence](evidence/presto-evaluation/README.md).

## Project reference

- [Dataset workflow, persistence and format](docs/DATASET_RUNS.md)
- [Interruption policy and evaluation](docs/PRECISION_CURATION.md)
- [Correction/cancellation evaluation](docs/PRESTO_EVALUATION.md)
- [Source dataset audit](docs/CORRECTION_DATASETS.md)
- [Earlier simulator and model probes](docs/EARLIER_WORK.md)

The repository retains earlier research for reproducibility. Local datasets, credentials, model weights and run artifacts are not committed. Source dataset licenses continue to apply to downloaded and curated material.
