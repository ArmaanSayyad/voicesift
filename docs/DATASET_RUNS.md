# Dataset-in, curated-ZIP-out

The default app at http://127.0.0.1:8766 now has one form and a run history. Choose a ZIP upload or a Hugging Face dataset URL, leave the fixed requirement `conversations where there is an interruption`, and submit. Other requirements are rejected by the API as well as being uneditable in the UI. No general-purpose natural-language planning is implemented yet.

The policy remains `successful-interruption-v1`: timed overlaps propose candidates; Gemini judges the audio and nearby transcript; only clear successful-interruption suggestions with no detected evidence truncation are shortlisted. A conversation is selected when at least one event is shortlisted. **Automatic ZIPs contain model-selected, unreviewed examples.** They are not labeled as human-confirmed. Review selected examples before using them as training labels.

## Supported sources

### Uploaded ZIP

`dataset.jsonl` must be at the root. Each line contains a conversation with a unique string ID, an audio path relative to the ZIP root, and speaker-labelled timed turns:

```json
{"id":"call-001","audio":"audio/call-001.wav","turns":[{"role":"assistant","text":"The first option is...","start_s":0.0,"end_s":3.0},{"role":"user","text":"Wait, let me clarify.","start_s":1.5,"end_s":4.0}]}
```

This illustrates the schema, not a verified positive example. Include the actual audio and original transcript. Optional metadata such as `source_group`, `split`, `provenance` and `timing_source` is retained. Default timing provenance is `transcript_segments`; timestamps are not independently validated against speech. Root LICENSE files and README.md are copied into the result's source notices.

User-on-assistant interruptions are searched for this format. Raw audio alone, untimed transcripts, arbitrary speaker names and arbitrary dataset schemas are not supported. Missing/malformed sources fail before paid classification begins. Mono/stereo audio readable by libsndfile is normalized to 16 kHz PCM16 WAV; selected conversations retain their entire supplied duration, not only the positive event.

### Hugging Face URL

Use the dataset page URL `https://huggingface.co/datasets/owner/name`, without a revision/file suffix. The loader supports repositories containing the same `dataset.jsonl` plus referenced audio, and a dedicated adapter for `mundo-ai/turn-benchmark-dev`. Unsupported schemas fail with a format message. No remote dataset code is executed.

The Hub resolves the current repository commit once, pins all file downloads to that revision, and caches the selected files locally. Only manifest-referenced audio and root license/README notices are downloaded for normalized datasets. TurnBench uses the FLAC audio embedded in its three parquet shards, plus the dataset notices. Annotator A supplies text/timing; event labels and other annotators are not passed to the model. Human speakers are mapped into both user/assistant directions when finding candidates, with original audio channel order preserved. Audio channel identity is not assumed by the model prompt.

Gated repositories use the existing local Hugging Face CLI authentication. Access terms must already be accepted by the user. Credential values are never copied into run records. Source licenses remain applicable to outputs.

## Persistence and execution

FastAPI serves the built React page and a background worker in one localhost process. No Docker, PostgreSQL, GPU, or extra service is needed for this single-user stack. Existing manual-review state remains in SQLite. New run state, cached model responses, source provenance, normalized audio, event decisions and immutable ZIPs persist under:

`artifacts/interruption-curation/dataset-runs/`

History is reconstructed from atomically replaced per-run status files. This implementation assumes **one server process**, not multiple Uvicorn workers. It allows one active dataset run, with up to four Gemini requests concurrently. Closing the browser does not stop the worker. A server restart marks unfinished runs interrupted; submitting again reuses exact-request caches. Automatic resumption, cancellation, deletion/retention controls, multi-user access and distributed workers are not implemented. Back up the whole artifacts directory to preserve history and downloads.

The upload is streamed to disk, ZIP extraction rejects traversal, symlinks, encryption and duplicate entries, and the API retains same-origin/session-token write protection. Audio hashes and final ZIP hashes are verified; exact model requests are cached independently of run identity. Failed model calls remain explicit errors, never negative labels. Completed runs with errors still offer a ZIP whose manifest lists those errors. Empty selections also produce a downloadable empty dataset and manifest.

Limits are explicit rather than silent truncation:

- Uploaded ZIP: 512 MB compressed, 2 GB expanded, 10,000 entries.
- Downloaded source files: 6 GB; normalized audio: 6 GB per run.
- Manifest: 20 MB, 1–500 conversations, 2–10,000 timed turns per conversation.
- Mono/stereo recordings: at most one hour each.
- Up to 10,000 detected candidate events per run. The full supported source is processed; no hidden sampling.

Costs scale with candidate count. The 10,000-event ceiling is a processing bound, not a dollar budget. Large runs can cost more than the small smoke tests.

## ZIP contents

Click a source name in History to expand its curated examples. The preview pages through 10 selected events at a time, with playable event excerpts, selection reasons, and nearby transcript context. Source timestamps and the selected turn are shown; shortened context is marked. Full conversations remain available in the ZIP. Empty and unfinished runs have explicit messages.

Each run has a feedback textbox and Save feedback button. Feedback persists in a separate `feedback.json` beside the run status, supports up to 10,000 characters, and can be edited or cleared. It is a run-level review note, not a new gold label: saving does not change the immutable ZIP, rerun curation, or retrain the model. Playback is opt-in. Only events in that run's exported selection can be played through the preview API, and clip hashes are checked.

- `dataset.jsonl`: full selected conversations, audio paths, and selection status/event IDs.
- `audio/`: full selected recordings normalized to 16 kHz PCM16 WAV.
- `clips/`: positive-event evidence clips, bounded by the existing analysis window.
- `selected-events.jsonl`: selected event results with source timing and evidence.
- `all-decisions.jsonl`: every proposed event's result, including errors and rejected suggestions.
- `manifest.json`: source revision/hash, policy, counts, coverage limitations and error IDs.
- `README.txt` and `source-notices/`: machine-label warning and source attribution/licenses.

`label_status` is explicitly `model_selected_not_human_verified`. No accuracy or purity claim is attached to these automatic exports.

## Validation

88 automated tests passed, including 22 dataset-workflow cases: source-to-ZIP, cache reuse, persistent history, replay after interrupted state, unsupported requirements, concurrent-submit rejection, missing audio, source traversal, streamed upload size limits, empty results, provider errors, immutable download hashes, pinned Hub file requests, TurnBench schema adaptation and gold-label exclusion. History-detail tests also exercise pagination, playable selected clips, invalid event access, feedback authorization/length validation, feedback persistence after recreating the run manager, and unchanged ZIP bytes after saving feedback.

The history-detail browser smoke test opened the existing selected-event run, played and paused its audio, expanded transcript context, saved a clearly marked workflow-test note, and confirmed the note survived a page reload. The empty-selection run displayed its empty state correctly. No new paid classification was needed for these UI checks.

The React/TypeScript production build passed. The actual page was exercised through browser upload, submit, history update and ZIP download. A two-clip real-audio smoke run processed five candidate events with zero errors and selected no conversations. A second smoke run used a full 150-second conversation window, processed seven events with zero errors, and selected one conversation via one event. The downloaded archive passed ZIP CRC checks and contained full audio, the event clip and source license. These are workflow smoke tests on previously inspected development data, **not accuracy measurements**.

Actual authenticated Hugging Face calls resolved TurnBench revision `c29aa4e6422122a8dccbe23598016a089bea2121` and five files totaling 4,215,901,555 bytes, reusing the existing local cache. The full 38-conversation dataset was not reclassified as part of this UI change. Unit tests separately cover the Hub-job integration and TurnBench adapter. Detailed runtime evidence is local in `artifacts/dataset-flow-check`; a text-free summary is committed under `evidence/dataset-runs`.

## Run locally

```sh
sh scripts/setup.sh core
(cd curation-web && npm install && npm run build)
# Configure GEMINI_API_KEY in the process environment, outside project files.
.venv/bin/curation-serve
```

Open http://127.0.0.1:8766. Existing artifacts survive restarts. To run tests: `.venv/bin/python -m pytest -q`.
