# Dataset-in, curated-ZIP-out

The default app at http://127.0.0.1:8766 now has one form and a run history. Choose a ZIP upload or a Hugging Face dataset URL, leave the fixed requirement `conversations where there is an interruption`, and check the dataset. After validation, click **Curate dataset**. Other requirements are rejected by the API as well as being uneditable in the UI. No general-purpose natural-language planning is implemented yet.

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

History is reconstructed from atomically replaced per-run status files. This implementation assumes **one server process**, not multiple Uvicorn workers. It allows one active dataset run, with up to four Gemini requests concurrently. Closing the browser does not stop the worker. A server restart marks unfinished runs interrupted; **Resume** reuses the frozen prepared source and completed event records. Exact-request caches are also retained. Resumption is explicit. Multi-user access and distributed workers are not implemented. Back up the whole artifacts directory to preserve history and downloads.

The upload is streamed to disk, ZIP extraction rejects traversal, symlinks, encryption and duplicate entries, and the API retains same-origin/session-token write protection. Audio hashes and final ZIP hashes are verified; exact model requests are cached independently of run identity. Failed model calls remain explicit errors, never negative labels. Completed runs with errors still offer a ZIP whose manifest lists those errors. Empty selections also produce a downloadable empty dataset and manifest.

Limits are explicit rather than silent truncation:

- Uploaded ZIP: 512 MB compressed, 2 GB expanded, 10,000 entries.
- Downloaded source files: 6 GB; normalized audio: 6 GB per run.
- Manifest: 20 MB, 1–500 conversations, 2–10,000 timed turns per conversation.
- Mono/stereo recordings: at most one hour each.
- Up to 10,000 detected candidate events per run. The full supported source is processed; no hidden sampling.

Costs scale with candidate count. The 10,000-event ceiling is a processing bound, not a dollar budget. Large runs can cost more than the small smoke tests.

## Minimal workflow and review

**Check dataset** creates a persistent run and downloads/extracts and validates the full supported source. It makes no model calls and works without a Gemini key. A compatible run stops at Ready, with conversation and candidate counts. **Curate dataset** starts paid classification. This is full validation, not a cheap remote metadata preview; checking a large Hugging Face source can download gigabytes. Unsupported schemas are rejected with an explanation rather than guessed column mappings. The Format link and Example ZIP show the expected structure; example audio is silence, not an interruption example.

Open a history row for:

- **Selected:** model-shortlisted events.
- **Not selected:** candidates the model rejected.
- **Unresolved:** pending, failed or uncertain candidates; these are not negative labels.
- **Not proposed:** incoming turns outside the detector's candidate groups. They did not receive a model judgment.
- **All examples:** these categories together. The unit is an event or incoming turn, not a distinct conversation.

Each page has at most ten examples, source-relative audio previews, model reasons where available and expandable transcript context. Sample order uses a fixed hash ordering per run, so paging is reproducible. It is for spot checks, not a statistically justified accuracy estimate. Coverage lists timed/untimed incoming turns, overlap candidates and unproposed turns. TurnBench searches both speaker directions; checking only rejected overlaps cannot establish full-dataset recall.

**Keep**, **Exclude** and **Unsure** persist a review decision per example. **Clear** removes its effective review; **Undo review** restores the previous choice from the last action in the current detail view. Reviews have monotonically increasing revisions and reject stale writes from another tab. The audit history retains changes. A reviewer can keep a rejected or unproposed example after listening; this does not change the original model decision.

**Prepare reviewed ZIP** snapshots the current review revision. Only Keep events receive positive annotations; unreviewed, excluded and unsure events are omitted. A conversation is included if it has at least one kept event. Full conversations may contain other speech that is not labeled positive. Reviewed ZIPs have their own immutable versions; changes to reviews mark older downloads as out of date without changing their bytes. An all-excluded or cleared review can produce an empty reviewed dataset. Creating a reviewed export waits until processing has stopped.

Run notes remain separate: they do not change labels or retrain anything. An unsaved note draft survives collapsing the run within the same browser tab; save it for durable server persistence.

## Recovery and history

- **Pause** stops scheduling new work. At most four model calls are in flight and may finish. During source preparation, pause takes effect at a progress checkpoint; an in-progress file download or audio normalization may finish first.
- **Resume** starts a Ready run or continues Paused, Interrupted or Failed runs. **Retry failed items** retries analysis failures. Completed event judgments are not re-requested.
- HTTP 429 and provider access errors stop further scheduling. Resume is manual, after quota/access is available; the app does not promise a reset time or automatically switch models.
- Partial ZIP manifests distinguish unresolved and pending items. Resuming creates a new model ZIP version; earlier versions remain downloadable from run details.
- **Run again from same source** creates a new Ready run with independent reviews and the same frozen validated data. Identical model requests reuse cache; this is not an independent fresh-model benchmark.
- History search matches source, requirement and run ID. Archive hides a run from the main list; Archived shows it again. Archive/restore preserve all files.
- **Delete local run** requires confirmation in the UI and removes that run's normalized audio, reviews, input copy and exports. Shared Hugging Face downloads and model caches remain because other runs may use them. Active runs cannot be archived or deleted.

Prepared snapshots pin source metadata, normalized audio hashes, candidate records, model and policy version. Resuming a snapshot from a different model/policy is rejected. This implementation remains a single-process local worker.

Existing older runs remain inspectable and reviewable without migration, but their available previews are limited to conversations retained in the original ZIP. Submit the source again for complete omission inspection. Older uploads can be re-prepared when their input ZIP remains available; older Hub jobs without resume metadata must be submitted again.

## ZIP contents

Model exports contain:

- `dataset.jsonl`: full selected conversations with audio paths and selected event IDs.
- `audio/`: full selected recordings normalized to 16 kHz PCM16 WAV.
- `clips/`: event evidence excerpts.
- `selected-events.jsonl`: model-selected events, evidence and timing.
- `all-decisions.jsonl`: every proposed event, including failures and pending items.
- `manifest.json`: provenance, policy, coverage, completion and error IDs.
- `README.txt` and `source-notices/`: interpretation and original source notices.

Reviewed exports have the same dataset/audio/clip structure but `selected-events.jsonl` contains only human-kept annotations (`human_reviewed_keep`). `review-audit.json` records the review revision/history. They intentionally omit `all-decisions.jsonl` so excluded model-positive judgments cannot be confused with reviewed positive annotations; original model ZIPs retain those decisions.

All export downloads verify their saved hash. Original source licenses remain applicable. Local data is never committed to Git.

## Validation

The suite has **109 passing tests**. The workflow tests use controlled model responses and synthetic audio; they check behavior, not acoustic classification accuracy. Coverage includes model-free preflight, source validation, pinned Hub download revisions, safe ZIP extraction, exact-result caching, bounded pause, HTTP 429 handling, retry-only-failures, restart persistence, error accounting, rejected/unproposed previews, review revision conflicts, kept-only exports, immutable earlier ZIPs, legacy history, archive/rerun/delete, source/export integrity, and write authorization.

Browser verification uses an isolated server with a controlled judge. It checks upload → compatibility → curate → listen → review → reviewed ZIP → undo, notes, sample/filter controls and history management. No classifier accuracy improvement is claimed by these UI changes; the interruption policy and model prompts are unchanged.

## Run locally

```sh
sh scripts/setup.sh core
(cd curation-web && npm install && npm run build)
# Configure GEMINI_API_KEY in the process environment, outside project files.
.venv/bin/curation-serve
```

Open http://127.0.0.1:8766. Existing artifacts survive restarts. To run tests: `.venv/bin/python -m pytest -q`.
