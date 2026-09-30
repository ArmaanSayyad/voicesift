# Gemini-assisted interruption curation: implementation and validation

> Historical development document. Its proposed scope and implementation state may be superseded. See [VoiceSift’s current purpose and capabilities](../../docs/README.md#project-scope-and-current-capabilities).

Built 2026-09-29. The local app at http://127.0.0.1:8766 now supports timed conversation import, source WAV attachment, Gemini background analysis, model-guided review, and JSONL/ZIP export. Laya is not in the default decision path.

## User workflow

1. Import JSONL conversations with roles, text, timestamps, source group, split, and provenance. Download the format example from the app. Timestamps and audio must share a clock. The current pilot requires known roles and supplied timing; it does not infer roles from arbitrary mixed recordings.
2. Attach the source WAV to a candidate. Mono and stereo WAVs up to 50 MB and 15 minutes are supported. Source audio is immutable after attachment. Supplied intervals generate candidates; user onset must be strictly inside an agent interval. Untimed turns remain in the corpus but are explicitly reported as unsearched.
3. Click **Analyze selected** or **Analyze pending audio (up to 50)**. The UI explains that source clips and nearby text are sent to Gemini. Each request receives up to 30 seconds of audio, including up to three seconds before onset and three after the user turn. Clips are resampled to 16 kHz with channels preserved. Truncation is recorded. Only nearby turn text, roles, timing and target location are included; source provenance, corpus labels and review annotations are not sent.
4. Inspect the model suggestion, evidence note, source timeline, original audio and extracted clip. Suggested floor-taking records appear first. Filters expose suggested interruptions, uncertain/error cases, unreviewed records, and included records. The model separately suggests interruption intent and observed agent outcome. These are review aids, not verified facts or calibrated confidence.
5. Save a human decision and observed agent outcome. Only an explicitly included review labeled interruption can enter an export. Model analysis never writes a review or sets inclusion. Unclear, failed, missing-audio, and backchannel cases remain accessible.
6. Download an immutable annotation JSONL or ZIP. The ZIP contains annotations, context WAV clips, and a clip manifest with source offsets and SHA-256 hashes. Source recordings remain intact. Old annotation exports retain their model suggestion and review snapshot even after later reviews.

## Implementation

- `curation/analysis.py`: evidence clipping, bounded Gemini requests, strict response parsing, sequential background jobs, content-addressed cache and failure isolation.
- SQLite additive tables retain jobs, immutable analysis history, and cached results. Existing datasets/reviews remain compatible. Review schema adds agent outcome with unknown as the default; export schema is now version 2.
- Jobs permit one active batch and 1–50 unique candidates. A restart marks abandoned jobs interrupted; rerunning a batch reuses matching successful results. Failures are retried only on an explicit subsequent run. There are no hidden automatic provider retries.
- Cache identity includes the audio bytes, input evidence, model identifier, prompt, and generation configuration. It cannot detect an upstream provider silently updating a fixed model identifier; a future model/prompt version should be changed deliberately to invalidate it.
- UI polls job progress, exposes errors, disables duplicate submissions and tracks human annotations independently. Cache hits avoid repeat provider calls. The $0.02-per-candidate UI figure is an estimate, not a billing cap. Requests are bounded to 1,024 output tokens and short audio/context windows.
- Model: `gemini-3.8-flash`. Existing provider configuration is server-side; the browser receives only a configured/not-configured flag. Start with `GEMINI_API_KEY` or `GOOGLE_API_KEY` configured in the server process, then run `.venv/bin/curation-serve`.

## Software verification

**38 automated tests pass.** Existing import atomicity, duplicate detection, split isolation, interval boundaries, review conflicts, source integrity, and export eligibility tests remain passing. New checks cover:

- actual API import → audio attachment → analysis → review → export;
- no automatic inclusion from model decisions;
- cache reuse without a second provider invocation;
- concurrency rejection and unique batch IDs;
- explicit provider errors and retry behavior without exposing provider error bodies;
- abandoned-job recovery;
- missing source audio and corrupted source hashes;
- 30-second clipping, truncation flags, 16 kHz output, and preserved stereo;
- schema-v2 annotations and ZIP clip manifests.

TypeScript/Vite production build succeeds. Browser smoke testing verified configured status, disabled analysis without audio, submitting a real Gemini request, running/completed progress, displaying an unclear result with evidence and audio, and saving an uncertain review without enabling export. One clearly marked constructed audio example remains in the UI. That smoke-test review is not human ground truth.

## Data-curation evidence: keep the two studies separate

### Earlier isolated-clip comparison

On 100 SID clips (50 positive, 50 negative), Gemini audio-only selected 48 positives and 8 negatives: **96% recall, 85.7% precision** at a 50% positive prevalence. Audio plus ASR had the same selection counts. This was interruption intent under an assumed speaking agent, not actual overlap detection. See `GEMINI_INTERRUPTION_EVALUATION.md` for the fixed protocol and all three arms.

### New integrated workflow experiment

Prepared 40 SID user clips not included in the earlier 100-clip Gemini experiment: 20 positive and 20 negative source labels. Each was placed over the same synthetic Kokoro agent utterance on a separate channel. Ten additional matched non-overlap controls were added. The user transcripts were cleaned reference transcripts, not ASR. Selection seed: 20261001. Dataset revision: `6eb13b573ad588646ce0de513d4255e55caf858b`. Kokoro uses the existing pinned `a71e4d38b236d968966a2002c4c895dbd12b1c3c` model and `af_heart` voice.

The actual importer, uploader, clipper, analysis jobs, real Gemini backend, reviewer endpoint and export packager were exercised in an isolated corpus. Results:

| Check | Observed result |
|---|---|
| Input conversations | 50 |
| Supplied-timestamp overlap candidates | 40/40 found |
| Non-overlap controls | 10/10 excluded |
| Gemini requests | 40 completed, zero errors |
| Source-positive clips selected as take_floor | 18/20 |
| Source-negative clips selected as take_floor | 1/20 |
| Selection precision / recall against source-label proxies | 94.7% / 90.0% |
| Unclear | 2 negative-source clips; retained for review |
| Cache rerun | 40/40 reused, with provider calls forbidden by test backend |
| Fixture review/export test | 20 expected records and 20 ZIP clips |
| Estimated successful-response API cost | $0.0524, excluding the separate UI smoke call |

The 20 exported fixture records were selected by explicit automated test annotations to check packaging. They are **not human-confirmed examples**, and the count is not the number Gemini selected. Model suggestion and fixture annotation remain separately recorded. Audio clips and source transcripts are not committed; source-free results and request hashes are under `evidence/curation-workflow/`.

The source labels are only proxies for the newly constructed conversations: inventing agent context can change the meaning of a user utterance. Thus **94.7%/90% is not a real-conversation accuracy claim**. Nor is it evidence of improvement over the earlier study: the sample, prompt, available context, transcript source, and audio composition differ. Two unclear negatives count as non-selections in the selection confusion matrix; counting abstentions as incorrect gives 35/40 strictly correct labels, rather than the 37/40 binary-selection accuracy. No calibrated probabilities or population confidence intervals are claimed.

A fixed four-word reference-transcript baseline selected 17/20 source positives and 1/20 negatives (94.4% precision, 85% recall). Gemini therefore recovered only one more positive at the same false-selection count on this sample; source-length confounding remains strong. Error inspection found one single-word filler selected as floor-taking, one short positive treated as a backchannel, and one substantive source-positive utterance interpreted as unrelated side speech against the invented agent context. That last judgment illustrates why labels cannot simply be transferred to synthetic conversations as unquestionable truth.

## What is ready, and what remains unproven

The review-assisted workflow is usable. It has demonstrated operational correctness and promising intent selection on limited data. Automatic acceptance or silent rejection is not justified. Candidate recall with inferred/noisy timestamps, speaker attribution on mixed audio, agent-outcome accuracy, natural full-dialogue semantic accuracy, and reviewer time saved remain unmeasured. The current strict-overlap detector excludes simultaneous onsets, boundary handoffs and untimed speech; those exclusions must be audited before broader deployment.

The next behavioral validation should use synchronized agent/user recordings or playback logs with human labels, including backchannels, attempted interruptions ignored by the agent, short objections, simultaneous starts, and ordinary handoffs. Keep conversation/source groups separate between development and test. Measure candidate-stage recall separately from classifier selection precision, report abstentions, and manually sample excluded conversations. Exported reviews are not automatically trustworthy merely because a reviewer identifier was supplied; this is a local pilot, not an authenticated multi-reviewer annotation service.

For the previously agreed human-recording workflow, capture a shared-clock stereo WAV (agent track on one channel, user on the other), retain verbatim transcripts, annotate speech intervals against the recording, import the matching JSONL and attach the WAV. Include both intentional cut-ins and supportive acknowledgments. This release supports upload/review of those recordings; it does not yet contain a microphone recording interface or automatic diarization.

## Reproduction

The existing pinned local Kokoro environment and SID annotation artifacts are prerequisites. Use a fresh evaluation corpus directory for a new fixture run; the script deliberately does not overwrite previous reviews.

```sh
.venv-audio/bin/python scripts/generate_workflow_agent.py
.venv/bin/python scripts/prepare_workflow_evaluation.py
# Configure GEMINI_API_KEY in this process before the next command.
.venv/bin/python scripts/evaluate_curation_workflow.py
.venv/bin/python -m pytest -q
cd curation-web
npm run build
```

Sources: [SID-Bench](https://github.com/xkx-hub/SID-bench), [Gemini audio documentation](https://ai.google.dev/gemini-api/docs/audio), [pricing](https://ai.google.dev/gemini-api/docs/pricing). Cost estimates use the observed token usage and published standard rates on 2026-09-29; they are not invoice verification.
