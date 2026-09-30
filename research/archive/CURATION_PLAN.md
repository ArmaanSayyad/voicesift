# Interruption curation — first release

> Historical development document. Its proposed scope and implementation state may be superseded. See [VoiceSift’s current purpose and capabilities](../../docs/README.md#project-scope-and-current-capabilities).

## Goal
Find datapoints where a user begins speaking while the agent is already speaking. A deterministic overlap detector finds candidates; a reviewer determines whether they are interruptions, backchannels, other overlaps, or uncertain. Correction/cancellation semantics and Laya integration are deferred. This supersedes the earlier semantic-mining first-release plan.

## User flow
1. Import JSONL with speaker-labelled turns and timestamps in seconds on one source-audio clock. Keep source_group, split and provenance. Untimed data is retained but not searched.
2. Inspect automatically detected overlap candidates and the agent/user interval plot. The detector merges overlapping or exactly adjacent spans for each speaker, so segmentation boundaries do not manufacture extra events. A candidate requires `agent_start < user_start < agent_end`; simultaneous starts, exact handoffs and agent-on-user overlaps are excluded.
3. Optionally attach the original mono/stereo WAV. Playback can start two seconds before the onset. No automatic microphone capture, ASR, diarization or remote fetching occurs.
4. Review context and source audio; label interruption/backchannel/other_overlap/uncertain. Inclusion is a separate explicit choice and allowed only for confirmed interruptions. Notes should state the evidence basis, especially without audio. Human annotations are not independently guaranteed correct.
5. Export an immutable JSONL snapshot of the latest included annotations. Each record preserves the full conversation, target indices, detector version, overlap interval, source hash, annotation version, and optional original-audio hash/reference. Audio bytes are not bundled; keep the original asset store when moving exports.

## Meaning and limits
The first operational definition is overlapping user onset during agent activity, not proof that the user took the floor or that the agent stopped. Backchannels also overlap. Turn timestamps that include pauses or generated-but-unplayed agent audio can produce false candidates. `timing_source` distinguishes speech intervals, transcript segments and manual annotations. Acoustic truth requires appropriate source annotations or listening.

This first goal measures candidate retrieval, not response quality, successful correction, cancellation, or model performance. It does not detect interruptions without overlap. No claim of recall or precision is made. Laya does not process audio and is not needed for this detector; its potential later role is classifying the semantics of reviewed/interruption candidates after task-specific evaluation.

## Input contract
Each JSONL line: id, source_group, split (`train/dev/test`), provenance, optional timing_source (`speech_intervals/transcript_segments/manual_annotations`), and turns. Each turn: role (`assistant/user`), text, optional paired start_s/end_s. Intervals must be positive-duration finite nonnegative values. Null timestamps are explicit missing data. Imported expected answers are rejected by the schema.

Limits for the local pilot: 2 MB JSONL body, 100 conversations and 2,000 turns per import, at most 1,000 overlap candidates in the corpus; exceeding a limit rejects the batch atomically. WAV attachments: 50 MB, 15 minutes, one/two channels, duration covering the annotated source clock. Attachments are immutable. No arbitrary local file paths or remote URLs are dereferenced.

Imports are transactional. Exact normalized-text-and-timing duplicates are skipped; near-duplicates are not automatically detected. Source groups and exact duplicates cannot cross splits. Users must assign stable source groups to related conversations. Reviews are append-only with optimistic version checks. Export hashes are checked at download. The application binds to loopback with same-origin write checks and a per-process token.

## Delivered and tested
- Separate curation app at port 8766; old workbench remains preserved.
- Import, timing detector, queue, source context, interval plot, WAV attachment/playback, review, immutable JSONL export and corpus counts.
- 30 total tests pass, including the earlier harness/application tests. New tests cover boundaries, direction, missing timing, segmentation invariance, backchannel candidates, transactional import/split rejection, review conflicts, export eligibility, audio attachment immutability and download integrity.
- Browser smoke: three constructed conversations imported; two candidates; clean handoff excluded. An automated UI test saved an uncertain review with its provenance explicitly stated; no example was admitted as human-validated training data.
- Frontend type check and production build pass. No natural conversation corpus or human-labelled evaluation has been run yet.

## Next steps
1. Exercise this flow with a small real, licensed or user-owned timestamped dialogue dataset.
2. Audit onset/offset accuracy and adjudicate a held-out sample, including backchannels and segmentation mistakes.
3. Add scalable dataset adapters and raw/channel-separated audio preprocessing only as needed.
4. Add Laya as an optional semantic classifier for correction/cancellation. Evaluate against rules/local Qwen; retain it only if useful. Preserve the separation between detection, human annotations and model suggestions.
