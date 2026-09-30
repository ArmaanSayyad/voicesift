# Freeform workflow verification

The web app accepts a selection objective, validates audio-only or timed-conversation sources, interprets freeform objectives into a saved rubric, evaluates each complete recording, and exports model-selected or explicitly human-kept data. The exact default interruption objective with timed transcripts continues through the existing precision workflow.

## Application checks

- **124 tests passed** with `.venv/bin/python -m pytest -q`.
- Frontend production build passed with `pnpm build` in `curation-web`.
- Ruff passed for changed Python modules and tests; `git diff --check` passed.
- Added contract tests cover model-free audio-only preflight, upload objective transport, objective validation, unsupported requests, three-way classification, hidden source labels, cache isolation across objectives, preserved plans on rerun, retry of malformed judgments, planner rate-limit recovery, complete-audio limits, playback, and reviewed export provenance.
- These tests use controlled answers and synthetic audio. They establish application behavior, not classifier accuracy.

## Live Gemini smoke test

Used 12 existing VocalSound recordings: the first two per class in source order, with opaque IDs and no source labels in the uploaded manifest or model requests. The objective was **“Recordings containing audible laughter or chuckling, but no coughing.”** Gemini generated criteria preserving both the laughter condition and the coughing exclusion.

Eight judgments completed: one laughter positive and seven negatives, all agreeing with the source class labels. Four remained unresolved after HTTP 429 responses, including the other laughter example. Completed decisions were retained across resumes; no failed request was silently made negative. Testing stopped rather than repeatedly retrying the persistent rate limit. See [record-level results](smoke-results.json).

This is a small, partial integration check on an already-used source, not a held-out evaluation or an accuracy claim for arbitrary objectives. Single-class source labels are also not exhaustive annotations for every co-occurring sound.

Browser verification on an isolated localhost server exercised editable objective input, audio-only ZIP upload, preflight, interpretation display, provider pause/resume, playback (readyState 4, no audio error), Keep review, reviewed ZIP creation and download. The downloaded ZIP's integrity, one kept recording, objective and rubric were verified. Editing the form objective after preflight returns the primary action to Check dataset; old runs keep their saved objective.

## Current boundaries

Freeform runs judge complete recordings, at most five minutes and 14 MB normalized WAV per recording. Supported sources remain the documented manifest ZIP/Hub layout and TurnBench adapter. No arbitrary Hub schema inference or automatic conversation splitting is claimed. Unsupported or ambiguous goals stop for clarification. Earlier preset benchmark scores do not validate this new automatic interpretation step.
