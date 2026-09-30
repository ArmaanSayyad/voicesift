# Minimal curation workflow verification

This change improves review and operation, not classifier accuracy. The Gemini model, interruption prompt and selection policy were not changed. No paid model calls were made for these checks.

## Automated checks

`python -m pytest -q`: 109 passing tests, including 13 new behavioral tests in `tests/test_curation_workflow.py`.

The new cases verify:

- Preflight validates audio/transcripts without calling the model, including when no key is configured.
- Reviewed ZIPs contain only explicitly kept events; exclude, unsure and unreviewed events cannot become positive annotations. Removing all kept events yields an empty dataset.
- Review revisions reject stale writes; undo/clear preserve original model ZIP bytes and earlier reviewed ZIPs. Reviews and exports survive manager recreation.
- Rejected and unproposed examples have inspectable audio and can be explicitly kept. Sample ordering is reproducible.
- Pause and HTTP 429 stop scheduling beyond the bounded in-flight batch. Resume preserves completed responses. A partial three-event run calls targets 1, 2, 3 initially and only target 2 on recovery.
- Shutdown drains only in-flight requests, saves partial state, and allows subsequent resume.
- Legacy history remains reviewable; source and export hashes detect corruption.
- Archive, rerun and deletion preserve independent run state and protect active work.
- New mutation routes require the local session token; invalid event IDs and review decisions are rejected.

Ruff checks and the React/TypeScript production build passed.

## Browser checks

An isolated localhost server on port 8767 used the same application with a controlled judge and a silent structural fixture. These are workflow checks, not acoustic or labeling ground truth.

Verified through browser controls:

1. Upload fixture ZIP, check compatibility, then start curation.
2. Open a history entry, activate audio playback and expand transcript context.
3. Keep an event, create and download a reviewed ZIP, then undo. The old ZIP remains available and is visibly marked as an older review version.
4. Retain an unsaved note while collapsing/reopening the entry; save it durably.
5. Filter rejected examples, show the empty state, and enable repeatable sample ordering.
6. Keep an event without collapsing expanded transcript context.
7. Archive/restore history, search for a nonexistent run, and create a new Ready run from the same validated source.
8. Inspect the form and expanded history at 390px viewport width: document width remained 390px, with no horizontal overflow. Restore the default viewport afterward.

Run deletion is covered by isolated automated tests; no existing user run was deleted. Existing production history is preserved. The real server remains a single local process on port 8766.

## Scope limits

Supported source formats and the interruption-only requirement remain unchanged. Source checking downloads and validates the full supported source, not arbitrary Hub schemas. Coverage counts and human-review counts are not accuracy estimates. Earlier run formats have limited omission previews, explicitly labeled in the UI. Shared source/model caches survive per-run deletion.
