# Larger interruption-curation evaluation

Date: 2026-09-29. Production classifier, prompt and model unchanged from the implemented workflow. No prompt optimization or selective reruns after observing results.

## Simple counts

- Submitted **400 constructed conversations**.
- The importer removed **19** duplicate text/timing records, retaining **381** conversations.
- **100** retained conversations have positive interruption-intent source labels; **281** have negative proxy labels.
- Flagged **111** as interruptions: **90** source positives and **21** false selections against the proxy labels.
- Missed **10** source positives in the interruption-selected set.
- Left **260** negatives unselected, including any unclear cases. Unselected is not necessarily a confident negative decision.

## How it was tested

The source pool is pinned SID-Bench English audio, revision `6eb13b573ad588646ce0de513d4255e55caf858b`. Deterministic seed 20261002 selected 100 positive and 200 negative clips, excluding all 140 source IDs in the earlier Gemini studies. These are new to Gemini evaluation, not necessarily unseen by earlier Laya experiments. The existing synthetic Kokoro agent track was reused, with 300 overlapping user clips and 100 matched handoff controls. Reference transcripts were supplied after stripping break markers; no ASR quality is measured in this experiment.

All 400 records were imported into one canonical corpus in batches of 100. Its existing text/timing deduplication retained 381 records, with **300 overlap candidates and 81 non-overlap controls**. Deduplication is applied before source WAV attachment, so distinct recordings with identical text and timing may collapse; this is a current product limitation, not independent evidence of duplicate audio. Nineteen nominal controls were removed. The preparation script does not change production deduplication to obtain a preferred denominator.

For bounded concurrency, the same 300 candidate analyses ran through four isolated application instances, each using the actual API importer, WAV uploader, clipper, background job service, cache, response validator, and Gemini backend. Production model inputs do not include local corpus IDs, gold labels, review annotations or source filenames. Each instance permits one active batch of up to 50 candidates. The final accounting uses the single canonical corpus membership, rather than summing partition-specific duplicate counts. No model calls were needed for excluded controls.

The first preflight check identified repeated control text/timing records. Reporting was corrected to use actual canonical importer membership; the model, prompt, overlapping examples and their labels were not altered. Some provider calls had already started when the preflight assertion was noticed. Results were consolidated from stored analyses without repeating provider requests.

No reviews were manufactured for this run: all records remain unreviewed and export is correctly refused. This is a selection evaluation, not a claim that a human reviewed a resulting training dataset. The production UI corpus was not replaced with these test records.

## Measurements

| Measure | Result |
|---|---:|
| Selection precision | 81.1% |
| Selection recall | 90.0% |
| False selections among overlapping negatives | 21/200 |
| API/analysis errors | 0 |
| Unclear source positives | 1 |
| Unclear source negatives | 11 |
| Truncated target clips | 0 |
| Estimated successful-response API cost | $0.3404 |

A fixed four-word reference-transcript baseline selected 86 of 100 positives and 5 negatives: 94.5% precision and 86.0% recall. This is a corpus-confounding diagnostic, not a proposed semantic production classifier. The model must demonstrate value beyond easy source-length separability on natural conversation data.

Unknown/error records remain available for manual review. If both positive suggestions and unclear/error records are retained, the review queue retains 91 positives and 32 negatives. This is review workload, not automatic selection precision.

## Decision after the larger run

The current Gemini classifier is not a clear winner over the simple baseline on this corpus. It gains four true selections in aggregate (90 versus 86) but adds sixteen false selections (21 versus 5). Its selection F1 is 85.3%, versus 90.1% for the four-word baseline. This does not make word count a reliable general interruption detector; it shows that the current dataset and classifier do not yet establish the value of audio reasoning. Keep the product review-assisted and do not claim autonomous curation quality from the earlier small result. Any prompt improvements should use a separate development set and be checked on fresh held-out conversations, not repeatedly optimized on this test.

The duplicate check also exposes a product limitation: transcript/timestamp equality can collapse distinct recordings before audio is attached. The 19 removed controls must not be counted as independently analyzed examples. A future importer should distinguish content similarity from audio identity, with an explicit policy, before production-scale curation.

## Interpretation limits

These are **constructed conversations with proxy source labels**, not independently human-labeled natural interruptions. The fixed artificial agent context may change whether a user utterance is directed at the agent. The controls share source audio with some overlap examples, acknowledgments repeat, speaker identities are not independently split, and the corpus is not a set of statistically independent conversations. Therefore no population confidence interval, natural-conversation accuracy, or agent-outcome accuracy is claimed.

All 300 overlaps are supplied-timestamp events. Finding them validates the interval contract, not acoustic segmentation or diarization. Natural timestamps can be noisy; this experiment does not test that. A larger sample reduces dependence on the earlier handful of examples but does not remove these design limitations.

Additional sources checked: the public Kyutai interactivity-alignment samples contain paired input/model audio, but do not provide the independently adjudicated interruption-event labels required to treat category membership as ground truth. The MagicLuke FDB model-output collection requires access approval and omits original stimuli. Neither was silently substituted into this measured result.

## Reproducibility

Scripts: `prepare_large_workflow_evaluation.py`, `evaluate_large_workflow.py`, `report_large_workflow.py`. The frozen protocol stores selected/excluded IDs, manifest and agent-audio hashes, production analysis source hash, model, prompt and schema. Source audio/transcripts stay in ignored artifacts; committed evidence contains hashes, labels, decisions and usage only. Existing tests remain 38/38 passing; new evaluation scripts pass Ruff.

Run from the repository root with the prior pinned SID annotations and synthetic agent track available, and the provider environment configured:

```sh
.venv/bin/python scripts/prepare_large_workflow_evaluation.py
.venv/bin/python scripts/evaluate_large_workflow.py
.venv/bin/python scripts/report_large_workflow.py
```

The evaluator resumes stored analyses and refuses a changed frozen protocol. Recorded errors are not automatically retried. API cost is estimated using observed token usage and previously verified standard Gemini 3.8 Flash rates, not a billing statement.

Sources: [SID-Bench](https://github.com/xkx-hub/SID-bench), [Kyutai audio samples](https://huggingface.co/datasets/kyutai/interactivity-alignment-samples), [FDB outputs](https://huggingface.co/datasets/MagicLuke/fdb-v1-outputs-v1).
