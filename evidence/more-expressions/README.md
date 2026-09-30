# Additional performed-expression evaluation

Six new binary audio curation goals: happy, surprised, fearful, disgust, calm and neutral. These are categories of acted vocal delivery, not judgments about a speaker's actual mental state. Calm and neutral are treated as distinct reference classes.

The test reuses the exact 240 normalized RAVDESS recordings from the earlier voice-lane evaluation: 30 per original class, including angry and sad examples as negatives. All 24 actors are represented. Each new goal therefore has 30 reference positives and 210 negatives. Each clip is sent three times with different requirement pairs, totaling 720 requests and 1,440 scored binary decisions. It is not 720 distinct recordings.

Pairs are happy/surprised, fearful/disgust, and calm/neutral. They use the existing audio-only judge, two goals per request, temperature zero, 2,048 output-token limit and six concurrent requests. The requests contain no transcript, source label, filename or actor metadata. Prompts and selection were frozen before the run. This is a follow-up on an already inspected benchmark sample, not an untouched holdout or a trained model comparison.

Source: [RAVDESS, Livingstone and Russo (2018)](https://zenodo.org/records/1188976), through [xbgoose/ravdess](https://huggingface.co/datasets/xbgoose/ravdess), pinned revision `a4a6c53ad083c4f16e92d1625e99113effe7569d`. Original license: CC BY-NC-SA 4.0; commercial use has separate licensing. Audio stays local and is not committed.

Reproduce from the repository root after core setup:

```sh
# Prepare the original frozen voice-lane sample if it is not already local; no API calls.
.venv/bin/python scripts/evaluate_voice_lanes.py
.venv/bin/python scripts/evaluate_more_expressions.py --run
# Optional uniform recovery pass, only if requests failed:
.venv/bin/python scripts/evaluate_more_expressions.py --retry-transport-errors
```

API credentials use the environment or an unechoed prompt. Data and responses are in ignored `artifacts/more-expressions/`. Source and normalized audio hashes are checked against the original frozen sample before running. Exact-request caches prevent rejudging completed answers. The optional recovery pass preserves initial failures and makes at most one additional attempt per transport/provider failure.

The engine reports unique audio clips separately from API requests. Accuracy counts only correct yes/no outcomes over all examples, with unclear/error outcomes counted as incorrect. Recall includes missed and unresolved positives. Precision uses only yes selections. Always rejecting yields 87.5% accuracy for each goal here, but zero recall. Speaker reuse and unknown model pretraining exposure limit generalization claims.

Each goal produces a local ZIP of selected, unverified audio clips. The output is not a timed conversation dataset for the existing interruption UI. See the complete follow-up report in [the research archive](../../research/archive/MORE_EXPRESSION_EVALUATION.md).
