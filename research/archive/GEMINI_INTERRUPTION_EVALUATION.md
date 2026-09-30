# Gemini interruption-intent comparison

Date: 2026-09-29. Model: Gemini 3.8 Flash. This is an offline matched input-modality experiment, not an integrated UI feature or validation of naturally observed agent interruptions.

## Fixed protocol

The same 100 SID-Bench clips (50 positive, 50 negative) used in the earlier Laya audio evaluation were sent to Gemini in three conditions: local Parakeet transcript only; original WAV only; WAV plus the same Parakeet transcript. No reference transcript, label, filename, break marker, break time, or duration field was sent. There is no paired agent timeline in these clips, so no timing evidence was invented or supplied. The prompt explicitly assumes an agent is speaking and asks about incoming speech intent.

One example across all three arms verified API/schema compatibility, then the remaining 297 requests ran with the identical prompt and generation configuration. No prompt tuning followed model results. Requests used temperature 0, a 1,024 output-token cap, structured labels, and at most four concurrent calls. Requests were independently sent with no shared chat context. The model can answer unclear; failures and unclear records are retained for review. This new experiment is exploratory because the subset and dataset behavior were already studied in the Laya evaluation.

## Results

| Input | TP | FP | TN | FN | Precision | Recall | Strict correct / 100 | Unclear | API/parse errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transcript | 47 | 9 | 41 | 3 | 83.9% | 94.0% | 86 | 3 | 0 |
| audio | 48 | 8 | 42 | 2 | 85.7% | 96.0% | 89 | 1 | 0 |
| audio_transcript | 48 | 8 | 42 | 2 | 85.7% | 96.0% | 88 | 2 | 0 |

Confusion matrices measure selection as interruption versus everything else. Unclear/errors are not confident negative labels: strict-correct counts treat them as incorrect regardless of gold class. Separate retain-for-review metrics are in summary.json.

## Paired differences

- audio: corrected 4 transcript-only errors and introduced 1 errors on cases the transcript arm got right.
- audio_transcript: corrected 2 transcript-only errors and introduced 0 errors on cases the transcript arm got right.

## Latency and estimated cost

| Input | Mean API seconds | Median | P95 | Estimated USD |
|---|---:|---:|---:|---:|
| transcript | 1.63 | 1.33 | 3.75 | $0.0528 |
| audio | 1.80 | 1.65 | 3.04 | $0.0484 |
| audio_transcript | 2.37 | 1.69 | 3.58 | $0.0517 |

Total estimated successful-response API cost: **$0.1530**. Uses published standard rates on the evaluation date: $0.75/million input tokens and $3.75/million output tokens including thinking. This is a token-accounting estimate, not billing verification. Failed requests can have unreported charges. API timings include network latency and concurrent service execution; ASR preprocessing time is excluded. No provider confidence score was requested or treated as calibrated.

## Limits and next implementation step

This sample cannot establish performance on full conversations, accurate overlap detection, speaker-role attribution, noise, new languages, or the intended deployment prevalence. Source utterances include repeated acknowledgments and length confounds; examples are not known to be statistically independent. No significance or population-generalization claim is made from small accuracy differences. No model was trained. Full-utterance audio may contain evidence unavailable at interruption onset; this is offline curation, not streaming interruption detection.

Keep role/timing candidate generation separate from semantic intent. For product integration, supply actual conversation context and evidence provenance, keep model and prompt versions on each decision, present audio/transcript evidence for reviewer confirmation, and export reviewed events. Do not make permanent data deletion or automatic rejection depend on this small study. The next validation set should contain real paired agent/user tracks or synchronized playback logs plus human-reviewed labels, split by conversation/source. Compare candidate recall and semantic precision separately, and audit excluded records.

Existing UI filtering remains the timestamp baseline; this commit provides the tested Gemini evaluation backend and evidence, not an automatic production filter.

## Reproduction and sources

With the prior audio/ASR artifacts prepared and GEMINI_API_KEY or GOOGLE_API_KEY configured in the process environment:

```sh
.venv/bin/python scripts/evaluate_interruption_gemini.py
.venv/bin/python scripts/summarize_gemini_evaluation.py
.venv/bin/python -m pytest -q
```

The runner resumes exactly matching protocols without repeating completed calls. API errors are recorded without automatic retries. Evidence retains only decisions, IDs, hashes, token counts, timing, and model metadata; source speech is not included.

- [Audio API documentation](https://ai.google.dev/gemini-api/docs/audio)
- [Standard model pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [SID-Bench](https://github.com/xkx-hub/SID-bench)
- Dataset revision: `6eb13b573ad588646ce0de513d4255e55caf858b`.
- Exact prompt/configuration and source artifact hashes: `evidence/gemini-interruption-eval/protocol.json`.
- Earlier Laya protocol and results: `research/archive/INTERRUPTION_EVALUATION.md`.

## Decision after this run

Gemini is a credible candidate for the semantic review stage: it recovers most positives on this subset, unlike the tested Laya configurations. Audio-only has 89 strictly correct decisions versus 86 for ASR-only; this small difference is not evidence of general superiority. Audio plus ASR does not improve interruption selection over audio alone here. Use the audio backend when audio is available, retain transcript-only as an alternative, and keep both experimental until validated on actual conversation events.

Audio-only still selects 8 of 50 negatives. If its measured 96% sensitivity and 16% false-positive rate held at a 10% positive deployment prevalence, expected selection precision would be only 40%; this is a hypothetical prevalence projection, not a measured deployment result. Human review remains necessary. For comparable historical context, base Laya found 0 of 50 positives on the same local-ASR subset, but the two systems have different prompts and output interfaces, so this is a pipeline comparison rather than an isolated architecture comparison.

Validation: 34 tests pass; Ruff checks pass for the new evaluation scripts and tests.
