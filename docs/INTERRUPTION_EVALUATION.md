# Laya interruption curation evaluation

Date: 2026-09-29. Decision: retain a generic decision-model interface, but do not use either tested Laya checkpoint to automatically discard interruption data. The deployed app still uses a deterministic timestamp detector; this evaluation runs Laya offline and does not imply it is integrated into the app.

## What the product should do

The requested first goal is to find a datapoint where an agent was interrupted. Separate two questions:

1. Did user speech begin during agent speech? Requires reliable speaker roles and timing (or equivalent logged events).
2. Was this an attempt to take the floor rather than a listening acknowledgment? Requires interpretation of speech and context; overlap alone cannot establish intent.

A transcript without timing cannot establish the first fact. A speech overlap can be a backchannel. Neither proves the agent stopped in response; that requires subsequent agent activity/event evidence. The dataset below evaluates the second question under an assumed speaking agent. It is not a natural-dialogue benchmark of the full first requirement.

The intended minimal architecture is **dataset adapter → evidence record → configurable decision model → review/export policy**. Evidence records hold conversation ID, roles, timestamps, transcript/context, source references, and optional precomputed features. Laya is a candidate decision backend. A versioned goal schema specifies the criterion and output labels. A policy separately decides what to prioritize for review or export. Missing evidence should produce an abstention, not a confident rejection. Expensive ASR is cached once per source, not repeated for each curation goal.

Simple timestamp arithmetic belongs in evidence preparation. The model should interpret semantic criteria, not be required to rediscover interval arithmetic. Adding correction or cancellation should reuse the same records and review/export path, changing the criterion and validation set. A decision backend must earn trust empirically; a binary API and returned probability do not establish reliability. Existing app behavior is a timing baseline, not the completed Laya-first product.

## Dataset and protocol

Sources: [SID-Bench repository](https://github.com/xkx-hub/SID-bench), [dataset](https://huggingface.co/datasets/kxxia/SID-bench), [paper](https://arxiv.org/abs/2603.24144). The authors describe human-recorded speech and labels distinguishing substantive interruption intent from backchannels. We use their labels, not independent human adjudication. The incoming clips lack paired agent timelines.

- English annotation set: 1,600 records. Five examples inspected for format were excluded: the first two records and first three negatives. Evaluation: **1,595 records, 1,098 positives, 497 negatives**.
- Git source revision: `0a9e5001e1fe534b3eec4f76c4455ae4bff9ccf1`.
- Hugging Face dataset revision: `6eb13b573ad588646ce0de513d4255e55caf858b`.
- Annotation SHA-256: `ceb18d41d2e014e225b3d729bd3a380514cb1fec911c9747325094138a104544`.
- Seed: `20260929`. Prompt and criteria fixed for the primary run. No parameter fitting or threshold optimization.
- Removed every `<break>` annotation and normalized whitespace. The marker and its position leak labels. Model input contains only cleaned user speech. No labels, filenames, durations, break times, or source identifiers reach Laya.
- A shuffled low-prevalence mixture uses 55 positives and 495 negatives from the same source. This is a mixture of records, not audio waveform splicing. Same-source backchannels are more relevant negatives than unrelated narration, though they still have length confounds.
- Fixed paired audio subset: 50 positives and 50 negatives, sampled before inference. Downloaded 100 WAVs totaling 8,250,700 bytes. Local Parakeet generated transcripts without reference transcripts or gold labels.
- Base Laya: `convaiinnovations/laya` at `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; published typed checkpoint: `convaiinnovations/laya-typed-decisions` at `1a793eb568e6718f15941d08f85432581df534e3`.
- Laya code revision: `6d942c92081fbc139e736bbd9ac0023223c29b7f`. CPU inference, four Torch threads. Parakeet: `mlx-community/parakeet-tdt-0.6b-v3`, revision `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`, local Metal. No paid inference or remote GPU.

## Results

Positive means interruption intent. Undefined precision means no positive predictions.

| Evaluation | TP | FP | TN | FN | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Laya, 1,595 reference transcripts | 4 | 0 | 497 | 1,094 | 100%* | 0.36% | 31.41% |
| Base Laya, 550-record 10%-positive mixture | 0 | 0 | 495 | 55 | Undefined | 0% | 90% |
| Base Laya, paired 100 reference transcripts | 1 | 0 | 50 | 49 | 100%* | 2% | 51% |
| Base Laya, same 100 through Parakeet | 0 | 0 | 50 | 50 | Undefined | 0% | 50% |
| Typed Laya, same 100 reference transcripts | 0 | 0 | 50 | 50 | Undefined | 0% | 50% |
| Fixed ≥4-word baseline, 1,595 references | 955 | 14 | 483 | 143 | 98.56% | 86.98% | 90.16% |
| Fixed ≥4-word baseline, 550-record mixture | 45 | 14 | 481 | 10 | 76.27% | 81.82% | 95.64% |

*Only four and one selections respectively: not evidence of dependable high precision. The 90% mixture accuracy is exactly the reject-everything baseline. Recall exposes the failure. The word-count baseline is a diagnostic of corpus separability, not a proposed production semantic classifier.

Base mean model-call latency was approximately 134 ms per reference example, excluding loading and ASR. Other work overlapped some runs; this is descriptive timing, not a controlled hardware comparison. Entire utterances were provided; these results do not measure streaming detection latency or SID-Bench's APT metric.

Reversing the option order on 200 fixed examples changed four decisions; accuracy remained 30% in both orders. This diagnostic was not selected as an improved configuration. It does not repair the failure.

Raw probabilities were unreliable: among 1,552 reference examples with maximum returned probability at least 0.8, average probability was **86.77%**, while accuracy was **32.02%**. Five-bin expected calibration error was 0.550; positive-class Brier score 0.515. These are descriptive values on this class distribution, not a fitted calibration or deployment guarantee. A confidence threshold cannot be justified from these outputs alone.

## Timing and actual platform checks

A separate constructed diagnostic paired 50 source transcripts with two invented agent/user timelines each: overlap and non-overlap. Laya received roles, texts, and timestamps, with a criterion explicitly asking only for overlap. It achieved TP=2, FP=2, TN=48, FN=48 (50% accuracy). The existing interval detector achieved TP=50, FP=0, TN=50, FN=0. This tests a simple supplied-evidence contract, not performance on naturally occurring interruptions or inferred diarization.

The actual corpus importer also processed 200 constructed conversations in two batches. Normalized duplicate text/timing handling reduced them to 154 unique conversations. It found all 77 unique overlaps, with no false candidates. This verifies ingestion, deduplication, and candidate detection, not acoustic accuracy. Identical speech content in overlap and handoff counterfactuals also demonstrates why text alone cannot establish observed interruption.

## Interpretation and limits

These results reject deployment of the tested zero-shot Laya configurations for automatic filtering. They do not prove every Laya prompt, training procedure, or future checkpoint will fail. Further prompt development must use a separate development set followed by fresh held-out evaluation; repeatedly optimizing this test would invalidate it.

The main evaluation measures semantic intent under a fixed assumption, whereas the product needs actual speaker/timing evidence. It does not establish performance on unseen speakers, actual company conversations, correction/cancellation, noise, multilingual data, or full-context dialogue. Complete transcripts may include information unavailable at the interruption onset. Dataset length and repeated-acknowledgment confounds are substantial; speaker/source dependence and possible training contamination are unknown. No independent human review or confidence intervals assuming independent samples are claimed.

Audio and source transcripts remain local ignored artifacts. Upstream redistribution/commercial rights were not established; the committed evidence contains hashes, predictions, protocols, and aggregates, not recordings or transcripts.

## Implementation decision

Keep the generic pipeline and Laya adapter boundary. Preserve a deterministic overlap feature for the first structural goal. Treat Laya scores as experimental review metadata until a separately validated configuration works. Do not silently replace failed semantic filtering with a length rule or export unreviewed examples as confirmed interruptions. Benchmark a stronger pluggable decision backend or task-adapted Laya on a development set, then evaluate against frozen held-out cases and human-recorded conversations with real role/timing evidence. No production classifier replacement was made in this evaluation.

## Reproduction

From the repository root, using the existing isolated environments and model manifest:

```sh
.venv/bin/python scripts/download_interruption_annotations.py
.venv/bin/python scripts/prepare_interruption_audio.py
.venv-laya/bin/python scripts/evaluate_interruption_laya.py
.venv-audio/bin/python scripts/transcribe_interruption_audio.py
.venv-laya/bin/python scripts/evaluate_interruption_laya.py --asr
.venv-laya/bin/python scripts/evaluate_interruption_laya.py --model convaiinnovations/laya-typed-decisions --revision 1a793eb568e6718f15941d08f85432581df534e3 --tag=-typed --subset-audio
PYTHONPATH=src .venv-laya/bin/python scripts/evaluate_laya_timing.py
.venv/bin/python scripts/summarize_interruption_eval.py
.venv/bin/python -m pytest -q
```

See feasibility documentation for isolated environment setup; `artifacts/models.json` points at locally downloaded models. Predictions resume for the same protocol; use a new tag for changed protocols. `evidence/interruption-eval/` preserves the evaluated outputs independently of future reruns. Validation: 31 tests pass; Ruff passes on the evaluation scripts and test.
