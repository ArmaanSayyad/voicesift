# TurnBench interruption-curation evaluation

## Results

The current curator is **not ready for automatic interruption curation**. It retrieves most successful interruptions but admits too many ordinary overlapping turns. Gemini gives only a small precision improvement over a cheap four-word rule, at lower recall.

### Conversation-level result

Of 38 original conversations, 33 contain at least one consensus successful interruption and five do not. The model flagged **all 38**: 33 true-positive conversations, five false-positive conversations, and zero missed positive conversations. This is poor discrimination despite the apparent 100% conversation recall. If unsuccessful attempts count, 37 are positive and the remaining one is also falsely flagged.

### Event-level result

The model analyzed 4,216 candidates and flagged 1,856. Of those flags, 1,723 match consensus events and 133 have no matching consensus event. Those 133 are unscored, not silently treated as correct or false. Across all candidates, 229 lack a consensus match.

| Ground-truth definition | Actual positives | Found | Missed | Known false flags | Precision among scored flags | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Successful interruption only | 347 | 341 | 6 | 1,382 | 19.8% | 98.3% |
| Successful or unsuccessful attempt | 706 | 593 | 113 | 1,130 | 34.4% | 84.0% |
| Successful, excluding attempts from scoring | 347 | 341 | 6 | 1,130 | 23.2% | 98.3% |

The six missed successful interruptions comprise two that never became timing candidates and four whose model responses failed. There were 91 model errors total (82 incomplete responses, nine timeouts) and 15 unclear judgments. They are counted as not automatically selected, but remain reviewable. Retaining errors and unclear judgments in the review queue raises successful-interruption coverage to 345/347 and attempt-inclusive coverage to 606/706, with more false alarms.

### Baselines on the same consensus events

| Selector | Attempts found / 706 | Attempts missed | False flags | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Timing only | 695 | 11 | 3,292 | 17.4% | 98.4% |
| At least four words | 606 | 100 | 1,263 | 32.4% | 85.8% |
| Gemini audio + transcript | 593 | 113 | 1,130 | 34.4% | 84.0% |

Gemini removes a net 133 false alarms compared with the word rule, but finds 13 fewer positive events. Its absolute precision gain is about two percentage points. This result does not justify calling the model filter substantially better. On successful interruptions alone, the word rule finds 345/347, versus Gemini's 341/347.

A 2,000-resample conversation-cluster bootstrap gives Gemini attempt-inclusive precision 27.0–41.4% and recall 79.7–88.2% (95% percentile intervals). Word-rule intervals are 24.9–39.4% and 82.7–88.7%. These are uncertainty intervals for this small corpus, not evidence of general production accuracy. Matching tolerances of 100, 200, and 300 ms produce the same TP/FP/FN counts here.

### What fails

Of the 1,130 false flags under the attempt-inclusive definition, **1,084 are consensus ordinary turns**, 44 backchannels, and two non-content events. The model is reasonably selective about acknowledgements (44/1,673 candidate backchannels flagged), but confuses ordinary overlapping floor transitions with interruptions (1,084/1,255 candidate ordinary turns flagged).

A secondary annotation audit found that 215 of those 1,084 ordinary-turn false flags occur while the opposing annotator-A segment is a backchannel. This supports a specific failure hypothesis: the pipeline treats any opposing speech as evidence that the opposing person holds the conversational floor. A speaker continuing over a listener's acknowledgement can consequently look like a new interruption. This is annotation-based analysis, not a new human listening adjudication. Annotator A sometimes disagrees with the consensus; the aggregate consensus remains the reference.

Attempt-inclusive precision varies by conversation type: 50.5% argumentative, 42.5% collaborative, 42.5% task-oriented, 31.6% instructional, 18.8% casual, and 15.0% narrative. Candidate-only precision is 35.5% on dataset-marked clean audio and 33.1% on noisy audio; poor selectivity is not confined to noisy recordings.

### Engineering verification and cost

- All 4,216 unique original-event predictions persisted across 866 isolated window stores. This does not remove the app's 1,000-candidate-per-store limit or validate a single unsegmented seven-hour import.
- A full persisted replay reproduced identical results with **zero model-backend calls**. No reviews were manufactured; every store refused unreviewed exports.
- **43 Python tests pass**, including new checks for matching isolation, one-to-one accounting, timing shifts, and label exclusion. Targeted Ruff checks pass.
- Original FLAC-to-WAV alignment and channel order were numerically checked for six windows across all three shards, within PCM16 rounding error.
- Among successful responses, 57 target spans were truncated by the existing evidence cap; zero text contexts were truncated and zero target turns were absent from the supplied text. Error records do not retain their evidence packet, so those truncation counts do not cover failed requests.
- Successful responses recorded 3,481,385 input tokens, 242,394 response tokens, and 948,175 thinking tokens. At the [published standard prices](https://ai.google.dev/gemini-api/docs/pricing), this is approximately **$7.08**. Failed/interrupted request usage is unavailable, so this is not the full billed amount.
- Local data and derived evaluation artifacts occupy about **19.7 GB**; the original download is 4.22 GB. Raw audio/transcripts remain local and ignored by Git.

### Recommended next experiment

Define the target as an interruption of an established floor holder, distinguish normal handoffs and continuation over backchannels, and test a revised selector against the existing cheap baseline. Keep this completed run as the unchanged baseline. Tune on development data and validate the revision on independent conversations before claiming an improvement. Improve error handling separately; retrying alone cannot fix the dominant ordinary-turn false positives.


## Question and scope

Can the existing interruption curator select real conversational interruptions, using supplied transcripts/timestamps and original two-channel audio? This tests the production timing detector, import/audio/analysis HTTP APIs, evidence builder, fixed Gemini prompt, response validation, persistence, and review/export boundaries.

It is an offline retrospective curation evaluation. It is **not** the official TurnBench online event/latency score, an evaluation of ASR/diarization, or proof of performance on an AI agent. Both participants are humans. Each speaker is evaluated as the incoming speaker, with role names mapped to the application's user/assistant schema.

## Data and ground truth

- Dataset: [Mundo AI TurnBench dev](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev), revision `c29aa4e6422122a8dccbe23598016a089bea2121`.
- 38 original conversations, 7.3075 hours, 20 marked clean and 18 noisy by the dataset.
- Three Parquet shards total 4,215,874,996 bytes (4.22 decimal GB). Downloaded locally using the user's authenticated Hugging Face access. Derived WAV windows and test stores require additional disk space.
- FLAC source audio is used; Opus previews are not used for predictions. Derived WAVs use the production input format, 16 kHz PCM16 stereo. There is no synthetic speech or artificial overlap.
- Consensus is generated by the official [Sesame TurnBench code](https://github.com/SesameAILabs/turnbench), pinned to `76ccd045f121ccfa921abac2ad3107027e736911`, using its canonical label mapping and two-of-three agreement with 200 ms endpoint tolerance.
- There are 347 consensus successful interruptions and 359 non-floor-taking attempts. The remaining canonical events include normal turns, backchannels, non-content sounds, laughter, and awkward silence.
- 33 conversations contain a successful interruption; 37 contain a successful interruption or unsuccessful attempt.

The dataset license and attribution are retained with the evidence. Access does not confer unrestricted commercial training rights.

## Frozen procedure

1. Annotator A's original nonempty transcript segments and timestamps provide the input. All semantic event labels are stripped. No majority-label selection is used to choose model inputs. Empty text segments cannot be represented by the application's current schema and are omitted; all consensus positives remain in the recall denominator.
2. Apply the unchanged production detector to the whole conversation in both speaker directions. This yields 4,216 distinct candidates.
3. Package candidates into 866 overlapping windows: 60-second cores, 30 seconds of preceding context, up to 60 seconds of following context. Preserve the original audio clock via an explicit offset. Only globally detected target onsets are evaluated, once each. Assert that each target's production evidence horizon survives windowing. Windows satisfy the existing application's input limits.
4. Import each window through FastAPI, attach the WAV through the audio endpoint, and run analysis through the production job endpoint. Only selected targets are analyzed; overlap duplicates near window boundaries are not counted twice.
5. Keep the fixed production `gemini-3.8-flash` prompt/schema, temperature 0, and 1,024 output-token limit. No prompt tuning or threshold selection on this dataset. The production evidence builder supplies up to 30 seconds of audio, three seconds before the onset, at most 20 transcript segments, and explicit truncation indicators.
6. Match candidates one-to-one to same-speaker, same-conversation consensus events by nearest onset within 200 ms. Unmatched candidates are unscored, not assumed negative. Record 100/300 ms matching sensitivity separately.
7. Compare the model with timing-only selection and a fixed baseline selecting target spans with at least four whitespace-separated words. Count errors and unclear responses as not automatically selected, and report a separate review queue retaining them.
8. Preserve model suggestions separately from reviews. No human reviews are manufactured, and the export endpoint must refuse unreviewed data in every evaluation store.

Input/gold/protocol hashes were frozen before the first model call. A three-candidate smoke test succeeded and was reused. The run started with eight workers and was restarted with 24 to reduce runtime. All persisted results, including errors, are reused; at most eight interrupted in-flight calls could have been issued again without a persisted result. This is recorded in `restart-audit.json`. There are no selective retries of unfavorable results or persisted model errors.

## Three label interpretations

- **Successful only:** positive is the consensus `Interruption` class. Unsuccessful attempts are negatives. This answers whether the other speaker actually lost the floor.
- **Attempts included:** both successful and non-floor-taking interruptions are positive. This most closely matches the existing model prompt's wording, “attempt to take the floor.”
- **Attempts excluded:** successful interruptions are positive; unsuccessful attempts are excluded, matching that aspect of the official TurnBench definition. This is still not the official online score.

Recall across all consensus positives includes timing-filter misses. Candidate-only metrics isolate the semantic filtering stage. Overall accuracy is not emphasized: silence/noise and easy non-overlap events would inflate it. Conversation-level counts complement event-level localization and do not establish that the model found the correct event inside each flagged conversation.

## Limitations

This is the publicly labeled development split, not the hidden official test split. Provider pretraining exposure is unknown. There are only 38 conversations, with correlated events and limited negative-conversation coverage. Annotated text/timing are better inputs than an unvalidated ASR/diarization pipeline. Speaker roles are human-human surrogates. Incomplete provider responses, disputed annotations, omitted empty transcripts, and evidence truncation are reported explicitly. Confidence intervals resample conversations rather than individual correlated events.

## Reproduction

From the repair-bench repository, install the usual project/test dependencies and:

```sh
uv pip install --python .venv/bin/python -r requirements/turnbench-eval.txt
```

Authenticate Hugging Face and accept access on the dataset page. Download the pinned snapshot into `artifacts/turnbench/source` using `huggingface_hub.snapshot_download` with `repo_type="dataset"` and the revision above. Clone the official reference repository outside this repository, then check out its pinned commit.

```sh
.venv/bin/python scripts/prepare_turnbench.py --reference /absolute/path/to/turnbench-reference
# Provide GEMINI_API_KEY through the process environment; never store it in this repo.
.venv/bin/python scripts/evaluate_turnbench.py --workers 24
.venv/bin/python scripts/report_turnbench.py
.venv/bin/python -m pytest -q
```

Resume evaluation with the same command: persisted analyses are reused. Do not modify frozen inputs after starting a run. Raw audio, transcripts, and local corpus databases stay under ignored `artifacts/turnbench`; committed evidence contains aggregate metrics and text-free predictions.
