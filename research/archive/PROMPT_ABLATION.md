# Interruption prompt and few-shot experiment

## Scope

This is a controlled development experiment on the existing TurnBench dev corpus. It tests whether clearer definitions and examples fix the curator's confusion between interruptions, ordinary handoffs, and continuation over a listener's acknowledgement. Production behavior is not changed by this experiment.

The first experiment has four arms: original prompt, clearer definitions/categories, definitions plus eight text demonstrations, and the same demonstrations with their original audio. Every arm receives the same 400 target packets and target audio from the production evidence builder. Original-arm request hashes exactly match the historical request bodies. The original is called fresh rather than recycling earlier predictions.

## Sampling and examples

A fixed seed (`20260930`) selects three of the 13 speaker pairs for the example pool. All their nine conversations are excluded from the evaluation sample. The remaining 29 conversations contain 3,506 eligible timing candidates; 400 are sampled uniformly without using model predictions or stratifying by labels. There is no speaker overlap between example and evaluation groups.

Of 400 target candidates, 374 match consensus labels: 37 successful interruptions, 36 non-floor-taking attempts, 126 ordinary turns, 134 backchannels, 23 non-content events, and 18 laughter events. The other 26 remain unscored. These are **candidate events, not 400 conversations**. Events missed by the original timing detector are outside this sample.

Eight examples cover two successful interruptions (competitive/cooperative), one unsuccessful attempt, two backchannels, two normal handoffs, and one continuation over a listener backchannel. Examples are selected from consensus labels plus fixed annotation-A/timing criteria, not model predictions. Handoff/continuation subcategories are operational interpretations of ordinary-turn examples, not independently human-adjudicated new dataset labels. No new listening adjudication was performed. All example targets fit their evidence clips without truncation. The audio demonstrations total about 91 seconds.

The first refined prompt explicitly distinguishes conversational floor ownership from the user/assistant role mapping; explains that annotated segment endpoints are not semantic completion points; defines normal handoffs and continuations; and preserves unsuccessful and cooperative interruptions as positives. It also identifies the stereo channel mapping. The refined response schema adds `normal_handoff` and `continuation`. Thus this arm tests a combined definitions/category/channel-instruction revision, not a single isolated sentence change.

Target clips, model (`gemini-3.8-flash`), temperature (0), and first-stage output cap (1,024 tokens) stay fixed. Text/audio demonstration arms use identical example packets, labels, explanations, and ordering; only example audio differs. Jobs are interleaved in a seeded random order. There are no selective retries of errors. Returned usage, including incomplete responses, is retained; transport errors without returned usage remain missing from the cost estimate.

## First-stage results: 1,024-token cap

The following counts include successful and unsuccessful interruption attempts (73 positives and 301 negatives among 374 scored candidates). Errors and unclear responses are not automatically selected.

| Configuration | Found / 73 | Missed | False flags | Precision | Recall | Errors / 400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original prompt, fresh run | 64 | 9 | 114 | 36.0% | 87.7% | 12 |
| Clearer definitions/categories | 49 | 24 | 18 | 73.1% | 67.1% | 18 |
| Definitions + text examples | 53 | 20 | 19 | 73.6% | 72.6% | 44 |
| Definitions + audio examples | 52 | 21 | 27 | 65.8% | 71.2% | 26 |
| Four-word baseline | 66 | 7 | 133 | 33.2% | 90.4% | N/A |

Unscored flags, excluded from precision denominators, are 16, 14, 13, and 14 respectively. Actual total flags across all 400 candidates are 194, 81, 85, and 93.

The main gain came from clearer definitions: ordinary-turn false flags fell from 108/126 to 16/126. Text examples recovered four additional positive attempts versus definitions alone, at the cost of one extra false flag. Audio examples did not outperform text examples in this run; this does not establish that audio examples never help.

For successful interruptions only (37 positives; unsuccessful attempts now counted as negatives), original/definitions/text/audio found 36/26/28/26, missed 1/11/9/11, and produced 142/41/44/53 known false flags. Successful-interruption precision was 20.2%/38.8%/38.9%/32.9%. A prompt targeting interruption attempts does not by itself establish successful floor takeover.

## Reliability and uncertainty

The text arm's 44 failures comprise 42 `MAX_TOKENS` responses and two timeouts. The original had eight token-limit responses and four timeouts; definitions had 15 and three; audio examples had 20 and six. The same cap is a useful operational control, but it can conceal improved classification behind extra response failures.

On the **303 scored candidates with complete responses in all four arms**, the positive count is 52. Original/definitions/text/audio recover 47/43/46/44, with 86/13/15/15 false flags. Text examples therefore retain nearly the original recall on this conditional subset while substantially reducing false alarms. This is a diagnostic subset selected by response success, not a replacement for the full-sample result.

The fresh original run changes 20/400 binary decisions versus its historical run, including changes caused by response failures. Historical sample counts were TP 66, FP 118, FN 7. Temperature zero did not make repeated service calls perfectly reproducible.

A paired 3,000-resample bootstrap clusters all conversations by their ten evaluation speaker pairs. Relative to the original, first-stage text examples have a precision difference interval of +29.4 to +45.5 percentage points, and a recall difference of -19.5 to -10.3 points (95% percentile intervals). This supports a precision/recall tradeoff on this development sample; it does not establish out-of-sample accuracy or prove superiority over definitions alone.

## Controlled token-budget follow-up

After observing the 42 token-cap failures in the text arm, a second protocol was frozen to rerun **all 400 candidates**, with both original and text-example arms given a 2,048-token cap. This is an adaptive development follow-up. It does not rerun only failures or replace the first-stage evidence. Inputs, example selection, prompts, schemas, and model remain the same. Definitions-only and audio-example arms are not rerun at the higher cap, so the follow-up cannot rank all possible configurations at that budget.

| Configuration, both at 2,048 tokens | Found / 73 | Missed | Known false flags | Precision | Recall | Errors / 400 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original prompt | 65 | 8 | 117 | 35.7% | 89.0% | 3 |
| Definitions + text examples | 56 | 17 | 26 | 68.3% | 76.7% | 1 |

Both arms flag 16 additional unmatched/unscored events. Total flags are 198 versus 98. Token-cap failures fall to zero in both arms; the remaining failures are timeouts. The revised prompt reduces known false alarms by **91 (77.8%)**, while finding nine fewer positive attempts. Precision is 1.91 times the original. Relative to the text arm's 1,024-token run, the larger cap finds three more positive events and seven more false alarms; response reliability improves, but classification is not solved.

For successful interruptions only, the original finds 36/37 with 146 known false flags; the revised prompt finds 29/37 with 53 known false flags. The revised successful-interruption recall is 78.4% and precision 35.4%. Thus the improvement is useful but does not satisfy a high-purity successful-interruption curation goal.

## Interpretation and recommendation

Clearer operational definitions address the dominant ordinary-turn confusion. Examples provide a smaller incremental benefit in the first-stage experiment; they are not the sole cause of the improvement. Audio examples did not outperform text examples under the tested cap. The higher output budget removes the observed truncation failures, but does not restore original recall.

The text-example configuration is a promising **review-queue filter**, not a validated automatic dataset selector. Definitions alone at a sufficient output budget remain an important cheaper comparison that this follow-up did not run. The next validation should compare those configurations on fresh conversations, keep successful takeovers distinct from attempts, and retain errors/unclear cases for review. Do not present this adaptively chosen development configuration as independently validated. The production app remains on its original prompt; experimental prompts are isolated in the scripts.

## Verification and cost

- 2,400 model calls across the four first-stage arms and the two complete follow-up arms; no selective retries.
- All 2,400 results replay identically with zero backend calls.
- 48 Python tests pass, including preservation of target packets, demonstration modality isolation, schema validation, and retention of usage for token-limit failures. Targeted Ruff checks pass.
- Frozen selection and production-code hashes remain unchanged. Text-free evidence contains predictions, protocol/input hashes, summaries, and sanitized example-selection metadata. Raw audio/transcripts stay local.
- Estimated cost from recorded usage is **$9.75** at the [published standard rates](https://ai.google.dev/gemini-api/docs/pricing). This includes returned usage for incomplete responses but excludes timeout calls with no returned usage; it is not a complete billing statement.



## Limits and reproduction

All 38 TurnBench conversations had already been used for diagnosis. Speaker separation prevents direct example/test speaker leakage, but does **not** turn this into a fresh held-out validation set. Only one sample, one example set/order, and one call per item/configuration are used. There are only ten evaluation speaker pairs and 37 successful positive events. Both participants are humans and inputs use reference transcripts/timestamps. Precision excludes unmatched events; candidate-only recall omits timing-stage misses.

The unchanged full-corpus baseline is documented in `TURNBENCH_EVALUATION.md`. Run from the repository after installing `requirements/turnbench-eval.txt` and preparing the pinned TurnBench inputs:

```sh
.venv/bin/python scripts/prepare_prompt_ablation.py
.venv/bin/python scripts/run_prompt_ablation.py --prepare-only
# Supply GEMINI_API_KEY through the process environment; never store it in the repo.
.venv/bin/python scripts/run_prompt_ablation.py --workers 24
.venv/bin/python scripts/report_prompt_ablation.py
.venv/bin/python scripts/run_prompt_budget_followup.py
```

The preparation script freezes speaker-separated selection before model calls. The first-stage protocol hashes the selection, input audio/text packets, production evidence code, prompts, and schemas. The follow-up references the frozen parent protocol. Raw demonstration/target content stays in ignored `artifacts/prompt-ablation`; committed evidence is text-free and retains the source dataset license/attribution.
