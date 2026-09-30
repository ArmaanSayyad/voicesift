# PRESTO correction/cancellation screening experiment

Completed 2026-09-29. Gemini `gemini-3.8-flash` evaluated 600 English PRESTO dialogue contexts with a fixed text-only prompt. This experimental screening script is separate from the live interruption-only curator. It measures recognition in the **final user turn**, not detection anywhere in a full conversation or audio understanding.

## Results

Of 400 examples tagged by PRESTO as a correction or cancellation, 343 were selected and 57 were not selected: **85.75% positive-tag recovery** on this deliberately balanced sample. Selection accepts either correction or cancellation, because the requested goal combines them.

| PRESTO category | Tested | Selected | Not selected | Positive-tag recovery |
|---|---:|---:|---:|---:|
| Within-turn correction | 100 | 98 | 2 | 98% |
| Argument correction | 100 | 88 | 12 | 88% |
| Action correction | 100 | 60 | 40 | 60% |
| Cancellation | 100 | 97 | 3 | 97% |
| Tagged-positive total | 400 | 343 | 57 | 85.75% |

Of the 57 not selected, 51 were classified neither and six unclear. All remain in the denominator. There were zero API/parse failures across all 600 requests.

Subtype distinction matters: ten of the 60 selected action-correction examples were called cancellation. Thus 246/300 correction-tagged examples were selected for the combined goal, but only 236/300 received the correction subtype. All 97 selected cancellation-tagged examples received the cancellation subtype. No response used both.

| Comparison group (binary truth unknown) | Tested | Selected | Not selected |
|---|---:|---:|---:|
| Untagged, human context | 100 | 11 | 89 |
| Disfluency-tagged | 100 | 8 | 92 |
| Total | 200 | 19 | 181 |

Across all 600 inputs, 362 were selected and 238 were not. **The 19 comparison selections are not established false positives. Precision and overall accuracy are not estimable from these labels.** These results do not imply 85.75% purity or 85.75% real-world accuracy.

## Error analysis and label fit

These are post-run analyst observations, not independent human adjudication. Original reference labels and scores remain unchanged; no cases were removed to improve the score.

- Action changes are the major disagreement. PRESTO sometimes tags a follow-up action after an apparently completed request as a correction. The conservative prompt often calls it a new task. Review example `correct-action` cases in the local response archive before changing this policy: maximizing tag agreement could lower usefulness for a narrower revision goal.
- Argument corrections can be implicit. In `ebbc008f1bfeb2ed71219e8586ed2f5a1d7e23531866a2cc407da1f32f202b6b`, an earlier named call recipient changes after another clarification. The model focuses on the clarification answer and rejects it. This suggests a risk of missing revisions spread across turns.
- Reference-label mismatch is visible in some positive examples. Within-turn tag `c36edd6f55fce2dcbf5031cbb0a1228226c504ae7bb28e1aa87089f32dd4867a` has no preceding dialogue and a plain request to pause an exercise app. Cancellation tag `8d67517683c98b8e1c6a237b13496c5d3bd282898dffaaaa0c9c3ba602efba3f` ends with “Begin note”. Neither visibly establishes the tagged event in the final turn.
- Comparison records can contain repairs. Untagged example `9f8821e6d2cda511477509cfaf8e9dbc1b87c67b757e93b5bda4b2aff8d55aa1` changes an event name within one utterance. Disfluency example `9f7941d22f4a5fa40d9ddb633f723c5e7fbc7ba6d4f250df1aef5c1996ab7a9a` changes a time expression. Counting these selections as automatic false positives would punish useful detection.
- Repetition and partial-word restarts remain a boundary risk. Some disfluency selections repair a partial word without changing the underlying intent. Our next annotation rule must distinguish any speech self-repair from a semantic correction of the user's request.

## Method and reproducibility

Source: [official PRESTO release](https://github.com/google-research-datasets/presto), English test member, CC BY 4.0. See [dataset audit](CORRECTION_DATASETS.md) for verified source sizes and full tag counts.

The source SHA256 is checked before sampling. Seed 20260929 selects 100 examples independently from each of six strata; all are shuffled before submission. The sample contains 600 distinct dialogue packets, 571 distinct final utterances, 400 human-context records and 200 synthetic-context records. Context-source metadata does not imply that every utterance was synthetically generated. Repeated final phrases remain; examples are not statistically independent merely because IDs differ.

The model sees only preceding dialogue and the final utterance. It sees no PRESTO labels, semantic targets, IDs, groups, seeded entity inventories, or expected answers. The full supplied dialogue text is retained without truncation. Missing non-dialogue context can limit interpretation.

The prompt defines correction, cancellation, both, neither and unclear; requires contextual evidence; treats source content as data; and prioritizes clear selection. It excludes unrelated new tasks, ordinary clarification answers and repetition without revision. It defines cancellation as withdrawing the ongoing request, rather than every command to cancel an existing object. It uses no PRESTO demonstrations. Prompt, selection and request configuration were frozen before inference and were not tuned or rerun after observing results.

Configuration: temperature zero, 2,048 maximum output tokens, structured JSON response, four concurrent requests. All 600 returned model version `gemini-3.8-flash`. No selective retries. Usage reported 200,477 prompt tokens, 25,937 candidate tokens and 126,544 thought tokens (352,958 total). No audio was sent.

The source test split had already been inspected during research. This is a development evaluation, not an untouched holdout claim. The four positive strata are equally sampled, so their combined percentage does not reflect PRESTO's natural category proportions or deployment prevalence.

Reproduction commands and frozen, text-free protocol/selection/results are in [evidence/presto-evaluation](../../evidence/presto-evaluation/README.md). Raw requests and evidence notes remain local under ignored `artifacts/presto-evaluation/`. The evaluator is [scripts/evaluate_presto.py](../../scripts/evaluate_presto.py). Automated tests check gold-label exclusion and prohibit reporting unknown comparisons as measured false positives.

## Decision

The approach is promising for finding explicit cancellations and within-turn corrections. It is not yet validated as a high-purity dataset curator. Do not deploy it as such based on this run.

Next, freeze the semantic boundary (request revision versus any speech self-repair), and create a model-blind human-adjudicated sample of both selected and rejected PRESTO examples, including action changes and comparison records. Then measure precision and recall on that independent reference. NC-Bench can provide a separate controlled pattern check, but cannot replace that validation. Keep audio evaluation separate. No change was made to the interruption UI or its production selection policy.
