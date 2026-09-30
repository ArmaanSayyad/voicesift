# Correction and cancellation evaluation: dataset audit

Inspected 2026-09-29. This is a dataset suitability audit, not a model accuracy report. No correction/cancellation classifier was evaluated in this investigation.

## Recommendation

Use PRESTO for a large semantic evaluation and NC-Bench for an independent, smaller pattern-based cross-check. Both are text-only. Neither licenses a claim about our audio pipeline's accuracy. Full-Duplex-Bench v3 is a possible recorded-audio follow-up for self-corrections, but does not supply a complete cancellation benchmark.

Do not silently turn absent phenomenon tags into negative labels. Freeze the meaning of correction/cancellation and independently audit negative examples before reporting curation precision. We have not verified a turnkey audio dataset with exhaustive gold labels for both goals.

## PRESTO

- [Official release and schema](https://github.com/google-research-datasets/presto), CC BY 4.0.
- [Paper](https://aclanthology.org/2023.emnlp-main.667/).
- [Hugging Face mirror](https://huggingface.co/datasets/DeepPavlov/presto), inspected revision `68f073c01e21c550d157c6ddfdb9c0f90f32a5fe`.
- [Official archive](https://storage.googleapis.com/gresearch/presto/presto_v1.zip): 415,990,813 bytes; ETag `5fb5bd7e437a07fbae4991b5b4a573f4`.

Downloaded the official English test member using HTTP byte ranges: about 8.1 MB compressed, 37,169,707 bytes expanded. Member `test_partitions/en-US/test.jsonl`, SHA256 `b71584b78f71e08aa1fea78a7a3f9d34db25402e20b2aa09ac51c89c02cdfa9b`.

Verified counts in that file:

| Phenomenon | Rows |
|---|---:|
| within-turn-correction | 7,203 |
| correct-argument | 928 |
| correct-action | 547 |
| cancel-action | 1,061 |
| disfluency | 6,176 |
| code-mixing | 2,091 |
| No phenomenon tag | 15,571 |
| Total | 33,577 |

There are 8,678 tagged corrections and 1,061 tagged cancellations. Human-context records number 8,614: 928 argument corrections, 547 action corrections, 1,061 cancellations, and 6,078 untagged records. The other 24,963 records have synthetic context. This metadata describes the origin of context, not whether every utterance was generated synthetically. PRESTO contains elicited task-oriented text, not released audio recordings.

Compared all 1,061 cancellation and 928 argument-correction rows in the downloaded mirror slices with the official file by example ID: zero final-input or phenomenon-label mismatches. Category/config partitions overlap; do not concatenate them without deduplication.

### Label limitations

The phenomenon label describes the final user query. It is not an exhaustive annotation of every earlier turn. Score final-turn decisions with preceding context; do not report those results as full-conversation detection accuracy.

Manual inspection found untagged examples whose final queries appear to express cancellation or correction. Many have the semantic target `Other()`. Consequently, an empty tag is not a reliable negative for our broader user requirement. Conversely, a request to cancel an existing object may differ from retracting the current request. Resolve that boundary using context and an explicit annotation rule.

Excluding `Other()` alone does not prove remaining negatives clean. Labels and semantic targets must stay out of classifier input. Use official train/development records for prompt examples; reserve test records for measurement. Individually inspected examples are audit/development evidence and should be excluded from any newly claimed untouched holdout.

## NC-Bench

- [Hugging Face dataset](https://huggingface.co/datasets/ibm-research/nc-bench), revision `af164c5d5999ecbc899d558cdd855d357d5656a6`.
- [Paper and construction method, section 3.2](https://arxiv.org/html/2601.06426v1).

Downloaded all three parquet files, approximately 62 KB total: Basic 180 rows, Complex Request 360, RAG 180; 720 total. The task labels identify 40 explicit self-correction examples and 60 abort examples. Another 20 RAG examples have a combined incremental/self-correction label: exclude or independently adjudicate these rather than automatically treating them as corrections.

The authors adapted existing material and manually authored scenarios. These are constructed text exchanges, not recorded spontaneous speech. The benchmark originally evaluates assistant continuations; our adaptation uses its scenario task labels, not its model-output judge scores. Remaining task categories are comparison candidates, not automatically exhaustive full-conversation negatives.

Basic alone provides a useful first controlled comparison: 20 self-corrections, 20 aborts, and 140 other final-turn patterns. Cancellation wording is repetitive: inspected Basic aborts contain only four distinct final utterances and RAG aborts only two. Strong results here would not establish broad deployment accuracy. Group shared scenario/template families when splitting to avoid leakage.

## Recorded-audio follow-up

[Full-Duplex-Bench v3](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/main/v3) documents 100 recorded examples, 79 unique scenarios, and 12 speakers. Downloaded scenario metadata has 100 entries; 21 carry `SELF_CORRECTION` in `disfluency_features`. The remaining 79 are not yet audited as clean negatives. The audio archive has not been downloaded or checked against these metadata counts.

This is a tool-calling benchmark with scenario/acting instructions. Treat it as elicited recorded speech, not unprompted natural conversations. It does not provide the cancellation class we need. Verify audio correspondence and audio redistribution terms before packaging it into exports.

## Evaluation protocol before implementation claims

1. Freeze the target: a speaker revises earlier content/intention, or withdraws an ongoing request. Label correction and cancellation separately; preserve a combined selection label. Specify whether ordinary cancellation of an existing reservation counts.
2. Adapt final-turn text benchmarks first. Pass conversation text and necessary context only; hide labels, task names, semantic targets, and label-bearing paths.
3. Independently annotate a fixed, stratified negative/comparison sample with model predictions hidden. Include hard negatives such as acknowledgements, clarification requests, new unrelated requests, and disfluency without revision. Adjudicate ambiguous items; do not let the evaluated Gemini model manufacture its own gold labels.
4. Keep dataset agreement and independently audited accuracy separate. Freeze prompts before held-out evaluation and group related dialogues/templates. Report natural/source prevalence as well as any artificially balanced sampling.
5. Report total assessed, actual positives, selected, true positives, false positives, missed positives, abstentions/errors, precision, recall, and uncertainty. Report correction and cancellation separately. Include failures in denominators rather than silently dropping them.
6. Evaluate audio independently once a labeled audio sample exists. Text results do not measure transcription quality, speaker attribution, timing, or acoustic interpretation. Full-conversation selection needs labels covering all evaluated turns.

Raw research downloads are local ignored artifacts, not committed dataset redistributions. The live interruption-only UI and filtering policy are unchanged by this audit.
