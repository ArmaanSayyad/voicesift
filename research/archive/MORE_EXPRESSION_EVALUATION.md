# Additional expression curation: partial evaluation

Status: **incomplete because the Gemini model/project daily request quota was exhausted**. Do not present these as final full-sample benchmark scores. Coughing is documented separately in the [published experiment page](../../docs/experiments/coughing.md).

## Scope

The six remaining RAVDESS expression classes were tested using the exact same 240 clips as the earlier anger/sadness evaluation. Each goal has 30 reference positives and 210 negatives in the planned set. Happy/surprised, fearful/disgust, and calm/neutral were tested as three separate requirement pairs, giving 720 intended requests and 1,440 binary decisions. Neutral and calm are separate dataset classes, not necessarily emotions in the ordinary sense.

The model was `gemini-3.8-flash`; input was audio only, with no transcript, filename or reference labels. The fixed judge/configuration was reused, with new plain-language requirements. No prompt tuning, examples, threshold selection or relabeling was performed. These are acted-expression labels, not actual psychological states.

## Completed-response results

560 requests completed; 154 returned HTTP 429 and six timed out. The following scores use **only completed responses**, with their denominators shown. They exclude the unresolved requests and therefore are not full-test accuracies. All 160 missing requests remain recorded separately; completed-response selection is not guaranteed representative.

| Expression | Completed / planned clips | Found / positives in completed clips | False positives | Accuracy on completed responses | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Happy | 189 / 240 | 6 / 19 | 8 | 88.9% | 42.9% | 31.6% |
| Surprised | 189 / 240 | 14 / 25 | 27 | 79.9% | 34.1% | 56.0% |
| Fearful | 184 / 240 | 22 / 25 | 44 | 74.5% | 33.3% | 88.0% |
| Disgusted | 184 / 240 | 9 / 22 | 15 | 84.8% | 37.5% | 40.9% |
| Calm | 187 / 240 | 20 / 26 | 26 | 82.9% | 43.5% | 76.9% |
| Neutral | 187 / 240 | 13 / 21 | 60 | 63.6% | 17.8% | 61.9% |

Every goal has substantial false positives in the completed portion. None currently supports a high-purity automatic curation claim. Happy delivery misses many positives; fear has high recall but low precision; neutral has especially many false selections. These are provisional observations, not conclusions from a complete held-out test.

## Coverage-aware accounting

The saved summary also reports strict accuracy over all 240 requested clips per goal, counting every unresolved response as incorrect. This measures operational completion plus classification, not pure model quality. Missing inputs are never silently counted as correct negative decisions.

| Expression | Missing requests | Strict correct / planned accuracy |
|---|---:|---:|
| Happy | 51 | 70.0% |
| Surprised | 51 | 62.9% |
| Fearful | 56 | 57.1% |
| Disgusted | 56 | 65.0% |
| Calm | 53 | 64.6% |
| Neutral | 53 | 49.6% |

## Quota evidence and continuation

A single diagnostic request also returned HTTP 429. Its structured QuotaFailure identified `GenerateRequestsPerDayPerProjectPerModel`, a value of 10,000 requests, model `gemini-3.8-flash`, and a RetryInfo delay of 76,220 seconds (about 21 hours at observation). This is the shared project/model quota, not the number used by this experiment. No further API calls were made after confirming the daily limit. Total attempts here: 720 experiment requests plus one diagnostic request.

Completed classifications and original failures are preserved. Once quota is available, run the following to retry each eligible failed request once with the unchanged payload; completed answers remain cached:

```sh
.venv/bin/python scripts/evaluate_more_expressions.py --retry-transport-errors
```

Using a different model would require a separate complete benchmark and must not be merged into this score. The user has been asked whether to preserve this model or run a separate model evaluation.

## Artifacts and checks

Frozen prompts, selection, per-request predictions/errors, quota details and both denominator conventions are in [evidence/more-expressions](../../evidence/more-expressions/README.md). Reference source/licensing and reproduction steps are recorded there. Source audio is not committed.

Partial selection ZIPs are local under `artifacts/more-expressions/exports/`; their manifests state requested clips and failed requests. They contain only successful yes decisions, with unverified model-selection status. They do not represent complete curation of the 240-clip set.

An export-path issue was found while adapting the shared runner to a separate experiment directory, fixed, and covered by a regression test. Saved predictions were unaffected; exports were regenerated offline. The engine now reports 240 unique audio clips separately from the 720 requested classifications.

The web UI and interruption policy were unchanged. This incomplete experiment is kept in the research archive, rather than promoted as a successful end-user benchmark.
