# Precision-first interruption curation

## Shipped behavior

The purpose is to extract a smaller, human-confirmed set of successful interruptions from a larger corpus. The model narrows the review queue; it does not establish a clean dataset by itself.

1. Import speaker-labelled, timestamped conversations and attach their source WAVs. Existing timing overlap detection proposes candidates; it does not decide intent. Untimed speech and events without detected overlap are outside this search.
2. Gemini 3.8 Flash receives the target audio clip and nearby annotated transcript, with explicit definitions of interruption attempts, backchannels, ordinary handoffs, and speakers continuing their own turn. Six invented textual illustrations anchor those distinctions. No TurnBench demonstration transcripts or recordings are embedded in the production prompt. No stereo channel-to-role mapping is assumed for arbitrary uploads.
3. The model separately reports whether an attempt succeeded, failed, or has an unknown outcome. A stopped segment is not proof that the speaker yielded. The output budget is 2,048 tokens; temperature is zero, which does not guarantee deterministic model output.
4. `successful-interruption-v1` shortlists only `take_floor` plus `successful`, with no detected target/context truncation and the target present in the evidence packet. Ambiguous or incomplete evidence stays in review; other candidates remain accessible. These are categorical criteria, not calibrated confidence scores.
5. The reviewer checks floor ownership, premature entry, and actual takeover, then explicitly confirms success and inclusion. New exports require both. Model decisions never write reviews or include records automatically.

This remains a local, bounded application: up to 100 imported conversations per request, 1,000 candidates per local corpus and 50 model requests per batch. It is not yet a streaming multi-million-record dataset service. No GPU or Laya dependency is introduced. Source audio is necessary for Gemini analysis; existing annotation-only manual workflows remain available and should not be mistaken for independent acoustic validation.

## Measured development results

The exact production request construction and validation were exercised on the existing frozen sample of 400 candidate events from 29 TurnBench conversations. This is **not 400 independent conversations**. There are 374 consensus-matched events, including 37 successful interruptions and 337 other events, plus 26 unmatched/unscored events. Matching uses the existing one-to-one, same-speaker, 200 ms onset rule. The sample and parent corpus had already been inspected: this is adaptive development evidence, not fresh held-out accuracy.

| Policy | True successful interruptions selected | False positives | Successful interruptions missed | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Earlier original prompt, 2,048 tokens | 36 | 146 | 1 | 19.8% | 97.3% |
| Earlier text-example prompt, 2,048 tokens | 29 | 53 | 8 | 35.4% | 78.4% |
| **Shipped explicit-success policy** | **22** | **17** | **15** | **56.4%** | **59.5%** |
| Experimental independent second pass | 16 | 11 | 21 | 59.3% | 43.2% |

Earlier prompts used `take_floor` as the positive selection rule; the shipped policy requires a successful outcome. These are comparisons of resulting curation policies, not a controlled claim about one prompt phrase. Earlier 68.3% attempt-inclusive precision is not successful-interruption precision.

The shipped policy shortlists **45 of 400** events: 22 known positives, 17 known false positives, and 6 without consensus matches. The latter six are not counted as correct or incorrect. The remaining 355 events comprise 344 not selected and 11 needing review. There were eight provider/validation failures (seven `ValueError`, one timeout); errors were kept out of the shortlist and counted as misses when their gold label was positive. This run did not preserve detailed reasons or usage for failed responses, so it cannot determine how many ValueErrors were token-limit versus schema/consistency failures.

The optional verification experiment made 45 additional requests, using the same model and audio but a separate adversarial prompt without the first answer. It removed six true positives and six false positives; two verification requests failed. That small precision gain and lost yield do not justify adding it to the default pipeline. It remains reproducible research code only.

The production run's recorded successful-response usage estimates $1.14 at the rates used in the prior evaluation; verification adds approximately $0.17. These are estimates, exclude missing usage from failed calls, and are not billing statements.

**Interpretation:** the shortlist is substantially less noisy than before, but 17 of its 39 scored selections are still false positives. A larger source corpus can recover more candidates; it cannot fix label purity. Human review is necessary. Neither reviewer accuracy nor final curated-dataset purity has been measured. TurnBench is human-human development data; transfer to agent-user audio is not established. No uncertainty interval or statistical significance claim is made for the small, conversation-clustered positive sample.

## Compatibility and traceability

- Production uses a separate versioned policy module. Legacy prompt/schema/request helpers remain intact for historical experiments; historical protocol source hashes intentionally detect modified code.
- Cache identity includes the model and complete request (prompt, schema, generation settings, audio, and text). New requests cannot reuse old-prompt results. Results save policy version, disposition, request hash, evidence hashes, usage, model version, and evidence note.
- Legacy analyses are marked for reanalysis in the UI and included in the pending batch action.
- Legacy included reviews without explicit successful-outcome confirmation do not count toward new exports. Review history is preserved; previously created immutable exports remain unchanged.
- New JSONL exports use schema version 3 and include `annotation.interruption_result`. Consumers must account for this field and version. Model suggestions remain distinct from human annotations.

## Verification and reproduction

- 66 automated tests pass, including export eligibility, legacy-review migration, invalid model labels, contradiction rejection, truncation/target omission, caching, API failures/retry, export integrity, and existing research tests.
- React/TypeScript production build passes. The live UI was inspected for the shortlist, success controls, and disabled uncertain inclusion.
- All 400 first-pass and 45 verification records replayed from request-hash-checked caches without API credentials or new calls. Result ordering is sorted; the verification protocol identifies first-pass inputs by per-event request hash and disposition. An initial file-byte hash was replaced because concurrent completion order is not stable; the original protocol is retained locally for audit. No request, response, target, or label was altered to make replay pass.

From the repository root, after preparing the gated TurnBench corpus and frozen sample as described in `PROMPT_ABLATION.md`:

```sh
.venv/bin/python scripts/evaluate_precision.py
.venv/bin/python scripts/verify_precision.py
.venv/bin/python -m pytest -q
(cd curation-web && npm run build)
```

Fresh API calls require `GEMINI_API_KEY` in the process environment. Existing result files are checked against exact request hashes; failed records are not selectively retried. `artifacts/precision-evaluation` contains local raw responses and protocols. `evidence/precision-curation` contains text-free predictions, summaries, request templates, and input hashes for audit. Dataset recordings and transcripts remain local under the gated dataset's license. See `evidence/turnbench-evaluation` and `research/archive/TURNBENCH_EVALUATION.md` for source revision and annotation protocol.

The next meaningful validation is a fresh, conversation-disjoint agent-user corpus with independently adjudicated successful-interruption labels, followed by a blinded audit of exported examples. Further tuning on these same 400 events would not establish generalization.
