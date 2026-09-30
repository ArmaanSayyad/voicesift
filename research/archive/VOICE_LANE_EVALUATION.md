# Audio-only curation: six-use-case evaluation

Completed 2026-09-29. This experiment uses the same Gemini model as the curator (`gemini-3.8-flash`) with a new reusable audio-goal judge and offline subset exporter. It evaluates **clip-level audio requirements**, without overlap gating or transcript input. The production UI remains interruption-only; these are experimental extensions, not six newly deployed UI features.

## Results

1,103 unique recordings were evaluated, with two independent requirements per recording (2,206 binary decisions). The table is scored against existing dataset labels. All scores below are after one uniform recovery pass for failed transport/provider requests.

| Goal | Evaluated | Reference positives | Found | Missed | False positives | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balance inquiries | 563 | 41 | 41 | 0 | 2 | 99.6% | 95.3% | 100.0% |
| Card-freeze / lost-card requests | 563 | 45 | 45 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| Laughter | 300 | 50 | 46 | 4 | 3 | 97.7% | 93.9% | 92.0% |
| Coughing | 300 | 50 | 43 | 7 | 11 | 94.0% | 79.6% | 86.0% |
| Angry performed expression | 240 | 30 | 14 | 16 | 16 | 86.7% | 46.7% | 46.7% |
| Sad performed expression | 240 | 30 | 8 | 22 | 34 | 76.7% | 19.0% | 26.7% |

Accuracy includes correctly rejected negatives. **Precision describes the fraction of exported selections matching the reference label; recall describes the fraction of reference positives found.** Exported sets remain machine-selected and unverified. No result establishes universal accuracy.

Always rejecting every clip would score 92.7% for balance, 92.0% for card freezing, 83.3% for each vocal-sound goal, and 87.5% for each expression goal. Both expression filters are below that trivial baseline on accuracy, although that baseline has zero recall. Their low selection precision is the more direct reason not to use them as automatic curators.

## Data and sampling

- [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14): all 563 US-English recordings in the available train partition, covering 14 banking intents. Balance has 41 positives; card freezing has 45. The source represents elicited spoken requests, not full customer-service conversations. We performed no training on this partition. License: CC BY 4.0.
- [VocalSound](https://github.com/YuanGongND/vocalsound), through the [DynamicSuperb mirror](https://huggingface.co/datasets/DynamicSuperb/VocalSoundRecognition_VocalSound): a fixed random sample of 50 from each of six classes, 300 total, from its 720-item test subset. Negatives include other human vocal events, including throat clearing and sneezing. This is not a test against every possible environmental sound or speech mixture. License: CC BY-SA 4.0.
- [RAVDESS](https://zenodo.org/records/1188976), through the [xbgoose mirror](https://huggingface.co/datasets/xbgoose/ravdess): 30 examples per emotion, 240 total from 1,440 audio-speech examples, covering all 24 actors. Reference labels describe acted expression. They do not establish a speaker's real mental state or measure customer frustration. Original license: CC BY-NC-SA 4.0, with separate commercial licensing.

Only relevant parquet files were downloaded: 582,466,480 bytes (about 582 MB). The test audio totals about 116.31 minutes. Audio was normalized to 16 kHz PCM16 without trimming. Source hashes are unique in the selected set. RAVDESS mirror label/actor fields matched source filename codes on all 1,440 records; this is a provenance consistency check rather than new perceptual annotation.

## Interpretation

**Spoken intent is the strongest lane here.** Card-freeze requests matched all 563 reference decisions; this is 45 observed positives, not proof of perfect generalization. Balance detection found every positive but also selected two ATM-limit examples. One asks how much can be withdrawn; another explicitly mentions balance as well as a withdrawal limit. We retained the original single-intent labels and counted both as false positives. This illustrates the boundary between primary-intent filtering and finding any mention of a topic.

**Laughter is promising; coughing needs review.** Laughter found 46/50 examples with three false positives. Of the eleven cough false positives, six are labeled throat clearing, three laughter, one sigh, and one sneeze. The distinction is acoustically difficult for this frozen judge.

**Emotion-based curation is weak.** Angry-expression false positives include nine disgust examples. Sad-expression false positives include thirteen calm, seven disgust, and six fearful examples. The model often conflates vocal delivery styles. Do not use these results to market reliable emotion or frustration filtering. These errors were inspected only after predictions; no labels were changed or difficult examples excluded.

## Uncertainty

Positive counts are small (30–50 per goal) and examples can share speakers. The following 95% Wilson intervals are descriptive binomial intervals for precision; they do not adjust for speaker clustering and should not be treated as population guarantees.

| Goal | Precision | Approximate 95% interval |
|---|---:|---:|
| Balance inquiries | 95.3% | 84.5%–98.7% |
| Card-freeze / lost-card requests | 100.0% | 92.1%–100.0% |
| Laughter | 93.9% | 83.5%–97.9% |
| Coughing | 79.6% | 67.1%–88.2% |
| Angry performed expression | 46.7% | 30.2%–63.9% |
| Sad performed expression | 19.0% | 10.0%–33.3% |

Category balancing in VocalSound/RAVDESS differs from real source prevalence. Precision may be lower when positives are rarer. Public-benchmark exposure during model pretraining is unknown. There was no training, prompt tuning, few-shot demonstration, or threshold selection on these results. Full dialogues, overlapping speakers, mixed acoustic events, languages beyond this intent sample, and domain transfer remain untested.

## Protocol and reliability

Each request contains audio bytes and two plain-language requirements. It contains no transcript, filename, source ID, benchmark instruction, reference label, or actor metadata. The schema asks for yes/no/unclear independently per requirement and a short evidence note. Requirements target the dominant event/expression or main intent, aligning with single-label datasets. Model temperature is zero, output limit 2,048 tokens, and concurrency six. Model generation is not guaranteed reproducible even with these settings.

There were 1,103 initial API attempts: 1,083 completed and 20 failed (18 URLError, one HTTPError, one TimeoutError). Every failed transport/provider request received exactly one separately recorded retry with its unchanged payload; all recovered. No completed answer was rerun. Final unresolved errors and unclear decisions: zero. Initial failures remain committed as evidence; their original records did not contain underlying HTTP/network diagnostics, so a more specific root cause is not established.

| Goal | First-attempt accuracy (failures count wrong) | After recovery |
|---|---:|---:|
| Balance inquiries | 98.0% | 99.6% |
| Card-freeze / lost-card requests | 98.4% | 100.0% |
| Laughter | 95.7% | 97.7% |
| Coughing | 92.3% | 94.0% |
| Angry performed expression | 84.6% | 86.7% |
| Sad performed expression | 75.0% | 76.7% |

Total attempted API calls: 1,123. Reported token usage for completed responses: 381,134 prompt, 46,283 candidate, 145,824 thought; 573,743 total. Failed attempts did not supply usage metadata.

## Artifacts and validation

Run `.venv/bin/python scripts/evaluate_voice_lanes.py --run` to download pinned source files, freeze the sample, classify audio, score results, and export selected subsets. The optional `--retry-transport-errors` pass preserves original attempts and retries only failed transport/provider requests once. See [frozen protocol, per-item results, hashes and recovery evidence](../../evidence/voice-lanes/README.md).

Six ZIPs are saved under `artifacts/voice-lanes/exports/` and copied to the project's `outputs/voice-lane-evaluation/` directory. They contain normalized audio, manifests and explicit unverified-selection status; reference labels are excluded. They are clip datasets without timed conversation transcripts, and are not directly supported by the current interruption-only upload schema. Each archive passed CRC verification.

Automated validation: 95 tests passed. New cases verify request isolation, strict model-output parsing, failure-aware metric denominators, selection-only exports without gold labels, and recovery that preserves the original failure and caches the completed answer. No production UI or existing interruption policy was changed.

**Recommendation:** prioritize a reviewed spoken-intent curation extension, followed by laughter. Keep cough filtering review-assisted. Do not enable automatic anger/sadness curation based on these results.
