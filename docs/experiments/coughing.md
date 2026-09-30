# Cough curation

Find clips whose dominant vocal event is coughing, distinguished from throat clearing, sneezing, sniffing, sighing and laughter.

**Final benchmark accuracy: 94.0%.** Precision: **79.6%**. Recall: **86.0%**.

## Dataset and results

[VocalSound via DynamicSuperb](https://huggingface.co/datasets/DynamicSuperb/VocalSoundRecognition_VocalSound). 300 recordings: 50 each of laughter, coughing, throat clearing, sneezing, sniffing and sighing, sampled from the 720-item benchmark subset. Source license: CC BY-SA 4.0.

| Measure | Count |
|---|---:|
| Audio clips evaluated | 300 |
| Reference positives | 50 |
| Correctly selected | 43 |
| Missed positives | 7 |
| Incorrectly selected | 11 |
| Correctly rejected | 239 |
| Total selected | 54 |

Accuracy counts all correct selections and rejections. Precision measures how many selected clips match the reference label; recall measures how many reference positives were found.

The selected subset contained 43 reference coughs and 11 other events. Six false positives were throat clearing, three laughter, one sigh, and one sneeze. Review selections when high dataset purity is required. This goal detects an acoustic event, not illness.

## How it was tested

The fixed `gemini-3.8-flash` judge received normalized 16 kHz audio and plain-language requirements. Transcripts, filenames and reference labels were withheld. Two requirements were evaluated independently per recording. No training or prompt tuning was performed on these results. Scores use the final completed predictions after one unchanged retry of transport/provider failures; completed predictions were never rerun. There were no unresolved errors or unclear decisions in the final scored set.

This test distinguishes isolated vocal events. It does not measure cough detection inside overlapping speech, long conversations, or arbitrary environmental audio. This is a clip-level benchmark result, not a full-conversation or production-wide accuracy guarantee. Source labels are the evaluation reference; exported selections remain unverified model judgments. Benchmark exposure during model pretraining is unknown.

## Reproduce and inspect

This goal is available through the evaluation CLI; it is not enabled in the interruption-only web UI. From the repository root, after the [core setup](../../README.md#run-locally):

```sh
.venv/bin/python scripts/evaluate_voice_lanes.py --run
# If transport/provider requests fail, recover them once without rejudging completed answers:
.venv/bin/python scripts/evaluate_voice_lanes.py --retry-transport-errors
```

The shared runner evaluates all configured voice goals and uses the Gemini API. It downloads pinned source files and caches results. The selected audio and manifest for this goal are saved to `artifacts/voice-lanes/exports/cough.zip`. Source licenses apply to the exported clips.

[Recorded summary](../../evidence/voice-lanes/summary.json) · [Frozen request and dataset protocol](../../evidence/voice-lanes/protocol.json) · [Experiment index](../README.md)
