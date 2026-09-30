# Balance inquiry curation

Find spoken requests to check a bank account balance.

**Final benchmark accuracy: 99.6%.** Precision: **95.3%**. Recall: **100.0%**.

## Dataset and results

[MINDS-14](https://huggingface.co/datasets/PolyAI/minds14). All 563 recordings in the US-English partition, covering 14 banking intents. Source license: CC BY 4.0.

| Measure | Count |
|---|---:|
| Audio clips evaluated | 563 |
| Reference positives | 41 |
| Correctly selected | 41 |
| Missed positives | 0 |
| Incorrectly selected | 2 |
| Correctly rejected | 520 |
| Total selected | 43 |

Accuracy counts all correct selections and rejections. Precision measures how many selected clips match the reference label; recall measures how many reference positives were found.

## How it was tested

The fixed `gemini-3.8-flash` judge received normalized 16 kHz audio and plain-language requirements. Transcripts, filenames and reference labels were withheld. Two requirements were evaluated independently per recording. No training or prompt tuning was performed on these results. Scores use the final completed predictions after one unchanged retry of transport/provider failures; completed predictions were never rerun. There were no unresolved errors or unclear decisions in the final scored set.

Classify the main request. A withdrawal-limit question can mention available money without having balance inquiry as its primary intent. This is a clip-level benchmark result, not a full-conversation or production-wide accuracy guarantee. Source labels are the evaluation reference; exported selections remain unverified model judgments. Benchmark exposure during model pretraining is unknown.

## Reproduce and inspect

This goal is configured in a preset audio-only evaluation script. It is not yet selectable in the web UI, and the script does not expose a general freeform-objective CLI. From the repository root, after the [core setup](../../README.md#get-started):

```sh
.venv/bin/python scripts/evaluate_voice_lanes.py --run
# If transport/provider requests fail, recover them once without rejudging completed answers:
.venv/bin/python scripts/evaluate_voice_lanes.py --retry-transport-errors
```

The shared runner evaluates all configured voice goals and uses the Gemini API. It downloads pinned source files and caches results. The selected audio and manifest for this goal are saved to `artifacts/voice-lanes/exports/balance.zip`. Source licenses apply to the exported clips.

[Recorded summary](../../evidence/voice-lanes/summary.json) · [Frozen request and dataset protocol](../../evidence/voice-lanes/protocol.json) · [Experiment index](../README.md)
