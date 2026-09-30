# Documentation

Start with the [project setup](../README.md#get-started), [setup help](GETTING_STARTED.md), and [dataset workflow](DATASET_RUNS.md).

## Project scope and current capabilities

VoiceSift's purpose is LALM-based curation of voice datasets from larger collections using freeform text objectives. The web app now supports freeform objectives as well as its optimized interruption workflow.

| Scope | Status |
|---|---|
| Describe an arbitrary objective, run it on a dataset, review and export | Implemented for supported sources and audible, per-recording objectives; up to five minutes per recording |
| Successful-interruption curation | Default objective with timed transcripts uses the existing overlap workflow |
| Other voice objectives | Freeform web objectives; preset scripts remain available for reproducing earlier benchmarks |

Freeform web runs do not require transcripts. Gemini produces a saved interpretation and criteria, then judges every complete recording yes/no/unclear. Unsupported or ambiguous requests stop before audio judging. Arbitrary Hugging Face schemas, dataset-wide ranking and editing audio are not supported; see [workflow limits](DATASET_RUNS.md).

## Selected experiments

These pages cover selected completed audio-only experiments. Accuracy, precision and recall are reported separately so readers can judge suitability for their curation task. This is a selected set of use cases, not an aggregate performance claim. Each page includes the source dataset, final counts, evaluation method and reproduction instructions.

| Experiment | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| [Balance inquiries](experiments/balance-inquiries.md) | 99.6% | 95.3% | 100% |
| [Card-freeze requests](experiments/card-freeze-requests.md) | 100% | 100% | 100% |
| [Laughter](experiments/laughter.md) | 97.7% | 93.9% | 92.0% |
| [Coughing](experiments/coughing.md) | 94.0% | 79.6% | 86.0% |

These experimental goals run through the preset evaluation scripts. The current web app also accepts freeform objectives, using an additional automatic planning step that these earlier benchmarks did not evaluate. Benchmark scores do not transfer automatically between goals, datasets or recording conditions.
