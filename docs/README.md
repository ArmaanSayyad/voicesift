# Documentation

Start with the [project setup](../README.md#get-started), [setup help](GETTING_STARTED.md), and [dataset workflow](DATASET_RUNS.md).

## Project scope and current capabilities

VoiceSift's purpose is LALM-based curation of voice datasets from larger collections using freeform text objectives. Interruption mining is its first implemented web workflow, not the boundary of the project.

| Scope | Status |
|---|---|
| Describe an arbitrary objective, run it on a dataset, review and export | Intended product workflow; not implemented end to end yet |
| Successful-interruption curation | Current web app; fixed requirement, timed transcripts plus audio |
| Other voice objectives | Preset audio-only evaluation scripts, including semantic requests and vocal events |

Audio-only experiments do not require the timed conversation transcripts used by the interruption workflow. The scripts accept their configured requirements; they do not expose a general freeform-objective command-line interface.

## Selected experiments

These pages cover selected completed audio-only experiments. Accuracy, precision and recall are reported separately so readers can judge suitability for their curation task. This is a selected set of use cases, not an aggregate performance claim. Each page includes the source dataset, final counts, evaluation method and reproduction instructions.

| Experiment | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| [Balance inquiries](experiments/balance-inquiries.md) | 99.6% | 95.3% | 100% |
| [Card-freeze requests](experiments/card-freeze-requests.md) | 100% | 100% | 100% |
| [Laughter](experiments/laughter.md) | 97.7% | 93.9% | 92.0% |
| [Coughing](experiments/coughing.md) | 94.0% | 79.6% | 86.0% |

These experimental goals run through the preset evaluation scripts. The current web app supports the fixed interruption objective. Benchmark scores do not transfer automatically between goals, datasets or recording conditions.
