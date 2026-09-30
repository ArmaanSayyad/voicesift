# Documentation

Start with the [project setup](../README.md#get-started), [setup help](GETTING_STARTED.md), and [dataset workflow](DATASET_RUNS.md).

## Selected experiments

These pages cover selected completed audio-only experiments. Accuracy, precision and recall are reported separately so readers can judge suitability for their curation task. This is a selected set of use cases, not an aggregate performance claim. Each page includes the source dataset, final counts, evaluation method and reproduction instructions.

| Experiment | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| [Balance inquiries](experiments/balance-inquiries.md) | 99.6% | 95.3% | 100% |
| [Card-freeze requests](experiments/card-freeze-requests.md) | 100% | 100% | 100% |
| [Laughter](experiments/laughter.md) | 97.7% | 93.9% | 92.0% |
| [Coughing](experiments/coughing.md) | 94.0% | 79.6% | 86.0% |

These experimental goals run through the CLI. The current web app supports interruption curation. Benchmark scores do not transfer automatically between goals, datasets or recording conditions.
