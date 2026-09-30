# PRESTO experiment evidence

Source: [PRESTO, Google Research](https://github.com/google-research-datasets/presto), [paper](https://aclanthology.org/2023.emnlp-main.667/), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

These files record a final-turn text screening experiment, not audio accuracy or human-verified curation purity. `protocol.json` freezes the model, prompt, request configuration, sampling seed and source/selection hashes. `selection.json` contains source example identifiers and sampling groups without dialogue text. Local raw packets and model notes remain in ignored `artifacts/presto-evaluation/`.

Source member: `test_partitions/en-US/test.jsonl` inside [presto_v1.zip](https://storage.googleapis.com/gresearch/presto/presto_v1.zip). The evaluator checks its exact SHA256 before running. No source labels, targets, example IDs, or sampling groups are sent to Gemini. Labels are used only for stratified sampling and evaluation.

Reproduce from the repository root after obtaining the source member:

```sh
.venv/bin/python scripts/evaluate_presto.py --source /path/to/test.jsonl --run
```

The script uses the existing Gemini API environment variable, or prompts without echoing the key. It freezes selection before inference, runs four requests concurrently, caches all responses (including errors), and does not selectively retry failures. Re-running the same output directory resumes cached requests. Use a different `--out` directory for a fresh run. Model generation is not guaranteed deterministic despite temperature zero.

The four tagged-positive strata contain 100 examples each. The two comparison strata contain 100 each and have **unknown binary truth for our goal**. Therefore precision and overall accuracy are intentionally null. The sample is deliberately stratified rather than representative of source prevalence. The source test split had previously been inspected during dataset research; this is a development experiment, not a claim of an untouched held-out evaluation.
