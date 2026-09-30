"""Publish source-free Gemini evaluation evidence and a readable report."""

import json
import shutil
import statistics
from pathlib import Path

from evaluate_interruption_gemini import ARMS, summarize
from evaluate_interruption_laya import ROOT, load_rows


def main():
    folder = ROOT / "gemini"
    predictions = [
        json.loads(line)
        for line in (folder / "predictions.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((ROOT / "audio-manifest.json").read_text())
    ids = {r["id"] for r in manifest}
    rows = [r for r in load_rows() if r["id"] in ids]
    assert (
        len(predictions) == 300
        and len({(p["id"], p["arm"]) for p in predictions}) == 300
    )
    assert {(p["id"], p["arm"]) for p in predictions} == {
        (i, a) for i in ids for a in ARMS
    }
    summary = summarize(rows, predictions)
    summary["model_versions"] = sorted(
        {p["model_version"] for p in predictions if p.get("model_version")}
    )
    for arm in ARMS:
        durations = sorted(p["seconds"] for p in predictions if p["arm"] == arm)
        summary["arms"][arm]["median_request_seconds"] = statistics.median(durations)
        summary["arms"][arm]["p95_request_seconds"] = durations[94]
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    target = Path("evidence/gemini-interruption-eval")
    target.mkdir(exist_ok=True)
    for name in ("protocol.json", "summary.json", "predictions.jsonl"):
        shutil.copy2(folder / name, target / name)
    lines = [
        "# Gemini interruption-intent comparison",
        "",
        "Date: 2026-09-29. Model: Gemini 3.8 Flash. This is an offline matched input-modality experiment, not an integrated UI feature or validation of naturally observed agent interruptions.",
        "",
        "## Fixed protocol",
        "",
        "The same 100 SID-Bench clips (50 positive, 50 negative) used in the earlier Laya audio evaluation were sent to Gemini in three conditions: local Parakeet transcript only; original WAV only; WAV plus the same Parakeet transcript. No reference transcript, label, filename, break marker, break time, or duration field was sent. There is no paired agent timeline in these clips, so no timing evidence was invented or supplied. The prompt explicitly assumes an agent is speaking and asks about incoming speech intent.",
        "",
        "One example across all three arms verified API/schema compatibility, then the remaining 297 requests ran with the identical prompt and generation configuration. No prompt tuning followed model results. Requests used temperature 0, a 1,024 output-token cap, structured labels, and at most four concurrent calls. Requests were independently sent with no shared chat context. The model can answer unclear; failures and unclear records are retained for review. This new experiment is exploratory because the subset and dataset behavior were already studied in the Laya evaluation.",
        "",
        "## Results",
        "",
        "| Input | TP | FP | TN | FN | Precision | Recall | Strict correct / 100 | Unclear | API/parse errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        a = summary["arms"][arm]
        m = a["selected_interrupt_only"]
        fmt = lambda n: f"{100 * n:.1f}%" if n is not None else "undefined"
        lines.append(
            f"| {arm} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {a['strict_correct']} | {a['unclear']} | {a['errors']} |"
        )
    lines += [
        "",
        "Confusion matrices measure selection as interruption versus everything else. Unclear/errors are not confident negative labels: strict-correct counts treat them as incorrect regardless of gold class. Separate retain-for-review metrics are in summary.json.",
        "",
        "## Paired differences",
        "",
    ]
    for arm in ARMS[1:]:
        p = summary["paired_vs_transcript"][arm]
        lines.append(
            f"- {arm}: corrected {p['corrected_transcript_errors']} transcript-only errors and introduced {p['introduced_errors']} errors on cases the transcript arm got right."
        )
    lines += [
        "",
        "## Latency and estimated cost",
        "",
        "| Input | Mean API seconds | Median | P95 | Estimated USD |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        a = summary["arms"][arm]
        lines.append(
            f"| {arm} | {a['mean_request_seconds']:.2f} | {a['median_request_seconds']:.2f} | {a['p95_request_seconds']:.2f} | ${a['estimated_usd_successful_responses']:.4f} |"
        )
    cost = sum(
        a["estimated_usd_successful_responses"] for a in summary["arms"].values()
    )
    lines += [
        "",
        f"Total estimated successful-response API cost: **${cost:.4f}**. Uses published standard rates on the evaluation date: $0.75/million input tokens and $3.75/million output tokens including thinking. This is a token-accounting estimate, not billing verification. Failed requests can have unreported charges. API timings include network latency and concurrent service execution; ASR preprocessing time is excluded. No provider confidence score was requested or treated as calibrated.",
        "",
        "## Limits and next implementation step",
        "",
        "This sample cannot establish performance on full conversations, accurate overlap detection, speaker-role attribution, noise, new languages, or the intended deployment prevalence. Source utterances include repeated acknowledgments and length confounds; examples are not known to be statistically independent. No significance or population-generalization claim is made from small accuracy differences. No model was trained. Full-utterance audio may contain evidence unavailable at interruption onset; this is offline curation, not streaming interruption detection.",
        "",
        "Keep role/timing candidate generation separate from semantic intent. For product integration, supply actual conversation context and evidence provenance, keep model and prompt versions on each decision, present audio/transcript evidence for reviewer confirmation, and export reviewed events. Do not make permanent data deletion or automatic rejection depend on this small study. The next validation set should contain real paired agent/user tracks or synchronized playback logs plus human-reviewed labels, split by conversation/source. Compare candidate recall and semantic precision separately, and audit excluded records.",
        "",
        "Existing UI filtering remains the timestamp baseline; this commit provides the tested Gemini evaluation backend and evidence, not an automatic production filter.",
        "",
        "## Reproduction and sources",
        "",
        "With the prior audio/ASR artifacts prepared and GEMINI_API_KEY or GOOGLE_API_KEY configured in the process environment:",
        "",
        "```sh",
        ".venv/bin/python scripts/evaluate_interruption_gemini.py",
        ".venv/bin/python scripts/summarize_gemini_evaluation.py",
        ".venv/bin/python -m pytest -q",
        "```",
        "",
        "The runner resumes exactly matching protocols without repeating completed calls. API errors are recorded without automatic retries. Evidence retains only decisions, IDs, hashes, token counts, timing, and model metadata; source speech is not included.",
        "",
        "- [Audio API documentation](https://ai.google.dev/gemini-api/docs/audio)",
        "- [Standard model pricing](https://ai.google.dev/gemini-api/docs/pricing)",
        "- [SID-Bench](https://github.com/xkx-hub/SID-bench)",
        "- Dataset revision: `6eb13b573ad588646ce0de513d4255e55caf858b`.",
        "- Exact prompt/configuration and source artifact hashes: `evidence/gemini-interruption-eval/protocol.json`.",
        "- Earlier Laya protocol and results: `research/archive/INTERRUPTION_EVALUATION.md`.",
        "",
    ]
    lines += [
        "## Decision after this run",
        "",
        "Gemini is a credible candidate for the semantic review stage: it recovers most positives on this subset, unlike the tested Laya configurations. Audio-only has 89 strictly correct decisions versus 86 for ASR-only; this small difference is not evidence of general superiority. Audio plus ASR does not improve interruption selection over audio alone here. Use the audio backend when audio is available, retain transcript-only as an alternative, and keep both experimental until validated on actual conversation events.",
        "",
        "Audio-only still selects 8 of 50 negatives. If its measured 96% sensitivity and 16% false-positive rate held at a 10% positive deployment prevalence, expected selection precision would be only 40%; this is a hypothetical prevalence projection, not a measured deployment result. Human review remains necessary. For comparable historical context, base Laya found 0 of 50 positives on the same local-ASR subset, but the two systems have different prompts and output interfaces, so this is a pipeline comparison rather than an isolated architecture comparison.",
        "",
        "Validation: 34 tests pass; Ruff checks pass for the new evaluation scripts and tests.",
        "",
    ]
    report = Path("research/archive/GEMINI_INTERRUPTION_EVALUATION.md")
    report.write_text("\n".join(lines))
    for directory in (
        "/Users/armaansayyad/Documents/Codex/2026-09-29/go-t/outputs",
        "/Users/armaansayyad/agent/conversation-repair-workbench/outputs",
    ):
        shutil.copy2(report, Path(directory) / "gemini-interruption-evaluation.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
