import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "presto_eval", Path(__file__).parents[1] / "scripts/evaluate_presto.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_packet_excludes_gold_and_preserves_context():
    row = {
        "inputs": "Actually Wednesday",
        "targets": "GOLD_PARSE",
        "metadata": {
            "linguistic_phenomena": "GOLD_LABEL",
            "example_id": "GOLD_ID",
            "previous_turns": [
                {"user_query": "Book Tuesday", "response_text": "What time?"}
            ],
        },
    }
    packet = evaluation.packet(row)
    assert packet == {
        "preceding_dialogue": [
            {"role": "user", "text": "Book Tuesday"},
            {"role": "assistant", "text": "What time?"},
        ],
        "final_user_utterance": "Actually Wednesday",
    }
    assert "GOLD" not in str(evaluation.body(packet))


def test_summary_never_counts_unknown_comparisons_as_false_positives():
    rows = [
        {"group": "cancel-action", "answer": {"label": "cancellation"}},
        {"group": "cancel-action", "status": "error"},
        {"group": "untagged_human_context", "answer": {"label": "correction"}},
    ]
    result = evaluation.summarize(rows)
    assert result["groups"]["cancel-action"]["total"] == 2
    assert result["groups"]["cancel-action"]["not_selected"] == 1
    assert result["groups"]["untagged_human_context"]["selected"] == 1
    assert result["precision"] is None and result["overall_accuracy"] is None
