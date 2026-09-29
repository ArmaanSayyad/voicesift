import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "eval_laya",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_interruption_laya.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_marker_removal_and_label_metrics():
    assert m.clean("Yep. <break> Wait!") == "Yep. Wait!"
    rows = [{"id": "a", "positive": True}, {"id": "b", "positive": False}]
    s = m.score(rows, {"a": True, "b": True})
    assert (s["tp"], s["fp"], s["tn"], s["fn"]) == (1, 1, 0, 0)
    assert s["precision"] == 0.5 and s["recall"] == 1 and s["balanced_accuracy"] == 0.5
    assert m.score(rows, {})["accuracy"] is None
