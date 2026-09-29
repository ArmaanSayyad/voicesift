import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "eval_gemini",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_interruption_gemini.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_modalities_are_isolated():
    text = m.payload("transcript", "AUTOMATIC SPEECH", b"AUDIO BYTES")
    audio = m.payload("audio", "AUTOMATIC SPEECH", b"AUDIO BYTES")
    assert "inlineData" not in str(text)
    assert "AUTOMATIC SPEECH" not in str(audio)
    assert (
        len(m.payload("audio_transcript", "words", b"a")["contents"][0]["parts"]) == 2
    )


def test_failed_or_invalid_answers_cannot_be_silent_negatives():
    with pytest.raises(ValueError):
        m.parse_response({"candidates": []})
    with pytest.raises(ValueError):
        m.parse_response({"candidates": [{"finishReason": "MAX_TOKENS"}]})
    rows = [{"id": "x", "positive": True}]
    ps = [
        {"id": "x", "arm": "audio", "positive": True, "status": "error", "seconds": 1}
    ]
    result = m.summarize(rows, ps)["arms"]["audio"]
    assert result["errors"] == 1
    assert result["selected_interrupt_only"]["fn"] == 1
    assert result["retained_for_review_including_unknowns"]["tp"] == 1


def test_answer_parser_ignores_thoughts_and_rejects_extra_fields():
    response = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": "private reasoning", "thought": True},
                        {"text": '{"intent":"interrupt"}'},
                    ]
                },
            }
        ]
    }
    assert m.parse_response(response) == "interrupt"
    response["candidates"][0]["content"]["parts"] = [
        {"text": '{"intent":"interrupt","extra":"ignored?"}'}
    ]
    with pytest.raises(ValueError):
        m.parse_response(response)
