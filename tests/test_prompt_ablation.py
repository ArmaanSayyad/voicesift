"""Check paired request construction without sending any model requests."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ablation", Path(__file__).parents[1] / "scripts/run_prompt_ablation.py"
)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def fixture():
    target = {
        "role": "user",
        "parts": [{"text": "target"}, {"inlineData": {"data": "target-audio"}}],
    }
    example = (
        {"answer": "normal_handoff", "note": "completed thought"},
        {
            "role": "user",
            "parts": [{"text": "example"}, {"inlineData": {"data": "example-audio"}}],
        },
    )
    return target, [example]


def test_all_arms_preserve_target_and_generation_budget():
    target, examples = fixture()
    for arm in ablation.ARMS:
        body = ablation.make_body(arm, target, examples)
        assert body["contents"][-1] == target
        assert body["generationConfig"]["temperature"] == 0
        assert body["generationConfig"]["maxOutputTokens"] == 1024
    assert len(target["parts"]) == 2


def test_text_and_audio_demonstrations_only_differ_by_example_audio():
    target, examples = fixture()
    text = ablation.make_body("text_examples", target, examples)
    audio = ablation.make_body("audio_examples", target, examples)
    assert text["systemInstruction"] == audio["systemInstruction"]
    assert text["generationConfig"] == audio["generationConfig"]
    assert text["contents"][0]["parts"] == audio["contents"][0]["parts"][:1]
    assert text["contents"][1:] == audio["contents"][1:]
    assert len(audio["contents"][0]["parts"]) == 2


def test_control_has_no_demonstrations_or_new_schema():
    target, examples = fixture()
    body = ablation.make_body("original", target, examples)
    assert body["contents"] == [target]
    assert body["systemInstruction"]["parts"][0]["text"] == ablation.PROMPT
    assert body["generationConfig"]["responseSchema"] == ablation.SCHEMA
    assert "normal_handoff" not in ablation.SCHEMA["properties"]["intent"]["enum"]


def test_incomplete_provider_response_retains_usage_and_is_not_a_negative(monkeypatch):
    import io
    import json

    monkeypatch.setenv("GEMINI_API_KEY", "unit-test")
    monkeypatch.setattr(
        ablation.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.StringIO(
            json.dumps(
                {
                    "candidates": [{"finishReason": "MAX_TOKENS"}],
                    "usageMetadata": {
                        "promptTokenCount": 50,
                        "thoughtsTokenCount": 1024,
                    },
                }
            )
        ),
    )
    target, examples = fixture()
    result = ablation.call(ablation.make_body("original", target, examples))
    assert result["status"] == "error"
    assert result["finish_reason"] == "MAX_TOKENS"
    assert result["usage"]["thoughtsTokenCount"] == 1024
    assert "answer" not in result


def test_refined_category_allowed_only_by_refined_schema(monkeypatch):
    import io
    import json

    monkeypatch.setenv("GEMINI_API_KEY", "unit-test")
    answer = {
        "intent": "normal_handoff",
        "agent_outcome": "unknown",
        "evidence_note": "completed thought",
    }
    monkeypatch.setattr(
        ablation.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.StringIO(
            json.dumps(
                {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": json.dumps(answer)}]},
                        }
                    ],
                }
            )
        ),
    )
    target, examples = fixture()
    assert (
        ablation.call(ablation.make_body("definitions", target, examples))["status"]
        == "complete"
    )
    assert (
        ablation.call(ablation.make_body("original", target, examples))["error"]
        == "invalid_response"
    )
