import json

import httpx
import pytest

from omni_agent.models.ollama import ModelError, OllamaModel


def scripted(responses):
    """A fake Ollama server returning the given (status, body) pairs for /api/chat, in order."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        status, payload = responses.pop(0)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler), calls


def test_empty_json_reply_retries_without_grammar():
    transport, calls = scripted([(200, {"message": {"content": ""}}), (200, {"message": {"content": '{"x": 1}'}})])
    m = OllamaModel("m", transport=transport)
    assert m.generate("p", []) == '{"x": 1}'
    assert "format" in calls[0][1] and "format" not in calls[1][1] and m.recoveries == 1


def test_repeat_limit_error_triggers_reload_then_succeeds():
    transport, calls = scripted([
        (500, {"error": "Unexpected empty grammar stack"}),
        (500, {"error": "prediction aborted, token repeat limit reached"}),
        (200, {"message": {"content": "ok"}}),
    ])
    m = OllamaModel("m", transport=transport)
    assert m.generate("p", []) == "ok"
    assert [c[0] for c in calls] == ["/api/chat", "/api/chat", "/api/generate", "/api/chat"]


def test_persistent_failure_raises_instead_of_returning_empty():
    transport, _ = scripted([(200, {"message": {"content": ""}})] * 3)
    with pytest.raises(ModelError):
        OllamaModel("m", transport=transport).generate("p", [])
