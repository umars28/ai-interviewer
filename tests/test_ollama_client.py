import httpx
import pytest

from interviewer.llm.base import LLMError, Role, user
from interviewer.llm.ollama import MAX_ATTEMPTS, OllamaClient, resolve_models
from interviewer.state import PlannedQuestion

GOOD_BODY = {"message": {"content": '{"goal_id": "g1", "question": "What happened?"}'}}


def client_with(handler) -> OllamaClient:
    client = OllamaClient(host="http://ollama.test")
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    client._sleep = lambda _seconds: None
    return client


def test_a_successful_call_parses_into_the_schema():
    def handler(_request):
        return httpx.Response(200, json=GOOD_BODY)

    result = client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert result.question == "What happened?"


def test_a_transient_server_error_is_retried_and_the_interview_survives():
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, text="model loading failed")
        return httpx.Response(200, json=GOOD_BODY)

    result = client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert result.question == "What happened?"
    assert calls["n"] == 3


def test_persistent_server_errors_give_up_after_the_attempt_budget():
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        return httpx.Response(500, text="still broken")

    with pytest.raises(LLMError, match=f"after {MAX_ATTEMPTS} attempts"):
        client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert calls["n"] == MAX_ATTEMPTS


def test_a_client_error_is_not_retried():
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        return httpx.Response(404, text="no such model")

    with pytest.raises(LLMError):
        client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert calls["n"] == 1


def test_connection_failures_are_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json=GOOD_BODY)

    result = client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert result.question == "What happened?"


def test_malformed_json_is_reported_with_the_offending_content():
    def handler(_request):
        return httpx.Response(200, json={"message": {"content": "not json at all"}})

    with pytest.raises(LLMError, match="not json at all"):
        client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)


def test_env_overrides_one_role_without_touching_the_others(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL_CRITIC", "llama3.1:70b")
    models = resolve_models()

    assert models[Role.CRITIC] == "llama3.1:70b"
    assert models[Role.RESPONDENT] == "qwen3:8b"


def test_a_blanket_override_applies_to_every_role(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    models = resolve_models()

    assert set(models.values()) == {"qwen3:8b"}
