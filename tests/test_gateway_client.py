import json

import httpx
import pytest

from interviewer.llm.base import LLMError, Role, user
from interviewer.llm.gateway import (
    CONNECT_TIMEOUT,
    DEADLINE_SECONDS,
    MAX_ATTEMPTS,
    READ_TIMEOUT,
    RETRY_MAX_SECONDS,
    GatewayClient,
    resolve_models,
)
from interviewer.state import PlannedQuestion

PAYLOAD = '{"goal_id": "g1", "question": "What happened?"}'
GOOD_BODY = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "emit", "arguments": PAYLOAD}}
                ],
            }
        }
    ]
}


def body_with(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def client_with(handler) -> GatewayClient:
    client = GatewayClient(host="https://gateway.test", api_key="k")
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    client._sleep = lambda _seconds: None
    return client


def test_a_successful_call_parses_into_the_schema():
    def handler(_request):
        return httpx.Response(200, json=GOOD_BODY)

    result = client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert result.question == "What happened?"


def test_the_schema_is_sent_as_a_forced_tool_call_not_response_format():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=GOOD_BODY)

    client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    body = seen["body"]
    assert "response_format" not in body
    assert body["tool_choice"] == {"type": "function", "function": {"name": "emit"}}
    assert body["tools"][0]["function"]["parameters"] == PlannedQuestion.model_json_schema()


def test_prose_instead_of_a_tool_call_is_retried():
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(200, json=body_with("Understood. Here is my question:"))
        return httpx.Response(200, json=GOOD_BODY)

    result = client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert result.question == "What happened?"
    assert calls["n"] == 3


def test_the_bearer_token_is_sent_on_every_request():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=GOOD_BODY)

    client = GatewayClient(host="https://gateway.test", api_key="secret-key")
    client._http = httpx.Client(
        transport=httpx.MockTransport(handler), headers=client._http.headers
    )
    client.structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert seen["auth"] == "Bearer secret-key"


def test_a_missing_host_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("LLM_GATEWAY_HOST", raising=False)

    with pytest.raises(LLMError, match="LLM_GATEWAY_HOST is not set"):
        GatewayClient(api_key="k")


def test_a_missing_api_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(LLMError, match="LLM_API_KEY is not set"):
        GatewayClient(host="https://gateway.test")


def test_a_transient_server_error_is_retried_and_the_interview_survives():
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, text="upstream overloaded")
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


def test_an_expired_key_is_not_retried():
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        return httpx.Response(401, text="invalid api key")

    with pytest.raises(LLMError):
        client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert calls["n"] == 1


def test_a_rate_limit_is_retried_rather_than_killing_the_interview():
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        return httpx.Response(200, json=GOOD_BODY)

    result = client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert result.question == "What happened?"
    assert calls["n"] == 3


def test_a_retry_after_header_overrides_the_exponential_backoff():
    slept: list[float] = []

    def handler(_request):
        return httpx.Response(429, headers={"retry-after": "7"}, text="slow down")

    client = client_with(handler)
    client._sleep = slept.append

    with pytest.raises(LLMError):
        client.structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert slept == [7.0, 7.0, 7.0]


def test_an_absurd_retry_after_is_capped():
    slept: list[float] = []

    def handler(_request):
        return httpx.Response(429, headers={"retry-after": "86400"}, text="tomorrow")

    client = client_with(handler)
    client._sleep = slept.append

    with pytest.raises(LLMError):
        client.structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert set(slept) == {RETRY_MAX_SECONDS}


def test_a_read_timeout_is_bounded_per_phase_not_by_one_blanket_value():
    client = GatewayClient(host="https://gateway.test", api_key="k")

    assert client._http.timeout.connect == CONNECT_TIMEOUT
    assert client._http.timeout.read == READ_TIMEOUT


def test_a_slow_gateway_stops_retrying_once_the_deadline_passes():
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        return httpx.Response(500, text="wedged")

    client = client_with(handler)
    ticks = iter([0.0, DEADLINE_SECONDS + 1.0])
    client._now = lambda: next(ticks)

    with pytest.raises(LLMError):
        client.structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

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


def test_json_wrapped_in_a_markdown_fence_is_still_accepted():
    def handler(_request):
        return httpx.Response(200, json=body_with(f"```json\n{PAYLOAD}\n```"))

    result = client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert result.question == "What happened?"


def test_a_schema_violation_is_retried_before_giving_up():
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(200, json=body_with("here you go: not json"))
        return httpx.Response(200, json=GOOD_BODY)

    result = client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert result.question == "What happened?"
    assert calls["n"] == 3


def test_persistent_schema_violations_are_reported_with_the_offending_content():
    def handler(_request):
        return httpx.Response(200, json=body_with("not json at all"))

    with pytest.raises(LLMError, match="not json at all"):
        client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)


def test_token_usage_accumulates_per_model_across_calls():
    def handler(_request):
        return httpx.Response(
            200,
            json={**GOOD_BODY, "usage": {"prompt_tokens": 100, "completion_tokens": 20}},
        )

    client = client_with(handler)
    client.structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)
    client.structured(Role.RESPONDENT, [user("go")], PlannedQuestion)
    client.structured(Role.RESPONDENT, [user("go")], PlannedQuestion)

    assert client.usage["claude-sonnet"] == {"calls": 1, "input": 100, "output": 20}
    assert client.usage["claude-haiku"] == {"calls": 2, "input": 200, "output": 40}
    assert "claude-haiku" in client.usage_summary()


def test_a_response_without_a_usage_block_still_counts_the_call():
    def handler(_request):
        return httpx.Response(200, json=GOOD_BODY)

    client = client_with(handler)
    client.structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert client.usage["claude-sonnet"] == {"calls": 1, "input": 0, "output": 0}


def test_retried_attempts_are_all_counted_not_just_the_one_that_worked():
    calls = {"n": 0}

    def handler(_request):
        calls["n"] += 1
        body = {"usage": {"prompt_tokens": 50, "completion_tokens": 10}}
        if calls["n"] < 3:
            return httpx.Response(200, json={**body_with("not json"), **body})
        return httpx.Response(200, json={**GOOD_BODY, **body})

    client = client_with(handler)
    client.structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)

    assert client.usage["claude-sonnet"] == {"calls": 3, "input": 150, "output": 30}


def test_check_ready_names_the_models_the_gateway_does_not_serve():
    def handler(_request):
        return httpx.Response(200, json={"data": [{"id": "claude-haiku"}]})

    with pytest.raises(LLMError, match="claude-sonnet"):
        client_with(handler).check_ready()


def test_check_ready_passes_when_every_configured_model_is_served():
    def handler(_request):
        return httpx.Response(
            200,
            json={"data": [{"id": "claude-haiku"}, {"id": "claude-sonnet"}]},
        )

    client_with(handler).check_ready()


def test_the_upstream_error_body_is_surfaced_not_just_the_status_line():
    def handler(_request):
        return httpx.Response(400, json={"error": {"message": "Extra inputs are not permitted"}})

    with pytest.raises(LLMError, match="Extra inputs are not permitted"):
        client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)


def test_a_client_error_reports_one_attempt_not_the_whole_budget():
    def handler(_request):
        return httpx.Response(400, text="bad payload")

    with pytest.raises(LLMError, match="after 1 attempts"):
        client_with(handler).structured(Role.INTERVIEWER, [user("go")], PlannedQuestion)


def test_env_overrides_one_role_without_touching_the_others(monkeypatch):
    monkeypatch.setenv("GATEWAY_MODEL_CRITIC", "claude-opus")
    models = resolve_models()

    assert models[Role.CRITIC] == "claude-opus"
    assert models[Role.RESPONDENT] == "claude-haiku"


def test_a_blanket_override_applies_to_every_role(monkeypatch):
    monkeypatch.setenv("GATEWAY_MODEL", "claude-haiku")
    models = resolve_models()

    assert set(models.values()) == {"claude-haiku"}
