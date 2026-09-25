import os
import time

import httpx
from pydantic import ValidationError

from interviewer.llm.base import LLMError, Msg, Role, T

MODELS: dict[Role, str] = {
    Role.INTERVIEWER: "claude-sonnet",
    Role.CRITIC: "claude-sonnet",
    Role.ASSESSOR: "claude-sonnet",
    Role.RESPONDENT: "claude-haiku",
    Role.SYNTHESIZER: "claude-sonnet",
}

DEFAULT_HOST = "https://llm.devopsinstitute.id"
TOOL_NAME = "emit"
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 120.0
WRITE_TIMEOUT = 30.0
POOL_TIMEOUT = 10.0
DEADLINE_SECONDS = 420.0
DEFAULT_MAX_TOKENS = 8192
MAX_ATTEMPTS = 4
RETRY_BASE_SECONDS = 2.0
RETRY_MAX_SECONDS = 60.0
RETRYABLE_STATUS = {408, 409, 429}


def resolve_models() -> dict[Role, str]:
    override_all = os.getenv("GATEWAY_MODEL")
    resolved = {}
    for role in Role:
        per_role = os.getenv(f"GATEWAY_MODEL_{role.name}")
        resolved[role] = per_role or override_all or MODELS[role]
    return resolved


def retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return min(max(float(raw), 0.0), RETRY_MAX_SECONDS)
    except ValueError:
        return None


def strip_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return body.rsplit("```", 1)[0].strip()


class GatewayClient:
    def __init__(
        self,
        host: str | None = None,
        api_key: str | None = None,
        models: dict[Role, str] | None = None,
    ) -> None:
        self.host = (host or os.getenv("LLM_GATEWAY_HOST") or DEFAULT_HOST).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or ""
        if not self.api_key:
            raise LLMError(
                "LLM_API_KEY is not set; export it before using LLM_BACKEND=gateway"
            )
        self.models = models or resolve_models()
        self.max_tokens = int(os.getenv("GATEWAY_MAX_TOKENS") or DEFAULT_MAX_TOKENS)
        self._http = httpx.Client(
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT,
                read=READ_TIMEOUT,
                write=WRITE_TIMEOUT,
                pool=POOL_TIMEOUT,
            ),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        self._sleep = time.sleep
        self._now = time.monotonic
        self.usage: dict[str, dict[str, int]] = {}

    def structured(self, role: Role, messages: list[Msg], schema: type[T]) -> T:
        payload = {
            "model": self.models[role],
            "messages": messages,
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "description": f"Emit one {schema.__name__} object.",
                        "parameters": schema.model_json_schema(),
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        }
        content = ""
        last: Exception | None = None
        attempts = 0
        delay: float | None = None
        started = self._now()
        for attempt in range(MAX_ATTEMPTS):
            if attempt and self._now() - started > DEADLINE_SECONDS:
                break
            attempts = attempt + 1
            delay = None
            try:
                content = self._post(payload)
                return schema.model_validate_json(strip_fence(content))
            except httpx.HTTPStatusError as exc:
                last = exc
                status = exc.response.status_code
                if status < 500 and status not in RETRYABLE_STATUS:
                    break
                delay = retry_after(exc.response)
            except httpx.TransportError as exc:
                last = exc
            except ValidationError as exc:
                last = exc
            if attempt + 1 < MAX_ATTEMPTS:
                backoff = min(RETRY_BASE_SECONDS * 2**attempt, RETRY_MAX_SECONDS)
                self._sleep(delay if delay is not None else backoff)
        if isinstance(last, ValidationError):
            raise LLMError(
                f"gateway returned invalid {schema.__name__} for role {role} after "
                f"{attempts} attempts: {content[:400]}"
            ) from last
        detail = last
        if isinstance(last, httpx.HTTPStatusError):
            detail = f"{last} :: {last.response.text[:400]}"
        raise LLMError(
            f"gateway request failed for role {role} "
            f"(model {self.models[role]}) after {attempts} attempts: {detail}"
        ) from last

    def _post(self, payload: dict) -> str:
        response = self._http.post(f"{self.host}/v1/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        self._record(payload["model"], body.get("usage") or {})
        message = body["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if calls:
            return calls[0]["function"]["arguments"]
        return message.get("content") or ""

    def _record(self, model: str, usage: dict) -> None:
        bucket = self.usage.setdefault(model, {"calls": 0, "input": 0, "output": 0})
        bucket["calls"] += 1
        bucket["input"] += int(usage.get("prompt_tokens") or 0)
        bucket["output"] += int(usage.get("completion_tokens") or 0)

    def usage_summary(self) -> str:
        if not self.usage:
            return "gateway usage: no calls recorded"
        header = f"{'model':<16}{'calls':>7}{'input tok':>12}{'output tok':>12}"
        lines = ["gateway usage:", header, "-" * len(header)]
        totals = {"calls": 0, "input": 0, "output": 0}
        for model in sorted(self.usage):
            b = self.usage[model]
            for key in totals:
                totals[key] += b[key]
            lines.append(f"{model:<16}{b['calls']:>7}{b['input']:>12,}{b['output']:>12,}")
        lines.append("-" * len(header))
        lines.append(
            f"{'total':<16}{totals['calls']:>7}{totals['input']:>12,}{totals['output']:>12,}"
        )
        return "\n".join(lines)

    def close(self) -> None:
        self._http.close()

    def available_models(self) -> set[str]:
        try:
            response = self._http.get(f"{self.host}/v1/models", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"cannot reach gateway at {self.host}: {exc}") from exc
        return {model["id"] for model in response.json().get("data", [])}

    def check_ready(self) -> None:
        available = self.available_models()
        missing = sorted(set(self.models.values()) - available)
        if missing:
            names = ", ".join(missing)
            offered = ", ".join(sorted(available)) or "none"
            raise LLMError(
                f"gateway {self.host} does not serve: {names}. Available: {offered}"
            )
