import os

import httpx
from pydantic import ValidationError

from interviewer.llm.base import LLMError, Msg, Role, T

MODELS: dict[Role, str] = {
    Role.INTERVIEWER: "qwen3:14b",
    Role.CRITIC: "qwen3:14b",
    Role.ASSESSOR: "qwen3:8b",
    Role.RESPONDENT: "qwen3:8b",
    Role.SYNTHESIZER: "qwen3:14b",
}

DEFAULT_HOST = "http://localhost:11434"
TIMEOUT_SECONDS = 300.0
DEFAULT_NUM_CTX = 16384


def resolve_models() -> dict[Role, str]:
    override_all = os.getenv("OLLAMA_MODEL")
    resolved = {}
    for role in Role:
        per_role = os.getenv(f"OLLAMA_MODEL_{role.name}")
        resolved[role] = per_role or override_all or MODELS[role]
    return resolved


class OllamaClient:
    def __init__(self, host: str | None = None, models: dict[Role, str] | None = None) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.models = models or resolve_models()
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX") or DEFAULT_NUM_CTX)
        self._http = httpx.Client(timeout=TIMEOUT_SECONDS)

    def structured(self, role: Role, messages: list[Msg], schema: type[T]) -> T:
        payload = {
            "model": self.models[role],
            "messages": messages,
            "stream": False,
            "format": schema.model_json_schema(),
            "think": False,
            "options": {"temperature": 0.7, "num_ctx": self.num_ctx},
        }
        try:
            response = self._http.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed for role {role}: {exc}") from exc

        content = response.json()["message"]["content"]
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            raise LLMError(
                f"ollama returned invalid {schema.__name__} for role {role}: {content[:400]}"
            ) from exc

    def close(self) -> None:
        self._http.close()

    def available_models(self) -> set[str]:
        try:
            response = self._http.get(f"{self.host}/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"cannot reach ollama at {self.host}: {exc}") from exc
        return {model["name"] for model in response.json().get("models", [])}

    def check_ready(self) -> None:
        available = self.available_models()
        missing = sorted(set(self.models.values()) - available)
        if missing:
            pulls = "\n".join(f"  ollama pull {name}" for name in missing)
            raise LLMError(f"missing ollama models:\n{pulls}")
