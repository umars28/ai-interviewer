import os

import httpx
from pydantic import ValidationError

from interviewer.llm.base import T, LLMError, Msg, Role

MODELS: dict[Role, str] = {
    Role.INTERVIEWER: "qwen3:14b",
    Role.CRITIC: "qwen3:14b",
    Role.ASSESSOR: "qwen3:8b",
    Role.RESPONDENT: "qwen3:8b",
    Role.SYNTHESIZER: "qwen3:14b",
}

DEFAULT_HOST = "http://localhost:11434"
TIMEOUT_SECONDS = 300.0


class OllamaClient:
    def __init__(self, host: str | None = None, models: dict[Role, str] | None = None) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.models = models or MODELS
        self._http = httpx.Client(timeout=TIMEOUT_SECONDS)

    def structured(self, role: Role, messages: list[Msg], schema: type[T]) -> T:
        payload = {
            "model": self.models[role],
            "messages": messages,
            "stream": False,
            "format": schema.model_json_schema(),
            "think": False,
            "options": {"temperature": 0.7},
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
