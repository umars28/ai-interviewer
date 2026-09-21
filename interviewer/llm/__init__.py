import os

from interviewer.llm.base import LLMClient, LLMError, Msg, Role, system, user

__all__ = ["LLMClient", "LLMError", "Msg", "Role", "build_client", "system", "user"]


def build_client(backend: str | None = None) -> LLMClient:
    name = (backend or os.getenv("LLM_BACKEND") or "ollama").lower()
    if name == "ollama":
        from interviewer.llm.ollama import OllamaClient

        return OllamaClient()
    if name == "anthropic":
        from interviewer.llm.anthropic import AnthropicClient

        return AnthropicClient()
    if name == "fake":
        from interviewer.llm.fake import FakeClient

        return FakeClient()
    raise LLMError(f"unknown LLM_BACKEND {name!r}; expected ollama, anthropic, or fake")
