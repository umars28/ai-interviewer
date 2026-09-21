from interviewer.llm.base import LLMError, Msg, Role, T

MODELS: dict[Role, str] = {
    Role.INTERVIEWER: "claude-opus-5",
    Role.CRITIC: "claude-opus-5",
    Role.ASSESSOR: "claude-opus-5",
    Role.RESPONDENT: "claude-haiku-4-5",
    Role.SYNTHESIZER: "claude-opus-5",
}


class AnthropicClient:
    def __init__(self, models: dict[Role, str] | None = None) -> None:
        self.models = models or MODELS

    def structured(self, role: Role, messages: list[Msg], schema: type[T]) -> T:
        raise LLMError(
            "AnthropicClient is not implemented yet. Set LLM_BACKEND=ollama, or implement "
            "this using anthropic.Anthropic().messages.parse() with output_config.format "
            "and an ANTHROPIC_API_KEY from console.anthropic.com."
        )
