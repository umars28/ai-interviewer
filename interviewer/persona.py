from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from interviewer import prompts
from interviewer.llm import LLMClient, Role, user
from interviewer.state import InterviewState
from interviewer.views import transcript_view


class Reply(BaseModel):
    text: str


@dataclass(frozen=True)
class Persona:
    id: str
    background: str
    voice: str
    knowledge: str
    hidden_facts: list[str]

    @classmethod
    def from_file(cls, path: Path) -> "Persona":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            id=data.get("id", path.stem),
            background=data["background"].strip(),
            voice=data["voice"].strip(),
            knowledge=data["knowledge"].strip(),
            hidden_facts=[fact.strip() for fact in data["hidden_facts"]],
        )


def load_personas(directory: Path) -> list[Persona]:
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise FileNotFoundError(f"no persona yaml files in {directory}")
    return [Persona.from_file(path) for path in paths]


def synthetic_respondent(persona: Persona, client: LLMClient):
    def respond(question: str, state: InterviewState) -> str:
        rendered = prompts.render(
            "respondent",
            background=persona.background,
            voice=persona.voice,
            knowledge=persona.knowledge,
            hidden_facts="\n".join(f"- {fact}" for fact in persona.hidden_facts),
            question=question,
        )
        history = transcript_view(state.get("transcript", []), limit=6)
        messages = [user(f"Conversation so far:\n{history}\n\n{rendered}")]
        return client.structured(Role.RESPONDENT, messages, Reply).text

    return respond
