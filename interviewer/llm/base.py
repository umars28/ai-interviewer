from enum import StrEnum
from typing import Protocol, TypedDict, TypeVar

from pydantic import BaseModel


class Role(StrEnum):
    INTERVIEWER = "interviewer"
    CRITIC = "critic"
    ASSESSOR = "assessor"
    RESPONDENT = "respondent"
    SYNTHESIZER = "synthesizer"


class Msg(TypedDict):
    role: str
    content: str


T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def structured(self, role: Role, messages: list[Msg], schema: type[T]) -> T: ...


class LLMError(RuntimeError):
    pass


def system(content: str) -> Msg:
    return {"role": "system", "content": content}


def user(content: str) -> Msg:
    return {"role": "user", "content": content}
