from collections import defaultdict, deque
from collections.abc import Callable

from pydantic import BaseModel

from interviewer.llm.base import LLMError, Msg, Role, T

Responder = Callable[[list[Msg], type[BaseModel]], BaseModel]


class FakeClient:
    def __init__(self, scripts: dict[Role, list[BaseModel]] | None = None) -> None:
        self._queues: dict[Role, deque[BaseModel]] = defaultdict(deque)
        self._responders: dict[Role, Responder] = {}
        self.calls: list[tuple[Role, list[Msg]]] = []
        for role, items in (scripts or {}).items():
            self._queues[role].extend(items)

    def script(self, role: Role, *items: BaseModel) -> "FakeClient":
        self._queues[role].extend(items)
        return self

    def respond_with(self, role: Role, responder: Responder) -> "FakeClient":
        self._responders[role] = responder
        return self

    def structured(self, role: Role, messages: list[Msg], schema: type[T]) -> T:
        self.calls.append((role, messages))
        if self._queues[role]:
            value = self._queues[role].popleft()
        elif role in self._responders:
            value = self._responders[role](messages, schema)
        else:
            raise LLMError(f"FakeClient has no scripted response for role {role}")
        if not isinstance(value, schema):
            raise LLMError(
                f"FakeClient produced {type(value).__name__} but the node asked for "
                f"{schema.__name__} (role {role}, call {len(self.calls)})"
            )
        return value

    def calls_for(self, role: Role) -> list[list[Msg]]:
        return [messages for called_role, messages in self.calls if called_role == role]
