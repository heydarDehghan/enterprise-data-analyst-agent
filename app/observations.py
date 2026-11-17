"""In-process observation hook. Only the diagnostic test installs an observer."""

from contextvars import ContextVar
from typing import Any, Protocol


class Observer(Protocol):
    def __call__(self, event: str, /, **values: Any) -> None: ...


observer: ContextVar[Observer | None] = ContextVar("agent_observer", default=None)


def observe(event: str, **values: Any) -> None:
    sink = observer.get()
    if sink is not None:
        sink(event, **values)
