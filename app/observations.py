"""In-process observation hook. Only the diagnostic test installs an observer."""

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any


Observer = Callable[[str, dict[str, Any]], None]
observer: ContextVar[Observer | None] = ContextVar("agent_observer", default=None)


def observe(event: str, **values: Any) -> None:
    sink = observer.get()
    if sink is not None:
        sink(event, values)
