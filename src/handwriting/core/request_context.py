from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)

def get_request_id() -> str | None:
    return _request_id.get()

@contextmanager
def request_context(request_id: str) -> Iterator[None]:
    token = _request_id.set(request_id)

    try:
        yield
    finally:
        _request_id.reset(token)