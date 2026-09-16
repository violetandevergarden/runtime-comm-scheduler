"""Application-facing asynchronous handle."""

from __future__ import annotations

import enum
import threading
import time
from datetime import timedelta
from typing import Any


class HandleState(enum.Enum):
    PENDING = "pending"
    GRANTED = "granted"
    BOUND = "bound"
    COMPLETED = "completed"
    FAILED = "failed"


class RuntimeHandle:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._condition = threading.Condition()
        self._state = HandleState.PENDING
        self._work: Any = None
        self._error: BaseException | None = None
        self.decision_seq: int | None = None

    @property
    def state(self) -> HandleState:
        with self._condition:
            return self._state

    def grant(self, decision_seq: int) -> None:
        with self._condition:
            if self._state is not HandleState.PENDING:
                raise RuntimeError(f"invalid grant for {self.task_id}: {self._state.value}")
            self.decision_seq = decision_seq
            self._state = HandleState.GRANTED
            self._condition.notify_all()

    def bind(self, work: Any) -> None:
        if work is None or not callable(getattr(work, "wait", None)):
            raise TypeError("runtime launch must return a Work-like object with wait()")
        with self._condition:
            if self._state is not HandleState.GRANTED or self._work is not None:
                raise RuntimeError(f"invalid bind for {self.task_id}: {self._state.value}")
            self._work = work
            self._state = HandleState.BOUND
            self._condition.notify_all()

    def fail(self, error: BaseException) -> bool:
        with self._condition:
            if self._state in (HandleState.COMPLETED, HandleState.FAILED):
                return False
            self._error = error
            self._state = HandleState.FAILED
            self._condition.notify_all()
            return True

    def mark_completed(self) -> bool:
        with self._condition:
            if self._state is HandleState.COMPLETED:
                return False
            if self._state is HandleState.FAILED:
                return False
            if self._work is None:
                raise RuntimeError(f"cannot complete unbound task {self.task_id}")
            self._state = HandleState.COMPLETED
            self._condition.notify_all()
            return True

    def wait_host(self, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._state not in (HandleState.BOUND, HandleState.COMPLETED, HandleState.FAILED):
                remaining = self._remaining(deadline)
                if remaining == 0:
                    return False
                self._condition.wait(remaining)
            self._raise_if_failed_locked()
            work = self._work
        if work is None:
            return self.state is HandleState.COMPLETED
        remaining = self._remaining(deadline)
        try:
            if remaining is None:
                ok = bool(work.wait())
            elif remaining == 0:
                ok = bool(getattr(work, "is_completed", lambda: False)())
            else:
                ok = bool(work.wait(timeout=timedelta(seconds=remaining)))
        except TypeError:
            ok = bool(work.wait())
        if not ok:
            return False
        return True

    def wait_on(self, stream: Any) -> None:
        """Establish a consumer dependency when the backend exposes one."""
        with self._condition:
            while self._work is None and self._error is None:
                self._condition.wait()
            self._raise_if_failed_locked()
            work = self._work
        if work is None:
            return
        wait_stream = getattr(stream, "wait_stream", None)
        if wait_stream is not None:
            wait_stream(work)
        elif callable(getattr(work, "wait", None)):
            work.wait()

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise self._error

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())
