"""Application-facing asynchronous handle."""

from __future__ import annotations

import enum
import threading
import time
from typing import Any


class HandleState(enum.Enum):
    PENDING = "pending"
    GRANTED = "granted"
    BOUND = "bound"
    COMPLETED = "completed"
    FAILED = "failed"

"""
应用调用 runtime.submit()
            │
            ▼
         PENDING
    等待 coordinator 调度
            │ grant()
            ▼
         GRANTED
    已准入，等待本地 launch
            │ bind(work)
            ▼
          BOUND
    已绑定真实 PyTorch Work
            │ mark_completed()
            ▼
        COMPLETED

  任意未结束状态 ── fail(error) ──> FAILED
"""


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
        has_work_wait = callable(getattr(work, "wait", None))
        has_stream_receipt = (callable(getattr(work, "is_completed", None))
                              and callable(getattr(work, "wait_on", None)))
        if work is None or not (has_work_wait or has_stream_receipt):
            raise TypeError("runtime launch must return a Work-like object or stream completion receipt")
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
            while self._state not in (HandleState.COMPLETED, HandleState.FAILED):
                remaining = self._remaining(deadline)
                if remaining == 0:
                    return False
                self._condition.wait(remaining)
            self._raise_if_failed_locked()
            return True

    def wait_on(self, stream: Any, timeout: float | None = 20.0) -> bool:
        """Establish a dependency on this collective for one consumer stream.

        Returns when the dependency has been enqueued, not when the GPU work
        has physically completed. The default binding deadline is finite.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while (self._work is None and self._error is None
                   and self._state not in (HandleState.COMPLETED, HandleState.FAILED)):
                remaining = self._remaining(deadline)
                if remaining == 0:
                    return False
                self._condition.wait(remaining)
            self._raise_if_failed_locked()
            work = self._work
        if work is None:
            raise RuntimeError(f"task {self.task_id} completed without an execution receipt")
        add_dependency = getattr(work, "wait_on", None)
        if not callable(add_dependency):
            raise NotImplementedError("backend execution receipt does not support CUDA stream dependencies")
        return bool(add_dependency(stream, timeout=timeout))

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise self._error

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())
