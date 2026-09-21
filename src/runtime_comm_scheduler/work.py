"""Deferred binding boundary for scheduler-managed collective work.

``ScheduledWork.wait`` preserves the underlying backend's wait contract. In
particular, ProcessGroupNCCL wait inserts a completion dependency into the
caller's current CUDA stream; it is not turned into a device-wide CPU wait.
Physical completion is observed independently by the scheduler's completion
probe.
"""

from __future__ import annotations

import threading
import time
from datetime import timedelta
from typing import Any, Callable, Optional

from .intent import CommIntent, IntentState
from .telemetry import CommTiming, now_us


class ScheduledWork:
    """A Work-like object that may be returned before ProcessGroup launch."""

    def __init__(
        self,
        intent: CommIntent,
        timing: CommTiming,
        *,
        on_error: Optional[Callable[["ScheduledWork", BaseException], None]] = None,
    ) -> None:
        self._intent = intent
        self._timing = timing
        self._on_error = on_error
        self._condition = threading.Condition()
        self._underlying: Optional[Any] = None
        self._error: Optional[BaseException] = None
        self._completed = False

    @property
    def key(self):
        return self._intent.key

    @property
    def timing(self) -> CommTiming:
        return self._timing

    @property
    def intent(self) -> CommIntent:
        return self._intent

    @property
    def is_bound(self) -> bool:
        with self._condition:
            return self._underlying is not None

    def bind(self, underlying: Any) -> None:
        """Bind the asynchronous Work returned by the launch executor."""
        with self._condition:
            if self._underlying is not None:
                raise RuntimeError(f"ScheduledWork {self.key} already bound")
            if self._error is not None:
                raise RuntimeError(f"ScheduledWork {self.key} already failed")
            self._underlying = underlying
            self._condition.notify_all()

    def fail(self, error: BaseException) -> bool:
        """Make all current/future waiters observe ``error``.

        Returns ``True`` only for the first transition to failure.
        """
        with self._condition:
            if self._error is not None or self._completed:
                return False
            self._error = error
            self._condition.notify_all()
            return True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for binding, then delegate to the underlying Work.

        For ProcessGroupNCCL this inserts the NCCL completion dependency into
        the CUDA stream current at this call site and can return before GPU
        execution physically completes. No device/stream synchronization is
        added by this wrapper.

        ``timeout`` is one total deadline covering both deferred binding and
        the underlying wait. c10d Work accepts a ``datetime.timedelta``.
        A positive ProcessGroupNCCL timeout is a backend failure boundary that
        can abort the communicator; it must not be used as a completion poll.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        # This is deliberately before the binding condition: time spent
        # waiting for the scheduler is application wait, not backend wait.
        self._timing.set_wait_start(now_us())
        deadline = None if timeout is None else time.monotonic() + timeout

        with self._condition:
            while self._underlying is None and self._error is None:
                remaining = self._remaining(deadline)
                if remaining == 0:
                    return False
                self._condition.wait(remaining)
            if self._error is not None:
                raise self._error
            underlying = self._underlying

        if underlying is None:  # defensive: condition predicate guarantees it
            return False
        try:
            self._timing.set_underlying_wait_start(now_us())
            remaining = self._remaining(deadline)
            if remaining == 0:
                if not bool(underlying.is_completed()):
                    return False
                # Preserve CUDA consumer-stream dependency insertion even for
                # a zero-timeout query that observes an already-complete Work.
                result = bool(underlying.wait())
            elif remaining is None:
                result = bool(underlying.wait())
            else:
                result = bool(underlying.wait(timeout=timedelta(seconds=remaining)))
            self._timing.set_wait_return(now_us())
            return result
        except BaseException as exc:  # noqa: BLE001 - preserve backend error
            first = self.fail(exc)
            if first and self._on_error is not None:
                self._on_error(self, exc)
            raise

    def is_completed(self) -> bool:
        """Return physical completion if bound, otherwise ``False``.

        Scheduler completion polling remains authoritative for lifecycle and
        telemetry updates; this observation does not itself release admission
        capacity.
        """
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._completed:
                return True
            underlying = self._underlying
        if underlying is None:
            return False
        try:
            return bool(underlying.is_completed())
        except BaseException as exc:  # noqa: BLE001
            first = self.fail(exc)
            if first and self._on_error is not None:
                self._on_error(self, exc)
            raise

    def underlying(self) -> Any:
        """Return the bound Work for scheduler-owned completion polling."""
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._underlying is None:
                raise RuntimeError(f"ScheduledWork {self.key} is not bound")
            return self._underlying

    def mark_completed(self) -> bool:
        """Record backend-observed physical completion exactly once."""
        with self._condition:
            if self._completed:
                return False
            if self._error is not None:
                return False
            if self._underlying is None:
                raise RuntimeError(f"cannot complete unbound work {self.key}")
            self._completed = True
            self._condition.notify_all()

        complete_ts = now_us()
        self._timing.set_completion_observed(complete_ts)
        if self._intent.state is IntentState.SUBMITTED:
            self._intent.transition(IntentState.COMPLETED)
        return True

    @staticmethod
    def _remaining(deadline: Optional[float]) -> Optional[float]:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())
