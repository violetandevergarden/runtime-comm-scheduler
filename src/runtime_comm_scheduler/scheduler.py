"""Worker-only admission scheduler and rank-wide host launch sequencer.

Every scheduler owns exactly one worker thread. Training threads only validate
and park intents; the worker is the sole component allowed to invoke launchers
for all locally managed process groups.
"""

from __future__ import annotations

import enum
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Collection, Optional

from .executor import (
    CompletionProbe,
    LaunchExecutor,
    WorkIsCompletedProbe,
    validate_underlying_work,
)
from .intent import CommIntent, IntentState, TaskKey
from .plan import Plan
from .telemetry import CommTiming, now_us
from .validate import ValidationError, check_plan_integrity, validate_intent
from .work import ScheduledWork


class SchedulerError(RuntimeError):
    """Base class for scheduler lifecycle failures."""


class SchedulerClosedError(SchedulerError):
    """Raised when work is submitted to, or stranded by, a closed scheduler."""


class SchedulerFailedError(SchedulerError):
    """Raised by work aborted as collateral damage after another task fails."""


class WindowIncompleteError(SchedulerError):
    """Raised when a scheduling window cannot be drained safely."""


class _SchedulerState(enum.Enum):
    RUNNING = "running"
    CLOSING = "closing"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class _PendingTask:
    intent: CommIntent
    work: ScheduledWork
    timing: CommTiming


class AdmissionScheduler:
    """Per-rank admission scheduler with one host launch worker."""

    def __init__(
        self,
        plan: Plan,
        *,
        local_group_ids: Collection[str],
        executor: LaunchExecutor,
        completion_probe: Optional[CompletionProbe] = None,
        max_outstanding: int = 0,
        completion_poll_interval_s: float = 0.001,
        selection_controller: Any = None,
        selection_serial: bool = False,
    ) -> None:
        if max_outstanding < 0:
            raise ValueError("max_outstanding must be non-negative")
        if completion_poll_interval_s <= 0:
            raise ValueError("completion_poll_interval_s must be positive")
        check_plan_integrity(plan)

        groups = frozenset(local_group_ids)
        unknown_groups = groups.difference(plan.group_ids())
        if unknown_groups:
            raise ValidationError(
                f"local groups are absent from plan v{plan.version}: "
                f"{sorted(unknown_groups)}"
            )

        probe = completion_probe or WorkIsCompletedProbe()
        if not probe.supports_physical_completion:
            raise ValueError(
                "completion_probe must provide physical-completion semantics"
            )

        self._plan = plan
        self._local_group_ids = groups
        self._executor = executor
        self._completion_probe = probe
        self._max_outstanding = max_outstanding
        self._completion_poll_interval_s = completion_poll_interval_s
        self._selection_controller = selection_controller
        self._selection_serial = selection_serial
        self._selection_last_key: Optional[TaskKey] = None

        projection = plan.local_projection(groups)
        self._local_projection = projection
        self._local_keys = frozenset(projection)
        self._remaining: deque[TaskKey] = deque(projection)
        self._pending: dict[TaskKey, _PendingTask] = {}
        self._launching: Optional[_PendingTask] = None
        self._inflight: dict[TaskKey, _PendingTask] = {}
        self._submitted: set[TaskKey] = set()
        self._launch_log: list[TaskKey] = []
        self._timings: dict[TaskKey, CommTiming] = {}
        self._timing_order: list[TaskKey] = []

        self._condition = threading.Condition()
        self._state = _SchedulerState.RUNNING
        self._failure: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="admission-worker",
            daemon=True,
        )
        self._thread.start()

    # ---- public API -------------------------------------------------

    def submit(self, intent: CommIntent) -> ScheduledWork:
        """Validate and park ``intent``; never invoke its launcher here."""
        submit_api_start_ts = now_us()
        validate_intent(intent, self._plan)
        key = intent.key
        with self._condition:
            self._raise_if_unavailable_locked()
            if key not in self._local_keys:
                raise ValidationError(
                    f"intent {key.as_list()} is not in this rank's plan projection"
                )
            if key in self._submitted:
                raise ValidationError(f"duplicate submit of {key.as_list()}")
            if intent.state is IntentState.CREATED:
                intent.transition(IntentState.READY)
            elif intent.state is not IntentState.READY:
                raise ValidationError(
                    f"intent {key.as_list()} in state {intent.state.name} "
                    "cannot be submitted"
                )
            intent.transition(IntentState.WAITING_FOR_ADMISSION)

            timestamp = now_us()
            timing = CommTiming(
                key=key,
                intent_ts=timestamp,
                ready_record_ts=timestamp,
                submit_api_start_ts=submit_api_start_ts,
            )
            work = ScheduledWork(intent, timing, on_error=self._on_work_error)
            task = _PendingTask(intent=intent, work=work, timing=timing)
            self._pending[key] = task
            self._submitted.add(key)
            self._timings[key] = timing
            self._timing_order.append(key)
            self._condition.notify_all()
            timing.submit_api_return_ts = now_us()
            timing.submit_api_duration_us = float(
                timing.submit_api_return_ts - submit_api_start_ts
            )
            return work

    def nudge(self) -> None:
        """Wake the worker after an external CPU-side ready condition changes."""
        with self._condition:
            self._raise_if_unavailable_locked()
            self._condition.notify_all()

    def sequence_log(self) -> list[TaskKey]:
        """Return the rank-wide host launch order."""
        with self._condition:
            self._raise_if_failed_locked()
            return list(self._launch_log)

    def group_sequence_log(self) -> dict[str, list[TaskKey]]:
        """Derive per-group launch sequences from the single launch log."""
        with self._condition:
            self._raise_if_failed_locked()
            return {
                group_id: [
                    key
                    for key in self._launch_log
                    if key.process_group_id == group_id
                ]
                for group_id in self._ordered_local_group_ids()
            }

    def timings(self) -> list[CommTiming]:
        """Return timing records in local intent submission order."""
        with self._condition:
            self._raise_if_failed_locked()
            return [self._timings[key] for key in self._timing_order]

    def finish_window(self, timeout: Optional[float] = None) -> None:
        """Verify all local tasks arrived and wait for physical completion."""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            self._raise_if_unavailable_locked()
            missing = [key for key in self._local_projection if key not in self._submitted]
            if missing:
                raise WindowIncompleteError(
                    "window has missing local intents: "
                    f"{[key.as_list() for key in missing]}"
                )
            while self._pending or self._launching is not None or self._inflight:
                self._raise_if_unavailable_locked()
                remaining = self._remaining_timeout(deadline)
                if remaining == 0:
                    raise TimeoutError(
                        "timed out draining scheduling window; "
                        f"pending={len(self._pending)} "
                        f"launching={self._launching is not None} "
                        f"inflight={len(self._inflight)}"
                    )
                self._condition.wait(remaining)

    def close(self) -> None:
        """Stop admission, fail unbound work, and join the launch worker."""
        stranded: list[_PendingTask] = []
        with self._condition:
            if self._state is _SchedulerState.CLOSED:
                failure = self._failure
            else:
                if self._state is _SchedulerState.RUNNING:
                    self._state = _SchedulerState.CLOSING
                stranded = list(self._pending.values())
                if self._launching is not None:
                    stranded.append(self._launching)
                self._pending.clear()
                self._launching = None
                self._condition.notify_all()
                failure = self._failure

        closed_error = SchedulerClosedError("scheduler closed before work was launched")
        for task in stranded:
            self._fail_task(task, closed_error, "shutdown")

        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise SchedulerError("scheduler worker did not stop within 5 seconds")

        with self._condition:
            self._state = _SchedulerState.CLOSED
            self._condition.notify_all()
            failure = failure or self._failure
        if failure is not None:
            raise failure

    def __enter__(self) -> "AdmissionScheduler":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    # ---- worker -----------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            if not self._poll_completions():
                return

            if self._selection_controller is not None:
                task, done = self._reserve_selected()
                if done:
                    return
                if task is None:
                    with self._condition:
                        if self._state is not _SchedulerState.RUNNING:
                            return
                        self._condition.wait(timeout=self._next_poll_timeout_locked())
                    continue
            else:
                with self._condition:
                    if self._state is not _SchedulerState.RUNNING:
                        return
                    task = self._reserve_next_locked()
                    if task is None:
                        self._condition.wait(timeout=self._next_poll_timeout_locked())
                        continue

            if not self._launch_reserved(task):
                return

    def _reserve_selected(self) -> tuple[Optional[_PendingTask], bool]:
        """Reserve the key selected by a cross-rank control plane."""
        with self._condition:
            if self._state is not _SchedulerState.RUNNING:
                return None, True
            last_key = self._selection_last_key
            if last_key is not None and self._selection_serial:
                local_done = (
                    last_key not in self._inflight
                    and (self._launching is None or self._launching.intent.key != last_key)
                )
            else:
                local_done = True
        if last_key is not None and self._selection_serial:
            if not local_done:
                return None, False
            self._selection_controller.wait_for_completion(last_key)
            with self._condition:
                self._selection_last_key = None

        with self._condition:
            if self._state is not _SchedulerState.RUNNING:
                return None, True
            if self._max_outstanding and self._outstanding_locked() >= self._max_outstanding:
                return None, False
            local_ready = [
                key for key in self._remaining
                if (task := self._pending.get(key)) is not None and self._cpu_ready(task.intent)
            ]
        done, selected = self._selection_controller.choose(local_ready)
        if done:
            with self._condition:
                drained = not (
                    self._pending or self._launching is not None or self._inflight
                )
            return None, drained
        if selected is None:
            return None, False
        with self._condition:
            task = self._pending.get(selected)
            if task is None or not self._cpu_ready(task.intent):
                # The controller only selects globally ready work. A local
                # rank can be a non-member and therefore has no task here.
                self._selection_last_key = selected
                return None, False
            self._remaining.remove(selected)
            del self._pending[selected]
            task.intent.transition(IntentState.ADMITTED)
            task.timing.admit_ts = now_us()
            self._launching = task
            self._selection_last_key = selected
            return task, False

    def _reserve_next_locked(self) -> Optional[_PendingTask]:
        if not self._remaining:
            return None
        if self._max_outstanding and self._outstanding_locked() >= self._max_outstanding:
            return None
        key = self._remaining[0]
        task = self._pending.get(key)
        if task is None or not self._cpu_ready(task.intent):
            return None

        self._remaining.popleft()
        del self._pending[key]
        task.intent.transition(IntentState.ADMITTED)
        task.timing.admit_ts = now_us()
        self._launching = task
        return task

    def _launch_reserved(self, task: _PendingTask) -> bool:
        task.timing.launch_start_ts = now_us()
        task.timing.collective_call_start_ts = task.timing.launch_start_ts
        try:
            underlying = self._executor.launch(task.intent)
            validate_underlying_work(underlying)
            task.timing.collective_call_return_ts = now_us()
            task.timing.submit_ts = task.timing.collective_call_return_ts
            task.timing.collective_call_duration_us = float(
                task.timing.collective_call_return_ts
                - task.timing.collective_call_start_ts
            )
        except BaseException as exc:  # noqa: BLE001 - scheduler is fail-stop
            self._fail_scheduler(exc, task, "launch")
            return False

        with self._condition:
            if (
                self._state is not _SchedulerState.RUNNING
                or self._launching is not task
            ):
                return False
            try:
                task.intent.transition(IntentState.SUBMITTED)
                task.work.bind(underlying)
            except BaseException as exc:  # noqa: BLE001
                bind_error = exc
                tasks = self._fail_scheduler_locked(exc, task)
            else:
                self._launching = None
                self._inflight[task.intent.key] = task
                self._launch_log.append(task.intent.key)
                self._condition.notify_all()
                return True

        self._fail_tasks(tasks, task, bind_error, "scheduler")
        return False

    def _poll_completions(self) -> bool:
        with self._condition:
            if self._state is not _SchedulerState.RUNNING:
                return False
            snapshot = list(self._inflight.items())

        completed: list[tuple[TaskKey, _PendingTask]] = []
        for key, task in snapshot:
            try:
                if self._completion_probe.is_completed(task.work.underlying()):
                    completed.append((key, task))
            except BaseException as exc:  # noqa: BLE001
                self._fail_scheduler(exc, task, "completion")
                return False

        for key, task in completed:
            try:
                task.work.mark_completed()
            except BaseException as exc:  # noqa: BLE001
                self._fail_scheduler(exc, task, "completion")
                return False
            with self._condition:
                if self._inflight.get(key) is task:
                    del self._inflight[key]
                    self._condition.notify_all()
        return True

    # ---- failure/state helpers -------------------------------------

    def _on_work_error(self, work: ScheduledWork, error: BaseException) -> None:
        with self._condition:
            task = self._inflight.get(work.key)
        self._fail_scheduler(error, task, "wait")

    def _fail_scheduler(
        self,
        error: BaseException,
        current: Optional[_PendingTask],
        stage: str,
    ) -> None:
        with self._condition:
            tasks = self._fail_scheduler_locked(error, current)
        self._fail_tasks(tasks, current, error, stage)

    def _fail_scheduler_locked(
        self,
        error: BaseException,
        current: Optional[_PendingTask],
    ) -> list[_PendingTask]:
        if self._state in (_SchedulerState.FAILED, _SchedulerState.CLOSED):
            return []
        self._state = _SchedulerState.FAILED
        self._failure = error
        tasks_by_key = {task.intent.key: task for task in self._pending.values()}
        if self._launching is not None:
            tasks_by_key[self._launching.intent.key] = self._launching
        for task in self._inflight.values():
            tasks_by_key[task.intent.key] = task
        if current is not None:
            tasks_by_key[current.intent.key] = current
        self._pending.clear()
        self._launching = None
        self._inflight.clear()
        self._remaining.clear()
        self._condition.notify_all()
        return list(tasks_by_key.values())

    def _fail_tasks(
        self,
        tasks: list[_PendingTask],
        current: Optional[_PendingTask],
        error: BaseException,
        stage: str,
    ) -> None:
        for task in tasks:
            task_error = error
            if task is not current:
                task_error = SchedulerFailedError(
                    f"scheduler failed during {stage}; work "
                    f"{task.intent.key.as_list()} was aborted"
                )
                task_error.__cause__ = error
            self._fail_task(task, task_error, stage)

    @staticmethod
    def _fail_task(task: _PendingTask, error: BaseException, stage: str) -> None:
        timestamp = now_us()
        if task.timing.error_ts is None:
            task.timing.error_ts = timestamp
            task.timing.error_stage = stage
        if task.intent.state not in (IntentState.COMPLETED, IntentState.FAILED):
            task.intent.transition(IntentState.FAILED)
        task.work.fail(error)

    def _raise_if_unavailable_locked(self) -> None:
        self._raise_if_failed_locked()
        if self._state is not _SchedulerState.RUNNING:
            raise SchedulerClosedError(f"scheduler is {self._state.value}")

    def _raise_if_failed_locked(self) -> None:
        if self._state is _SchedulerState.FAILED:
            assert self._failure is not None
            raise self._failure

    def _outstanding_locked(self) -> int:
        return len(self._inflight) + (self._launching is not None)

    def _next_poll_timeout_locked(self) -> Optional[float]:
        if self._selection_controller is not None:
            return self._completion_poll_interval_s
        if self._inflight:
            return self._completion_poll_interval_s
        if self._remaining:
            task = self._pending.get(self._remaining[0])
            if task is not None and isinstance(task.intent.ready_event, threading.Event):
                return self._completion_poll_interval_s
        return None

    @staticmethod
    def _cpu_ready(intent: CommIntent) -> bool:
        event = intent.ready_event
        return not isinstance(event, threading.Event) or event.is_set()

    def _ordered_local_group_ids(self) -> tuple[str, ...]:
        return tuple(
            group_id
            for group_id in self._plan.group_ids()
            if group_id in self._local_group_ids
        )

    @staticmethod
    def _remaining_timeout(deadline: Optional[float]) -> Optional[float]:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())
