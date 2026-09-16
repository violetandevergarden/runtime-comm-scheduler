"""Rank-local runtime: application API, grant queue, launcher and probe."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from .coordinator import CoordinatorError
from .executor import DirectExecutor, WorkIsCompletedProbe
from .handle import RuntimeHandle
from .model import GroupSpec, LocalBinding, TaskHint, TaskSpec
from .transport import ControlClient


@dataclass
class _LocalTask:
    spec: TaskSpec
    hint: TaskHint
    binding: LocalBinding | None
    handle: RuntimeHandle


class RankRuntime:
    def __init__(
        self,
        rank: int,
        epoch: int,
        transport: ControlClient,
        executor: Any | None = None,
        completion_probe: Any | None = None,
        completion_poll_interval_s: float = 0.001,
    ) -> None:
        if completion_poll_interval_s <= 0:
            raise ValueError("completion_poll_interval_s must be positive")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        self.rank = rank
        self.epoch = epoch
        self.transport = transport
        self.executor = executor or DirectExecutor()
        self.completion_probe = completion_probe or WorkIsCompletedProbe()
        if not getattr(self.completion_probe, "supports_physical_completion", False):
            raise ValueError("completion probe must provide physical completion semantics")
        self.poll_interval = completion_poll_interval_s
        self._groups: dict[str, GroupSpec] = {}
        self._tasks: dict[str, _LocalTask] = {}
        self._launch_queue: queue.Queue[str | None] = queue.Queue()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._failure: BaseException | None = None
        self._started = False
        self._closed = False
        self._finish_received = False
        self._grant_order: list[str] = []
        self._reader: threading.Thread | None = None
        self._launcher: threading.Thread | None = None
        self._completion: threading.Thread | None = None

    def start(self, timeout: float = 20.0) -> None:
        if self._started:
            return
        self.transport.connect()
        ready = self.transport.receive(timeout)
        if ready.get("kind") != "READY":
            raise RuntimeError(f"expected READY, got {ready.get('kind')}")
        self._started = True
        self._reader = threading.Thread(target=self._control_loop, name=f"runtime-control-{self.rank}", daemon=True)
        self._launcher = threading.Thread(target=self._launch_loop, name=f"runtime-launch-{self.rank}", daemon=True)
        self._completion = threading.Thread(target=self._completion_loop, name=f"runtime-completion-{self.rank}", daemon=True)
        for thread in (self._reader, self._launcher, self._completion):
            thread.start()
        for group in self._groups.values():
            self._send_group(group)

    def register_group(self, group: GroupSpec, process_group: Any = None) -> None:
        if group.epoch != self.epoch or self.rank not in group.ranks:
            raise ValueError(f"rank {self.rank} cannot register group {group.group_id}")
        old = self._groups.get(group.group_id)
        if old is not None and old != group:
            raise ValueError(f"group metadata mismatch for {group.group_id}")
        self._groups[group.group_id] = group
        if self._started:
            self._send_group(group)

    def declare(self, spec: TaskSpec, hint: TaskHint) -> None:
        self._check_spec(spec)
        with self._condition:
            existing = self._tasks.get(spec.task_id)
            if existing is not None and existing.spec != spec:
                raise ValueError(f"task metadata mismatch for {spec.task_id}")
            if existing is None:
                self._tasks[spec.task_id] = _LocalTask(spec, hint, None, RuntimeHandle(spec.task_id))
        self.transport.send("DECLARE", {"task": spec.to_dict(), "hint": hint.to_dict()})

    def submit(self, spec: TaskSpec, binding: LocalBinding, hint: TaskHint) -> RuntimeHandle:
        self._check_spec(spec)
        if not callable(binding.launch):
            raise TypeError("LocalBinding.launch must be callable")
        with self._condition:
            self._raise_if_failed_locked()
            existing = self._tasks.get(spec.task_id)
            if existing is not None and existing.spec != spec:
                raise ValueError(f"task metadata mismatch for {spec.task_id}")
            if existing is not None and existing.binding is not None:
                raise ValueError(f"duplicate submit for {spec.task_id}")
            handle = existing.handle if existing is not None else RuntimeHandle(spec.task_id)
            self._tasks[spec.task_id] = _LocalTask(spec, hint, binding, handle)
        self.transport.send("OFFER", {"task": spec.to_dict(), "hint": hint.to_dict()})
        return handle

    def finish_epoch(self, timeout: float = 20.0) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        self.transport.send("INPUT_CLOSED", {"task_count": len(self._tasks)})
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._finish_received and self._failure is None:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining == 0:
                    raise TimeoutError("timed out waiting for FINISHED")
                self._condition.wait(remaining)
            self._raise_if_failed_locked()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._launch_queue.put(None)
        self.transport.close()
        for thread in (self._reader, self._launcher, self._completion):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2)
        with self._condition:
            error = self._failure or RuntimeError("runtime closed")
            for task in self._tasks.values():
                task.handle.fail(error)
            self._condition.notify_all()

    @property
    def failure(self) -> BaseException | None:
        with self._condition:
            return self._failure

    @property
    def grant_order(self) -> list[str]:
        with self._condition:
            return list(self._grant_order)

    def _send_group(self, group: GroupSpec) -> None:
        self.transport.send("REGISTER_GROUP", {"group": group.to_dict()})

    def _check_spec(self, spec: TaskSpec) -> None:
        if spec.epoch != self.epoch:
            raise ValueError(f"task {spec.task_id} has wrong epoch")
        group = self._groups.get(spec.group_id)
        if group is None:
            raise ValueError(f"group {spec.group_id} is not registered")
        if spec.group_id != group.group_id:
            raise ValueError("task group mismatch")

    def _control_loop(self) -> None:
        while not self._stop.is_set():
            try:
                message = self.transport.receive(0.1)
            except queue.Empty:
                continue
            except BaseException as exc:  # noqa: BLE001
                self._fail(exc)
                return
            kind = message.get("kind")
            try:
                if kind == "GRANT":
                    task_id = message["payload"]["task"]["task_id"]
                    decision_seq = int(message["payload"]["decision_seq"])
                    with self._condition:
                        task = self._tasks.get(task_id)
                        if task is None or task.binding is None:
                            raise CoordinatorError(f"grant for unsubmitted local task {task_id}")
                        task.handle.grant(decision_seq)
                        self._grant_order.append(task_id)
                    self._launch_queue.put(task_id)
                elif kind == "FINISHED":
                    with self._condition:
                        self._finish_received = True
                        self._condition.notify_all()
                elif kind == "FAILED":
                    payload = message.get("payload", {})
                    self._fail(RuntimeError(f"coordinator failed: {payload}"))
                    return
                elif kind != "READY":
                    raise CoordinatorError(f"unknown control message {kind!r}")
            except BaseException as exc:  # noqa: BLE001
                self._fail(exc)
                return

    def _launch_loop(self) -> None:
        while not self._stop.is_set():
            task_id = self._launch_queue.get()
            if task_id is None:
                return
            try:
                with self._condition:
                    task = self._tasks[task_id]
                    binding = task.binding
                if binding is None:
                    raise RuntimeError(f"task {task_id} has no local binding")
                work = self.executor.launch(binding)
                task.handle.bind(work)
                self.transport.send("SUBMITTED", {"task_id": task_id, "decision_seq": task.handle.decision_seq})
            except BaseException as exc:  # noqa: BLE001
                task = self._tasks.get(task_id)
                if task is not None:
                    task.handle.fail(exc)
                self._fail(exc, stage="launch", task_id=task_id)

    def _completion_loop(self) -> None:
        while not self._stop.is_set():
            did_work = False
            for task_id, task in list(self._tasks.items()):
                if task.handle.state.value not in {"bound"}:
                    continue
                try:
                    work = task.handle._work  # protected access is local to this runtime worker
                    if work is not None and self.completion_probe.is_completed(work):
                        task.handle.mark_completed()
                        self.transport.send("COMPLETED", {"task_id": task_id, "decision_seq": task.handle.decision_seq})
                        did_work = True
                except BaseException as exc:  # noqa: BLE001
                    task.handle.fail(exc)
                    self._fail(exc, stage="completion", task_id=task_id)
            if not did_work:
                self._stop.wait(self.poll_interval)

    def _fail(self, error: BaseException, **details: Any) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = error
                for task in self._tasks.values():
                    task.handle.fail(error)
                self._condition.notify_all()
        if self._started and not self._closed:
            try:
                self.transport.send("FAILED", {"stage": details.get("stage", "runtime"), "error": str(error), **details})
            except BaseException:
                pass

    def _raise_if_failed_locked(self) -> None:
        if self._failure is not None:
            raise self._failure
