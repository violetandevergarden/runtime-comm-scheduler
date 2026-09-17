"""Rank-local runtime: application API, grant queue, launcher and probe."""

from __future__ import annotations

import queue
import enum
import threading
import time
from dataclasses import dataclass
from typing import Any

from .coordinator import CoordinatorError
from .executor import DirectExecutor, WorkIsCompletedProbe
from .handle import RuntimeHandle
from .model import GroupSpec, LocalBinding, TaskHint, TaskSpec
from .telemetry import EventLog
from .transport import ControlClient


class RuntimeState(enum.Enum):
    CREATED = "created"
    RUNNING = "running"
    INPUT_CLOSED = "input_closed"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class _LocalTask:
    spec: TaskSpec
    hint: TaskHint
    binding: LocalBinding | None
    handle: RuntimeHandle
    declared: bool = False
    offered: bool = False


class RankRuntime:
    def __init__(
        self,
        rank: int,
        epoch: int,
        transport: ControlClient,
        executor: Any | None = None,
        completion_probe: Any | None = None,
        completion_poll_interval_s: float = 0.001,
        event_log: EventLog | None = None,
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
        self._process_groups: dict[str, Any] = {}
        self._tasks: dict[str, _LocalTask] = {}
        self._active_task_ids: set[str] = set()
        self._launch_queue: queue.Queue[str | None] = queue.Queue()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._failure: BaseException | None = None
        self._state = RuntimeState.CREATED
        self._started = False
        self._closed = False
        self._finish_received = False
        self._grant_order: list[str] = []
        self._launch_order: list[str] = []
        self.event_log = event_log or EventLog("runtime", rank)
        self._reader: threading.Thread | None = None
        self._launcher: threading.Thread | None = None
        self._completion: threading.Thread | None = None

    def start(self, timeout: float = 20.0) -> None:
        with self._condition:
            if self._state is RuntimeState.RUNNING:
                return
            self._raise_if_terminal_locked()
            if self._state is not RuntimeState.CREATED:
                raise RuntimeError(f"cannot start runtime in {self._state.value} state")
        try:
            self.transport.connect()
            ready = self.transport.receive(timeout)
            if ready.get("kind") != "READY":
                raise RuntimeError(f"expected READY, got {ready.get('kind')}")
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc, stage="start")
            raise
        with self._condition:
            self._state = RuntimeState.RUNNING
            self._started = True
        self._reader = threading.Thread(target=self._control_loop, name=f"runtime-control-{self.rank}", daemon=True)
        self._launcher = threading.Thread(target=self._launch_loop, name=f"runtime-launch-{self.rank}", daemon=True)
        self._completion = threading.Thread(target=self._completion_loop, name=f"runtime-completion-{self.rank}", daemon=True)
        for thread in (self._reader, self._launcher, self._completion):
            thread.start()
        try:
            for group in self._groups.values():
                self._send_group(group)
        except BaseException as exc:  # noqa: BLE001
            self._fail(exc, stage="register_group")
            raise

    @property
    def state(self) -> RuntimeState:
        with self._condition:
            return self._state

    def register_group(self, group: GroupSpec, process_group: Any = None) -> None:
        send_error: BaseException | None = None
        with self._condition:
            self._raise_if_terminal_locked()
            if self._state not in (RuntimeState.CREATED, RuntimeState.RUNNING):
                raise RuntimeError(f"cannot register group in {self._state.value} state")
            if process_group is None:
                raise ValueError("process_group must not be None")
            if group.epoch != self.epoch or self.rank not in group.ranks:
                raise ValueError(f"rank {self.rank} cannot register group {group.group_id}")
            old = self._groups.get(group.group_id)
            if old is not None and old != group:
                raise ValueError(f"group metadata mismatch for {group.group_id}")
            old_process_group = self._process_groups.get(group.group_id)
            if old_process_group is not None and old_process_group is not process_group:
                raise ValueError(f"process group mismatch for {group.group_id}")
            self._groups[group.group_id] = group
            self._process_groups[group.group_id] = process_group
            if self._state is RuntimeState.RUNNING:
                try:
                    self._send_group(group)
                except BaseException as exc:  # noqa: BLE001
                    send_error = exc
        self.event_log.record("group_registered", group_id=group.group_id)
        if send_error is not None:
            self._fail(send_error, stage="register_group")
            raise send_error

    def declare(self, spec: TaskSpec, hint: TaskHint) -> None:
        send_error: BaseException | None = None
        with self._condition:
            self._check_spec_locked(spec)
            existing = self._tasks.get(spec.task_id)
            if existing is not None:
                if existing.spec != spec:
                    raise ValueError(f"task metadata mismatch for {spec.task_id}")
                raise ValueError(f"duplicate declaration for {spec.task_id}")
            self._tasks[spec.task_id] = _LocalTask(
                spec, hint, None, RuntimeHandle(spec.task_id), declared=True
            )
            try:
                self.transport.send("DECLARE", {"task": spec.to_dict(), "hint": hint.to_dict()})
            except BaseException as exc:  # noqa: BLE001
                send_error = exc
        if send_error is not None:
            self._fail(send_error, stage="declare", task_id=spec.task_id)
            raise send_error
        self.event_log.record("declared", task_id=spec.task_id)

    def submit(self, spec: TaskSpec, binding: LocalBinding, hint: TaskHint) -> RuntimeHandle:
        send_error: BaseException | None = None
        with self._condition:
            self._check_spec_locked(spec)
            self._validate_binding_locked(spec, binding)
            self._raise_if_failed_locked()
            existing = self._tasks.get(spec.task_id)
            if existing is not None and existing.spec != spec:
                raise ValueError(f"task metadata mismatch for {spec.task_id}")
            if existing is not None and existing.offered:
                raise ValueError(f"duplicate submit for {spec.task_id}")
            handle = existing.handle if existing is not None else RuntimeHandle(spec.task_id)
            self._tasks[spec.task_id] = _LocalTask(
                spec, hint, binding, handle,
                declared=existing.declared if existing is not None else False,
                offered=True,
            )
            try:
                self.transport.send("OFFER", {"task": spec.to_dict(), "hint": hint.to_dict()})
            except BaseException as exc:  # noqa: BLE001
                send_error = exc
        if send_error is not None:
            self._fail(send_error, stage="offer", task_id=spec.task_id)
            raise send_error
        self.event_log.record("offered", task_id=spec.task_id)
        return handle

    def finish_epoch(self, timeout: float = 20.0) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        send_error: BaseException | None = None
        with self._condition:
            self._raise_if_terminal_locked()
            if self._state is not RuntimeState.RUNNING:
                raise RuntimeError(f"cannot finish runtime in {self._state.value} state")
            self._state = RuntimeState.INPUT_CLOSED
            task_count = len(self._tasks)
            try:
                self.transport.send("INPUT_CLOSED", {"task_count": task_count})
            except BaseException as exc:  # noqa: BLE001
                send_error = exc
        if send_error is not None:
            self._fail(send_error, stage="input_closed")
            raise send_error
        self.event_log.record("input_closed")
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._finish_received and self._failure is None:
                remaining = max(0.0, deadline - time.monotonic())
                if remaining == 0:
                    error = TimeoutError("timed out waiting for FINISHED")
                    self._fail(error, stage="finish_epoch")
                    raise error
                self._condition.wait(remaining)
            self._raise_if_failed_locked()

    def close(self) -> None:
        with self._condition:
            if self._state is RuntimeState.CLOSED:
                return
            self._state = RuntimeState.CLOSED
            self._closed = True
            error = self._failure or RuntimeError("runtime closed")
            for task in self._tasks.values():
                task.handle.fail(error)
                task.binding = None
            self._active_task_ids.clear()
            self._condition.notify_all()
        self._stop.set()
        self._launch_queue.put(None)
        self.transport.close()
        for thread in (self._reader, self._launcher, self._completion):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2)

    @property
    def failure(self) -> BaseException | None:
        with self._condition:
            return self._failure

    @property
    def grant_order(self) -> list[str]:
        with self._condition:
            return list(self._grant_order)

    @property
    def launch_order(self) -> list[str]:
        with self._condition:
            return list(self._launch_order)

    def _send_group(self, group: GroupSpec) -> None:
        self.transport.send("REGISTER_GROUP", {"group": group.to_dict()})

    def _check_spec_locked(self, spec: TaskSpec) -> None:
        self._raise_if_terminal_locked()
        if self._state is not RuntimeState.RUNNING:
            raise RuntimeError(f"runtime is not accepting tasks in {self._state.value} state")
        if spec.epoch != self.epoch:
            raise ValueError(f"task {spec.task_id} has wrong epoch")
        group = self._groups.get(spec.group_id)
        if group is None:
            raise ValueError(f"group {spec.group_id} is not registered")
        if spec.group_id != group.group_id or self.rank not in group.ranks:
            raise ValueError("task group mismatch")

    def _check_spec(self, spec: TaskSpec) -> None:
        with self._condition:
            self._check_spec_locked(spec)

    def _validate_binding_locked(self, spec: TaskSpec, binding: LocalBinding) -> None:
        process_group = self._process_groups.get(spec.group_id)
        if binding.process_group is not process_group:
            raise ValueError(f"LocalBinding process group mismatch for {spec.task_id}")
        if not callable(binding.launch):
            raise TypeError("LocalBinding.launch must be callable")
        tensor = binding.tensor
        shape = getattr(tensor, "shape", None)
        if shape is None or tuple(int(dim) for dim in shape) != spec.collective.shape:
            raise ValueError(f"tensor shape mismatch for {spec.task_id}")
        numel = getattr(tensor, "numel", None)
        numel = numel() if callable(numel) else numel
        if not isinstance(numel, int) or isinstance(numel, bool) or numel != spec.collective.numel:
            raise ValueError(f"tensor numel mismatch for {spec.task_id}")
        element_size = getattr(tensor, "element_size", None)
        element_size = element_size() if callable(element_size) else element_size
        if (
            not isinstance(element_size, int)
            or isinstance(element_size, bool)
            or element_size <= 0
            or element_size * numel != spec.collective.num_bytes
        ):
            raise ValueError(f"tensor bytes mismatch for {spec.task_id}")
        if _dtype_name(getattr(tensor, "dtype", None)) != spec.collective.dtype:
            raise ValueError(f"tensor dtype mismatch for {spec.task_id}")
        tensor_device = getattr(tensor, "device", None)
        if binding.device is not None and tensor_device is not None:
            if _device_kind(binding.device) != _device_kind(tensor_device):
                raise ValueError(f"tensor device mismatch for {spec.task_id}")
        if binding.producer_event is not None and not getattr(
            self.executor, "supports_producer_dependency", False
        ):
            raise ValueError("executor does not support producer dependency")

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
                if message.get("epoch") != self.epoch:
                    raise CoordinatorError("grant has wrong epoch")
                if kind == "GRANT":
                    payload = message["payload"]
                    granted_spec = TaskSpec.from_dict(payload["task"])
                    decision_seq = payload.get("decision_seq")
                    if (
                        not isinstance(decision_seq, int)
                        or isinstance(decision_seq, bool)
                        or decision_seq <= 0
                    ):
                        raise CoordinatorError("GRANT decision_seq must be positive")
                    with self._condition:
                        task = self._tasks.get(granted_spec.task_id)
                        if task is None or not task.offered:
                            raise CoordinatorError(
                                f"grant for unsubmitted local task {granted_spec.task_id}"
                            )
                        if task.spec != granted_spec:
                            raise CoordinatorError(
                                f"grant metadata mismatch for {granted_spec.task_id}"
                            )
                        group = self._groups.get(granted_spec.group_id)
                        if group is None or self.rank not in group.ranks:
                            raise CoordinatorError("grant is for a non-member group")
                        task.handle.grant(decision_seq)
                        self._grant_order.append(granted_spec.task_id)
                    self.event_log.record("grant_received", task_id=granted_spec.task_id, decision_seq=decision_seq)
                    self._launch_queue.put(granted_spec.task_id)
                elif kind == "FINISHED":
                    with self._condition:
                        self._finish_received = True
                        self._condition.notify_all()
                    self.event_log.record("finished_received")
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
            task: _LocalTask | None = None
            try:
                with self._condition:
                    if self._failure is not None or self._stop.is_set():
                        continue
                    task = self._tasks.get(task_id)
                    if task is None or task.handle.state.value != "granted":
                        raise RuntimeError(f"task {task_id} is not ready to launch")
                    binding = task.binding
                    if binding is None:
                        raise RuntimeError(f"task {task_id} has no local binding")
                    self._launch_order.append(task_id)
                    decision_seq = task.handle.decision_seq
                self.event_log.record("launch_start", task_id=task_id, decision_seq=decision_seq)
                work = self.executor.launch(binding)
                task.handle.bind(work)
                self.transport.send("SUBMITTED", {"task_id": task_id, "decision_seq": decision_seq})
                self.event_log.record("submitted_sent", task_id=task_id, decision_seq=decision_seq)
                with self._condition:
                    self._active_task_ids.add(task_id)
            except BaseException as exc:  # noqa: BLE001
                if task is not None:
                    task.handle.fail(exc)
                self._fail(exc, stage="launch", task_id=task_id)
            finally:
                binding = None

    def _completion_loop(self) -> None:
        while not self._stop.is_set():
            did_work = False
            with self._condition:
                active_ids = tuple(self._active_task_ids)
            for task_id in active_ids:
                if self._stop.is_set():
                    return
                with self._condition:
                    task = self._tasks.get(task_id)
                    if task is None or task_id not in self._active_task_ids:
                        continue
                    work = task.handle._work
                    decision_seq = task.handle.decision_seq
                try:
                    if work is not None and self.completion_probe.is_completed(work):
                        with self._condition:
                            self._active_task_ids.discard(task_id)
                        self.event_log.record("completion_observed", task_id=task_id, decision_seq=decision_seq)
                        if not task.handle.mark_completed():
                            continue
                        self.transport.send("COMPLETED", {"task_id": task_id, "decision_seq": decision_seq})
                        with self._condition:
                            task.binding = None
                        self.event_log.record("completed_sent", task_id=task_id, decision_seq=decision_seq)
                        did_work = True
                except BaseException as exc:  # noqa: BLE001
                    task.handle.fail(exc)
                    self._fail(exc, stage="completion", task_id=task_id)
            if not did_work:
                self._stop.wait(self.poll_interval)

    def _fail(self, error: BaseException, **details: Any) -> None:
        first = False
        with self._condition:
            if self._state is RuntimeState.CLOSED:
                return
            if self._failure is None:
                first = True
                self._failure = error
                self._state = RuntimeState.FAILED
                self._active_task_ids.clear()
                for task in self._tasks.values():
                    task.handle.fail(error)
                    task.binding = None
                self._condition.notify_all()
        if first:
            self.event_log.record("failed", error=f"{type(error).__name__}: {error}")
        self._stop.set()
        self._launch_queue.put(None)
        if first and self._started and self.state is not RuntimeState.CLOSED:
            try:
                self.transport.send("FAILED", {"stage": details.get("stage", "runtime"), "error": str(error), **details})
            except BaseException:
                pass

    def _raise_if_terminal_locked(self) -> None:
        self._raise_if_failed_locked()
        if self._state is RuntimeState.CLOSED:
            raise RuntimeError("runtime is closed")

    def _raise_if_failed_locked(self) -> None:
        if self._failure is not None:
            raise self._failure


def _dtype_name(value: Any) -> str:
    name = str(value).lower()
    if name.startswith("torch."):
        name = name[6:]
    return {
        "float": "float32",
        "half": "float16",
        "double": "float64",
        "int": "int32",
        "long": "int64",
    }.get(name, name)


def _device_kind(value: Any) -> str:
    return str(value).lower().split(":", 1)[0]
