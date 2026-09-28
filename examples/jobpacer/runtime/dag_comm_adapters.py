"""Communication-engine adapters consumed by the shared DAG runner."""

from __future__ import annotations

import threading
import time
from datetime import timedelta
from typing import Any, Mapping

import torch
import torch.distributed as dist

from runtime_comm_scheduler import AdmissionScheduler, CommIntent, Plan, TaskKey, TorchProcessGroupExecutor
from runtime_comm_scheduler.dag import (
    CommNode, DagGraph, DagJob, build_static_order, validate_static_order,
)
from runtime_comm_scheduler.runtime import EventLog, HandleState, LocalBinding, TaskHint, TaskSpec
from runtime_comm_scheduler.runtime.executor import CudaCollectiveExecutor


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


class EpochFailureSignal:
    """Cross-rank fail-fast notification over the rendezvous Store, not a job collective."""

    def __init__(self, epoch: int, rank: int, world_size: int):
        self.rank, self.world_size = rank, world_size
        self._store = dist.distributed_c10d._get_default_store()
        self._prefix = f"jobpacer/phase3/epoch/{epoch}"
        self._key = f"{self._prefix}/failure"

    def publish(self, exc: BaseException) -> None:
        payload = f"rank={self.rank} {type(exc).__name__}: {exc}".encode("utf-8", errors="replace")
        self._store.set(self._key, payload)

    def failure(self) -> BaseException | None:
        if not self._store.check([self._key]):
            return None
        try:
            payload = self._store.get(self._key).decode("utf-8", errors="replace")
        except BaseException as exc:
            return RuntimeError(f"peer failure notification could not be read: {exc}")
        return RuntimeError(f"peer rank failure: {payload}")

    def publish_teardown_ready(self) -> None:
        self._store.set(f"{self._prefix}/teardown/{self.rank}", b"ready")

    def wait_for_teardown_ready(self, deadline: float) -> bool:
        keys = [f"{self._prefix}/teardown/{rank}" for rank in range(self.world_size)]
        while time.monotonic() < deadline:
            if self._store.check(keys):
                return True
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return self._store.check(keys)


def build_legacy_plan(graph: DagGraph, policy: str, *, epoch: int,
                      tails: Mapping[str, float] | None = None,
                      order: tuple[str, ...] | None = None) -> tuple[Plan, dict[str, TaskKey]]:
    order = (validate_static_order(order, graph) if order is not None
             else build_static_order(graph, policy, tails=tails))
    job_positions = {job.job_id: index for index, job in enumerate(graph.jobs)}
    node_positions = {f"{job.job_id}/{node.node_id}": index
                      for job in graph.jobs for index, node in enumerate(job.nodes)}
    comms = {f"{job.job_id}/{node.node_id}": (job, node)
             for job in graph.jobs for node in job.nodes if isinstance(node, CommNode)}
    task_keys: dict[str, TaskKey] = {}
    entries = []
    for task_id in order:
        job, node = comms[task_id]
        key = TaskKey(
            iteration=epoch,
            microbatch=job_positions[job.job_id],
            parallelism="jobpacer-gpu-dag-v1",
            process_group_id=node.group_id,
            layer_id=node_positions[task_id],
            bucket_id=node.group_seq,
            ordinal=node.group_seq,
        )
        if key in task_keys.values():
            raise ValueError(f"legacy TaskKey collision while mapping {task_id}")
        task_keys[task_id] = key
        entries.append((key, node.collective.op, node.collective.num_bytes))
    return Plan(version=1, window_id=epoch, entries=tuple(entries)), task_keys


class CudaEventCompletionProbe:
    """Bridge old ProcessGroup Work into an explicit stream event exactly once."""

    supports_physical_completion = True

    def __init__(self, device: str):
        self.device = torch.device(device)
        self._lock = threading.Lock()
        self._receipts: dict[int, tuple[Any, Any, Any]] = {}
        self._streams: dict[str, Any] = {}

    def is_completed(self, work: Any) -> bool:
        key = id(work)
        with self._lock:
            receipt = self._receipts.get(key)
            if receipt is None:
                if not callable(getattr(work, "wait", None)):
                    raise TypeError("old GPU Work must provide wait() for CUDA completion bridging")
                stream = torch.cuda.Stream(device=self.device)
                done = torch.cuda.Event(blocking=False, interprocess=False)
                with torch.cuda.device(self.device), torch.cuda.stream(stream):
                    work.wait()
                    done.record(stream)
                receipt = (work, stream, done)
                self._receipts[key] = receipt
        return bool(receipt[2].query())


class _LegacyHandle:
    def __init__(self, task_id: str, work, key: TaskKey, probe: CudaEventCompletionProbe):
        self.task_id, self.work, self.key, self.probe = task_id, work, key, probe
        self.decision_seq = None

    @property
    def state(self):
        return self.work.intent.state

    def wait_host(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.work.intent.state.value == "failed":
                return self.work.is_completed()  # raises the original work failure
            # The legacy scheduler may retain a submitted intent behind the
            # static Plan head.  An unbound ScheduledWork is pending, not a
            # failed completion probe.
            if not self.work.is_bound:
                if timeout == 0:
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                time.sleep(min(0.001, remaining) if remaining is not None else 0.001)
                continue
            if self.probe.is_completed(self.work.underlying()):
                return True
            if timeout == 0:
                return False
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            time.sleep(min(0.001, remaining) if remaining is not None else 0.001)


class LegacySchedulerDagAdapter:
    def __init__(self, graph: DagGraph, *, policy: str, epoch: int, rank: int,
                 group_ids: set[str], tails: Mapping[str, float], device: str,
                 poll_interval_s: float, event_log: EventLog,
                 order: tuple[str, ...] | None = None,
                 failure_signal: EpochFailureSignal | None = None,
                 failure_drain_timeout_s: float = 1.0):
        self.plan, self.task_keys = build_legacy_plan(graph, policy, epoch=epoch,
                                                      tails=tails, order=order)
        digests: list[str | None] = [None] * dist.get_world_size()
        dist.all_gather_object(digests, self.plan.digest())
        if len(set(digests)) != 1:
            raise RuntimeError(f"old scheduler GPU Plan differs across ranks: {digests}")
        self.rank, self.graph, self.event_log = rank, graph, event_log
        self.failure_signal = failure_signal
        self.failure_drain_timeout_s = max(0.01, failure_drain_timeout_s)
        self._replay_deadline: float | None = None
        self._cleanup_deadline: float | None = None
        self.probe = CudaEventCompletionProbe(device)
        self.completion_probe = self.probe
        self.executor = TorchProcessGroupExecutor(torch.device(device))
        self.scheduler = AdmissionScheduler(
            self.plan, local_group_ids=group_ids,
            executor=self.executor,
            completion_probe=self.probe, max_outstanding=1,
            completion_poll_interval_s=poll_interval_s,
        )
        self._task_by_key = {key: task_id for task_id, key in self.task_keys.items()}
        self._handles: dict[str, _LegacyHandle] = {}
        self._handles_lock = threading.Lock()
        self._started = False
        self._closed = False

    @property
    def failure(self):
        return (getattr(self.scheduler, "_failure", None)
                or (self.failure_signal.failure() if self.failure_signal is not None else None))

    @property
    def grant_order(self) -> list[str]:
        return []

    @property
    def launch_order(self) -> list[str]:
        try:
            return [self._task_by_key[key] for key in self.scheduler.sequence_log()]
        except BaseException:
            return []

    def register_group(self, *_args, **_kwargs) -> None:
        return None

    def start(self, *_args, **_kwargs) -> None:
        self._started = True

    @property
    def cleanup_deadline(self) -> float | None:
        if self._cleanup_deadline is not None:
            return self._cleanup_deadline
        return self._replay_deadline

    def set_replay_deadline(self, deadline: float) -> None:
        self._replay_deadline = deadline

    def submit(self, spec: TaskSpec, binding: LocalBinding, hint: TaskHint):
        del hint
        task_id = spec.task_id
        key = self.task_keys.get(task_id)
        if key is None:
            raise ValueError(f"task {task_id} is absent from the shared old Plan")
        metadata = self.plan.metadata(key)
        if metadata != (spec.collective.op, spec.collective.num_bytes):
            raise ValueError(f"task metadata differs from the shared old Plan for {task_id}")
        intent = CommIntent(
            key=key, op=spec.collective.op, tensor=binding.tensor,
            process_group=binding.process_group, num_bytes=spec.collective.num_bytes,
            launch_fn=binding.launch, producer=task_id, consumer=task_id + ":consumer",
            ready_event=binding.producer_event, device=binding.device, keepalive=binding.keepalive,
        )
        with self._handles_lock:
            if self._closed:
                raise RuntimeError("old scheduler adapter is closed")
            work = self.scheduler.submit(intent)
            handle = _LegacyHandle(task_id, work, key, self.probe)
            self._handles[task_id] = handle
            self.event_log.record("old_plan_submit_return", task_id=task_id, key=key.as_list())
            return handle

    def declare(self, *_args, **_kwargs):
        raise NotImplementedError("old scheduler adapter does not support Lookahead DECLARE")

    def finish_epoch(self, timeout: float | None = None) -> None:
        self.scheduler.finish_window(timeout=timeout)

    def abort(self, exc: BaseException, **context) -> None:
        deadline = context.pop("deadline", None)
        if deadline is None:
            deadline = self._replay_deadline
        if deadline is None:
            deadline = time.monotonic() + self.failure_drain_timeout_s
        if self._cleanup_deadline is not None:
            deadline = min(deadline, self._cleanup_deadline)
        self._cleanup_deadline = deadline
        self.event_log.record("old_adapter_abort", error_type=type(exc).__name__, **context)
        if self.failure_signal is not None:
            try:
                self.failure_signal.publish(exc)
            except BaseException as signal_error:
                self.event_log.record("peer_failure_signal_error", error_type=type(signal_error).__name__)
        with self._handles_lock:
            if self._closed:
                return
            self._closed = True
            handles = tuple(self._handles.values())
        # A committed NCCL Work cannot be cancelled. Ask the backend for a
        # bounded failure boundary before destroying its communicator.
        for handle in handles:
            remaining = _remaining(deadline)
            if remaining <= 0:
                break
            if not handle.work.is_bound:
                continue
            try:
                handle.work.wait(timeout=remaining)
            except BaseException:
                pass
        try:
            self.scheduler.close(timeout=_remaining(deadline))
        except BaseException:
            pass

    def close(self) -> None:
        with self._handles_lock:
            if self._closed:
                return
            self._closed = True
        deadline = self._cleanup_deadline
        if deadline is None:
            self.scheduler.close()
        else:
            self.scheduler.close(timeout=_remaining(deadline))


class _OrderedHandle:
    def __init__(self, task_id: str, binding: LocalBinding):
        self.task_id, self.binding = task_id, binding
        self.decision_seq = None
        self.state = HandleState.PENDING
        self.receipt = None
        self.error: BaseException | None = None
        self.condition = threading.Condition()

    def bind(self, receipt) -> None:
        with self.condition:
            self.receipt = receipt
            if self.state is not HandleState.FAILED:
                self.state = HandleState.BOUND
            self.condition.notify_all()

    def complete(self) -> None:
        with self.condition:
            self.state = HandleState.COMPLETED
            self.condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self.condition:
            if self.state is not HandleState.COMPLETED:
                self.error = error
                self.state = HandleState.FAILED
                self.condition.notify_all()

    def wait_host(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while self.state not in {HandleState.COMPLETED, HandleState.FAILED}:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self.condition.wait(remaining)
            if self.error is not None:
                raise self.error
            return True


class RawOrderedDagAdapter:
    """Bare-policy-free static host order; an explicitly named raw-ordered reference."""

    def __init__(self, graph: DagGraph, *, order: tuple[str, ...], rank: int,
                 group_ranks: Mapping[str, tuple[int, ...]], device: str,
                 deadline_s: float, poll_interval_s: float, event_log: EventLog,
                 failure_signal: EpochFailureSignal | None = None,
                 failure_drain_timeout_s: float = 1.0):
        observed: list[tuple[str, ...] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(observed, tuple(order))
        if any(item != tuple(order) for item in observed):
            raise RuntimeError(f"raw-ordered static sequence differs across ranks: {observed}")
        self.graph, self.order, self.rank = graph, tuple(order), rank
        self.group_ranks, self.event_log = dict(group_ranks), event_log
        self.failure_signal = failure_signal
        self.failure_drain_timeout_s = max(0.01, failure_drain_timeout_s)
        self.deadline_s, self.poll_interval_s = deadline_s, poll_interval_s
        self._replay_deadline: float | None = None
        self._cleanup_deadline: float | None = None
        self.executor = CudaCollectiveExecutor(device)
        self._condition = threading.Condition()
        self._handles: dict[str, _OrderedHandle] = {}
        self._failure: BaseException | None = None
        self._deadline: float | None = None
        self._stopping = False
        self._started = False
        self._closed = False
        self._launch_order: list[str] = []
        self._comm_by_task = {
            f"{job.job_id}/{node.node_id}": node
            for job in graph.jobs for node in job.nodes if isinstance(node, CommNode)
        }
        self._thread = threading.Thread(target=self._run, name="raw-ordered-dag-dispatch", daemon=True)
        self._thread.start()

    @property
    def failure(self):
        return self._failure or (self.failure_signal.failure()
                                 if self.failure_signal is not None else None)

    @property
    def grant_order(self) -> list[str]:
        return []

    @property
    def launch_order(self) -> list[str]:
        with self._condition:
            return list(self._launch_order)

    def register_group(self, *_args, **_kwargs) -> None:
        return None

    def start(self, *_args, **_kwargs) -> None:
        with self._condition:
            self._deadline = time.monotonic() + self.deadline_s
            self._replay_deadline = self._deadline
            self._started = True
            self._condition.notify_all()

    @property
    def cleanup_deadline(self) -> float | None:
        if self._cleanup_deadline is not None:
            return self._cleanup_deadline
        return self._replay_deadline

    def set_replay_deadline(self, deadline: float) -> None:
        with self._condition:
            self._replay_deadline = deadline
            self._deadline = deadline
            self._condition.notify_all()

    def submit(self, spec: TaskSpec, binding: LocalBinding, hint: TaskHint):
        del hint
        task_id = spec.task_id
        if task_id not in self._comm_by_task:
            raise ValueError(f"raw-ordered task {task_id} is absent from common sequence")
        if self.rank not in self.group_ranks[spec.group_id]:
            raise ValueError(f"rank {self.rank} submitted non-member task {task_id}")
        with self._condition:
            self._raise_failure_locked()
            if task_id in self._handles:
                raise ValueError(f"duplicate raw-ordered submit for {task_id}")
            handle = _OrderedHandle(task_id, binding)
            self._handles[task_id] = handle
            self.event_log.record("raw_ordered_submit_return", task_id=task_id)
            self._condition.notify_all()
            return handle

    def declare(self, *_args, **_kwargs):
        raise NotImplementedError("raw-ordered adapter does not support Lookahead DECLARE")

    def finish_epoch(self, timeout: float | None = None) -> None:
        deadline = self._deadline
        if deadline is None:
            deadline = time.monotonic() + (self.deadline_s if timeout is None else timeout)
        elif timeout is not None:
            deadline = min(deadline, time.monotonic() + timeout)
        expected = [task_id for task_id in self.order
                    if self.rank in self.group_ranks[self._comm_by_task[task_id].group_id]]
        with self._condition:
            while (not self._failure and (self._launch_order != expected
                                          or any(handle.state is not HandleState.COMPLETED
                                                 for handle in self._handles.values()))):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("raw-ordered adapter did not drain the common plan")
                self._condition.wait(min(remaining, self.poll_interval_s))
            self._raise_failure_locked()
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(max(0.0, deadline - time.monotonic()))
        if self._thread.is_alive():
            raise TimeoutError("raw-ordered dispatcher did not stop")

    def abort(self, exc: BaseException, **context) -> None:
        deadline = context.pop("deadline", None)
        if deadline is None:
            deadline = self._replay_deadline
        if deadline is None:
            deadline = time.monotonic() + self.failure_drain_timeout_s
        if self._cleanup_deadline is not None:
            deadline = min(deadline, self._cleanup_deadline)
        self._cleanup_deadline = deadline
        if self.failure_signal is not None:
            try:
                self.failure_signal.publish(exc)
            except BaseException as signal_error:
                self.event_log.record("peer_failure_signal_error", error_type=type(signal_error).__name__)
        with self._condition:
            if self._failure is None:
                self._failure = exc
            for handle in self._handles.values():
                if handle.state is not HandleState.COMPLETED:
                    handle.fail(exc)
            self._stopping = True
            self.event_log.record("raw_ordered_abort", error_type=type(exc).__name__, **context)
            self._condition.notify_all()
        # A launch can be between leaving the lock and binding its backend
        # receipt. Join the dispatcher first so the following drain sees every
        # committed Work before the process groups are destroyed.
        if threading.current_thread() is not self._thread:
            self._thread.join(_remaining(deadline))
        # Committed backend Work cannot be cancelled at the adapter boundary.
        # Use its bounded backend wait as the failure boundary before PG teardown.
        for handle in tuple(self._handles.values()):
            remaining = _remaining(deadline)
            if remaining <= 0:
                break
            receipt = handle.receipt
            backend_work = getattr(receipt, "backend_work", None)
            if backend_work is None:
                continue
            try:
                backend_work.wait(timeout=timedelta(seconds=remaining))
            except BaseException:
                pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        deadline = self._cleanup_deadline
        self._thread.join(5.0 if deadline is None else _remaining(deadline))
        if self._thread.is_alive():
            raise TimeoutError("raw-ordered dispatcher did not stop")

    def _run(self) -> None:
        for task_id in self.order:
            node = self._comm_by_task[task_id]
            if self.rank not in self.group_ranks[node.group_id]:
                continue
            with self._condition:
                while task_id not in self._handles and not self._stopping:
                    self._check_deadline_locked()
                    self._condition.wait(self._next_wait_locked())
                if self._stopping:
                    return
                self._raise_failure_locked()
                handle = self._handles[task_id]
            try:
                receipt = self.executor.launch(handle.binding)
                handle.bind(receipt)
                self.event_log.record("raw_ordered_launch", task_id=task_id,
                                      completion_source=receipt.completion_source)
                while not receipt.is_completed():
                    with self._condition:
                        self._check_deadline_locked()
                        if self._stopping:
                            return
                        self._condition.wait(self._next_wait_locked())
                handle.complete()
                with self._condition:
                    self._launch_order.append(task_id)
                    self._condition.notify_all()
            except BaseException as exc:
                handle.fail(exc)
                self.abort(exc, task_id=task_id)
                return

    def _next_wait_locked(self) -> float:
        if self._deadline is None:
            return self.poll_interval_s
        return max(0.0, min(self.poll_interval_s, self._deadline - time.monotonic()))

    def _check_deadline_locked(self) -> None:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            self._failure = TimeoutError("raw-ordered adapter exceeded fixed replay deadline")
            self._condition.notify_all()
            self._raise_failure_locked()

    def _raise_failure_locked(self) -> None:
        if self._failure is not None:
            raise self._failure
