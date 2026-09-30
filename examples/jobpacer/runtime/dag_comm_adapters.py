"""Communication-engine adapters consumed by the shared DAG runner."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping

import torch
import torch.distributed as dist

from runtime_comm_scheduler import AdmissionScheduler, CommIntent, Plan, TaskKey, TorchProcessGroupExecutor
from runtime_comm_scheduler.dag import (
    CommNode, ComputeNode, DagGraph, DagJob, build_layered_fifo_order,
    build_static_order, validate_graph, validate_static_order,
)
from runtime_comm_scheduler.runtime import EventLog, HandleState, LocalBinding, TaskHint, TaskSpec
from runtime_comm_scheduler.runtime.executor import CudaCollectiveExecutor


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


BARE_CONTRACT_VERSION = "bare-ordered-v2-layered-round-robin"
NCCL_LAUNCH_ORDER_ENV = "NCCL_LAUNCH_ORDER_IMPLICIT"


def validate_bare_backend(*, nccl_version: tuple[int, ...], cuda_version: str | None,
                          implicit: str | None, blocking_wait: str | None) -> None:
    """Supported implicit-order path, checked before any communicator is created."""
    if tuple(nccl_version) < (2, 26, 0):
        raise ValueError("bare implicit launch ordering requires NCCL >= 2.26")
    try:
        cuda = tuple(int(part) for part in (cuda_version or "").split(".")[:2])
    except ValueError as exc:
        raise ValueError("bare requires a known CUDA runtime version") from exc
    if cuda < (12, 3):
        raise ValueError("bare multi-communicator contract requires CUDA >= 12.3")
    if (implicit or "").strip() != "1":
        raise ValueError("bare requires NCCL_LAUNCH_ORDER_IMPLICIT=1")
    if (blocking_wait or "").strip().lower() in {"1", "true", "yes", "on"}:
        raise ValueError("bare asynchronous completion bridge rejects TORCH_NCCL_BLOCKING_WAIT")


@dataclass(frozen=True)
class BareOrder:
    contract_version: str
    input_hash: str
    job_order: tuple[str, ...]
    sequence: tuple[str, ...]
    digest: str


def build_bare_order(graph: DagGraph, *, submit_after: Mapping[str, tuple[str, ...]] | None = None,
                     input_hash: str | None = None) -> BareOrder:
    """Freeze the layered, estimate-independent communication projection of the DAG."""
    validate_graph(graph)
    node_by_id = {
        f"{job.job_id}/{node.node_id}": node
        for job in graph.jobs for node in job.nodes
    }
    normalized_submit_after: dict[str, tuple[str, ...]] = {}
    extra_predecessors: dict[str, tuple[str, ...]] = {}
    for compute_id, comm_ids in (submit_after or {}).items():
        compute = node_by_id.get(compute_id)
        if not isinstance(compute, ComputeNode):
            raise ValueError(f"bare submit_after key {compute_id!r} is not a compute node")
        if len(comm_ids) != len(set(comm_ids)):
            raise ValueError(f"bare submit_after for {compute_id} contains duplicate communication nodes")
        for comm_id in comm_ids:
            comm = node_by_id.get(comm_id)
            if not isinstance(comm, CommNode) or comm_id.split("/", 1)[0] != compute_id.split("/", 1)[0]:
                raise ValueError(f"bare submit_after for {compute_id} references invalid comm {comm_id!r}")
            # A submit gate means this request must have been accepted before
            # the compute starts. Model that acceptance-before-start relation
            # while freezing the communication sequence so a queue head cannot
            # depend on compute that waits for a later request to be accepted.
        extra_predecessors[compute_id] = tuple(comm_ids)
        normalized_submit_after[compute_id] = tuple(sorted(comm_ids))
    job_order = tuple(job.job_id for job in graph.jobs)
    try:
        sequence = build_layered_fifo_order(
            graph, extra_predecessors=extra_predecessors, job_order=job_order,
        )
    except ValueError as exc:
        if "constraints form a cycle" in str(exc):
            raise ValueError(f"bare default order conflicts with DAG/submit_after constraints: {exc}") from exc
        raise
    groups = {
        group.group_id: list(group.ranks)
        for group in sorted(graph.groups, key=lambda item: item.group_id)
    }
    resolved_input_hash = input_hash or _bare_graph_hash(graph)
    payload = {
        "contract_version": BARE_CONTRACT_VERSION,
        "input_hash": resolved_input_hash,
        "group_members": groups,
        "job_order": list(job_order),
        "submit_after": {key: list(value) for key, value in sorted(normalized_submit_after.items())},
        "task_ids": list(sequence),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    return BareOrder(BARE_CONTRACT_VERSION, resolved_input_hash, job_order, sequence, digest)


def _bare_graph_hash(graph: DagGraph) -> str:
    """Stable fallback for direct unit-test graphs without a parsed input hash."""
    rows = []
    for job in graph.jobs:
        nodes = []
        for node in job.nodes:
            row: dict[str, Any] = {
                "node_id": node.node_id,
                "kind": "comm" if isinstance(node, CommNode) else "compute",
                "deps": list(node.deps),
            }
            if isinstance(node, CommNode):
                row.update(group_id=node.group_id, group_seq=node.group_seq,
                           collective=node.collective.to_dict())
            nodes.append(row)
        rows.append({"job_id": job.job_id, "nodes": nodes})
    payload = {
        "groups": {group.group_id: list(group.ranks)
                   for group in sorted(graph.groups, key=lambda item: item.group_id)},
        "jobs": rows,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class EpochFailureSignal:
    """Cross-rank fail-fast notification over the rendezvous Store, not a job collective."""

    def __init__(self, epoch: int, rank: int, world_size: int):
        self.rank, self.world_size = rank, world_size
        self._store = dist.distributed_c10d._get_default_store()
        self._prefix = f"jobpacer/phase3/epoch/{epoch}"
        self._key = f"{self._prefix}/failure"
        self._publish_lock = threading.Lock()
        self._published = False

    def publish(self, exc: BaseException) -> None:
        # Keep the first epoch failure.  Several ranks and DAG workers can
        # observe the same stop concurrently; a follow-on cancellation must
        # not overwrite the original cause.
        with self._publish_lock:
            if self._published:
                return
            payload = f"rank={self.rank} {type(exc).__name__}: {exc}"
            self._store.compare_set(self._key, "", payload)
            self._published = True

    def failure(self) -> BaseException | None:
        # compare_set makes the epoch's first failure immutable.
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

    def verify_common_contract(self, name: str, payload: Mapping[str, Any], deadline: float) -> str:
        """Compare a frozen contract through the rendezvous store before app release."""
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        keys = [f"{self._prefix}/contract/{name}/rank/{rank}" for rank in range(self.world_size)]
        self._store.set(keys[self.rank], serialized.encode("utf-8"))
        while time.monotonic() < deadline and not self._store.check(keys):
            peer_failure = self.failure()
            if peer_failure is not None:
                raise peer_failure
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        if not self._store.check(keys):
            error = TimeoutError(f"timed out verifying common {name} contract across ranks")
            self.publish(error)
            raise error
        values = [self._store.get(key).decode("utf-8", errors="replace") for key in keys]
        if len(set(values)) != 1:
            error = ValueError(f"ranks loaded different {name} contracts: {values}")
            self.publish(error)
            raise error
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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


class _BareHandle:
    def __init__(self, task_id: str, group_id: str, group_seq: int,
                 binding: LocalBinding, arrival_seq: int):
        self.task_id, self.group_id, self.group_seq = task_id, group_id, group_seq
        self.binding, self.arrival_seq = binding, arrival_seq
        self.decision_seq = None
        self._state = HandleState.PENDING
        self.launch_started = False
        self.receipt = None
        self.error: BaseException | None = None
        self.condition = threading.Condition()

    @property
    def state(self):
        with self.condition:
            return self._state

    def bind(self, receipt) -> None:
        with self.condition:
            if self._state is not HandleState.PENDING:
                raise RuntimeError(f"invalid bare bind for {self.task_id}: {self._state.value}")
            self.receipt = receipt
            self._state = HandleState.BOUND
            self.condition.notify_all()

    def complete(self) -> None:
        with self.condition:
            if self._state is HandleState.BOUND:
                self._state = HandleState.COMPLETED
                self.condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self.condition:
            if self._state not in {HandleState.COMPLETED, HandleState.FAILED}:
                self.error = error
                self._state = HandleState.FAILED
                self.condition.notify_all()

    def wait_host(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.condition:
            while self._state not in {HandleState.COMPLETED, HandleState.FAILED}:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self.condition.wait(remaining)
            if self.error is not None:
                raise self.error
            return True


class BareDagAdapter:
    """Direct NCCL issue in a frozen framework-default order, without admission.

    ``submit`` only accepts a locally ready request and returns its handle.
    One dispatcher per rank issues the local projection of the common order;
    a separate observer reports physical completion. The dispatcher never
    waits for an earlier operation to complete before issuing the next one.
    """

    def __init__(self, graph: DagGraph, *, rank: int,
                 group_ranks: Mapping[str, tuple[int, ...]], device: str,
                 order: BareOrder, submit_after: Mapping[str, tuple[str, ...]] | None,
                 deadline_s: float, poll_interval_s: float, event_log: EventLog,
                 failure_signal: EpochFailureSignal | None = None,
                 completion_probe: Any | None = None,
                 failure_drain_timeout_s: float = 0.5):
        validate_graph(graph)
        if deadline_s <= 0 or poll_interval_s <= 0:
            raise ValueError("bare deadline and poll interval must be positive")
        manifest_group_ranks = {group.group_id: group.ranks for group in graph.groups}
        if dict(group_ranks) != manifest_group_ranks:
            raise ValueError("bare group membership differs from the validated DAG")
        expected_order = build_bare_order(graph, submit_after=submit_after,
                                          input_hash=order.input_hash)
        if order.contract_version != BARE_CONTRACT_VERSION:
            raise ValueError(f"unsupported bare contract {order.contract_version!r}")
        if order != expected_order:
            raise ValueError("bare order differs from the validated DAG/default-order contract")
        self.graph, self.rank = graph, rank
        self.group_ranks, self.event_log = dict(group_ranks), event_log
        self.order = order
        self.submit_after = dict(submit_after or {})
        self.failure_signal = failure_signal
        self.completion_probe = completion_probe
        self.failure_drain_timeout_s = max(0.01, min(0.5, failure_drain_timeout_s))
        self.deadline_s, self.poll_interval_s = deadline_s, poll_interval_s
        self._replay_deadline: float | None = None
        self._cleanup_deadline: float | None = None
        self.executor = CudaCollectiveExecutor(device)
        self._condition = threading.Condition()
        self._handles: dict[str, _BareHandle] = {}
        self._comm_by_task = {
            f"{job.job_id}/{node.node_id}": node
            for job in graph.jobs for node in job.nodes if isinstance(node, CommNode)
        }
        self._local_order = tuple(task_id for task_id in order.sequence
                                  if rank in self.group_ranks[self._comm_by_task[task_id].group_id])
        self._local_expected = set(self._local_order)
        self._next_order_index = 0
        self._head_wait_started: dict[str, float] = {}
        self._head_wait_logged_tasks: set[str] = set()
        self._failure: BaseException | None = None
        self._abort_started = False
        self._deadline: float | None = None
        self._stopping = False
        self._started = False
        self._closed = False
        self._launch_order: list[str] = []
        self._orphan_receipts: list[Any] = []
        self._inflight = 0
        self.peak_inflight = 0
        self._arrival_seq = 0
        self._dispatcher = threading.Thread(target=self._dispatch, name="bare-dag-dispatch", daemon=True)
        self._thread = threading.Thread(target=self._observe, name="bare-dag-completion", daemon=True)
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

    @property
    def local_order(self) -> tuple[str, ...]:
        return self._local_order

    @property
    def cleanup_deadline(self) -> float | None:
        return self._cleanup_deadline or self._replay_deadline

    def register_group(self, *_args, **_kwargs) -> None:
        return None

    def start(self, *_args, **_kwargs) -> None:
        with self._condition:
            if self._closed or self._stopping:
                raise RuntimeError("cannot start a closed bare adapter")
            if self._started:
                raise RuntimeError("bare adapter has already started")
            self._deadline = time.monotonic() + self.deadline_s
            self._replay_deadline = self._deadline
            self._started = True
            self._condition.notify_all()
            self._dispatcher.start()

    def set_replay_deadline(self, deadline: float) -> None:
        with self._condition:
            self._replay_deadline = deadline
            self._deadline = deadline
            self._condition.notify_all()

    def submit(self, spec: TaskSpec, binding: LocalBinding, hint: TaskHint):
        del hint
        task_id = spec.task_id
        node = self._comm_by_task.get(task_id)
        if node is None:
            raise ValueError(f"bare task {task_id} is absent from the validated DAG")
        if (spec.group_id != node.group_id or spec.group_seq != node.group_seq
                or spec.collective != node.collective):
            raise ValueError(f"bare task metadata differs from the common DAG for {task_id}")
        if self.rank not in self.group_ranks[spec.group_id]:
            raise ValueError(f"rank {self.rank} submitted non-member task {task_id}")
        if task_id not in self._local_expected:
            raise ValueError(f"bare task {task_id} is not in this rank's frozen launch projection")
        peer_failure = self.failure
        if peer_failure is not None:
            raise peer_failure
        with self._condition:
            if not self._started:
                raise RuntimeError("bare submit before adapter start")
            self._raise_failure_locked()
            if self._stopping:
                raise RuntimeError("bare submit after adapter termination")
            if task_id in self._handles:
                raise ValueError(f"duplicate bare submit for {task_id}")
            self._arrival_seq += 1
            handle = _BareHandle(task_id, spec.group_id, spec.group_seq,
                                 binding, self._arrival_seq)
            self._handles[task_id] = handle
            self.event_log.record("bare_submit_accepted", task_id=task_id,
                                  group_id=spec.group_id, group_seq=spec.group_seq,
                                  local_arrival_seq=handle.arrival_seq,
                                  expected_order_index=self._local_order.index(task_id))
            self.event_log.record("bare_submit_return", task_id=task_id,
                                  state=handle.state.value)
            self._condition.notify_all()
        return handle

    def _dispatch(self) -> None:
        while True:
            peer_failure = self.failure
            if peer_failure is not None:
                self.abort(peer_failure, stage="peer_failure", publish_failure=False)
                return
            error: BaseException | None = None
            handle: _BareHandle | None = None
            with self._condition:
                if self._stopping:
                    return
                if not self._started:
                    self._condition.wait(self.poll_interval_s)
                    continue
                deadline = self._deadline
                remaining = float("inf") if deadline is None else deadline - time.monotonic()
                if remaining <= 0:
                    next_task = (self._local_order[self._next_order_index]
                                 if self._next_order_index < len(self._local_order) else None)
                    error = TimeoutError(
                        f"bare ordered dispatch deadline expired at task {next_task!r}; "
                        f"launched={self._next_order_index}/{len(self._local_order)}"
                    )
                elif self._next_order_index >= len(self._local_order):
                    self._condition.wait(min(remaining, self.poll_interval_s))
                    continue
                else:
                    task_id = self._local_order[self._next_order_index]
                    handle = self._handles.get(task_id)
                    if handle is None:
                        if task_id not in self._head_wait_logged_tasks:
                            self._head_wait_logged_tasks.add(task_id)
                            self._head_wait_started[task_id] = time.monotonic()
                            self.event_log.record("bare_order_head_wait", task_id=task_id,
                                                  order_index=self._next_order_index)
                        self._condition.wait(min(remaining, self.poll_interval_s))
                        continue
                    handle.launch_started = True
                    wait_start = self._head_wait_started.pop(task_id, None)
                    if wait_start is not None:
                        self.event_log.record("bare_order_head_released", task_id=task_id,
                                              order_wait_s=max(0.0, time.monotonic() - wait_start))
                    self.event_log.record("bare_launch_call", task_id=handle.task_id,
                                          group_id=handle.group_id, group_seq=handle.group_seq,
                                          launch_index=self._next_order_index,
                                          local_arrival_seq=handle.arrival_seq)
            if error is not None:
                self.abort(error, stage="bare_dispatch_deadline")
                return
            assert handle is not None
            try:
                receipt = self.executor.launch(handle.binding)
            except BaseException as exc:
                self.abort(exc, stage="bare_launch", task_id=handle.task_id)
                return
            with self._condition:
                handle.receipt = receipt
                issued_after_abort = self._stopping or handle.state is HandleState.FAILED
                if not issued_after_abort:
                    handle.bind(receipt)
                else:
                    self._orphan_receipts.append(receipt)
                self._launch_order.append(handle.task_id)
                self._next_order_index += 1
                self._inflight += 1
                self.peak_inflight = max(self.peak_inflight, self._inflight)
                self.event_log.record("bare_launch", task_id=handle.task_id,
                                      group_id=handle.group_id, group_seq=handle.group_seq,
                                      launch_index=len(self._launch_order) - 1,
                                      inflight=self._inflight,
                                      issued_after_abort=issued_after_abort)
                self._condition.notify_all()

    def declare(self, *_args, **_kwargs):
        raise NotImplementedError("bare adapter does not implement coordinator DECLARE")

    def finish_epoch(self, timeout: float | None = None) -> None:
        deadline = self._deadline
        if deadline is None:
            deadline = time.monotonic() + (self.deadline_s if timeout is None else timeout)
        elif timeout is not None:
            deadline = min(deadline, time.monotonic() + timeout)
        expected = self._local_expected
        while True:
            peer_failure = self.failure
            if peer_failure is not None:
                self.abort(peer_failure, stage="peer_failure", publish_failure=False,
                           deadline=deadline)
                raise peer_failure
            error: BaseException | None = None
            with self._condition:
                self._raise_failure_locked()
                launched = set(self._launch_order)
                all_completed = all(handle.state is HandleState.COMPLETED
                                    for handle in self._handles.values())
                if launched == expected and len(self._handles) == len(expected) and all_completed:
                    self._stopping = True
                    self._condition.notify_all()
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = TimeoutError(
                        f"bare epoch did not drain: launched={len(launched)}/{len(expected)}, "
                        f"handles={len(self._handles)}/{len(expected)}, "
                        f"next={self._local_order[self._next_order_index:self._next_order_index + 1]}"
                    )
                else:
                    self._condition.wait(min(remaining, self.poll_interval_s))
            if error is not None:
                self.abort(error, stage="bare_drain", deadline=deadline)
                raise error
        self._dispatcher.join(_remaining(deadline))
        self._thread.join(_remaining(deadline))
        if self._dispatcher.is_alive() or self._thread.is_alive():
            error = TimeoutError("bare dispatcher or completion observer did not stop")
            self.abort(error, stage="bare_worker_join", deadline=deadline)
            raise error

    def abort(self, exc: BaseException, **context) -> None:
        deadline = context.pop("deadline", None)
        publish_failure = context.pop("publish_failure", True)
        now = time.monotonic()
        if deadline is None:
            deadline = self._replay_deadline
        cleanup_limit = now + self.failure_drain_timeout_s
        if deadline is None or deadline <= now:
            deadline = cleanup_limit
        else:
            deadline = min(deadline, cleanup_limit)
        if self._cleanup_deadline is not None:
            deadline = min(deadline, self._cleanup_deadline)
        self._cleanup_deadline = deadline
        with self._condition:
            if self._abort_started:
                self._stopping = True
                self._condition.notify_all()
                return
            self._abort_started = True
            if self._failure is None:
                self._failure = exc
            root_cause = self._failure
            for handle in self._handles.values():
                if handle.state is not HandleState.COMPLETED:
                    handle.fail(root_cause)
            self._stopping = True
            self.event_log.record("bare_abort", error_type=type(root_cause).__name__, **context)
            self._condition.notify_all()
        if publish_failure and self.failure_signal is not None:
            try:
                self.failure_signal.publish(root_cause)
            except BaseException as signal_error:
                self.event_log.record("peer_failure_signal_error",
                                      error_type=type(signal_error).__name__)
        for worker in (self._dispatcher, self._thread):
            if worker.ident is not None and threading.current_thread() is not worker:
                worker.join(_remaining(deadline))
        receipts = [handle.receipt for handle in tuple(self._handles.values())
                    if handle.receipt is not None]
        receipts.extend(self._orphan_receipts)
        for receipt in receipts:
            remaining = _remaining(deadline)
            if remaining <= 0:
                break
            backend_work = getattr(receipt, "backend_work", None)
            if backend_work is None:
                continue
            try:
                backend_work.wait(timeout=timedelta(seconds=remaining))
            except BaseException:
                pass

    def close(self) -> None:
        peer_failure = self.failure
        with self._condition:
            if self._closed:
                return
            fully_drained = (
                self._started
                and set(self._launch_order) == self._local_expected
                and len(self._handles) == len(self._local_expected)
                and all(handle.state is HandleState.COMPLETED
                        for handle in self._handles.values())
            )
            close_error = None
            if self._started and not fully_drained and self._failure is None:
                close_error = peer_failure or RuntimeError(
                    "bare adapter closed before the epoch drained"
                )
            self._stopping = True
            self._condition.notify_all()
        if close_error is not None:
            self.abort(close_error, stage="bare_close_before_drain")
        with self._condition:
            self._closed = True
        deadline = self.cleanup_deadline
        if self._dispatcher.ident is not None:
            self._dispatcher.join(1.0 if deadline is None else _remaining(deadline))
        self._thread.join(1.0 if deadline is None else _remaining(deadline))
        if self._dispatcher.is_alive() or self._thread.is_alive():
            raise TimeoutError("bare dispatcher or completion observer did not stop")

    def _observe(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
                started = self._started
                deadline = self._deadline
                handles = [handle for handle in self._handles.values()
                           if handle.state is HandleState.BOUND]
            if not started:
                with self._condition:
                    self._condition.wait(self.poll_interval_s)
                continue
            peer_failure = self.failure_signal.failure() if self.failure_signal is not None else None
            if peer_failure is not None:
                self.abort(peer_failure, stage="peer_failure", publish_failure=False)
                return
            if deadline is not None and time.monotonic() >= deadline:
                self.abort(TimeoutError("bare adapter exceeded fixed replay deadline"), stage="deadline")
                return
            for handle in handles:
                try:
                    completed = (self.completion_probe.is_completed(handle.receipt)
                                 if self.completion_probe is not None
                                 else bool(handle.receipt.is_completed()))
                except BaseException as exc:
                    self.abort(exc, stage="completion_probe", task_id=handle.task_id)
                    return
                if completed:
                    with self._condition:
                        if handle.state is HandleState.BOUND:
                            handle.complete()
                            self._inflight -= 1
                            self.event_log.record("bare_physical_complete", task_id=handle.task_id,
                                                  group_id=handle.group_id,
                                                  inflight=self._inflight)
                            self._condition.notify_all()
            with self._condition:
                if not self._stopping:
                    self._condition.wait(self.poll_interval_s)

    def _raise_failure_locked(self) -> None:
        if self._failure is not None:
            raise self._failure
