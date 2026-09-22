"""Advance one DAG job while the runtime remains the admission authority."""

from __future__ import annotations

import concurrent.futures
import enum
import threading
import time
from typing import Any, Callable, Mapping

from runtime_comm_scheduler.runtime import EventLog, LocalBinding, RankRuntime, monotonic_us

from .model import (
    CommNode,
    ComputeNode,
    DagGraph,
    DagJob,
    Node,
    compute_tails,
    dag_task_hint,
    dag_task_spec,
)


class NodeState(enum.Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DagRunner:
    """Advance one job using an application-supplied CPU compute callable."""

    def __init__(self, graph: DagGraph, job: DagJob, runtime: RankRuntime, *,
                 epoch: int, rank: int,
                 compute_fn: Callable[[ComputeNode, threading.Event], None],
                 make_binding: Callable[[CommNode], LocalBinding],
                 deadline: float, poll_interval: float = 0.001,
                 tails: Mapping[str, float] | None = None,
                 enable_lookahead: bool = False,
                 stop_event: threading.Event | None = None,
                 event_log: EventLog | None = None) -> None:
        if deadline <= 0 or poll_interval <= 0:
            raise ValueError("deadline and dag poll interval must be positive")
        if not callable(compute_fn):
            raise TypeError("compute_fn must be callable")
        self.graph, self.job, self.runtime = graph, job, runtime
        self.epoch, self.rank = epoch, rank
        self.compute_fn, self.make_binding = compute_fn, make_binding
        self.deadline, self.poll_interval = deadline, poll_interval
        self.enable_lookahead = enable_lookahead
        self.stop_event = stop_event or threading.Event()
        self.event_log = event_log or EventLog("dag", rank)
        self.tails = tails if tails is not None else compute_tails(graph)
        self.node_by_id = {node.node_id: node for node in job.nodes}
        self.states = {node.node_id: NodeState.PENDING for node in job.nodes}
        self.remaining_deps = {node.node_id: len(node.deps) for node in job.nodes}
        self.successors = {node.node_id: [] for node in job.nodes}
        for node in job.nodes:
            for dep in node.deps:
                self.successors[dep].append(node.node_id)
        self.future: concurrent.futures.Future[None] | None = None
        self.compute_node: ComputeNode | None = None
        self.compute_started: dict[str, int] = {}
        self.handles: dict[str, Any] = {}
        self.bindings: dict[str, LocalBinding] = {}
        self.declared: set[str] = set()
        self.failed_node: str | None = None
        self._node_key = {node.node_id: index for index, node in enumerate(job.nodes)}

    @property
    def completed_node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.job.nodes if self.states[node.node_id] is NodeState.COMPLETED)

    def run(self) -> dict[str, Any]:
        started = time.perf_counter_ns() // 1000
        self.event_log.record("job_started", job_id=self.job.job_id)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"dag-compute-{self.job.job_id}")
        try:
            for node in self.job.nodes:
                if not self.remaining_deps[node.node_id]:
                    self.states[node.node_id] = NodeState.READY
                    self._record_ready(node)
            while any(state is not NodeState.COMPLETED for state in self.states.values()):
                self._check_stop()
                progressed = self._collect_compute()
                progressed |= self._collect_comms()
                for comm in self._ready_comms():
                    task_id = f"{self.job.job_id}/{comm.node_id}"
                    self.failed_node = comm.node_id
                    binding = self.make_binding(comm)
                    spec = dag_task_spec(self.job, comm, epoch=self.epoch)
                    hint = dag_task_hint(comm, tail_s=self.tails[task_id])
                    self.event_log.record("comm_submit_call", task_id=task_id, job_id=self.job.job_id,
                                          node_id=comm.node_id, group_id=comm.group_id, group_seq=comm.group_seq)
                    handle = self.runtime.submit(spec, binding, hint)
                    self.bindings[comm.node_id] = binding
                    self.handles[comm.node_id] = handle
                    self.states[comm.node_id] = NodeState.RUNNING
                    self.event_log.record("comm_submit_return", task_id=task_id, job_id=self.job.job_id,
                                          node_id=comm.node_id, group_id=comm.group_id, group_seq=comm.group_seq)
                    self.failed_node = None
                    progressed = True
                if self.future is None:
                    ready_compute = [node for node in self.job.nodes
                                     if isinstance(node, ComputeNode) and self.states[node.node_id] is NodeState.READY]
                    if ready_compute:
                        node = min(ready_compute, key=lambda item: self._node_key[item.node_id])
                        self.compute_node = node
                        self.compute_started[node.node_id] = monotonic_us()
                        self.states[node.node_id] = NodeState.RUNNING
                        self.event_log.record("compute_started", job_id=self.job.job_id, node_id=node.node_id,
                                              estimated_duration_s=node.estimated_duration_s)
                        self.future = pool.submit(self.compute_fn, node, self.stop_event)
                        progressed = True
                if self.enable_lookahead and self._declare_safe_frontier():
                    progressed = True
                if not progressed:
                    remaining = self.deadline - time.monotonic()
                    if remaining <= 0:
                        self._raise_timeout()
                    self.stop_event.wait(min(self.poll_interval, remaining))
            self.event_log.record("job_completed", job_id=self.job.job_id,
                                  duration_s=(time.perf_counter_ns() // 1000 - started) / 1_000_000)
            return {"job_id": self.job.job_id, "status": "ok", "node_count": len(self.job.nodes),
                    "completed_node_ids": list(self.completed_node_ids)}
        except BaseException as exc:
            if self.failed_node is not None:
                self.states[self.failed_node] = NodeState.FAILED
                self.event_log.record("node_failed", job_id=self.job.job_id, node_id=self.failed_node,
                                      error_type=type(exc).__name__, error=str(exc))
            self.event_log.record("job_failed", job_id=self.job.job_id, error_type=type(exc).__name__, error=str(exc))
            self.runtime.abort(exc, stage="dag_runner", job_id=self.job.job_id, node_id=self.failed_node)
            raise
        finally:
            if self.future is not None and not self.future.done():
                self.stop_event.set()
            pool.shutdown(wait=self.future is None or self.future.done(), cancel_futures=True)

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise RuntimeError(f"DAG job {self.job.job_id} stopped after a peer failure")
        if self.runtime.failure is not None:
            raise self.runtime.failure
        if time.monotonic() >= self.deadline:
            self._raise_timeout()

    def _raise_timeout(self) -> None:
        detail = {
            "pending": {node: self.remaining_deps[node] for node, state in self.states.items()
                        if state is not NodeState.COMPLETED},
            "states": {key: value.value for key, value in self.states.items()},
            "running_handles": {node: handle.state.value for node, handle in self.handles.items()},
        }
        self.event_log.record("job_timeout", job_id=self.job.job_id, **detail)
        raise TimeoutError(f"DAG job {self.job.job_id} exceeded replay deadline: {detail}")

    def _collect_compute(self) -> bool:
        future, node = self.future, self.compute_node
        if future is None or node is None or not future.done():
            return False
        self.failed_node = node.node_id
        future.result()
        self.failed_node = None
        self.event_log.record("compute_completed", job_id=self.job.job_id, node_id=node.node_id)
        self.future, self.compute_node = None, None
        self._complete(node.node_id)
        return True

    def _collect_comms(self) -> bool:
        progressed = False
        for node in self.job.nodes:
            if not isinstance(node, CommNode) or self.states[node.node_id] is not NodeState.RUNNING:
                continue
            handle = self.handles[node.node_id]
            self.failed_node = node.node_id
            if handle.wait_host(0):
                self.failed_node = None
                self.event_log.record("comm_completed_observed", task_id=f"{self.job.job_id}/{node.node_id}",
                                      job_id=self.job.job_id, node_id=node.node_id, handle_state=handle.state.value)
                self._complete(node.node_id)
                progressed = True
            else:
                self.failed_node = None
        return progressed

    def _ready_comms(self) -> list[CommNode]:
        return [node for node in self.job.nodes if isinstance(node, CommNode) and self.states[node.node_id] is NodeState.READY]

    def _declare_safe_frontier(self) -> bool:
        changed = False
        for node in self.job.nodes:
            if not isinstance(node, CommNode) or self.states[node.node_id] is not NodeState.PENDING or node.node_id in self.declared:
                continue
            deps = [self.node_by_id[dep_id] for dep_id in node.deps]
            unfinished = [dep for dep in deps if self.states[dep.node_id] is not NodeState.COMPLETED]
            if not unfinished or any(not isinstance(dep, ComputeNode) or self.states[dep.node_id] is not NodeState.RUNNING
                                     for dep in unfinished):
                continue
            running = [dep for dep in unfinished if dep.node_id in self.compute_started]
            if len(running) != len(unfinished):
                continue
            prediction_base_us = monotonic_us()
            ready_after = max(
                max(0.0, dep.estimated_duration_s
                    - (prediction_base_us - self.compute_started[dep.node_id]) / 1_000_000.0)
                for dep in running
            )
            predicted_ready_at_us = prediction_base_us + round(ready_after * 1_000_000)
            task_id = f"{self.job.job_id}/{node.node_id}"
            self.runtime.declare(
                dag_task_spec(self.job, node, epoch=self.epoch),
                dag_task_hint(node, tail_s=self.tails[task_id], ready_after_s=ready_after),
            )
            self.declared.add(node.node_id)
            self.event_log.record("comm_declared", task_id=task_id, job_id=self.job.job_id,
                                  node_id=node.node_id, ready_after_s=ready_after,
                                  prediction_base_us=prediction_base_us,
                                  predicted_ready_at_us=predicted_ready_at_us,
                                  tail_s=self.tails[task_id], prediction_basis=[dep.node_id for dep in running])
            changed = True
        return changed

    def _complete(self, node_id: str) -> None:
        if self.states[node_id] is not NodeState.RUNNING:
            raise RuntimeError(f"DAG node {self.job.job_id}/{node_id} completed from {self.states[node_id].value}")
        self.states[node_id] = NodeState.COMPLETED
        for successor_id in self.successors[node_id]:
            self.remaining_deps[successor_id] -= 1
            if self.remaining_deps[successor_id] < 0:
                raise RuntimeError(f"DAG node {self.job.job_id}/{successor_id} dependency count underflow")
            if self.remaining_deps[successor_id] == 0:
                self.states[successor_id] = NodeState.READY
                self._record_ready(self.node_by_id[successor_id])

    def _record_ready(self, node: Node) -> None:
        self.event_log.record("node_ready", job_id=self.job.job_id, node_id=node.node_id,
                              node_kind="comm" if isinstance(node, CommNode) else "compute",
                              remaining_deps=self.remaining_deps[node.node_id])
