"""Validated DAG models, scheduling summaries, and local CPU replay runner."""

from __future__ import annotations

import concurrent.futures
import enum
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Callable, Mapping

from runtime_comm_scheduler.runtime import (
    CollectiveSpec,
    EventLog,
    GroupSpec,
    LocalBinding,
    TaskHint,
    TaskSpec,
    monotonic_us,
)


@dataclass(frozen=True)
class ComputeNode:
    node_id: str
    deps: tuple[str, ...]
    estimated_duration_s: float


@dataclass(frozen=True)
class CommNode:
    node_id: str
    deps: tuple[str, ...]
    group_id: str
    group_seq: int
    estimated_comm_s: float
    collective: CollectiveSpec


Node = ComputeNode | CommNode


class NodeState(enum.Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class DagJob:
    job_id: str
    nodes: tuple[Node, ...]


@dataclass(frozen=True)
class ReplayExecutionConfig:
    compute_duration_s: Mapping[str, float]


@dataclass(frozen=True)
class DagInput:
    schema_version: int
    name: str
    seed: int
    groups: tuple[GroupSpec, ...]
    jobs: tuple[DagJob, ...]
    execution: ReplayExecutionConfig
    manifest_digest: str
    canonical_json: str
    tails: Mapping[str, float]
    node_order: Mapping[str, tuple[str, int, int]]
    group_ranks: Mapping[str, tuple[int, ...]]

    @property
    def expected_task_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{job.job_id}/{node.node_id}"
            for job in self.jobs for node in job.nodes if isinstance(node, CommNode)
        )

    @property
    def expected_node_ids(self) -> tuple[str, ...]:
        return tuple(f"{job.job_id}/{node.node_id}" for job in self.jobs for node in job.nodes)


def load_dag(path: str | Path, *, epoch: int = 0, world_size: int | None = None) -> DagInput:
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load DAG {path}: {exc}") from exc
    return parse_dag(raw, epoch=epoch, world_size=world_size)


def parse_dag(raw: Any, *, epoch: int = 0, world_size: int | None = None) -> DagInput:
    root = _object(raw, "DAG", {"schema_version", "name", "seed", "groups", "execution", "jobs"})
    version = root["schema_version"]
    if not _is_int(version) or version != 1:
        raise ValueError("DAG.schema_version must be integer 1")
    name = _text(root["name"], "DAG.name")
    seed = root["seed"]
    if not _is_int(seed):
        raise ValueError("DAG.seed must be an integer")
    if not _is_int(epoch) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if world_size is not None and (not _is_int(world_size) or world_size <= 0):
        raise ValueError("world_size must be a positive integer")

    groups: list[GroupSpec] = []
    group_ranks: dict[str, tuple[int, ...]] = {}
    for i, item in enumerate(_array(root["groups"], "DAG.groups")):
        value = _object(item, f"groups[{i}]", {"group_id", "ranks"})
        group_id = _text(value["group_id"], f"groups[{i}].group_id")
        ranks_raw = _array(value["ranks"], f"groups[{i}].ranks")
        if not ranks_raw or any(not _is_int(rank) or rank < 0 for rank in ranks_raw):
            raise ValueError(f"groups[{i}].ranks must be non-empty non-negative integers")
        if len(set(ranks_raw)) != len(ranks_raw):
            raise ValueError(f"groups[{i}].ranks contains duplicates")
        ranks = tuple(sorted(ranks_raw))
        if world_size is not None and ranks[-1] >= world_size:
            raise ValueError(f"group {group_id} contains rank {ranks[-1]} >= world_size {world_size}")
        if group_id in group_ranks:
            raise ValueError(f"duplicate group_id {group_id}")
        group_ranks[group_id] = ranks
        groups.append(GroupSpec(epoch, group_id, ranks))
    groups.sort(key=lambda group: group.group_id)

    execution_raw = _object(root["execution"], "execution", {"compute_duration_s"})
    execution_raw = _object(execution_raw["compute_duration_s"], "execution.compute_duration_s", None)
    execution: dict[str, float] = {}
    for key, value in execution_raw.items():
        execution[_text(key, "execution compute node id")] = _duration(value, f"execution.compute_duration_s[{key}]")

    jobs: list[DagJob] = []
    job_ids: set[str] = set()
    node_order: dict[str, tuple[str, int, int]] = {}
    compute_ids: set[str] = set()
    task_ids: set[str] = set()
    for ji, item in enumerate(_array(root["jobs"], "DAG.jobs")):
        value = _object(item, f"jobs[{ji}]", {"job_id", "nodes"})
        job_id = _identifier(value["job_id"], f"jobs[{ji}].job_id")
        if job_id in job_ids:
            raise ValueError(f"duplicate job_id {job_id}")
        job_ids.add(job_id)
        node_items = _array(value["nodes"], f"jobs[{ji}].nodes")
        nodes: list[Node] = []
        node_ids: set[str] = set()
        for ni, node_raw in enumerate(node_items):
            kind = node_raw.get("kind") if isinstance(node_raw, dict) else None
            if kind == "compute":
                node_value = _object(node_raw, f"{job_id}.nodes[{ni}]", {"node_id", "kind", "deps", "estimated_duration_s"})
                node_id = _identifier(node_value["node_id"], f"{job_id}.node_id")
                deps = _deps(node_value["deps"], f"{job_id}/{node_id}.deps")
                node = ComputeNode(node_id, deps, _duration(node_value["estimated_duration_s"], f"{job_id}/{node_id}.estimated_duration_s"))
                compute_ids.add(f"{job_id}/{node_id}")
            elif kind == "comm":
                node_value = _object(node_raw, f"{job_id}.nodes[{ni}]", {"node_id", "kind", "deps", "group_id", "group_seq", "estimated_comm_s", "collective"})
                node_id = _identifier(node_value["node_id"], f"{job_id}.node_id")
                deps = _deps(node_value["deps"], f"{job_id}/{node_id}.deps")
                group_id = _text(node_value["group_id"], f"{job_id}/{node_id}.group_id")
                seq = node_value["group_seq"]
                if not _is_int(seq) or seq < 0:
                    raise ValueError(f"{job_id}/{node_id}.group_seq must be a non-negative integer")
                if group_id not in group_ranks:
                    raise ValueError(f"{job_id}/{node_id} references unknown group {group_id}")
                collective = _collective(node_value["collective"], f"{job_id}/{node_id}.collective")
                node = CommNode(node_id, deps, group_id, seq, _duration(node_value["estimated_comm_s"], f"{job_id}/{node_id}.estimated_comm_s"), collective)
                task_id = f"{job_id}/{node_id}"
                if task_id in task_ids:
                    raise ValueError(f"duplicate task_id {task_id}")
                task_ids.add(task_id)
            else:
                raise ValueError(f"{job_id}.nodes[{ni}].kind must be compute or comm")
            if node_id in node_ids:
                raise ValueError(f"duplicate node_id {job_id}/{node_id}")
            node_ids.add(node_id)
            nodes.append(node)
            node_order[f"{job_id}/{node_id}"] = (job_id, ji, ni)
        if not any(isinstance(node, CommNode) for node in nodes):
            raise ValueError(f"job {job_id} must contain at least one comm node")
        for node in nodes:
            missing = set(node.deps) - node_ids
            if missing:
                raise ValueError(f"{job_id}/{node.node_id} has missing deps: {sorted(missing)}")
            if node.node_id in node.deps:
                raise ValueError(f"{job_id}/{node.node_id} depends on itself")
        _check_job_cycle(job_id, nodes)
        used_groups = {node.group_id for node in nodes if isinstance(node, CommNode)}
        if len({group_ranks[group_id] for group_id in used_groups}) != 1:
            raise ValueError(f"job {job_id} references groups with different rank membership")
        jobs.append(DagJob(job_id, tuple(nodes)))

    if not jobs:
        raise ValueError("DAG.jobs must not be empty")
    expected_compute = compute_ids
    if set(execution) != expected_compute:
        raise ValueError(
            "execution.compute_duration_s keys must exactly cover compute nodes; "
            f"missing={sorted(expected_compute - set(execution))}, extra={sorted(set(execution) - expected_compute)}"
        )

    # Build the cross-job graph, adding each group's canonical communication order.
    predecessors: dict[str, set[str]] = {
        qid: {f"{job.job_id}/{dep}" for dep in node.deps}
        for job in jobs for node in job.nodes
        for qid in (f"{job.job_id}/{node.node_id}",)
    }
    group_comms: dict[str, list[tuple[int, str]]] = {}
    for job in jobs:
        for node in job.nodes:
            if isinstance(node, CommNode):
                group_comms.setdefault(node.group_id, []).append((node.group_seq, f"{job.job_id}/{node.node_id}"))
    for group_id, sequence in group_comms.items():
        sequence.sort()
        seqs = [seq for seq, _ in sequence]
        if seqs != list(range(len(seqs))):
            raise ValueError(f"group {group_id} group_seq must be unique and contiguous from 0; got {seqs}")
        for (_, previous), (_, current) in zip(sequence, sequence[1:]):
            predecessors[current].add(previous)
    try:
        tuple(TopologicalSorter(predecessors).static_order())
    except CycleError as exc:
        cycle = exc.args[1] if len(exc.args) > 1 else ()
        raise ValueError(f"combined DAG/group ordering cycle: {cycle}") from exc

    tails = _compute_tails(jobs)
    canonical = _canonical_document(name, seed, groups, jobs, execution)
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical_json.encode()).hexdigest()
    return DagInput(1, name, seed, tuple(groups), tuple(jobs), ReplayExecutionConfig(execution), digest,
                    canonical_json, tails, node_order, group_ranks)


def load_static_order(path: str | Path, dag: DagInput) -> tuple[str, ...]:
    try:
        values = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load static order {path}: {exc}") from exc
    return validate_static_order(values, dag)


def validate_static_order(values: Any, dag: DagInput) -> tuple[str, ...]:
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ValueError("static order must be a JSON array of task IDs")
    expected = set(dag.expected_task_ids)
    if len(values) != len(set(values)) or set(values) != expected:
        raise ValueError(f"static order must cover all communication tasks exactly once; expected={sorted(expected)}")
    positions = {task_id: i for i, task_id in enumerate(values)}
    predecessors = _joint_predecessors(dag)
    by_task_id = {task_id: node for job in dag.jobs for node in job.nodes
                  for task_id in (f"{job.job_id}/{node.node_id}",)}
    for task_id in dag.expected_task_ids:
        pending = list(predecessors[task_id])
        seen: set[str] = set()
        while pending:
            ancestor = pending.pop()
            if ancestor in seen:
                continue
            seen.add(ancestor)
            if isinstance(by_task_id[ancestor], CommNode) and positions[ancestor] >= positions[task_id]:
                raise ValueError(f"static order violates communication dependency: {ancestor} before {task_id}")
            pending.extend(predecessors[ancestor])
    return tuple(values)


def build_static_order(dag: DagInput, policy: str) -> tuple[str, ...]:
    """Generate one deterministic legal topological order for Static FIFO/LTF."""
    if policy not in {"static_fifo", "static_ltf"}:
        raise ValueError("static order generation requires static_fifo or static_ltf")
    predecessors: dict[str, set[str]] = {
        f"{job.job_id}/{node.node_id}": {f"{job.job_id}/{dep}" for dep in node.deps}
        for job in dag.jobs for node in job.nodes
    }
    by_group: dict[str, list[tuple[int, str]]] = {}
    for job in dag.jobs:
        for node in job.nodes:
            if isinstance(node, CommNode):
                by_group.setdefault(node.group_id, []).append((node.group_seq, f"{job.job_id}/{node.node_id}"))
    for seq in by_group.values():
        seq.sort()
        for (_, previous), (_, current) in zip(seq, seq[1:]):
            predecessors[current].add(previous)
    successors: dict[str, set[str]] = {node_id: set() for node_id in predecessors}
    remaining = {node_id: len(deps) for node_id, deps in predecessors.items()}
    for node_id, deps in predecessors.items():
        for dep in deps:
            successors[dep].add(node_id)
    node_by_id = {f"{job.job_id}/{node.node_id}": node for job in dag.jobs for node in job.nodes}
    jobs_by_id = {job.job_id: job for job in dag.jobs}
    ready = {node_id for node_id, count in remaining.items() if count == 0}
    order: list[str] = []
    while ready:
        compute_ready = sorted((qid for qid in ready if isinstance(node_by_id[qid], ComputeNode)), key=lambda qid: dag.node_order[qid])
        if compute_ready:
            current = compute_ready[0]
            ready.remove(current)
        else:
            comm_ready = [qid for qid in ready if isinstance(node_by_id[qid], CommNode)]
            if policy == "static_ltf":
                current = min(comm_ready, key=lambda qid: (-dag.tails[qid], dag.node_order[qid], qid))
            else:
                current = min(comm_ready, key=lambda qid: (dag.node_order[qid], qid))
            ready.remove(current)
            order.append(current)
        for child in successors[current]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.add(child)
    return validate_static_order(order, dag)


def dag_task_spec(job: DagJob, comm: CommNode, *, epoch: int) -> TaskSpec:
    return TaskSpec(epoch, job.job_id, f"{job.job_id}/{comm.node_id}", comm.group_id,
                    comm.group_seq, comm.collective)


def dag_task_hint(dag: DagInput, comm: CommNode, *, job_id: str, ready_after_s: float = 0.0) -> TaskHint:
    return TaskHint(ready_after_s, comm.estimated_comm_s, dag.tails[f"{job_id}/{comm.node_id}"])


def dag_group_specs(dag: DagInput, *, epoch: int) -> tuple[GroupSpec, ...]:
    return tuple(GroupSpec(epoch, group.group_id, group.ranks) for group in dag.groups)


def sample_compute_duration(seed: int, epoch: int, job_id: str, node_id: str, rank: int,
                            base_s: float, jitter: float) -> float:
    if not 0 <= jitter < 1:
        raise ValueError("compute_jitter must be in [0, 1)")
    if jitter == 0 or base_s == 0:
        return base_s
    key = f"{seed}:{epoch}:{job_id}:{node_id}:{rank}".encode()
    fraction = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return base_s * (1 + jitter * (2 * fraction - 1))


class DagRunner:
    """Advance one job; the runtime remains the sole communication admission authority."""

    def __init__(self, dag: DagInput, job: DagJob, runtime: Any, *, epoch: int, rank: int,
                 make_binding: Callable[[CommNode], LocalBinding], timeout: float,
                 poll_interval: float = 0.001, compute_jitter: float = 0.0,
                 enable_lookahead: bool = False,
                 stop_event: threading.Event | None = None,
                 compute_fn: Callable[[ComputeNode, float, threading.Event], None] | None = None,
                 event_log: EventLog | None = None, deadline: float | None = None) -> None:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and dag poll interval must be positive")
        if not 0 <= compute_jitter < 1:
            raise ValueError("compute_jitter must be in [0, 1)")
        self.dag, self.job, self.runtime = dag, job, runtime
        self.epoch, self.rank = epoch, rank
        self.make_binding = make_binding
        self.timeout, self.poll_interval = timeout, poll_interval
        self.compute_jitter = compute_jitter
        self.enable_lookahead = enable_lookahead
        self.stop_event = stop_event or threading.Event()
        self.compute_fn = compute_fn or (lambda _node, duration, stop: stop.wait(duration))
        self.event_log = event_log or EventLog("dag", rank)
        self.deadline = deadline if deadline is not None else time.monotonic() + timeout
        self.states = {node.node_id: NodeState.PENDING for node in job.nodes}
        self.remaining_deps = {node.node_id: len(node.deps) for node in job.nodes}
        self.successors = {node.node_id: [] for node in job.nodes}
        for node in job.nodes:
            for dep in node.deps:
                self.successors[dep].append(node.node_id)
        self.future: concurrent.futures.Future[None] | None = None
        self.compute_node: ComputeNode | None = None
        self.compute_started: dict[str, int] = {}
        self.compute_samples: dict[str, float] = {}
        self.handles: dict[str, Any] = {}
        self.bindings: dict[str, LocalBinding] = {}
        self.declared: set[str] = set()
        self.failed_node: str | None = None
        self._node_key = {node.node_id: i for i, node in enumerate(job.nodes)}

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
                    hint = dag_task_hint(self.dag, comm, job_id=self.job.job_id)
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
                        duration = sample_compute_duration(
                            self.dag.seed, self.epoch, self.job.job_id, node.node_id, self.rank,
                            self.dag.execution.compute_duration_s[f"{self.job.job_id}/{node.node_id}"],
                            self.compute_jitter,
                        )
                        self.compute_node = node
                        self.compute_started[node.node_id] = monotonic_us()
                        self.compute_samples[node.node_id] = duration
                        self.states[node.node_id] = NodeState.RUNNING
                        self.event_log.record("compute_started", job_id=self.job.job_id, node_id=node.node_id,
                                              estimated_duration_s=node.estimated_duration_s,
                                              sampled_duration_s=duration)
                        self.future = pool.submit(self.compute_fn, node, duration, self.stop_event)
                        progressed = True
                if self.enable_lookahead and self._declare_safe_frontier():
                    progressed = True
                if not progressed:
                    remaining = self.deadline - time.monotonic()
                    if remaining <= 0:
                        pending = {node: self.remaining_deps[node] for node, state in self.states.items()
                                   if state is not NodeState.COMPLETED}
                        detail = {"pending": pending, "states": {key: value.value for key, value in self.states.items()},
                                  "running_handles": {node: handle.state.value for node, handle in self.handles.items()}}
                        self.event_log.record("job_timeout", job_id=self.job.job_id, **detail)
                        raise TimeoutError(f"DAG job {self.job.job_id} timed out: {detail}")
                    self.stop_event.wait(min(self.poll_interval, remaining))
            self.event_log.record("job_completed", job_id=self.job.job_id,
                                  duration_s=(time.perf_counter_ns() // 1000 - started) / 1_000_000)
            return {"job_id": self.job.job_id, "status": "ok", "node_count": len(self.job.nodes),
                    "completed_node_ids": list(self.completed_node_ids), "compute_samples_s": dict(self.compute_samples)}
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
        try:
            future.result()
        except BaseException:
            raise
        self.failed_node = None
        self.event_log.record("compute_completed", job_id=self.job.job_id, node_id=node.node_id,
                              sampled_duration_s=self.compute_samples[node.node_id])
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
            deps = [next(dep for dep in self.job.nodes if dep.node_id == dep_id) for dep_id in node.deps]
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
            self.runtime.declare(dag_task_spec(self.job, node, epoch=self.epoch),
                                 dag_task_hint(self.dag, node, job_id=self.job.job_id, ready_after_s=ready_after))
            self.declared.add(node.node_id)
            self.event_log.record("comm_declared", task_id=f"{self.job.job_id}/{node.node_id}",
                                  job_id=self.job.job_id, node_id=node.node_id, ready_after_s=ready_after,
                                  prediction_base_us=prediction_base_us,
                                  predicted_ready_at_us=predicted_ready_at_us,
                                  tail_s=self.dag.tails[f"{self.job.job_id}/{node.node_id}"],
                                  prediction_basis=[dep.node_id for dep in running])
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
                node = next(item for item in self.job.nodes if item.node_id == successor_id)
                self._record_ready(node)

    def _record_ready(self, node: Node) -> None:
        self.event_log.record("node_ready", job_id=self.job.job_id, node_id=node.node_id,
                              node_kind="comm" if isinstance(node, CommNode) else "compute",
                              remaining_deps=self.remaining_deps[node.node_id])


def _compute_tails(jobs: list[DagJob]) -> dict[str, float]:
    tails: dict[str, float] = {}
    for job in jobs:
        nodes = {node.node_id: node for node in job.nodes}
        predecessors: dict[str, set[str]] = {node.node_id: set(node.deps) for node in job.nodes}
        comms: dict[str, list[CommNode]] = {}
        for node in job.nodes:
            if isinstance(node, CommNode):
                comms.setdefault(node.group_id, []).append(node)
        for group_nodes in comms.values():
            group_nodes.sort(key=lambda node: node.group_seq)
            for previous, current in zip(group_nodes, group_nodes[1:]):
                predecessors[current.node_id].add(previous.node_id)
        successors: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for node_id, deps in predecessors.items():
            for dep in deps:
                successors[dep].add(node_id)
        try:
            order = tuple(TopologicalSorter(predecessors).static_order())
        except CycleError as exc:
            raise ValueError(f"job {job.job_id} cycle after group ordering") from exc
        for node_id in reversed(order):
            tails[f"{job.job_id}/{node_id}"] = max(
                (
                    (nodes[child].estimated_duration_s if isinstance(nodes[child], ComputeNode)
                     else nodes[child].estimated_comm_s)
                    + tails[f"{job.job_id}/{child}"]
                    for child in successors[node_id]
                ),
                default=0.0,
            )
    return tails


def _check_job_cycle(job_id: str, nodes: list[Node]) -> None:
    try:
        TopologicalSorter({node.node_id: set(node.deps) for node in nodes}).prepare()
    except CycleError as exc:
        cycle = exc.args[1] if len(exc.args) > 1 else ()
        raise ValueError(f"job {job_id} has dependency cycle: {cycle}") from exc


def _joint_predecessors(dag: DagInput) -> dict[str, set[str]]:
    predecessors = {
        f"{job.job_id}/{node.node_id}": {f"{job.job_id}/{dep}" for dep in node.deps}
        for job in dag.jobs for node in job.nodes
    }
    groups: dict[str, list[tuple[int, str]]] = {}
    for job in dag.jobs:
        for node in job.nodes:
            if isinstance(node, CommNode):
                groups.setdefault(node.group_id, []).append((node.group_seq, f"{job.job_id}/{node.node_id}"))
    for sequence in groups.values():
        sequence.sort()
        for (_, previous), (_, current) in zip(sequence, sequence[1:]):
            predecessors[current].add(previous)
    return predecessors


def _canonical_document(name: str, seed: int, groups: list[GroupSpec], jobs: list[DagJob], execution: Mapping[str, float]) -> dict[str, Any]:
    canonical_jobs = []
    for job in jobs:
        nodes = []
        for node in job.nodes:
            common = {"node_id": node.node_id, "deps": sorted(node.deps)}
            if isinstance(node, ComputeNode):
                nodes.append({**common, "kind": "compute", "estimated_duration_s": node.estimated_duration_s})
            else:
                nodes.append({**common, "kind": "comm", "group_id": node.group_id,
                              "group_seq": node.group_seq, "estimated_comm_s": node.estimated_comm_s,
                              "collective": node.collective.to_dict()})
        canonical_jobs.append({"job_id": job.job_id, "nodes": nodes})
    return {"schema_version": 1, "name": name, "seed": seed,
            "groups": [{"group_id": group.group_id, "ranks": list(group.ranks)} for group in groups],
            "execution": {"compute_duration_s": dict(sorted(execution.items()))}, "jobs": canonical_jobs}


def _collective(value: Any, where: str) -> CollectiveSpec:
    fields = {"op", "numel", "num_bytes", "dtype", "shape", "reduction"}
    data = _object(value, where, fields, required=fields - {"reduction"})
    required = fields - {"reduction"}
    if required - set(data):
        raise ValueError(f"{where} missing fields: {sorted(required - set(data))}")
    if not isinstance(data["op"], str) or not isinstance(data["dtype"], str) or not isinstance(data.get("reduction", "sum"), str):
        raise ValueError(f"{where} op, dtype and reduction must be strings")
    if not _is_int(data["numel"]) or not _is_int(data["num_bytes"]) or any(
        not _is_int(dim) for dim in _array(data["shape"], f"{where}.shape")
    ):
        raise ValueError(f"{where} numel, num_bytes and shape dimensions must be integers")
    try:
        return CollectiveSpec(data["op"], data["numel"], data["num_bytes"], data["dtype"],
                              tuple(data["shape"]), data.get("reduction", "sum"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {where}: {exc}") from exc


def _object(value: Any, where: str, allowed: set[str] | None,
            required: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    if allowed is not None and set(value) - allowed:
        raise ValueError(f"{where} has unknown fields {sorted(set(value) - allowed)}")
    required = allowed if required is None else required
    if required is not None:
        missing = required - set(value)
        if missing:
            raise ValueError(f"{where} missing fields {sorted(missing)}")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _identifier(value: Any, where: str) -> str:
    text = _text(value, where)
    if "/" in text:
        raise ValueError(f"{where} must not contain '/'")
    return text


def _deps(value: Any, where: str) -> tuple[str, ...]:
    deps = _array(value, where)
    if any(not isinstance(dep, str) or not dep for dep in deps):
        raise ValueError(f"{where} must contain non-empty node IDs")
    if len(deps) != len(set(deps)):
        raise ValueError(f"{where} contains duplicate dependencies")
    return tuple(sorted(deps))


def _duration(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{where} must be a finite non-negative number")
    return float(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
