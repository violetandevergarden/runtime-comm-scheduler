"""Validated DAG models and scheduling summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any, Mapping

from runtime_comm_scheduler.runtime import CollectiveSpec, GroupSpec, TaskHint, TaskSpec
from runtime_comm_scheduler.runtime.policy import ltf_score


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


@dataclass(frozen=True)
class DagJob:
    job_id: str
    nodes: tuple[Node, ...]


@dataclass(frozen=True)
class DagGraph:
    groups: tuple[GroupSpec, ...]
    jobs: tuple[DagJob, ...]

    @property
    def expected_task_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{job.job_id}/{node.node_id}"
            for job in self.jobs for node in job.nodes if isinstance(node, CommNode)
        )

    @property
    def expected_node_ids(self) -> tuple[str, ...]:
        return tuple(f"{job.job_id}/{node.node_id}" for job in self.jobs for node in job.nodes)


def validate_graph(graph: DagGraph, *, world_size: int | None = None) -> None:
    """Validate directly-created graphs as strictly as parsed replay inputs."""
    if not isinstance(graph, DagGraph):
        raise TypeError("graph must be a DagGraph")
    if not isinstance(graph.groups, tuple) or not isinstance(graph.jobs, tuple):
        raise TypeError("DAG groups and jobs must be tuples")
    if not graph.jobs:
        raise ValueError("DAG.jobs must not be empty")
    if world_size is not None and (not _is_int(world_size) or world_size <= 0):
        raise ValueError("world_size must be a positive integer")

    groups: dict[str, GroupSpec] = {}
    for group in graph.groups:
        if not isinstance(group, GroupSpec):
            raise TypeError("DAG groups must be GroupSpec values")
        if not isinstance(group.group_id, str) or not group.group_id.strip():
            raise ValueError("group_id must be a non-empty string")
        if not _is_int(group.epoch) or group.epoch < 0:
            raise ValueError(f"group {group.group_id} epoch must be a non-negative integer")
        if group.group_id in groups:
            raise ValueError(f"duplicate group_id {group.group_id}")
        if not group.ranks or any(not _is_int(rank) or rank < 0 for rank in group.ranks):
            raise ValueError(f"group {group.group_id} ranks must be non-empty non-negative integers")
        if len(set(group.ranks)) != len(group.ranks):
            raise ValueError(f"group {group.group_id} ranks contain duplicates")
        if world_size is not None and max(group.ranks) >= world_size:
            raise ValueError(f"group {group.group_id} contains rank {max(group.ranks)} >= world_size {world_size}")
        groups[group.group_id] = group

    job_ids: set[str] = set()
    epochs = {group.epoch for group in graph.groups}
    if len(epochs) > 1:
        raise ValueError("all DAG groups must use the same epoch")
    for job in graph.jobs:
        if not isinstance(job, DagJob):
            raise TypeError("DAG jobs must be DagJob values")
        _identifier(job.job_id, "job_id")
        if job.job_id in job_ids:
            raise ValueError(f"duplicate job_id {job.job_id}")
        job_ids.add(job.job_id)
        if not job.nodes:
            raise ValueError(f"job {job.job_id} must contain at least one comm node")
        if not isinstance(job.nodes, tuple):
            raise TypeError(f"job {job.job_id}.nodes must be a tuple")
        node_ids: set[str] = set()
        for node in job.nodes:
            if not isinstance(node, (ComputeNode, CommNode)):
                raise TypeError(f"job {job.job_id} contains an unsupported node")
            _identifier(node.node_id, f"{job.job_id}.node_id")
            if node.node_id in node_ids:
                raise ValueError(f"duplicate node_id {job.job_id}/{node.node_id}")
            node_ids.add(node.node_id)
            if not isinstance(node.deps, tuple) or any(not isinstance(dep, str) or not dep for dep in node.deps):
                raise ValueError(f"{job.job_id}/{node.node_id}.deps must contain non-empty node IDs")
            if len(node.deps) != len(set(node.deps)):
                raise ValueError(f"{job.job_id}/{node.node_id}.deps contains duplicate dependencies")
            if isinstance(node, ComputeNode):
                _duration(node.estimated_duration_s, f"{job.job_id}/{node.node_id}.estimated_duration_s")
            else:
                _duration(node.estimated_comm_s, f"{job.job_id}/{node.node_id}.estimated_comm_s")
                if not isinstance(node.group_id, str) or not node.group_id.strip():
                    raise ValueError(f"{job.job_id}/{node.node_id}.group_id must be a non-empty string")
                if not _is_int(node.group_seq) or node.group_seq < 0:
                    raise ValueError(f"{job.job_id}/{node.node_id}.group_seq must be a non-negative integer")
                if node.group_id not in groups:
                    raise ValueError(f"{job.job_id}/{node.node_id} references unknown group {node.group_id}")
                if not isinstance(node.collective, CollectiveSpec):
                    raise TypeError(f"{job.job_id}/{node.node_id}.collective must be a CollectiveSpec")
        if not any(isinstance(node, CommNode) for node in job.nodes):
            raise ValueError(f"job {job.job_id} must contain at least one comm node")
        for node in job.nodes:
            missing = set(node.deps) - node_ids
            if missing:
                raise ValueError(f"{job.job_id}/{node.node_id} has missing deps: {sorted(missing)}")
            if node.node_id in node.deps:
                raise ValueError(f"{job.job_id}/{node.node_id} depends on itself")
        try:
            TopologicalSorter({node.node_id: set(node.deps) for node in job.nodes}).prepare()
        except CycleError as exc:
            cycle = exc.args[1] if len(exc.args) > 1 else ()
            raise ValueError(f"job {job.job_id} has dependency cycle: {cycle}") from exc
        used_groups = {node.group_id for node in job.nodes if isinstance(node, CommNode)}
        if len({groups[group_id].ranks for group_id in used_groups}) != 1:
            raise ValueError(f"job {job.job_id} references groups with different rank membership")

    group_sequences: dict[str, list[int]] = {}
    for job in graph.jobs:
        for node in job.nodes:
            if isinstance(node, CommNode):
                group_sequences.setdefault(node.group_id, []).append(node.group_seq)
    for group_id, seqs in group_sequences.items():
        ordered = sorted(seqs)
        if ordered != list(range(len(ordered))):
            raise ValueError(f"group {group_id} group_seq must be unique and contiguous from 0; got {ordered}")

    try:
        tuple(TopologicalSorter(joint_predecessors(graph)).static_order())
    except CycleError as exc:
        cycle = exc.args[1] if len(exc.args) > 1 else ()
        raise ValueError(f"combined DAG/group ordering cycle: {cycle}") from exc


def joint_predecessors(graph: DagGraph) -> dict[str, set[str]]:
    """Return data-dependency and canonical group-order edges for the full graph."""
    return _predecessors_for_jobs(graph.jobs)


def _predecessors_for_jobs(jobs: tuple[DagJob, ...]) -> dict[str, set[str]]:
    predecessors = {
        f"{job.job_id}/{node.node_id}": {f"{job.job_id}/{dep}" for dep in node.deps}
        for job in jobs for node in job.nodes
    }
    groups: dict[str, list[tuple[int, str]]] = {}
    for job in jobs:
        for node in job.nodes:
            if isinstance(node, CommNode):
                groups.setdefault(node.group_id, []).append((node.group_seq, f"{job.job_id}/{node.node_id}"))
    for sequence in groups.values():
        sequence.sort()
        for (_, previous), (_, current) in zip(sequence, sequence[1:]):
            predecessors[current].add(previous)
    return predecessors


def compute_tails(graph: DagGraph) -> dict[str, float]:
    """
    Compute per-node remaining critical paths, excluding the node itself.
    这里的 tail 定义是：从当前节点执行完成之后，到该 job 结束的最长后继路径估计时长。
    tail(v) = max(duration(u) + tail(u) for u in successors(v))
    """
    tails: dict[str, float] = {}
    for job in graph.jobs:
        predecessors = _predecessors_for_jobs((job,))
        successors: dict[str, set[str]] = {node.node_id: set() for node in job.nodes}
        for qualified, deps in predecessors.items():
            node_id = qualified.split("/", 1)[1]
            for dep in deps:
                successors[dep.split("/", 1)[1]].add(node_id)
        order = tuple(TopologicalSorter(predecessors).static_order())
        nodes = {node.node_id: node for node in job.nodes}
        for qualified in reversed(order):
            node_id = qualified.split("/", 1)[1]
            tails[f"{job.job_id}/{node_id}"] = max(
                (
                    _duration_of(nodes[child]) + tails[f"{job.job_id}/{child}"]
                    for child in successors[node_id]
                ),
                default=0.0,
            )
    return tails


def build_layered_fifo_order(
    graph: DagGraph,
    *,
    extra_predecessors: Mapping[str, tuple[str, ...]] | None = None,
    job_order: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Return a fixed, estimate-independent FIFO projection of the DAG.

    The complete node graph, including group-order and caller-supplied edges,
    determines topological layers. Nodes in a layer are then emitted by a
    round-robin over jobs; each job's nodes retain their input order. Only
    communication nodes are returned, and their legality is checked again.
    """
    validate_graph(graph)
    predecessors = joint_predecessors(graph)
    node_by_id = {
        f"{job.job_id}/{node.node_id}": node
        for job in graph.jobs for node in job.nodes
    }
    node_order = {node_id: index for index, node_id in enumerate(node_by_id)}
    for node_id, extra in (extra_predecessors or {}).items():
        if node_id not in predecessors:
            raise ValueError(f"extra predecessor target {node_id!r} is absent from the DAG")
        if len(extra) != len(set(extra)):
            raise ValueError(f"extra predecessors for {node_id!r} contain duplicates")
        for predecessor in extra:
            if predecessor not in predecessors:
                raise ValueError(
                    f"extra predecessor {predecessor!r} for {node_id!r} is absent from the DAG"
                )
            predecessors[node_id].add(predecessor)

    successors: dict[str, list[str]] = {node_id: [] for node_id in predecessors}
    indegree = {node_id: len(deps) for node_id, deps in predecessors.items()}
    for node_id, deps in predecessors.items():
        for dependency in deps:
            successors[dependency].append(node_id)

    layers: list[list[str]] = []
    ready = sorted(
        (node_id for node_id, degree in indegree.items() if degree == 0),
        key=node_order.__getitem__,
    )
    visited = 0
    while ready:
        layer = ready
        layers.append(layer)
        visited += len(layer)
        next_ready: list[str] = []
        for node_id in layer:
            for child in successors[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_ready.append(child)
        ready = sorted(next_ready, key=node_order.__getitem__)
    if visited != len(predecessors):
        blocked = sorted(node_id for node_id, degree in indegree.items() if degree)
        raise ValueError(f"layered FIFO constraints form a cycle: {blocked}")

    resolved_job_order = (tuple(job.job_id for job in graph.jobs)
                          if job_order is None else job_order)
    graph_job_ids = tuple(job.job_id for job in graph.jobs)
    if (len(resolved_job_order) != len(graph_job_ids)
            or set(resolved_job_order) != set(graph_job_ids)):
        raise ValueError("job_order must be a permutation of the DAG job IDs")
    job_for_node = {node_id: node_id.split("/", 1)[0] for node_id in node_by_id}

    ordered_nodes: list[str] = []
    for layer in layers:
        queues: dict[str, list[str]] = {job_id: [] for job_id in resolved_job_order}
        for node_id in sorted(layer, key=node_order.__getitem__):
            queues[job_for_node[node_id]].append(node_id)
        offsets = {job_id: 0 for job_id in resolved_job_order}
        while any(offsets[job_id] < len(queues[job_id]) for job_id in resolved_job_order):
            for job_id in resolved_job_order:
                offset = offsets[job_id]
                if offset < len(queues[job_id]):
                    ordered_nodes.append(queues[job_id][offset])
                    offsets[job_id] += 1

    sequence = tuple(
        node_id for node_id in ordered_nodes if isinstance(node_by_id[node_id], CommNode)
    )
    position = {node_id: index for index, node_id in enumerate(ordered_nodes)}
    if any(position[predecessor] >= position[node_id]
           for node_id, dependencies in predecessors.items()
           for predecessor in dependencies):
        raise ValueError("layered FIFO projection violates a full-DAG predecessor constraint")
    return _validate_static_order(sequence, graph)


def build_static_order(graph: DagGraph, policy: str, *,
                       tails: Mapping[str, float] | None = None,
                       extra_predecessors: Mapping[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    """Generate one deterministic legal topological order for Static FIFO/LTF."""
    if policy not in {"static_fifo", "static_ltf"}:
        raise ValueError("static order generation requires static_fifo or static_ltf")
    if policy == "static_fifo":
        return build_layered_fifo_order(graph, extra_predecessors=extra_predecessors)
    validate_graph(graph)
    predecessors = joint_predecessors(graph)
    for node_id, deps in (extra_predecessors or {}).items():
        if node_id not in predecessors or any(dep not in predecessors for dep in deps):
            raise ValueError("static order has unknown acceptance constraint nodes")
        predecessors[node_id].update(deps)
    successors: dict[str, set[str]] = {node_id: set() for node_id in predecessors}
    remaining = {node_id: len(deps) for node_id, deps in predecessors.items()}
    for node_id, deps in predecessors.items():
        for dep in deps:
            successors[dep].add(node_id)
    node_by_id = {
        f"{job.job_id}/{node.node_id}": node
        for job in graph.jobs for node in job.nodes
    }
    node_order = {
        f"{job.job_id}/{node.node_id}": (job_index, node_index)
        for job_index, job in enumerate(graph.jobs)
        for node_index, node in enumerate(job.nodes)
    }
    tails = (tails if tails is not None else compute_tails(graph)) if policy == "static_ltf" else {}
    ready = {node_id for node_id, count in remaining.items() if count == 0}
    order: list[str] = []
    while ready:
        compute_ready = sorted(
            (qid for qid in ready if isinstance(node_by_id[qid], ComputeNode)),
            key=node_order.__getitem__,
        )
        if compute_ready:
            current = compute_ready[0]
            ready.remove(current)
        else:
            comm_ready = (qid for qid in ready if isinstance(node_by_id[qid], CommNode))
            if policy == "static_ltf":
                current = min(
                    comm_ready,
                    key=lambda qid: (
                        -ltf_score(node_by_id[qid].estimated_comm_s, tails[qid]),
                        node_order[qid], qid,
                    ),
                )
            else:
                current = min(comm_ready, key=lambda qid: (node_order[qid], qid))
            ready.remove(current)
            order.append(current)
        for child in successors[current]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.add(child)
    return _validate_static_order(tuple(order), graph)


def validate_static_order(values: Any, graph: DagGraph, *,
                          extra_predecessors: Mapping[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    validate_graph(graph)
    return _validate_static_order(values, graph, extra_predecessors=extra_predecessors)


def _validate_static_order(values: Any, graph: DagGraph, *,
                           extra_predecessors: Mapping[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or any(not isinstance(item, str) for item in values):
        raise ValueError("static order must be an array of task IDs")
    expected = set(graph.expected_task_ids)
    if len(values) != len(set(values)) or set(values) != expected:
        raise ValueError(f"static order must cover all communication tasks exactly once; expected={sorted(expected)}")
    positions = {task_id: index for index, task_id in enumerate(values)}
    predecessors = joint_predecessors(graph)
    for node_id, deps in (extra_predecessors or {}).items():
        if node_id not in predecessors or any(dep not in predecessors for dep in deps):
            raise ValueError("static order has unknown acceptance constraint nodes")
        predecessors[node_id].update(deps)
    node_by_id = {
        f"{job.job_id}/{node.node_id}": node
        for job in graph.jobs for node in job.nodes
    }
    for task_id in graph.expected_task_ids:
        pending = list(predecessors[task_id])
        seen: set[str] = set()
        while pending:
            ancestor = pending.pop()
            if ancestor in seen:
                continue
            seen.add(ancestor)
            if isinstance(node_by_id[ancestor], CommNode) and positions[ancestor] >= positions[task_id]:
                raise ValueError(f"static order violates communication dependency: {ancestor} before {task_id}")
            pending.extend(predecessors[ancestor])
    return tuple(values)


def dag_task_spec(job: DagJob, comm: CommNode, *, epoch: int) -> TaskSpec:
    return TaskSpec(epoch, job.job_id, f"{job.job_id}/{comm.node_id}", comm.group_id,
                    comm.group_seq, comm.collective)


def dag_task_hint(comm: CommNode, *, tail_s: float,
                  ready_after_s: float | None = 0.0) -> TaskHint:
    return TaskHint(ready_after_s, comm.estimated_comm_s, tail_s)


def _duration_of(node: Node) -> float:
    return node.estimated_duration_s if isinstance(node, ComputeNode) else node.estimated_comm_s


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or "/" in value:
        raise ValueError(f"{where} must be a non-empty identifier without '/'")
    return value


def _duration(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{where} must be a finite non-negative number")
    return float(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
