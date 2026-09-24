"""Translate replay inputs and local execution to the Phase 3 runtime."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from runtime_comm_scheduler.dag import (
    CommNode,
    ComputeNode,
    DagGraph,
    DagJob,
    validate_graph,
    validate_static_order,
)
from runtime_comm_scheduler.runtime import CollectiveSpec, GroupSpec, LocalBinding, TaskHint, TaskSpec
from examples.jobpacer.workloads import CollectiveComm, Job, Workload


@dataclass(frozen=True)
class ReplayExecutionConfig:
    compute_duration_s: Mapping[str, float]
    # Optional bridge to the linear producer sampling key.
    linear_sample_keys: Mapping[str, tuple[int, str]] | None = None


@dataclass(frozen=True)
class DagInput:
    graph: DagGraph
    name: str
    seed: int
    execution: ReplayExecutionConfig
    manifest_digest: str
    canonical_json: str

    @property
    def expected_task_ids(self) -> tuple[str, ...]:
        return self.graph.expected_task_ids

    @property
    def expected_node_ids(self) -> tuple[str, ...]:
        return self.graph.expected_node_ids

    @property
    def group_ranks(self) -> dict[str, tuple[int, ...]]:
        return {group.group_id: group.ranks for group in self.graph.groups}

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

    groups = []
    for index, item in enumerate(_array(root["groups"], "DAG.groups")):
        value = _object(item, f"groups[{index}]", {"group_id", "ranks"})
        group_id = _text(value["group_id"], f"groups[{index}].group_id")
        ranks_raw = _array(value["ranks"], f"groups[{index}].ranks")
        if not ranks_raw or any(not _is_int(rank) or rank < 0 for rank in ranks_raw):
            raise ValueError(f"groups[{index}].ranks must be non-empty non-negative integers")
        if len(set(ranks_raw)) != len(ranks_raw):
            raise ValueError(f"groups[{index}].ranks contains duplicates")
        groups.append(GroupSpec(epoch, group_id, tuple(ranks_raw)))
    groups.sort(key=lambda group: group.group_id)

    execution_raw = _object(root["execution"], "execution", {"compute_duration_s", "linear_sample_keys"},
                            required={"compute_duration_s"})
    sample_keys_raw = _object(execution_raw.get("linear_sample_keys", {}),
                              "execution.linear_sample_keys", None)
    sample_keys: dict[str, tuple[int, str]] = {}
    for key, value in sample_keys_raw.items():
        if (not isinstance(value, str) or not value.startswith("comm-")
                or ":" not in value):
            raise ValueError(f"invalid linear sample key for {key!r}: {value!r}")
        ordinal_text, segment = value.split(":", 1)
        if not ordinal_text[5:].isdigit() or segment not in {"producer", "consumer"}:
            raise ValueError(f"invalid linear sample key for {key!r}: {value!r}")
        sample_keys[key] = (int(ordinal_text[5:]), segment)
    execution_raw = _object(execution_raw["compute_duration_s"], "execution.compute_duration_s", None)
    execution: dict[str, float] = {}
    for key, value in execution_raw.items():
        execution[_text(key, "execution compute node id")] = _duration(
            value, f"execution.compute_duration_s[{key}]"
        )

    jobs = []
    for job_index, item in enumerate(_array(root["jobs"], "DAG.jobs")):
        value = _object(item, f"jobs[{job_index}]", {"job_id", "nodes"})
        job_id = _identifier(value["job_id"], f"jobs[{job_index}].job_id")
        nodes = []
        for node_index, node_raw in enumerate(_array(value["nodes"], f"{job_id}.nodes")):
            kind = node_raw.get("kind") if isinstance(node_raw, dict) else None
            if kind == "compute":
                node_value = _object(node_raw, f"{job_id}.nodes[{node_index}]",
                                     {"node_id", "kind", "deps", "estimated_duration_s"})
                node_id = _identifier(node_value["node_id"], f"{job_id}.node_id")
                node = ComputeNode(
                    node_id,
                    _deps(node_value["deps"], f"{job_id}/{node_id}.deps"),
                    _duration(node_value["estimated_duration_s"], f"{job_id}/{node_id}.estimated_duration_s"),
                )
            elif kind == "comm":
                node_value = _object(node_raw, f"{job_id}.nodes[{node_index}]",
                                     {"node_id", "kind", "deps", "group_id", "group_seq",
                                      "estimated_comm_s", "collective"})
                node_id = _identifier(node_value["node_id"], f"{job_id}.node_id")
                seq = node_value["group_seq"]
                if not _is_int(seq) or seq < 0:
                    raise ValueError(f"{job_id}/{node_id}.group_seq must be a non-negative integer")
                node = CommNode(
                    node_id,
                    _deps(node_value["deps"], f"{job_id}/{node_id}.deps"),
                    _text(node_value["group_id"], f"{job_id}/{node_id}.group_id"),
                    seq,
                    _duration(node_value["estimated_comm_s"], f"{job_id}/{node_id}.estimated_comm_s"),
                    _collective(node_value["collective"], f"{job_id}/{node_id}.collective"),
                )
            else:
                raise ValueError(f"{job_id}.nodes[{node_index}].kind must be compute or comm")
            nodes.append(node)
        jobs.append(DagJob(job_id, tuple(nodes)))

    graph = DagGraph(tuple(groups), tuple(jobs))
    validate_graph(graph, world_size=world_size)
    expected_compute = {
        f"{job.job_id}/{node.node_id}"
        for job in graph.jobs for node in job.nodes if isinstance(node, ComputeNode)
    }
    if set(execution) != expected_compute:
        raise ValueError(
            "execution.compute_duration_s keys must exactly cover compute nodes; "
            f"missing={sorted(expected_compute - set(execution))}, extra={sorted(set(execution) - expected_compute)}"
        )
    if set(sample_keys) - expected_compute:
        raise ValueError("linear_sample_keys contains unknown compute nodes")

    canonical = _canonical_document(name, seed, graph, execution, sample_keys)
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical_json.encode()).hexdigest()
    return DagInput(graph, name, seed, ReplayExecutionConfig(execution, sample_keys), digest, canonical_json)


def apply_dag_profile(dag: DagInput, profile, environment: Mapping[str, Any], *, strict: bool = True) -> DagInput:
    """Apply a communication profile to every DAG comm node by strict signature."""
    import warnings
    from examples.jobpacer.comm_profile import CommSignature

    mismatches = [
        f"{key}: profile={profile.environment.get(key)!r}, replay={environment.get(key)!r}"
        for key in ("backend", "world_size", "device_type")
        if profile.environment.get(key) != environment.get(key)
    ]
    if mismatches:
        raise ValueError("profile environment mismatch: " + "; ".join(mismatches))
    expected_groups: list[list[int]] = []
    for group in dag.graph.groups:
        ranks = list(group.ranks)
        if ranks not in expected_groups:
            expected_groups.append(ranks)
    if profile.environment.get("group_ranks") != expected_groups:
        raise ValueError(
            "profile environment mismatch: group_ranks: "
            f"profile={profile.environment.get('group_ranks')!r}, replay={expected_groups!r}"
        )
    records = {record.signature: record for record in profile.records}
    groups = {group.group_id: group for group in dag.graph.groups}
    missing: list[str] = []
    jobs = []
    for job in dag.graph.jobs:
        nodes = []
        for node in job.nodes:
            if not isinstance(node, CommNode):
                nodes.append(node)
                continue
            signature = CommSignature(
                node.collective.op, node.collective.num_bytes, node.collective.dtype,
                len(groups[node.group_id].ranks), str(environment["backend"]),
                str(environment["device_type"]), node.collective.reduction,
            )
            record = records.get(signature)
            if record is None:
                missing.append(f"{job.job_id}/{node.node_id} {signature}")
                nodes.append(node)
            else:
                nodes.append(replace(node, estimated_comm_s=record.p50_s))
        jobs.append(replace(job, nodes=tuple(nodes)))
    if missing and strict:
        raise ValueError("profile is missing DAG communication signatures: " + ", ".join(missing))
    if missing:
        warnings.warn("profile fallback to DAG manifest for: " + ", ".join(missing), stacklevel=2)
    graph = DagGraph(dag.graph.groups, tuple(jobs))
    execution = dict(dag.execution.compute_duration_s)
    canonical = _canonical_document(dag.name, dag.seed, graph, execution,
                                    dag.execution.linear_sample_keys or {})
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical_json.encode()).hexdigest()
    return DagInput(graph, dag.name, dag.seed, dag.execution, digest, canonical_json)


def load_static_order(path: str | Path, graph: DagGraph) -> tuple[str, ...]:
    try:
        values = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load static order {path}: {exc}") from exc
    return validate_static_order(values, graph)


def sample_compute_duration(seed: int, epoch: int, job_id: str, node_id: str, rank: int,
                            base_s: float, jitter: float) -> float:
    if not 0 <= jitter < 1:
        raise ValueError("compute_jitter must be in [0, 1)")
    if jitter == 0 or base_s == 0:
        return base_s
    key = f"{seed}:{epoch}:{job_id}:{node_id}:{rank}".encode()
    fraction = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return base_s * (1 + jitter * (2 * fraction - 1))


def make_replay_compute(job: DagJob, execution: ReplayExecutionConfig, *, seed: int, epoch: int,
                        rank: int, jitter: float, samples: MutableMapping[str, float], event_log):
    """Bind deterministic CPU sleep sampling to a DAG job, outside the runner."""
    def compute(node: ComputeNode, stop_event) -> None:
        node_key = f"{job.job_id}/{node.node_id}"
        base_s = execution.compute_duration_s[node_key]
        linear_key = (execution.linear_sample_keys or {}).get(node_key)
        if linear_key is None:
            duration = sample_compute_duration(seed, epoch, job.job_id, node.node_id, rank, base_s, jitter)
        else:
            from examples.jobpacer.workloads import sample_linear_duration
            duration = sample_linear_duration(seed, epoch, job.job_id, linear_key[0], rank,
                                              linear_key[1], base_s, jitter)
        samples[node.node_id] = duration
        event_log.record("compute_sampled", job_id=job.job_id, node_id=node.node_id,
                         base_duration_s=base_s, jitter=jitter, sampled_duration_s=duration)
        stop_event.wait(duration)

    return compute


def make_collective_binding(spec: TaskSpec, process_group: Any, *, rank: int, device: str) -> LocalBinding:
    """Build the rank-local tensor and async all-reduce binding for supported inputs."""
    if spec.collective.reduction != "sum":
        raise ValueError(f"reduction {spec.collective.reduction!r} is not supported by the replay binding")
    import torch
    import torch.distributed as dist

    try:
        dtype = getattr(torch, spec.collective.dtype)
    except AttributeError as exc:
        raise ValueError(f"unsupported tensor dtype {spec.collective.dtype!r}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported tensor dtype {spec.collective.dtype!r}")
    tensor = torch.full(spec.collective.shape, float(rank + 1), dtype=dtype, device=device)

    def launch():
        return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group, async_op=True)

    return LocalBinding(tensor, process_group, launch, device=device, keepalive=(tensor,))


def linear_static_order(workload: Workload, policy: str) -> tuple[str, ...]:
    """Bridge the historical linear Plan builder to runtime task IDs only."""
    policy_map = {"static_fifo": "fifo", "static_ltf": "ltf"}
    try:
        builder_policy = policy_map[policy]
    except KeyError as exc:
        raise ValueError("linear static order requires static_fifo or static_ltf") from exc
    from examples.jobpacer.runtime.plan_builder import build_plan
    plan = build_plan(workload, builder_policy)
    return tuple(f"{key.process_group_id}/comm-{key.ordinal}" for key in plan.keys)


def group_spec(job: Job, world_size: int, *, epoch: int = 0) -> GroupSpec:
    ranks = tuple(range(world_size)) if job.ranks is None else tuple(job.ranks)
    return GroupSpec(epoch, job.job_id, ranks)


def task_spec(job: Job, comm: CollectiveComm, *, epoch: int = 0) -> TaskSpec:
    numel = comm.num_bytes // 4
    return TaskSpec(
        epoch=epoch,
        job_id=job.job_id,
        task_id=f"{job.job_id}/comm-{comm.id}",
        group_id=job.job_id,
        group_seq=comm.id,
        collective=CollectiveSpec("all_reduce", numel, comm.num_bytes, "float32", (numel,)),
    )


def remaining_tail(job: Job, index: int) -> float:
    current = job.communications[index]
    return current.consumer_compute_s + sum(
        item.producer_compute_s + item.estimated_comm_s + item.consumer_compute_s
        for item in job.communications[index + 1 :]
    )


def task_hint(job: Job, index: int) -> TaskHint:
    comm = job.communications[index]
    return TaskHint(comm.producer_compute_s, comm.estimated_comm_s, remaining_tail(job, index))


def all_specs(workload: Workload, *, epoch: int = 0) -> tuple[tuple[Job, int, TaskSpec, TaskHint], ...]:
    return tuple(
        (job, index, task_spec(job, comm, epoch=epoch), task_hint(job, index))
        for job in workload.jobs
        for index, comm in enumerate(job.communications)
    )


def _canonical_document(name: str, seed: int, graph: DagGraph,
                        execution: Mapping[str, float],
                        sample_keys: Mapping[str, tuple[int, str]]) -> dict[str, Any]:
    canonical_jobs = []
    for job in graph.jobs:
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
    execution_document: dict[str, Any] = {"compute_duration_s": dict(sorted(execution.items()))}
    if sample_keys:
        execution_document["linear_sample_keys"] = {
            key: f"comm-{ordinal}:{segment}"
            for key, (ordinal, segment) in sorted(sample_keys.items())
        }
    return {"schema_version": 1, "name": name, "seed": seed,
            "groups": [{"group_id": group.group_id, "ranks": list(group.ranks)} for group in graph.groups],
            "execution": execution_document, "jobs": canonical_jobs}


def _collective(value: Any, where: str) -> CollectiveSpec:
    fields = {"op", "numel", "num_bytes", "dtype", "shape", "reduction"}
    data = _object(value, where, fields, required=fields - {"reduction"})
    required = fields - {"reduction"}
    if required - set(data):
        raise ValueError(f"{where} missing fields: {sorted(required - set(data))}")
    if not isinstance(data["op"], str) or not isinstance(data["dtype"], str) or not isinstance(data.get("reduction", "sum"), str):
        raise ValueError(f"{where} op, dtype and reduction must be strings")
    reduction = data.get("reduction", "sum")
    if reduction != "sum":
        raise ValueError(f"{where}.reduction {reduction!r} is unsupported; replay only supports 'sum'")
    if not _is_int(data["numel"]) or not _is_int(data["num_bytes"]) or any(
        not _is_int(dim) for dim in _array(data["shape"], f"{where}.shape")
    ):
        raise ValueError(f"{where} numel, num_bytes and shape dimensions must be integers")
    try:
        return CollectiveSpec(data["op"], data["numel"], data["num_bytes"], data["dtype"],
                              tuple(data["shape"]), reduction)
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
