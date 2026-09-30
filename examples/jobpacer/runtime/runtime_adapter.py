"""Translate replay inputs and local execution to the Phase 3 runtime."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
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
    compute_duration_s: Mapping[str, float] = field(default_factory=dict)
    # Optional bridge to the linear producer sampling key.
    linear_sample_keys: Mapping[str, tuple[int, str]] | None = None
    mode: str = "host-sleep"
    sample_id: str | None = None
    compute_model: str = "one-active-compute-per-job"
    buffers: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(default_factory=dict)
    compute_programs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    comm_bindings: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    submit_after: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    profiles: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1


@dataclass(frozen=True)
class DagInput:
    graph: DagGraph
    name: str
    seed: int
    execution: ReplayExecutionConfig
    manifest_digest: str
    canonical_json: str
    input_hash: str | None = None
    estimate_view_hash: str | None = None

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
    if not _is_int(version) or version not in {1, 2}:
        raise ValueError("DAG.schema_version must be integer 1 or 2")
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

    if version == 2:
        return _parse_dag_v2(root, name=name, seed=seed, epoch=epoch, world_size=world_size,
                             groups=groups)

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
    execution_config = ReplayExecutionConfig(execution, sample_keys)
    return DagInput(graph, name, seed, execution_config, digest, canonical_json,
                    input_hash=digest, estimate_view_hash=_estimate_view_hash(graph))


def _parse_dag_v2(root: Mapping[str, Any], *, name: str, seed: int, epoch: int,
                  world_size: int | None, groups: list[GroupSpec]) -> DagInput:
    execution_raw = _object(
        root["execution"], "execution",
        {"mode", "sample_id", "compute_model", "buffers", "compute_programs",
         "comm_bindings", "submit_after", "profiles"},
        required={"mode", "sample_id", "compute_model", "buffers", "compute_programs", "comm_bindings"},
    )
    mode = _text(execution_raw["mode"], "execution.mode")
    if mode != "cuda-program":
        raise ValueError("execution.mode must be 'cuda-program' for DAG schema v2")
    sample_id = _text(execution_raw["sample_id"], "execution.sample_id")
    compute_model = _text(execution_raw["compute_model"], "execution.compute_model")
    if compute_model != "one-active-compute-per-job":
        raise ValueError("execution.compute_model must be 'one-active-compute-per-job'")

    jobs = []
    for job_index, item in enumerate(_array(root["jobs"], "DAG.jobs")):
        value = _object(item, f"jobs[{job_index}]", {"job_id", "nodes"})
        job_id = _identifier(value["job_id"], f"jobs[{job_index}].job_id")
        nodes = []
        for node_index, node_raw in enumerate(_array(value["nodes"], f"{job_id}.nodes")):
            kind = node_raw.get("kind") if isinstance(node_raw, dict) else None
            where = f"{job_id}.nodes[{node_index}]"
            if kind == "compute":
                node_value = _object(node_raw, where,
                                     {"node_id", "kind", "deps", "estimated_duration_s"})
                node_id = _identifier(node_value["node_id"], f"{where}.node_id")
                node = ComputeNode(node_id, _deps(node_value["deps"], f"{job_id}/{node_id}.deps"),
                                   _duration(node_value["estimated_duration_s"],
                                             f"{job_id}/{node_id}.estimated_duration_s"))
            elif kind == "comm":
                node_value = _object(node_raw, where,
                                     {"node_id", "kind", "deps", "group_id", "group_seq",
                                      "estimated_comm_s", "collective"})
                node_id = _identifier(node_value["node_id"], f"{where}.node_id")
                seq = node_value["group_seq"]
                if not _is_int(seq) or seq < 0:
                    raise ValueError(f"{job_id}/{node_id}.group_seq must be a non-negative integer")
                node = CommNode(node_id, _deps(node_value["deps"], f"{job_id}/{node_id}.deps"),
                                _text(node_value["group_id"], f"{job_id}/{node_id}.group_id"), seq,
                                _duration(node_value["estimated_comm_s"],
                                          f"{job_id}/{node_id}.estimated_comm_s"),
                                _collective(node_value["collective"], f"{job_id}/{node_id}.collective"))
            else:
                raise ValueError(f"{where}.kind must be compute or comm")
            nodes.append(node)
        jobs.append(DagJob(job_id, tuple(nodes)))
    graph = DagGraph(tuple(groups), tuple(jobs))
    validate_graph(graph, world_size=world_size)

    job_ids = {job.job_id for job in graph.jobs}
    buffers_raw = _object(execution_raw["buffers"], "execution.buffers", None)
    if set(buffers_raw) != job_ids:
        raise ValueError("execution.buffers keys must exactly cover jobs; "
                         f"missing={sorted(job_ids - set(buffers_raw))}, extra={sorted(set(buffers_raw) - job_ids)}")
    buffers: dict[str, dict[str, dict[str, Any]]] = {}
    for job in graph.jobs:
        raw_job_buffers = _object(buffers_raw[job.job_id], f"execution.buffers.{job.job_id}", None)
        if not raw_job_buffers:
            raise ValueError(f"execution.buffers.{job.job_id} must not be empty")
        buffers[job.job_id] = {}
        for buffer_id, buffer_raw in raw_job_buffers.items():
            _identifier(buffer_id, f"execution.buffers.{job.job_id} key")
            spec = _object(buffer_raw, f"buffer {job.job_id}/{buffer_id}",
                           {"shape", "dtype", "init"})
            shape = _shape(spec["shape"], f"buffer {job.job_id}/{buffer_id}.shape", allow_scalar=True)
            dtype = _text(spec["dtype"], f"buffer {job.job_id}/{buffer_id}.dtype")
            if dtype != "float32":
                raise ValueError(f"buffer {job.job_id}/{buffer_id}.dtype only supports float32")
            init = _text(spec["init"], f"buffer {job.job_id}/{buffer_id}.init")
            if init not in {"seeded-random", "zeros", "empty"}:
                raise ValueError(f"buffer {job.job_id}/{buffer_id}.init is unsupported: {init!r}")
            buffers[job.job_id][buffer_id] = {"shape": list(shape), "dtype": dtype, "init": init}

    compute_nodes = {f"{job.job_id}/{node.node_id}": (job, node)
                     for job in graph.jobs for node in job.nodes if isinstance(node, ComputeNode)}
    comm_nodes = {f"{job.job_id}/{node.node_id}": (job, node)
                  for job in graph.jobs for node in job.nodes if isinstance(node, CommNode)}
    compute_programs = _parse_compute_programs(execution_raw["compute_programs"], compute_nodes, buffers)
    comm_bindings = _parse_comm_bindings(execution_raw["comm_bindings"], comm_nodes, buffers)
    _validate_buffer_hazards(graph, buffers, compute_programs, comm_bindings)
    submit_after = _parse_submit_after(execution_raw.get("submit_after", {}), graph)
    profiles_raw = execution_raw.get("profiles", {})
    profiles = _object(profiles_raw, "execution.profiles", None)

    config = ReplayExecutionConfig(
        mode=mode, sample_id=sample_id, compute_model=compute_model,
        buffers=buffers, compute_programs=compute_programs, comm_bindings=comm_bindings,
        submit_after=submit_after, profiles=profiles, schema_version=2,
    )
    canonical = _canonical_v2_document(name, seed, graph, config)
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical_json.encode()).hexdigest()
    return DagInput(graph, name, seed, config, digest, canonical_json,
                    input_hash=digest, estimate_view_hash=_estimate_view_hash(graph))


def _parse_compute_programs(raw: Any, compute_nodes: Mapping[str, tuple[DagJob, ComputeNode]],
                            buffers: Mapping[str, Mapping[str, Mapping[str, Any]]]
                            ) -> dict[str, dict[str, Any]]:
    values = _object(raw, "execution.compute_programs", None)
    if set(values) != set(compute_nodes):
        raise ValueError("execution.compute_programs keys must exactly cover compute nodes; "
                         f"missing={sorted(set(compute_nodes) - set(values))}, "
                         f"extra={sorted(set(values) - set(compute_nodes))}")
    parsed: dict[str, dict[str, Any]] = {}
    for key, raw_program in values.items():
        job, _node = compute_nodes[key]
        program = _object(raw_program, f"compute program {key}", None)
        op = _text(program.get("op"), f"compute program {key}.op")
        if op == "matmul":
            _object(program, f"compute program {key}", {"op", "inputs", "output", "repeats"})
            inputs = _buffer_ids(program["inputs"], f"compute program {key}.inputs", count=2)
            output = _buffer_id(program["output"], f"compute program {key}.output")
            repeats = program["repeats"]
            if not _is_int(repeats) or repeats <= 0:
                raise ValueError(f"compute program {key}.repeats must be a positive integer")
            referenced = set(inputs) | {output}
            missing = referenced - set(buffers[job.job_id])
            if missing:
                raise ValueError(f"compute program {key} references unknown buffers: {sorted(missing)}")
            left, right, target = (buffers[job.job_id][item] for item in (*inputs, output))
            if any(tuple(spec["shape"]) == () or len(spec["shape"]) != 2
                   for spec in (left, right, target)):
                raise ValueError(f"compute program {key} matmul requires rank-2 buffers")
            m, k = left["shape"]
            right_k, n = right["shape"]
            if k != right_k or target["shape"] != [m, n]:
                raise ValueError(f"compute program {key} matmul buffer shapes are incompatible")
            parsed[key] = {"op": op, "inputs": list(inputs), "output": output, "repeats": repeats}
        elif op == "fill":
            _object(program, f"compute program {key}", {"op", "output"})
            output = _buffer_id(program["output"], f"compute program {key}.output")
            parsed[key] = {"op": op, "inputs": [], "output": output}
        elif op == "sum_join":
            _object(program, f"compute program {key}", {"op", "inputs", "output"})
            inputs = _buffer_ids(program["inputs"], f"compute program {key}.inputs", minimum=1)
            output = _buffer_id(program["output"], f"compute program {key}.output")
            missing = (set(inputs) | {output}) - set(buffers[job.job_id])
            if missing:
                raise ValueError(f"compute program {key} references unknown buffers: {sorted(missing)}")
            if buffers[job.job_id][output]["shape"] != []:
                raise ValueError(f"compute program {key}.output must be a scalar buffer")
            parsed[key] = {"op": op, "inputs": list(inputs), "output": output}
        else:
            raise ValueError(f"compute program {key}.op must be fill, matmul, or sum_join")
        referenced = set(parsed[key]["inputs"]) | {parsed[key]["output"]}
        missing = referenced - set(buffers[job.job_id])
        if missing:
            raise ValueError(f"compute program {key} references unknown buffers: {sorted(missing)}")
        if parsed[key]["output"] in parsed[key]["inputs"]:
            raise ValueError(
                f"compute program {key} inputs and output must use distinct buffers"
            )
    return parsed


def _parse_comm_bindings(raw: Any, comm_nodes: Mapping[str, tuple[DagJob, CommNode]],
                         buffers: Mapping[str, Mapping[str, Mapping[str, Any]]]
                         ) -> dict[str, dict[str, str]]:
    values = _object(raw, "execution.comm_bindings", None)
    if set(values) != set(comm_nodes):
        raise ValueError("execution.comm_bindings keys must exactly cover comm nodes; "
                         f"missing={sorted(set(comm_nodes) - set(values))}, "
                         f"extra={sorted(set(values) - set(comm_nodes))}")
    parsed: dict[str, dict[str, str]] = {}
    for key, raw_binding in values.items():
        job, node = comm_nodes[key]
        binding = _object(raw_binding, f"comm binding {key}", {"buffer"})
        buffer_id = _buffer_id(binding["buffer"], f"comm binding {key}.buffer")
        if buffer_id not in buffers[job.job_id]:
            raise ValueError(f"comm binding {key} references unknown buffer {buffer_id!r}")
        buffer = buffers[job.job_id][buffer_id]
        shape = tuple(buffer["shape"])
        if (shape != node.collective.shape or buffer["dtype"] != node.collective.dtype
                or _product(shape) != node.collective.numel
                or _product(shape) * 4 != node.collective.num_bytes):
            raise ValueError(f"comm binding {key} buffer {buffer_id!r} does not match collective shape/dtype/size")
        parsed[key] = {"buffer": buffer_id}
    return parsed


def _validate_buffer_hazards(graph: DagGraph,
                             buffers: Mapping[str, Mapping[str, Mapping[str, Any]]],
                             programs: Mapping[str, Mapping[str, Any]],
                             comm_bindings: Mapping[str, Mapping[str, str]]) -> None:
    accessed: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    node_lookup = {f"{job.job_id}/{node.node_id}": (job, node)
                   for job in graph.jobs for node in job.nodes}
    predecessors: dict[str, set[str]] = {}
    for job in graph.jobs:
        for node in job.nodes:
            key = f"{job.job_id}/{node.node_id}"
            predecessors[key] = {f"{job.job_id}/{dep}" for dep in node.deps}
    ancestors: dict[str, set[str]] = {}
    for key in node_lookup:
        found: set[str] = set()
        pending = list(predecessors[key])
        while pending:
            parent = pending.pop()
            if parent in found:
                continue
            found.add(parent)
            pending.extend(predecessors[parent])
        ancestors[key] = found

    for key, program in programs.items():
        job_id = key.split("/", 1)[0]
        for buffer_id in program["inputs"]:
            accessed.setdefault((job_id, buffer_id), []).append((key, "read", program["op"]))
        accessed.setdefault((job_id, program["output"]), []).append((key, "write", program["op"]))
    for key, binding in comm_bindings.items():
        job_id = key.split("/", 1)[0]
        buffer_id = binding["buffer"]
        accessed.setdefault((job_id, buffer_id), []).extend(((key, "read", "all_reduce"),
                                                              (key, "write", "all_reduce")))

    for job_id, job_buffers in buffers.items():
        for buffer_id, spec in job_buffers.items():
            operations = accessed.get((job_id, buffer_id), [])
            if not operations:
                raise ValueError(f"buffer {job_id}/{buffer_id} is unused")
            writers = sorted({key for key, access, _op in operations if access == "write"})
            readers = sorted({key for key, access, _op in operations if access == "read"})
            for index, first in enumerate(writers):
                for second in writers[index + 1:]:
                    if first not in ancestors[second] and second not in ancestors[first]:
                        raise ValueError(f"buffer {job_id}/{buffer_id} has unordered writers {first} and {second}")
            for writer in writers:
                _writer_job, writer_node = node_lookup[writer]
                if isinstance(writer_node, CommNode) and any(
                    writer in ancestors[later] for later in writers if later != writer
                ):
                    raise ValueError(f"buffer {job_id}/{buffer_id} is overwritten after communication {writer}")
            for reader in readers:
                for writer in writers:
                    if reader == writer:
                        continue
                    if writer not in ancestors[reader] and reader not in ancestors[writer]:
                        raise ValueError(f"buffer {job_id}/{buffer_id} has unordered read/write access "
                                         f"between {reader} and {writer}")
                ancestor_writers = [writer for writer in writers if writer in ancestors[reader]]
                if not ancestor_writers and spec["init"] == "empty":
                    raise ValueError(f"buffer {job_id}/{buffer_id} is read by {reader} before any writer")
                if len(ancestor_writers) > 1:
                    latest = [writer for writer in ancestor_writers
                              if not any(writer in ancestors[other] for other in ancestor_writers if other != writer)]
                    if len(latest) != 1:
                        raise ValueError(f"buffer {job_id}/{buffer_id} has no unique reaching writer for {reader}")


def _parse_submit_after(raw: Any, graph: DagGraph) -> dict[str, tuple[str, ...]]:
    values = _object(raw, "execution.submit_after", None)
    node_lookup = {f"{job.job_id}/{node.node_id}": (job, node)
                   for job in graph.jobs for node in job.nodes}
    if set(values) - {key for key, (_job, node) in node_lookup.items() if isinstance(node, ComputeNode)}:
        raise ValueError("execution.submit_after contains a non-compute node key")
    result: dict[str, tuple[str, ...]] = {}
    for compute_key, raw_comms in values.items():
        job, compute = node_lookup[compute_key]
        comm_keys = _array(raw_comms, f"execution.submit_after.{compute_key}")
        if any(not isinstance(key, str) for key in comm_keys) or len(comm_keys) != len(set(comm_keys)):
            raise ValueError(f"execution.submit_after.{compute_key} must contain unique communication IDs")
        ancestors: dict[str, set[str]] = {}
        for node in job.nodes:
            pending = list(node.deps)
            found: set[str] = set()
            while pending:
                parent = pending.pop()
                if parent in found:
                    continue
                found.add(parent)
                parent_node = next(item for item in job.nodes if item.node_id == parent)
                pending.extend(parent_node.deps)
            ancestors[f"{job.job_id}/{node.node_id}"] = {f"{job.job_id}/{item}" for item in found}
        compute_ancestors = ancestors[compute_key]
        for comm_key in comm_keys:
            entry = node_lookup.get(comm_key)
            if entry is None or entry[0].job_id != job.job_id or not isinstance(entry[1], CommNode):
                raise ValueError(f"execution.submit_after.{compute_key} references invalid comm {comm_key!r}")
            if comm_key in compute_ancestors:
                raise ValueError(f"submit_after comm {comm_key} is already a completion predecessor of {compute_key}")
            comm_ancestors = ancestors[comm_key]
            if not comm_ancestors.issubset(compute_ancestors):
                missing = sorted(comm_ancestors - compute_ancestors)
                raise ValueError(f"submit_after {comm_key} has completion predecessors absent from {compute_key}: {missing}")
        result[compute_key] = tuple(sorted(comm_keys))
    return result


def _canonical_v2_document(name: str, seed: int, graph: DagGraph,
                           execution: ReplayExecutionConfig) -> dict[str, Any]:
    base = _canonical_graph_document(name, seed, graph)
    base["schema_version"] = 2
    base["execution"] = {
        "mode": execution.mode,
        "sample_id": execution.sample_id,
        "compute_model": execution.compute_model,
        "buffers": {job_id: {buffer_id: dict(spec) for buffer_id, spec in sorted(job_buffers.items())}
                    for job_id, job_buffers in sorted(execution.buffers.items())},
        "compute_programs": {key: dict(value) for key, value in sorted(execution.compute_programs.items())},
        "comm_bindings": {key: dict(value) for key, value in sorted(execution.comm_bindings.items())},
        "submit_after": {key: list(value) for key, value in sorted(execution.submit_after.items())},
        "profiles": dict(execution.profiles),
    }
    return base


def _canonical_graph_document(name: str, seed: int, graph: DagGraph) -> dict[str, Any]:
    jobs = []
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
        jobs.append({"job_id": job.job_id, "nodes": nodes})
    return {"schema_version": 1, "name": name, "seed": seed,
            "groups": [{"group_id": group.group_id, "ranks": list(group.ranks)} for group in graph.groups],
            "jobs": jobs}


def _estimate_view_hash(graph: DagGraph) -> str:
    payload = json.dumps(_canonical_graph_document("", 0, graph), sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _buffer_ids(value: Any, where: str, *, count: int | None = None,
                minimum: int | None = None) -> tuple[str, ...]:
    raw = _array(value, where)
    if count is not None and len(raw) != count:
        raise ValueError(f"{where} must contain exactly {count} buffers")
    if minimum is not None and len(raw) < minimum:
        raise ValueError(f"{where} must contain at least {minimum} buffer")
    ids = tuple(_buffer_id(item, where) for item in raw)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{where} contains duplicate buffers")
    return ids


def _buffer_id(value: Any, where: str) -> str:
    return _identifier(value, where)


def _shape(value: Any, where: str, *, allow_scalar: bool = False) -> tuple[int, ...]:
    raw = _array(value, where)
    if not raw and not allow_scalar:
        raise ValueError(f"{where} must not be scalar")
    if any(not _is_int(dim) or dim <= 0 for dim in raw):
        raise ValueError(f"{where} dimensions must be positive integers")
    return tuple(raw)


def _product(shape: tuple[int, ...] | list[int]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


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
    if dag.execution.schema_version == 2:
        canonical = _canonical_v2_document(dag.name, dag.seed, graph, dag.execution)
    else:
        canonical = _canonical_document(dag.name, dag.seed, graph,
                                        dict(dag.execution.compute_duration_s),
                                        dag.execution.linear_sample_keys or {})
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(canonical_json.encode()).hexdigest()
    manifest_digest = digest if dag.execution.schema_version == 1 else dag.manifest_digest
    return DagInput(graph, dag.name, dag.seed, dag.execution, manifest_digest, canonical_json,
                    input_hash=dag.input_hash or dag.manifest_digest,
                    estimate_view_hash=_estimate_view_hash(graph))


def apply_dag_compute_profile(dag: DagInput, profile, *, device_uuids: tuple[str, ...],
                              software: Mapping[str, Any], strict: bool = True) -> DagInput:
    """Apply one shared compute estimate per node, conservatively across visible GPUs."""
    from examples.jobpacer.runtime.gpu_compute_profile import DAG_PROFILE_VERSION, dag_compute_profile_signature

    if dag.execution.schema_version != 2:
        raise ValueError("GPU compute profiles apply only to schema-v2 DAG inputs")
    if profile.raw.get("schema_version") != DAG_PROFILE_VERSION:
        raise ValueError("schema-v2 DAG compute requires a DAG-node compute profile (version 3)")
    if not device_uuids or any(not isinstance(uuid, str) or not uuid for uuid in device_uuids):
        raise ValueError("compute profile application requires visible GPU UUIDs")
    updates: dict[str, float] = {}
    missing: list[str] = []
    jobs = []
    for job in dag.graph.jobs:
        nodes = []
        for node in job.nodes:
            if not isinstance(node, ComputeNode):
                nodes.append(node)
                continue
            key = f"{job.job_id}/{node.node_id}"
            program = dict(dag.execution.compute_programs[key])
            nominal = dag.execution.profiles.get("nominal_compute_repeats", {})
            if nominal:
                expected_keys = {name for name, item in dag.execution.compute_programs.items()
                                 if item["op"] == "matmul"}
                if set(nominal) != expected_keys or any(
                        not _is_int(value) or value <= 0 for value in nominal.values()):
                    raise ValueError("nominal compute repeats must cover every matmul with positive integers")
                if program["op"] == "matmul":
                    program["repeats"] = nominal[key]
            signature = dag_compute_profile_signature(program, dag.execution.buffers[job.job_id])
            samples = []
            for uuid in device_uuids:
                try:
                    record = profile.record(device_uuid=uuid, stage="compute", spec=signature,
                                            software=software)
                    samples.append(record.device_event_p50_s)
                except ValueError as exc:
                    missing.append(f"{key} on {uuid}: {exc}")
            if samples:
                duration = max(samples)
                updates[key] = duration
                nodes.append(replace(node, estimated_duration_s=duration))
            else:
                nodes.append(node)
        jobs.append(replace(job, nodes=tuple(nodes)))
    if missing and strict:
        raise ValueError("GPU compute profile is missing DAG signatures: " + "; ".join(missing))
    graph = DagGraph(dag.graph.groups, tuple(jobs))
    canonical = _canonical_v2_document(dag.name, dag.seed, graph, dag.execution)
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return DagInput(graph, dag.name, dag.seed, dag.execution, dag.manifest_digest, canonical_json,
                    input_hash=dag.input_hash or dag.manifest_digest,
                    estimate_view_hash=_estimate_view_hash(graph))


def load_static_order(path: str | Path, graph: DagGraph, *,
                      extra_predecessors: Mapping[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    try:
        values = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load static order {path}: {exc}") from exc
    return validate_static_order(values, graph, extra_predecessors=extra_predecessors)


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
                        rank: int, jitter: float, samples: MutableMapping[str, float], event_log,
                        gpu_programs: Mapping[str, Any] | None = None):
    """Bind deterministic host or preallocated CUDA compute to a DAG job."""
    gpu_programs = gpu_programs or {}

    def compute(node: ComputeNode, stop_event):
        node_key = f"{job.job_id}/{node.node_id}"
        program = gpu_programs.get(node.node_id)
        if program is not None:
            event_log.record("gpu_compute_program", job_id=job.job_id, node_id=node.node_id,
                             device=str(getattr(program, "device", "unknown")),
                             operation=getattr(program, "op", "matmul"),
                             program_seed=getattr(program, "seed", None),
                             work_kind="cuda-program")
            return program.submit()
        if execution.mode == "cuda-program":
            raise RuntimeError(f"GPU compute program is not prepared for {node_key}")
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
                         base_duration_s=base_s, jitter=jitter, sampled_duration_s=duration,
                         work_kind="host_sleep")
        stop_event.wait(duration)
        return None

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
    """Construct a runtime-only static order without changing Phase 2 scores."""
    if policy == "static_fifo":
        return tuple(
            f"{job.job_id}/comm-{index}"
            for index in range(max(len(job.communications) for job in workload.jobs))
            for job in workload.jobs if index < len(job.communications)
        )
    if policy != "static_ltf":
        raise ValueError("linear static order requires static_fifo or static_ltf")
    positions = {job.job_id: 0 for job in workload.jobs}
    order: list[str] = []
    total = sum(len(job.communications) for job in workload.jobs)
    while len(order) < total:
        candidates = []
        for job in workload.jobs:
            index = positions[job.job_id]
            if index >= len(job.communications):
                continue
            comm = job.communications[index]
            tail = remaining_tail(job, index)
            candidates.append((-(comm.estimated_comm_s + tail), job.job_id, index))
        _negative_score, job_id, index = min(candidates)
        order.append(f"{job_id}/comm-{index}")
        positions[job_id] = index + 1
    return tuple(order)


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
    return max(current.consumer_compute_s - current.estimated_comm_s, 0.0) + sum(
        item.producer_compute_s + max(item.estimated_comm_s, item.consumer_compute_s)
        for item in job.communications[index + 1 :]
    )


def task_hint(job: Job, index: int) -> TaskHint:
    comm = job.communications[index]
    return TaskHint(comm.producer_compute_s, comm.estimated_comm_s,
                    remaining_tail(job, index))


def linear_ltf_estimates(workload: Workload) -> list[dict[str, Any]]:
    """Expose the c/u/tail inputs and score used by new linear LTF."""
    rows = []
    for job in workload.jobs:
        for index, comm in enumerate(job.communications):
            tail = remaining_tail(job, index)
            rows.append({
                "task_id": f"{job.job_id}/comm-{comm.id}",
                "estimated_comm_s": comm.estimated_comm_s,
                "independent_consumer_s": comm.consumer_compute_s,
                "remaining_tail_s": tail,
                "ltf_score_s": comm.estimated_comm_s + tail,
            })
    return rows


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
