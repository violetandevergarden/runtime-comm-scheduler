"""Strict schema and immutable models for the phase 3 GPU linear workload."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from runtime_comm_scheduler.runtime import CollectiveSpec


SCHEMA = "jobpacer-gpu-linear"
SCHEMA_VERSION = 1
FIFO_ESTIMATOR_VERSION = "gpu-fifo-unused-v1"
GPU_TAIL_ESTIMATOR_VERSION = "gpu-linear-static-tail-v1"


@dataclass(frozen=True)
class GpuComputeSpec:
    op: str
    dtype: str = "float32"
    layout: str = "contiguous"
    m: int | None = None
    n: int | None = None
    k: int | None = None
    repeats: int = 1
    input_role: str | None = None
    output_role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op}
        if self.op == "matmul":
            result.update({"m": self.m, "n": self.n, "k": self.k, "dtype": self.dtype,
                           "layout": self.layout, "repeats": self.repeats})
        if self.input_role is not None:
            result["input_role"] = self.input_role
        if self.output_role is not None:
            result["output_role"] = self.output_role
        return result


@dataclass(frozen=True)
class GpuEstimates:
    compute_profile_ref: str | None
    comm_profile_ref: str | None
    estimator_version: str

    def to_dict(self) -> dict[str, Any]:
        return {"compute_profile_ref": self.compute_profile_ref,
                "comm_profile_ref": self.comm_profile_ref,
                "estimator_version": self.estimator_version}


@dataclass(frozen=True)
class GpuSegment:
    segment_id: str
    group_id: str
    group_seq: int
    producer_compute: GpuComputeSpec
    collective: CollectiveSpec
    independent_compute: GpuComputeSpec | None
    dependent_compute: tuple[str, ...] | None
    estimates: GpuEstimates

    @property
    def task_id(self) -> str:
        return f"{self._job_id}/{self.segment_id}"

    # The parser binds a frozen job id into the spec; it is deliberately absent
    # from the wire-level compute/task protocol.
    _job_id: str = ""


@dataclass(frozen=True)
class GpuJob:
    job_id: str
    segments: tuple[GpuSegment, ...]


@dataclass(frozen=True)
class GpuGroup:
    group_id: str
    ranks: tuple[int, ...]


@dataclass(frozen=True)
class GpuLinearInput:
    name: str
    execution_seed: int
    groups: tuple[GpuGroup, ...]
    jobs: tuple[GpuJob, ...]
    execution_contract: Mapping[str, Any]
    canonical_json: str
    manifest_digest: str
    raw_digest: str
    source_path: str | None = None

    @property
    def group_ranks(self) -> dict[str, tuple[int, ...]]:
        return {group.group_id: group.ranks for group in self.groups}

    @property
    def expected_task_ids(self) -> tuple[str, ...]:
        return tuple(segment.task_id for job in self.jobs for segment in job.segments)

    def local_jobs(self, rank: int) -> tuple[GpuJob, ...]:
        return tuple(job for job in self.jobs
                     if rank in self.group_ranks[job.segments[0].group_id])


def load_gpu_linear(path: str | Path, *, world_size: int | None = None) -> GpuLinearInput:
    source = Path(path)
    try:
        raw_bytes = source.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load GPU linear manifest {source}: {exc}") from exc
    return parse_gpu_linear(raw, world_size=world_size, source_path=str(source),
                            raw_digest=hashlib.sha256(raw_bytes).hexdigest())


def parse_gpu_linear(raw: Any, *, world_size: int | None = None,
                     source_path: str | None = None, raw_digest: str | None = None) -> GpuLinearInput:
    root = _object(raw, "GPU manifest", {"schema", "schema_version", "name", "execution_seed",
                                         "groups", "execution_contract", "jobs"})
    if root["schema"] != SCHEMA or not _integer(root["schema_version"]) or root["schema_version"] != 1:
        raise ValueError(f"GPU manifest schema must be {SCHEMA!r} version 1")
    name = _text(root["name"], "name")
    seed = root["execution_seed"]
    if not _integer(seed):
        raise ValueError("execution_seed must be an integer")
    if world_size is not None and (not _integer(world_size) or world_size <= 0):
        raise ValueError("world_size must be a positive integer")

    contract = _object(root["execution_contract"], "execution_contract",
                       {"producer_readiness", "submit_order", "segment_advance", "max_inflight"})
    if not _integer(contract["max_inflight"]):
        raise ValueError("execution_contract.max_inflight must be a non-boolean integer")
    expected_contract = {"producer_readiness": "physical-ready",
                         "submit_order": "comm-request-before-independent",
                         "segment_advance": "terminal-event-complete", "max_inflight": 1}
    if contract != expected_contract:
        raise ValueError(f"unsupported GPU execution_contract; expected {expected_contract}")

    groups: list[GpuGroup] = []
    seen_groups: set[str] = set()
    for index, raw_group in enumerate(_array(root["groups"], "groups")):
        value = _object(raw_group, f"groups[{index}]", {"group_id", "ranks"})
        group_id = _identifier(value["group_id"], f"groups[{index}].group_id")
        ranks_raw = _array(value["ranks"], f"groups[{index}].ranks")
        if not ranks_raw or any(not _integer(rank) or rank < 0 for rank in ranks_raw):
            raise ValueError(f"groups[{index}].ranks must contain non-negative integers")
        ranks = tuple(sorted(ranks_raw))
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"groups[{index}].ranks contains duplicates")
        if world_size is not None and max(ranks) >= world_size:
            raise ValueError(f"groups[{index}] rank is outside world_size={world_size}")
        if group_id in seen_groups:
            raise ValueError(f"duplicate group_id {group_id}")
        seen_groups.add(group_id)
        groups.append(GpuGroup(group_id, ranks))
    if not groups:
        raise ValueError("groups must not be empty")

    jobs: list[GpuJob] = []
    seen_jobs: set[str] = set()
    seen_tasks: set[str] = set()
    group_sequences: dict[str, list[int]] = {group.group_id: [] for group in groups}
    for job_index, raw_job in enumerate(_array(root["jobs"], "jobs")):
        job_value = _object(raw_job, f"jobs[{job_index}]", {"job_id", "segments"})
        job_id = _identifier(job_value["job_id"], f"jobs[{job_index}].job_id")
        if job_id in seen_jobs:
            raise ValueError(f"duplicate job_id {job_id}")
        seen_jobs.add(job_id)
        segments: list[GpuSegment] = []
        seen_segments: set[str] = set()
        for segment_index, raw_segment in enumerate(_array(job_value["segments"], f"{job_id}.segments")):
            where = f"{job_id}.segments[{segment_index}]"
            value = _object(raw_segment, where, {"segment_id", "group_id", "group_seq",
                                                  "producer_compute", "collective",
                                                  "independent_compute", "dependent_compute", "estimates"})
            segment_id = _identifier(value["segment_id"], f"{where}.segment_id")
            if segment_id in seen_segments:
                raise ValueError(f"duplicate segment_id {job_id}/{segment_id}")
            seen_segments.add(segment_id)
            group_id = _identifier(value["group_id"], f"{where}.group_id")
            if group_id not in seen_groups:
                raise ValueError(f"{where} references unknown group {group_id}")
            group_seq = value["group_seq"]
            if not _integer(group_seq) or group_seq < 0:
                raise ValueError(f"{where}.group_seq must be a non-negative integer")
            group_sequences[group_id].append(group_seq)
            task_id = f"{job_id}/{segment_id}"
            if task_id in seen_tasks:
                raise ValueError(f"duplicate task id {task_id}")
            seen_tasks.add(task_id)
            collective = _collective(value["collective"], f"{where}.collective")
            producer = _compute(value["producer_compute"], f"{where}.producer_compute", role="producer",
                                collective_shape=collective.shape)
            independent_raw = value["independent_compute"]
            independent = (None if independent_raw is None else
                           _compute(independent_raw, f"{where}.independent_compute", role="independent",
                                    collective_shape=collective.shape))
            dependent = _dependent(value["dependent_compute"], f"{where}.dependent_compute", independent)
            estimates = _estimates(value["estimates"], f"{where}.estimates")
            segments.append(GpuSegment(segment_id, group_id, group_seq, producer, collective,
                                       independent, dependent, estimates, job_id))
        if not segments:
            raise ValueError(f"job {job_id} must contain at least one segment")
        jobs.append(GpuJob(job_id, tuple(segments)))
    if not jobs:
        raise ValueError("jobs must not be empty")
    for group_id, sequences in group_sequences.items():
        if not sequences:
            raise ValueError(f"group {group_id} is unused")
        if sorted(sequences) != list(range(len(sequences))):
            raise ValueError(f"group {group_id} group_seq must be unique and contiguous from 0")
    group_members = {group.group_id: group.ranks for group in groups}
    for job in jobs:
        memberships = {group_members[segment.group_id] for segment in job.segments}
        if len(memberships) != 1:
            raise ValueError(f"all segments in job {job.job_id} must have the same member set")

    # Canonical form is stable under object key order and insignificant float spelling.
    canonical_doc = {
        "schema": SCHEMA, "schema_version": 1, "name": name, "execution_seed": seed,
        "groups": [{"group_id": g.group_id, "ranks": list(g.ranks)} for g in sorted(groups, key=lambda x: x.group_id)],
        "execution_contract": expected_contract,
        "jobs": [{"job_id": job.job_id, "segments": [
            {"segment_id": seg.segment_id, "group_id": seg.group_id, "group_seq": seg.group_seq,
             "producer_compute": seg.producer_compute.to_dict(), "collective": seg.collective.to_dict(),
             "independent_compute": (seg.independent_compute.to_dict() if seg.independent_compute else None),
             "dependent_compute": (None if seg.dependent_compute is None else {"op": "sum_join", "inputs": list(seg.dependent_compute)}),
             "estimates": seg.estimates.to_dict()} for seg in job.segments]}
            for job in jobs],
    }
    canonical_json = json.dumps(canonical_doc, sort_keys=True, separators=(",", ":"), allow_nan=False)
    canonical_digest = hashlib.sha256(canonical_json.encode()).hexdigest()
    if raw_digest is None:
        raw_digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
    return GpuLinearInput(name, seed, tuple(sorted(groups, key=lambda x: x.group_id)), tuple(jobs),
                          expected_contract, canonical_json, canonical_digest, raw_digest, source_path)


def _collective(raw: Any, where: str) -> CollectiveSpec:
    value = _object(raw, where, {"op", "reduction", "shape", "dtype", "numel", "num_bytes"})
    if value["op"] != "all_reduce" or value["reduction"] != "sum" or value["dtype"] != "float32":
        raise ValueError(f"{where} supports only float32 SUM all_reduce")
    shape_raw = _array(value["shape"], f"{where}.shape")
    if not shape_raw or any(not _integer(dim) or dim <= 0 for dim in shape_raw):
        raise ValueError(f"{where}.shape must contain positive integers")
    numel, num_bytes = value["numel"], value["num_bytes"]
    if not _integer(numel) or not _integer(num_bytes):
        raise ValueError(f"{where}.numel/num_bytes must be integers")
    return CollectiveSpec("all_reduce", numel, num_bytes, "float32", tuple(shape_raw), "sum")


def _compute(raw: Any, where: str, *, role: str, collective_shape: tuple[int, ...]) -> GpuComputeSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object")
    op = raw.get("op")
    if role == "producer" and op == "fill":
        value = _object(raw, where, {"op", "dtype", "layout", "output_role"}, required={"op"})
        if value.get("dtype", "float32") != "float32" or value.get("layout", "contiguous") != "contiguous":
            raise ValueError(f"{where} supports only contiguous float32")
        if value.get("output_role", "collective_input") != "collective_input":
            raise ValueError(f"{where}.output_role must be collective_input")
        return GpuComputeSpec("fill", output_role="collective_input")
    if op != "matmul":
        raise ValueError(f"{where}.op must be {'fill or matmul' if role == 'producer' else 'matmul'}")
    allowed = {"op", "m", "n", "k", "dtype", "layout", "repeats", "output_role"}
    if role == "independent":
        allowed.add("input_role")
    value = _object(raw, where, allowed)
    m, n, k, repeats = value["m"], value["n"], value["k"], value["repeats"]
    if any(not _integer(item) or item <= 0 for item in (m, n, k, repeats)):
        raise ValueError(f"{where} m/n/k/repeats must be positive integers")
    if (m, n) != collective_shape:
        raise ValueError(f"{where}.m/n must match collective shape {collective_shape}")
    if value.get("dtype") != "float32" or value.get("layout") != "contiguous":
        raise ValueError(f"{where} supports only contiguous float32")
    expected_input = "private_inputs" if role == "independent" else None
    expected_output = "independent_output" if role == "independent" else "collective_input"
    if value.get("output_role") != expected_output or value.get("input_role") != expected_input:
        raise ValueError(f"{where} has an invalid input/output role for {role}")
    return GpuComputeSpec("matmul", "float32", "contiguous", m, n, k, repeats,
                          expected_input, expected_output)


def _dependent(raw: Any, where: str, independent: GpuComputeSpec | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    value = _object(raw, where, {"op", "inputs"})
    if value["op"] != "sum_join":
        raise ValueError(f"{where}.op must be sum_join")
    inputs = _array(value["inputs"], f"{where}.inputs")
    allowed = {"collective_output"} | ({"independent_output"} if independent else set())
    if not inputs or any(not isinstance(item, str) for item in inputs) or len(set(inputs)) != len(inputs):
        raise ValueError(f"{where}.inputs must contain unique input roles")
    if set(inputs) - allowed or "collective_output" not in inputs:
        raise ValueError(f"{where}.inputs may reference only available collective/independent outputs")
    return tuple(sorted(inputs))


def _estimates(raw: Any, where: str) -> GpuEstimates:
    value = _object(raw, where, {"compute_profile_ref", "comm_profile_ref", "estimator_version"})
    compute_ref = value.get("compute_profile_ref")
    comm_ref = value.get("comm_profile_ref")
    for key, item in (("compute_profile_ref", compute_ref), ("comm_profile_ref", comm_ref)):
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise ValueError(f"{where}.{key} must be a non-empty string or null")
    version = value["estimator_version"]
    if not isinstance(version, str) or version not in {FIFO_ESTIMATOR_VERSION, GPU_TAIL_ESTIMATOR_VERSION}:
        raise ValueError(f"{where}.estimator_version must be a supported GPU estimator version")
    if version == GPU_TAIL_ESTIMATOR_VERSION and (not compute_ref or not comm_ref):
        raise ValueError(f"{where} GPU tail estimates require compute and comm profile references")
    return GpuEstimates(compute_ref, comm_ref, version)


def _object(raw: Any, where: str, allowed: set[str], required: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object")
    unknown = set(raw) - allowed
    missing = (required if required is not None else allowed) - set(raw)
    if unknown or missing:
        raise ValueError(f"{where} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}")
    return raw


def _array(raw: Any, where: str) -> list[Any]:
    if not isinstance(raw, list):
        raise ValueError(f"{where} must be an array")
    return raw


def _text(raw: Any, where: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return raw


def _identifier(raw: Any, where: str) -> str:
    value = _text(raw, where)
    if "/" in value:
        raise ValueError(f"{where} must not contain '/'")
    return value


def _integer(raw: Any) -> bool:
    return isinstance(raw, int) and not isinstance(raw, bool)
