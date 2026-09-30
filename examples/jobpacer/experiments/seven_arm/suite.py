"""Build and verify frozen inputs for the Phase 3 GPU seven-arm study."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Mapping

from examples.jobpacer.runtime.runtime_adapter import (
    apply_dag_compute_profile,
    apply_dag_profile,
    load_dag,
)
from runtime_comm_scheduler.dag import build_static_order
from runtime_comm_scheduler.dag.model import compute_tails


ROOT = Path(__file__).resolve().parents[4]
SCENARIOS = (
    "L0-balanced", "L1-skew-tail", "D0-fork-join",
    "D1-asymmetric-frontiers", "D2-cross-job-skew", "D3-order-and-sinks",
)
FORMAL_WORKLOAD_SEEDS = (8101, 8102, 8103, 8104, 8105)
PILOT_WORKLOAD_SEEDS = (9101, 9102, 9103)
ORDER_SEED = 20260927
ARMS = (
    "bare-ordered", "old-static-fifo", "old-static-ltf", "new-static-fifo",
    "new-static-ltf", "new-dynamic-fifo", "new-dynamic-ltf",
)
MECHANISM_PILOT_ARMS = (
    "old-static-fifo", "old-static-ltf", "new-static-fifo", "new-static-ltf",
    "new-dynamic-fifo", "new-dynamic-ltf",
)
SUPPORTED_ARMS = ARMS
RAW_ORDERED_ARM = "raw-ordered-static-fifo"
BARE_ARM = "bare-ordered"
CONTRACT_VERSION = "gpu-seven-arm-v2"
GENERATOR_VERSION = "gpu-seven-arm-inputs-v5-mechanisms"
TENSOR_SEED_VERSION = "sha256-domain-separated-tensor-seed-v1"
EXECUTION_CONTRACT = {
    "backend": "nccl", "dtype": "float32", "op": "all_reduce",
    "reduction": "sum", "world_size": 2,
    "contract_version": CONTRACT_VERSION,
    "capacity_by_engine": {"bare": None, "old": "local-one", "new": "global-one"},
    "warmup_iterations": 5, "measurement_lane": "DAG CUDA program with physical completion",
    "static_fifo_order": "topological-layer-job-round-robin-v1",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256(data)


class _DagBuilder:
    def __init__(self, scenario: str, *, seed: int = 0, sample_id: str = "template"):
        self.scenario = scenario
        self.seed = seed
        self.sample_id = sample_id
        self.groups = [
            {"group_id": "g0", "ranks": [0, 1]},
            {"group_id": "g1", "ranks": [0, 1]},
        ]
        self.jobs: list[dict[str, Any]] = []
        self.buffers: dict[str, dict[str, dict[str, Any]]] = {}
        self.programs: dict[str, dict[str, Any]] = {}
        self.comm_bindings: dict[str, dict[str, str]] = {}
        self.explicit_group_sequences: dict[str, int] = {}
        self.submit_after: dict[str, list[str]] = {}
        self.comm_shape = [32, 32]

    def new_job(self, job_id: str) -> dict[str, Any]:
        job = {"job_id": job_id, "nodes": []}
        self.jobs.append(job)
        self.buffers[job_id] = {}
        return job

    def _buffer(self, job_id: str, name: str, shape: list[int], init: str) -> None:
        self.buffers[job_id][name] = {"shape": list(shape), "dtype": "float32", "init": init}

    def compute(self, job: dict[str, Any], node_id: str, deps: list[str], program: dict[str, Any],
                output: str, output_shape: list[int], *, estimate_s: float = 0.001) -> str:
        job_id = job["job_id"]
        key = f"{job_id}/{node_id}"
        self._buffer(job_id, output, output_shape, "zeros")
        job["nodes"].append({"node_id": node_id, "kind": "compute", "deps": list(deps),
                             "estimated_duration_s": estimate_s})
        self.programs[key] = {**program, "output": output}
        return node_id

    def matmul(self, job: dict[str, Any], node_id: str, deps: list[str], *, output: str,
               repeats: int, shape: int = 32, estimate_s: float = 0.001) -> str:
        job_id = job["job_id"]
        left, right = f"{node_id}-left", f"{node_id}-right"
        self._buffer(job_id, left, [shape, shape], "seeded-random")
        self._buffer(job_id, right, [shape, shape], "seeded-random")
        return self.compute(job, node_id, deps,
                            {"op": "matmul", "inputs": [left, right], "repeats": repeats},
                            output, [shape, shape], estimate_s=estimate_s)

    def fill(self, job: dict[str, Any], node_id: str, deps: list[str], *, output: str,
             shape: list[int] | None = None) -> str:
        shape = list(self.comm_shape if shape is None else shape)
        return self.compute(job, node_id, deps, {"op": "fill"}, output, shape)

    def join(self, job: dict[str, Any], node_id: str, deps: list[str], inputs: list[str]) -> str:
        return self.compute(job, node_id, deps, {"op": "sum_join", "inputs": inputs},
                            f"{node_id}-scalar", [])

    def comm(self, job: dict[str, Any], node_id: str, deps: list[str], group: str,
             buffer: str, *, group_seq: int | None = None) -> str:
        job_id = job["job_id"]
        job["nodes"].append({
            "node_id": node_id, "kind": "comm", "deps": list(deps), "group_id": group,
            "group_seq": 0,
            "estimated_comm_s": 0.001,
            "collective": {"op": "all_reduce", "reduction": "sum", "shape": list(self.comm_shape),
                           "dtype": "float32", "numel": math.prod(self.comm_shape),
                           "num_bytes": 4 * math.prod(self.comm_shape)},
        })
        self.comm_bindings[f"{job_id}/{node_id}"] = {"buffer": buffer}
        if group_seq is not None:
            self.explicit_group_sequences[f"{job_id}/{node_id}"] = group_seq
        return node_id

    def document(self) -> dict[str, Any]:
        seqs = {group["group_id"]: 0 for group in self.groups}
        for job in self.jobs:
            for node in job["nodes"]:
                if node["kind"] == "comm":
                    group = node["group_id"]
                    explicit = self.explicit_group_sequences.get(
                        f"{job['job_id']}/{node['node_id']}"
                    )
                    if explicit is None:
                        node["group_seq"] = seqs[group]
                        seqs[group] += 1
                    else:
                        node["group_seq"] = explicit
        return {
            "schema_version": 2,
            "name": f"gpu-seven-arm-{self.scenario}-{self.sample_id}",
            "seed": self.seed,
            "groups": self.groups,
            "jobs": self.jobs,
            "execution": {
                "mode": "cuda-program", "sample_id": self.sample_id,
                "compute_model": "one-active-compute-per-job",
                "buffers": self.buffers, "compute_programs": self.programs,
                "comm_bindings": self.comm_bindings, "submit_after": self.submit_after,
                "profiles": {},
            },
        }


# These are initial pilot parameters, not calibrated duration promises. Revision
# is explicit and bounded by the pilot gate; formal seeds never select parameters.
SCENARIO_PARAMETERS = {
    "L0-balanced": {"matrix": 256, "stages": 3, "producer_repeats": 8, "tail_repeats": 8},
    "L1-skew-tail": {"matrix": 256, "stages": 3, "producer_repeats": 8, "skew_repeats": 160},
    "D0-fork-join": {"matrix": 256, "stages": 2, "independent_repeats": 32},
    "D1-asymmetric-frontiers": {"matrix": 256, "occupancy_bytes": 64 * 1024 * 1024,
                               "short_tail_repeats": 8, "long_tail_repeats": 96},
    "D2-cross-job-skew": {"matrix": 256, "stages": 3, "skew_repeats": 96,
                           "independent_repeats": 24},
    "D3-order-and-sinks": {"matrix": 256, "stages": 2, "independent_repeats": 16},
}


def make_template(scenario: str) -> dict[str, Any]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown seven-arm scenario {scenario!r}")
    builder = _DagBuilder(scenario)
    params = SCENARIO_PARAMETERS[scenario]
    size = params["matrix"]
    builder.comm_shape = [size, size]
    if scenario == "D1-asymmetric-frontiers":
        # A real large predecessor occupies admission while two independent jobs
        # produce frontiers. No policy-specific delay or synthetic OFFER exists.
        builder.groups.insert(0, {"group_id": "occupancy", "ranks": [0, 1]})
        occupy = builder.new_job("occupancy")
        shape = [params["occupancy_bytes"] // 4]
        builder._buffer("occupancy", "wire", shape, "seeded-random")
        builder.comm_shape = shape
        builder.comm(occupy, "comm", [], "occupancy", "wire")
        builder.join(occupy, "sink", ["comm"], ["wire"])
        builder.comm_shape = [size, size]
        for index in range(2):
            job = builder.new_job(f"job-{index}")
            builder.matmul(job, "producer", [], output="wire", shape=size, repeats=8 + index * 4)
            builder.comm(job, "frontier", ["producer"], f"g{index}", "wire")
            builder.matmul(job, "independent", ["producer"], output="private", shape=size, repeats=8)
            builder.submit_after[f"job-{index}/independent"] = [f"job-{index}/frontier"]
            builder.matmul(job, "tail", ["frontier"], output="tail", shape=size,
                           repeats=params["long_tail_repeats" if index else "short_tail_repeats"])
            builder.join(job, "sink", ["tail", "independent"], ["wire", "private", "tail"])
    else:
        for index in range(2):
            job = builder.new_job(f"job-{index}")
            previous = None
            for stage in range(params["stages"]):
                suffix = f"s{stage}"
                producer, comm, join = f"producer-{suffix}", f"comm-{suffix}", f"join-{suffix}"
                repeats = params.get("producer_repeats", 8)
                if index == 0 and stage == 0:
                    repeats = params.get("skew_repeats", repeats)
                builder.matmul(job, producer, [previous] if previous else [],
                               output=f"wire-{suffix}", shape=size, repeats=repeats)
                # D3 crosses groups and shares each group's canonical sequence.
                group = f"g{(index + stage) % 2}" if scenario.startswith("D3") else f"g{index}"
                builder.comm(job, comm, [producer], group, f"wire-{suffix}")
                dependencies, inputs = [comm], [f"wire-{suffix}"]
                if scenario.startswith("D"):
                    independent = f"independent-{suffix}"
                    builder.matmul(job, independent, [producer], output=f"private-{suffix}",
                                   shape=size, repeats=params["independent_repeats"])
                    builder.submit_after[f"job-{index}/{independent}"] = [f"job-{index}/{comm}"]
                    dependencies.append(independent)
                    inputs.append(f"private-{suffix}")
                if scenario.startswith("D2") or scenario.startswith("L"):
                    tail = f"tail-{suffix}"
                    builder.matmul(job, tail, [comm], output=f"tail-{suffix}", shape=size,
                                   repeats=(8 if index == 0 else 32) if scenario.startswith("D2") else 8)
                    # L0/L1 are genuine chains: comm -> tail -> consumer.
                    if scenario.startswith("L"):
                        dependencies = [tail]
                    else:
                        dependencies.append(tail)
                    inputs.append(f"tail-{suffix}")
                builder.join(job, join, dependencies, inputs)
                previous = join
            if scenario.startswith("D3"):
                builder.fill(job, "sink-a", [previous], output="sink-a-buffer")
                builder.fill(job, "sink-b", ["join-s0"], output="sink-b-buffer")
    document = builder.document()
    document["execution"]["profiles"] = {
        "nominal_compute_repeats": {key: (8 if "producer" in key else program["repeats"])
                                    for key, program in builder.programs.items()
                                    if program["op"] == "matmul"},
        "parameters": params, "design_revision": 0,
    }
    for job in document["jobs"]:
        for node in job["nodes"]:
            key = f"{job['job_id']}/{node['node_id']}"
            nominal = document["execution"]["profiles"]["nominal_compute_repeats"].get(key)
            if nominal is not None:
                node["estimated_duration_s"] = nominal * 0.00025
    return document


def _execution_hash(document: Mapping[str, Any]) -> str:
    """Hash concrete execution work and topology without policy estimates."""
    graph = []
    for job in document["jobs"]:
        nodes = []
        for node in job["nodes"]:
            row = {key: value for key, value in node.items()
                   if key not in {"estimated_duration_s", "estimated_comm_s"}}
            nodes.append(row)
        graph.append({"job_id": job["job_id"], "nodes": nodes})
    execution = document["execution"]
    payload = {
        "groups": document["groups"], "jobs": graph,
        "mode": execution["mode"], "sample_id": execution["sample_id"],
        "compute_model": execution["compute_model"], "buffers": execution["buffers"],
        "compute_programs": execution["compute_programs"],
        "comm_bindings": execution["comm_bindings"], "submit_after": execution["submit_after"],
        "seed": document["seed"],
    }
    return _sha256(_json_bytes(payload))


def topology_hash(document: Mapping[str, Any]) -> str:
    """Hash graph, shapes, buffers, and program roles while ignoring seed/repeats/estimates."""
    jobs = []
    for job in document["jobs"]:
        nodes = []
        for node in job["nodes"]:
            nodes.append({key: value for key, value in node.items()
                          if key not in {"estimated_duration_s", "estimated_comm_s"}})
        jobs.append({"job_id": job["job_id"], "nodes": nodes})
    programs = {key: {field: value for field, value in program.items() if field != "repeats"}
                for key, program in document["execution"]["compute_programs"].items()}
    payload = {
        "groups": document["groups"], "jobs": jobs,
        "execution": {"mode": document["execution"]["mode"],
                      "compute_model": document["execution"]["compute_model"],
                      "buffers": document["execution"]["buffers"],
                      "compute_programs": programs,
                      "comm_bindings": document["execution"]["comm_bindings"],
                      "submit_after": document["execution"]["submit_after"]},
    }
    return _sha256(_json_bytes(payload))


def _sample_pattern(seeds: tuple[int, ...]) -> dict[int, int]:
    # Stratification gives each requested seed a stable unique workload codeword;
    # it never redraws a seed based on runtime or performance.
    ordered = sorted(seeds, key=lambda seed: _sha256(f"workload-seed:{seed}".encode()))
    return {seed: index for index, seed in enumerate(ordered)}


def _tensor_seed(scenario: str, workload_seed: int) -> int:
    raw = f"{TENSOR_SEED_VERSION}\0{scenario}\0{workload_seed}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def expand_sample(template: Mapping[str, Any], scenario: str, workload_seed: int,
                  pattern_index: int) -> dict[str, Any]:
    document = json.loads(json.dumps(template))
    tensor_seed = _tensor_seed(scenario, workload_seed)
    document["name"] = f"gpu-seven-arm-{scenario}-workload-{workload_seed}"
    document["seed"] = tensor_seed
    document["execution"]["sample_id"] = f"{GENERATOR_VERSION}:{scenario}:tensor-v1"
    matmuls = sorted(key for key, program in document["execution"]["compute_programs"].items()
                     if program["op"] == "matmul")
    if len(matmuls) < 2:
        raise ValueError(f"{scenario} needs two matmul programs for distinct workload samples")
    # Factors are integer-preserving for the template's multiple-of-four repeats.
    codewords = ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0))
    if not 0 <= pattern_index < len(codewords):
        raise ValueError("the formal workload sample set has exactly five stratified seeds")
    fixed = codewords[pattern_index]
    factors = (0.75, 1.0, 1.25)
    for index, key in enumerate(matmuls):
        if index < 2:
            factor = factors[fixed[index]]
        else:
            digest = hashlib.sha256(f"workload-node:{workload_seed}:{key}".encode()).digest()
            factor = factors[digest[0] % len(factors)]
        program = document["execution"]["compute_programs"][key]
        base = program["repeats"]
        program["repeats"] = max(1, int(math.floor(base * factor + 0.5)))
    return document


def prepare_suite(output_dir: Path, *, seeds: tuple[int, ...] = FORMAL_WORKLOAD_SEEDS) -> dict[str, Any]:
    if (len(seeds) not in {1, 3, 5} or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)):
        raise ValueError("prepared suite requires one diagnostic, three pilot or five formal integer workload seeds")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty input directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = output_dir / "templates"
    samples_dir = output_dir / "inputs"
    templates_dir.mkdir(exist_ok=True)
    pattern = _sample_pattern(seeds)
    scenarios: dict[str, Any] = {}
    execution_hashes: list[str] = []
    for scenario in SCENARIOS:
        template = make_template(scenario)
        template_path = templates_dir / f"{scenario}.json"
        template_sha = _write_json(template_path, template)
        samples = []
        sample_hashes = []
        execution_seen: set[str] = set()
        for seed in seeds:
            sample = expand_sample(template, scenario, seed, pattern[seed])
            sample_path = samples_dir / scenario / f"workload-{seed}.json"
            file_sha = _write_json(sample_path, sample)
            parsed = load_dag(sample_path, world_size=2)
            if parsed.execution.schema_version != 2 or parsed.execution.mode != "cuda-program":
                raise ValueError(f"generated sample is not schema-v2 CUDA DAG: {sample_path}")
            execution_sha = _execution_hash(sample)
            if execution_sha in execution_seen:
                raise ValueError(f"{scenario} generated duplicate execution samples")
            execution_seen.add(execution_sha)
            sample_hashes.append(execution_sha)
            fifo = build_static_order(
                parsed.graph, "static_fifo",
                extra_predecessors=parsed.execution.submit_after,
            )
            fifo_path = samples_dir / scenario / f"workload-{seed}.fifo-order.json"
            fifo_sha = _write_json(fifo_path, list(fifo))
            samples.append({
                "scenario": scenario, "workload_seed": seed, "tensor_seed": sample["seed"],
                "sample_id": sample["execution"]["sample_id"],
                "path": sample_path.relative_to(output_dir).as_posix(),
                "input_sha256": file_sha, "input_hash": parsed.input_hash,
                "execution_sample_hash": execution_sha,
                "topology_hash": topology_hash(sample),
                "estimate_view_hash": parsed.estimate_view_hash,
                "fifo_order_path": fifo_path.relative_to(output_dir).as_posix(),
                "fifo_order_sha256": fifo_sha, "fifo_sequence": list(fifo),
                "pattern_index": pattern[seed],
                "default_order_rule": EXECUTION_CONTRACT["static_fifo_order"],
                "actual_matmul_repeats": {key: program["repeats"] for key, program
                                           in sample["execution"]["compute_programs"].items()
                                           if program["op"] == "matmul"},
                "nominal_matmul_repeats": sample["execution"]["profiles"]["nominal_compute_repeats"],
            })
        if len(set(sample_hashes)) != len(seeds):
            raise ValueError(f"{scenario} generated duplicate execution samples")
        execution_hashes.extend(sample_hashes)
        scenarios[scenario] = {"template_path": template_path.relative_to(output_dir).as_posix(),
                                "template_sha256": template_sha, "samples": samples}
    suite = {
        "schema": "jobpacer-gpu-seven-arm-input-suite", "schema_version": 1,
        "status": "prepared-awaiting-profiles", "scope": "formal" if len(seeds) == 5 else "pilot" if len(seeds) == 3 else "diagnostic",
        "generator_version": GENERATOR_VERSION,
        "tensor_seed_derivation": TENSOR_SEED_VERSION,
        "workload_seeds": list(seeds), "order_seed": ORDER_SEED,
        "scenarios": scenarios, "execution_contract": dict(EXECUTION_CONTRACT),
        "bare": {"status": "unqualified", "evidence": None,
                 "available_reference": RAW_ORDERED_ARM},
        "contract_version": CONTRACT_VERSION,
        "bare_qualification": {"arm": BARE_ARM, "status": "pending-G1", "evidence": None},
        "readiness": {"software": "pending", "backend": "pending", "mechanism": "pending",
                      "measurement": "pending", "rehearsal": "pending", "recovery": "pending"},
        "pilot_revision": 0, "max_pilot_parameter_revisions": 2,
        "aggregate_execution_sample_hash": _sha256("\n".join(execution_hashes).encode()),
    }
    _write_json(output_dir / "suite-inputs.json", suite)
    return suite


def _copy_profile(source: Path, target: Path) -> str:
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target.resolve():
        shutil.copyfile(source, target)
    return _sha256(target.read_bytes())


def verify_compute_profile_inputs(calibration: Mapping[str, Any], suite_input_dir: Path) -> str:
    """Prove the compute profile was calibrated against this exact 30-sample set."""
    suite_input_dir = suite_input_dir.resolve(strict=True)
    suite_path = suite_input_dir / "suite-inputs.json"
    suite = json.loads(suite_path.read_text())
    if (suite.get("scope") != "formal"
            or set(suite.get("workload_seeds", [])) != set(FORMAL_WORKLOAD_SEEDS)):
        raise ValueError("compute calibration inputs must be the five-seed formal candidate suite")
    expected: list[tuple[str, str, str]] = []
    for scenario in SCENARIOS:
        scenario_record = suite.get("scenarios", {}).get(scenario)
        if not scenario_record:
            raise ValueError(f"compute calibration input suite is missing scenario {scenario}")
        for sample in scenario_record.get("samples", []):
            input_path = suite_input_dir / sample["path"]
            if not input_path.is_file() or _sha256(input_path.read_bytes()) != sample.get("input_sha256"):
                raise ValueError(f"compute calibration input is missing or changed: {input_path}")
            expected.append((sample["input_sha256"], sample["input_hash"], sample["estimate_view_hash"]))
    recorded_rows = calibration.get("inputs")
    if not isinstance(recorded_rows, list) or len(recorded_rows) < len(expected):
        raise ValueError(f"compute profile must record at least {len(expected)} calibration inputs")
    recorded = []
    recorded_paths = set()
    for row in recorded_rows:
        if not isinstance(row, Mapping):
            raise ValueError("compute profile calibration input record is malformed")
        path = row.get("path")
        if not isinstance(path, str) or path in recorded_paths:
            raise ValueError("compute profile calibration input paths must be unique")
        recorded_paths.add(path)
        recorded.append((row.get("file_sha256"), row.get("input_hash"), row.get("estimate_view_hash")))
    for extra in recorded_rows:
        identity = (extra.get("file_sha256"), extra.get("input_hash"), extra.get("estimate_view_hash"))
        if identity not in expected:
            path = Path(extra["path"])
            if not path.is_file() or _sha256(path.read_bytes()) != extra.get("file_sha256"):
                raise ValueError("additional calibration input is missing or changed")
            parsed = load_dag(path, world_size=2)
            if (parsed.input_hash, parsed.estimate_view_hash) != identity[1:]:
                raise ValueError("additional calibration input identity differs")
    if not set(expected).issubset(set(recorded)) or len(set(recorded)) != len(recorded):
        raise ValueError("compute profile calibration inputs do not match the complete formal sample set")
    return _sha256(suite_path.read_bytes())


def freeze_suite(input_dir: Path, compute_profile_path: Path, comm_profile_path: Path,
                 *, output_dir: Path | None = None, world_size: int = 2,
                 calibration_input_dir: Path | None = None) -> dict[str, Any]:
    input_dir = input_dir.resolve(strict=True)
    output_dir = (output_dir or input_dir).resolve()
    if output_dir != input_dir:
        raise ValueError("suite profiles and static orders must be frozen into the prepared input directory")
    suite_path = input_dir / "suite-inputs.json"
    suite = json.loads(suite_path.read_text())
    if suite.get("schema") != "jobpacer-gpu-seven-arm-input-suite" or suite.get("schema_version") != 1:
        raise ValueError("unsupported GPU seven-arm input suite")
    manifest_path = input_dir / "suite-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace a frozen suite manifest: {manifest_path}")
    profile_dir = output_dir / "profiles"
    compute_path = profile_dir / "gpu-compute-profile.json"
    comm_path = profile_dir / "nccl-communication-profile.json"
    compute_sha = _copy_profile(compute_profile_path, compute_path)
    comm_sha = _copy_profile(comm_profile_path, comm_path)
    from examples.jobpacer.gpu.gpu_compute_profile import load_gpu_compute_profile
    from examples.jobpacer.comm_profile import load_profile
    compute_profile = load_gpu_compute_profile(compute_path)
    comm_profile = load_profile(comm_path)
    if int(comm_profile.environment.get("world_size", -1)) != world_size:
        raise ValueError("NCCL communication profile world-size differs from frozen suite")
    if comm_profile.environment.get("backend") != "nccl":
        raise ValueError("seven-arm GPU suite requires an NCCL communication profile")
    if len(compute_profile.devices) != world_size:
        raise ValueError("compute profile must cover exactly the two acquired GPU devices")
    if len(comm_profile.environment.get("device_uuids", [])) != world_size:
        raise ValueError("communication profile must record the two acquired GPU UUIDs")
    compute_uuids = tuple(compute_profile.devices)
    comm_uuids = tuple(comm_profile.environment.get("device_uuids", []))
    if (len(set(compute_uuids)) != world_size or any(not uuid for uuid in compute_uuids)
            or len(set(comm_uuids)) != world_size or any(not uuid for uuid in comm_uuids)
            or set(compute_uuids) != set(comm_uuids)):
        raise ValueError("compute and communication profiles must cover the same two unique GPU UUIDs")
    if compute_profile.raw.get("software", {}).get("matmul_precision") != "highest":
        raise ValueError("seven-arm suite currently freezes matmul_precision=highest")
    compute_sidecar = Path(str(compute_path) + ".manifest.json")
    source_sidecar = Path(str(compute_profile_path) + ".manifest.json")
    if not source_sidecar.is_file():
        raise ValueError("compute profile calibration sidecar is required for a reproducible freeze")
    if source_sidecar.resolve() != compute_sidecar.resolve():
        shutil.copyfile(source_sidecar, compute_sidecar)
    calibration = json.loads(compute_sidecar.read_text())
    calibration_settings = calibration.get("settings", {})
    if calibration.get("profile_sha256") != compute_sha:
        raise ValueError("compute profile calibration sidecar hash does not match the profile")
    if set(calibration.get("device_uuids", [])) != set(compute_uuids):
        raise ValueError("compute profile calibration sidecar GPU UUIDs do not match the profile")
    if (calibration_settings.get("matmul_precision") != "highest"
            or calibration_settings.get("warmup", 0) < 5
            or calibration_settings.get("iterations", 0) < 30):
        raise ValueError("compute profile calibration requires highest precision, warmup >= 5, iterations >= 30")
    calibration_input_hash = verify_compute_profile_inputs(
        calibration, calibration_input_dir or input_dir)
    all_sample_rows = []
    for scenario, scenario_record in suite["scenarios"].items():
        for sample in scenario_record["samples"]:
            input_path = input_dir / sample["path"]
            if _sha256(input_path.read_bytes()) != sample["input_sha256"]:
                raise ValueError(f"frozen input changed: {input_path}")
            dag = load_dag(input_path, world_size=world_size)
            comm_applied = apply_dag_profile(
                dag, comm_profile,
                {"backend": "nccl", "device_type": "cuda", "world_size": world_size},
                strict=True,
            )
            visible_uuids = tuple(compute_profile.devices)
            compute_applied = apply_dag_compute_profile(
                comm_applied, compute_profile, device_uuids=visible_uuids,
                software=compute_profile_software_from_profile(compute_profile), strict=True,
            )
            ltf = build_static_order(compute_applied.graph, "static_ltf",
                                     tails=compute_tails(compute_applied.graph),
                                     extra_predecessors=dag.execution.submit_after)
            fifo = tuple(sample["fifo_sequence"])
            sample_dir = (output_dir / Path(sample["path"]).parent)
            ltf_path = sample_dir / f"workload-{sample['workload_seed']}.ltf-order.json"
            ltf_sha = _write_json(ltf_path, list(ltf))
            row = dict(sample)
            row.update({
                "estimate_view_hash": compute_applied.estimate_view_hash,
                "comm_profile_sha256": comm_sha, "compute_profile_sha256": compute_sha,
                "profiled_fifo_order": list(fifo), "profiled_ltf_order": list(ltf),
                "ltf_order_path": ltf_path.relative_to(output_dir).as_posix(),
                "ltf_order_sha256": ltf_sha,
                "canonical_static_sequence_sha256": _sha256(_json_bytes({"fifo": list(fifo), "ltf": list(ltf)})),
            })
            all_sample_rows.append(row)
    manifest = {
        **suite,
        "status": ("profiled-awaiting-E2-pilot" if suite.get("scope") == "formal"
                   else "frozen-inputs-pilot-awaiting-batch"),
        "source_suite_sha256": _sha256(suite_path.read_bytes()),
        "profiles": {
            "compute": {"path": compute_path.relative_to(output_dir).as_posix(),
                        "sha256": compute_sha, "schema_version": compute_profile.raw.get("schema_version"),
                        "calibration": calibration,
                        "calibration_input_suite_sha256": calibration_input_hash},
            "communication": {"path": comm_path.relative_to(output_dir).as_posix(),
                               "sha256": comm_sha, "schema_version": comm_profile.schema_version,
                               "settings": dict(comm_profile.settings),
                               "environment": dict(comm_profile.environment)},
        },
        "samples": all_sample_rows,
        "arm_contracts": {arm: {"status": "pending-G1" if arm == BARE_ARM else "implemented",
                                 "supported": arm in SUPPORTED_ARMS}
                          for arm in ARMS},
    }
    _write_json(manifest_path, manifest)
    return manifest


def compute_profile_software_from_profile(profile) -> dict[str, Any]:
    # apply_dag_compute_profile compares only fields present in this mapping.
    return dict(profile.software)


def make_order_table(manifest: Mapping[str, Any], *, order_seed: int = ORDER_SEED,
                     repeats: int = 5, arms: tuple[str, ...] = ARMS) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms) or not arms:
        raise ValueError("invalid seven-arm suite arm set")
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("suite contract version differs; regenerate v2 inputs")
    blocks = []
    for sample in manifest.get("samples", []):
        for repeat in range(repeats):
            blocks.append({"scenario": sample["scenario"], "workload_seed": sample["workload_seed"],
                           "repeat": repeat, "sample_path": sample["path"],
                           "fifo_order_path": sample["fifo_order_path"],
                           "ltf_order_path": sample.get("ltf_order_path"),
                           "fifo_sequence": sample.get("profiled_fifo_order", sample.get("fifo_sequence")),
                           "ltf_sequence": sample.get("profiled_ltf_order"),
                           "input_sha256": sample["input_sha256"],
                           "execution_sample_hash": sample["execution_sample_hash"],
                           "estimate_view_hash": sample["estimate_view_hash"],
                           "compute_profile_sha256": sample["compute_profile_sha256"],
                           "comm_profile_sha256": sample["comm_profile_sha256"]})
    expected_blocks = len(SCENARIOS) * len(manifest.get("workload_seeds", [])) * repeats
    if len(blocks) != expected_blocks or len({(b["scenario"], b["workload_seed"], b["repeat"]) for b in blocks}) != len(blocks):
        raise ValueError("suite manifest does not contain a unique complete block product")
    random.Random(order_seed).shuffle(blocks)
    base = list(arms)
    random.Random(f"{order_seed}:arms").shuffle(base)
    planned = []
    for index, block in enumerate(blocks):
        shift = index % len(base)
        ordered_arms = base[shift:] + base[:shift]
        rows = []
        block_key = f"{block['scenario']}-w{block['workload_seed']}-r{block['repeat']}"
        for arm_position, arm in enumerate(ordered_arms):
            rows.append({
                **block,
                "block_id": block_key,
                "arm": arm,
                "arm_position": arm_position,
                "epoch": 0,
                "run_id": f"{block_key}-{arm}",
            })
        planned.append({**block, "block_id": block_key, "arms": rows})
    position_counts = {arm: [0] * len(arms) for arm in arms}
    for block in planned:
        for row in block["arms"]:
            position_counts[row["arm"]][row["arm_position"]] += 1
    run_count = sum(len(block["arms"]) for block in planned)
    return {
        "schema": "jobpacer-gpu-seven-arm-order", "schema_version": 1,
        "suite_manifest_sha256": _sha256(_json_bytes(manifest)),
        "order_seed": order_seed, "repeats_per_sample": repeats,
        "block_count": len(planned), "run_count": run_count,
        "arms": list(arms), "arm_position_counts": position_counts,
        "blocks": planned,
    }


def verify_order_table(order: Mapping[str, Any]) -> None:
    if order.get("schema") != "jobpacer-gpu-seven-arm-order" or order.get("schema_version") != 1:
        raise ValueError("unsupported seven-arm order table")
    if (not order.get("arms") or len(set(order["arms"])) != len(order["arms"])
            or any(arm not in ARMS for arm in order["arms"])):
        raise ValueError("order contains unsupported or duplicate arm IDs")
    blocks = order.get("blocks")
    arms = order.get("arms")
    if not isinstance(blocks, list) or not isinstance(arms, list):
        raise ValueError("seven-arm order requires block and arm arrays")
    seen_runs: set[str] = set()
    seen_blocks: set[str] = set()
    for block in blocks:
        block_id = block.get("block_id")
        if block_id in seen_blocks:
            raise ValueError(f"duplicate block {block_id}")
        seen_blocks.add(block_id)
        run_rows = block.get("arms")
        if not isinstance(run_rows, list) or [row.get("arm") for row in run_rows] not in (arms, list(reversed(arms))):
            # Position balancing permits arbitrary cyclic rotations of the base order.
            if not isinstance(run_rows, list) or len(run_rows) != len(arms) or set(row.get("arm") for row in run_rows) != set(arms):
                raise ValueError(f"block {block_id} does not contain each planned arm exactly once")
        for row in run_rows:
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or run_id in seen_runs:
                raise ValueError(f"invalid or duplicate run ID {run_id!r}")
            seen_runs.add(run_id)
    if len(blocks) != order.get("block_count") or len(seen_runs) != order.get("run_count"):
        raise ValueError("order table declared counts differ from its rows")


def make_estimate_view(dag_path: Path, compute_profile_path: Path, comm_profile_path: Path,
                       *, world_size: int = 2):
    """Load a sample with the same strict profile path used by the replay runner."""
    from examples.jobpacer.gpu.gpu_compute_profile import load_gpu_compute_profile
    from examples.jobpacer.comm_profile import load_profile

    dag = load_dag(dag_path, world_size=world_size)
    comm_profile = load_profile(comm_profile_path)
    dag = apply_dag_profile(dag, comm_profile,
                            {"backend": "nccl", "device_type": "cuda", "world_size": world_size},
                            strict=True)
    compute_profile = load_gpu_compute_profile(compute_profile_path)
    return apply_dag_compute_profile(
        dag, compute_profile, device_uuids=tuple(compute_profile.devices),
        software=dict(compute_profile.software), strict=True,
    )
