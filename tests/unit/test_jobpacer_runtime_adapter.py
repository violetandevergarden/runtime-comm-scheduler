from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from examples.jobpacer.runtime.runtime_adapter import (
    apply_dag_compute_profile,
    linear_static_order,
    remaining_tail,
    task_hint,
    load_dag,
    make_collective_binding,
    make_replay_compute,
    parse_dag,
    sample_compute_duration,
)
from examples.jobpacer.runtime.plan_builder import build_plan
from examples.jobpacer.workloads import built_workload
from runtime_comm_scheduler.dag import ComputeNode
from runtime_comm_scheduler.runtime import CollectiveSpec, EventLog, TaskSpec


ROOT = Path(__file__).resolve().parents[2]
DAG_SMOKE = ROOT / "benchmark/phase3/experiments/dag-semantics/smoke"


@pytest.mark.parametrize(("name", "digest"), [
    ("diamond", "eca9e79f4ca656be0226c92da2793d2413b0c36aaca190d1336ca2436915fe24"),
    ("linear", "addc43a85e8116336ed3e4e17eee03887ccc77ad0063197d36b894e0e35cb9b7"),
    ("multi-group", "e3fa6b60b22a1e9bda8e1277968c12d01d3e8206646691e0f95afe2a5db9431a"),
])
def test_existing_dag_canonical_digests_are_unchanged(name, digest):
    parsed = load_dag(DAG_SMOKE / f"{name}.json", world_size=2)
    assert parsed.manifest_digest == digest
    assert parse_dag(json.loads(parsed.canonical_json)).manifest_digest == digest


def test_parse_rejects_unsupported_reduction_before_runtime_start():
    raw = json.loads((DAG_SMOKE / "linear.json").read_text())
    collective = next(node["collective"] for job in raw["jobs"] for node in job["nodes"]
                      if node["kind"] == "comm")
    collective["reduction"] = "max"
    with pytest.raises(ValueError, match="reduction.*unsupported"):
        parse_dag(raw, world_size=2)


def test_compute_sampling_and_historical_linear_order_remain_stable():
    sample = sample_compute_duration(7, 3, "job", "compute", 0, 2.0, 0.5)
    assert sample == sample_compute_duration(7, 3, "job", "compute", 0, 2.0, 0.5)
    assert 1.0 <= sample <= 3.0
    workload = built_workload("tail")
    for policy, old_policy in (("static_fifo", "fifo"), ("static_ltf", "ltf")):
        expected = tuple(f"{key.process_group_id}/comm-{key.ordinal}"
                         for key in build_plan(workload, old_policy).keys)
        assert linear_static_order(workload, policy) == expected
    with pytest.raises(ValueError, match="requires static"):
        linear_static_order(workload, "fifo")


def test_linear_postcompletion_tail_overlap_estimate_and_hint():
    from examples.jobpacer.workloads import CollectiveComm, Job, Workload

    job = Job("scores", (
        CollectiveComm(0, estimated_comm_s=2.0, consumer_compute_s=5.0),
        CollectiveComm(1, producer_compute_s=4.0, estimated_comm_s=3.0, consumer_compute_s=2.0),
        CollectiveComm(2, producer_compute_s=1.0, estimated_comm_s=0.0, consumer_compute_s=0.0),
    ))
    workload = Workload("scores", (job,))
    assert remaining_tail(job, 0) == pytest.approx(11.0)
    assert remaining_tail(job, 1) == pytest.approx(1.0)
    assert remaining_tail(job, 2) == 0.0  # u == c == 0
    hint = task_hint(job, 0)
    assert hint.remaining_tail_s == pytest.approx(11.0)
    assert set(hint.to_dict()) == {"ready_after_s", "estimated_comm_s", "remaining_tail_s"}
    assert linear_static_order(workload, "static_ltf") == tuple(
        f"scores/comm-{index}" for index in range(3)
    )


def test_replay_compute_owns_sampled_duration_and_sampling_event():
    dag = load_dag(DAG_SMOKE / "linear.json", world_size=2)
    job = dag.graph.jobs[0]
    node = next(node for node in job.nodes if isinstance(node, ComputeNode))
    samples = {}
    events = EventLog("dag", 0)
    stop = threading.Event()
    stop.set()
    compute = make_replay_compute(job, dag.execution, seed=dag.seed, epoch=0, rank=0,
                                  jitter=0.0, samples=samples, event_log=events)
    compute(node, stop)
    assert samples[node.node_id] == dag.execution.compute_duration_s[f"{job.job_id}/{node.node_id}"]
    assert events.events[0]["kind"] == "compute_sampled"
    assert events.events[0]["sampled_duration_s"] == samples[node.node_id]


def test_collective_binding_obeys_shape_dtype_and_rejects_unsupported_reduction():
    torch = pytest.importorskip("torch")
    group = object()
    collective = CollectiveSpec("all_reduce", 4, 16, "float32", (2, 2))
    spec = TaskSpec(0, "job", "job/comm", "g", 0, collective)
    binding = make_collective_binding(spec, group, rank=1, device="cpu")
    assert binding.process_group is group
    assert tuple(binding.tensor.shape) == collective.shape
    assert binding.tensor.dtype == torch.float32
    unsupported = TaskSpec(0, "job", "job/comm", "g", 0,
                           CollectiveSpec("all_reduce", 4, 16, "float32", (2, 2), "max"))
    with pytest.raises(ValueError, match="not supported"):
        make_collective_binding(unsupported, group, rank=1, device="cpu")


def test_formal_dag_import_has_no_replay_or_torch_dependency():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys; import runtime_comm_scheduler.dag; "
        "assert 'torch' not in sys.modules; "
        "assert not any(name == 'examples' or name.startswith('examples.') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True, timeout=10)


def _v2_gpu_dag():
    return {
        "schema_version": 2, "name": "v2", "seed": 19,
        "groups": [{"group_id": "g0", "ranks": [0, 1]}],
        "jobs": [{"job_id": "job-0", "nodes": [
            {"node_id": "p", "kind": "compute", "deps": [], "estimated_duration_s": 0.01},
            {"node_id": "c", "kind": "comm", "deps": ["p"], "group_id": "g0", "group_seq": 0,
             "estimated_comm_s": 0.02,
             "collective": {"op": "all_reduce", "shape": [2, 2], "numel": 4,
                            "num_bytes": 16, "dtype": "float32", "reduction": "sum"}},
            {"node_id": "u", "kind": "compute", "deps": ["p"], "estimated_duration_s": 0.03},
            {"node_id": "d", "kind": "compute", "deps": ["c", "u"], "estimated_duration_s": 0.04},
        ]}],
        "execution": {
            "mode": "cuda-program", "sample_id": "fixed-19",
            "compute_model": "one-active-compute-per-job",
            "buffers": {"job-0": {
                "a": {"shape": [2, 2], "dtype": "float32", "init": "seeded-random"},
                "b": {"shape": [2, 2], "dtype": "float32", "init": "seeded-random"},
                "x": {"shape": [2, 2], "dtype": "float32", "init": "empty"},
                "y": {"shape": [2, 2], "dtype": "float32", "init": "empty"},
                "z": {"shape": [], "dtype": "float32", "init": "empty"},
            }},
            "compute_programs": {
                "job-0/p": {"op": "matmul", "inputs": ["a", "b"], "output": "x", "repeats": 2},
                "job-0/u": {"op": "matmul", "inputs": ["a", "b"], "output": "y", "repeats": 3},
                "job-0/d": {"op": "sum_join", "inputs": ["x", "y"], "output": "z"},
            },
            "comm_bindings": {"job-0/c": {"buffer": "x"}},
            "submit_after": {"job-0/u": ["job-0/c"]},
        },
    }


def test_v2_gpu_dag_normalizes_program_buffers_and_submit_gate_without_epoch_identity():
    first = parse_dag(_v2_gpu_dag(), epoch=0, world_size=2)
    second = parse_dag(_v2_gpu_dag(), epoch=7, world_size=2)
    assert first.execution.schema_version == 2
    assert first.execution.compute_programs["job-0/p"]["op"] == "matmul"
    assert first.execution.comm_bindings == {"job-0/c": {"buffer": "x"}}
    assert first.execution.submit_after == {"job-0/u": ("job-0/c",)}
    assert first.manifest_digest == second.manifest_digest
    assert first.estimate_view_hash == second.estimate_view_hash
    assert parse_dag(json.loads(first.canonical_json), world_size=2).manifest_digest == first.manifest_digest


def test_v2_gpu_dag_rejects_incomplete_program_mapping_and_unordered_buffer_hazard():
    missing_program = _v2_gpu_dag()
    del missing_program["execution"]["compute_programs"]["job-0/u"]
    with pytest.raises(ValueError, match="compute_programs keys must exactly cover"):
        parse_dag(missing_program, world_size=2)

    unordered = _v2_gpu_dag()
    node = unordered["jobs"][0]["nodes"][-1]
    node["deps"] = ["c"]  # reads y even though its writer u is unordered
    with pytest.raises(ValueError, match="unordered read/write access"):
        parse_dag(unordered, world_size=2)


@pytest.mark.parametrize(("node_id", "program"), [
    ("p", {"op": "matmul", "inputs": ["a", "b"], "output": "a", "repeats": 2}),
    ("d", {"op": "sum_join", "inputs": ["z"], "output": "z"}),
])
def test_v2_gpu_dag_rejects_compute_input_output_alias(node_id, program):
    raw = _v2_gpu_dag()
    raw["execution"]["compute_programs"][f"job-0/{node_id}"] = program

    with pytest.raises(ValueError, match="inputs and output must use distinct buffers"):
        parse_dag(raw, world_size=2)


def test_v2_submit_after_rejects_completion_dependency_and_future_predecessors():
    completion_gate = _v2_gpu_dag()
    completion_gate["execution"]["submit_after"] = {"job-0/d": ["job-0/c"]}
    with pytest.raises(ValueError, match="already a completion predecessor"):
        parse_dag(completion_gate, world_size=2)

    future_predecessor = _v2_gpu_dag()
    nodes = future_predecessor["jobs"][0]["nodes"]
    nodes[2]["deps"] = []
    nodes[3]["deps"] = ["c", "u"]
    future_predecessor["execution"]["submit_after"] = {"job-0/u": ["job-0/c"]}
    with pytest.raises(ValueError, match="completion predecessors absent"):
        parse_dag(future_predecessor, world_size=2)


def test_v2_compute_profile_uses_one_shared_estimate_across_rank_devices(tmp_path):
    from examples.jobpacer.runtime.gpu_compute_profile import (
        dag_compute_profile_signature,
        load_gpu_compute_profile,
    )

    dag = parse_dag(_v2_gpu_dag(), world_size=2)
    records = []
    for key, program in dag.execution.compute_programs.items():
        signature = dag_compute_profile_signature(program, dag.execution.buffers["job-0"])
        records.append({"stage": "compute", "signature": signature,
                        "device_event_ms_samples": [0.2], "host_enqueue_us_samples": [2.0],
                        "preparation_us": 3.0})
    raw = {
        "schema": "jobpacer-gpu-compute-profile", "schema_version": 3,
        "software": {"pytorch_version": "2.x", "cuda_version": "12.x",
                     "matmul_precision": "highest", "allow_tf32": False},
        "devices": [{"device_uuid": uuid, "records": [dict(record,
                      device_event_ms_samples=[duration + index * 0.1]) for record, duration in
                      zip(records, (0.2, 0.3, 0.4))]}
                     for index, uuid in enumerate(("GPU-0", "GPU-1"))],
    }
    profile_path = tmp_path / "dag-profile.json"
    profile_path.write_text(json.dumps(raw))
    profile = load_gpu_compute_profile(profile_path)
    estimated = apply_dag_compute_profile(
        dag, profile, device_uuids=("GPU-0", "GPU-1"), software=raw["software"])
    estimates = {f"{job.job_id}/{node.node_id}": node.estimated_duration_s
                 for job in estimated.graph.jobs for node in job.nodes if isinstance(node, ComputeNode)}
    assert sorted(estimates.values()) == pytest.approx([0.0003, 0.0004, 0.0005])
    assert estimated.input_hash == dag.input_hash
    assert estimated.estimate_view_hash != dag.estimate_view_hash
    assert estimated.execution.submit_after == dag.execution.submit_after
