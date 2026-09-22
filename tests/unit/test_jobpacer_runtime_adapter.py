from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from examples.jobpacer.runtime_adapter import (
    linear_static_order,
    load_dag,
    make_collective_binding,
    make_replay_compute,
    parse_dag,
    sample_compute_duration,
)
from examples.jobpacer.plan_builder import build_plan
from examples.jobpacer.workloads import built_workload
from runtime_comm_scheduler.dag import ComputeNode
from runtime_comm_scheduler.runtime import CollectiveSpec, EventLog, TaskSpec


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(("name", "digest"), [
    ("diamond", "eca9e79f4ca656be0226c92da2793d2413b0c36aaca190d1336ca2436915fe24"),
    ("linear", "addc43a85e8116336ed3e4e17eee03887ccc77ad0063197d36b894e0e35cb9b7"),
    ("multi-group", "e3fa6b60b22a1e9bda8e1277968c12d01d3e8206646691e0f95afe2a5db9431a"),
])
def test_existing_dag_canonical_digests_are_unchanged(name, digest):
    parsed = load_dag(ROOT / "benchmark/phase3" / f"{name}.json", world_size=2)
    assert parsed.manifest_digest == digest
    assert parse_dag(json.loads(parsed.canonical_json)).manifest_digest == digest


def test_parse_rejects_unsupported_reduction_before_runtime_start():
    raw = json.loads((ROOT / "benchmark/phase3/linear.json").read_text())
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


def test_replay_compute_owns_sampled_duration_and_sampling_event():
    dag = load_dag(ROOT / "benchmark/phase3/linear.json", world_size=2)
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
