"""Opt-in two-rank CUDA event contract test for the Phase 3 executor."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("nccl_semantics_worker.py")


@pytest.mark.skipif(os.environ.get("RUN_JOBPACER_RUNTIME_NCCL") != "1",
                    reason="set RUN_JOBPACER_RUNTIME_NCCL=1 to run the dual-GPU NCCL test")
def test_dual_rank_cuda_producer_consumer_and_unrelated_stream_contract(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nnodes=1", "--nproc_per_node=2", str(WORKER),
               "--output-dir", str(tmp_path)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=90, check=False)
    assert result.returncode == 0, (
        f"NCCL worker failed ({result.returncode})\n"
        f"stdout:\n{result.stdout[-6000:]}\nstderr:\n{result.stderr[-6000:]}"
    )
    ranks = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert len({item["device_uuid"] for item in ranks}) == 2
    for item in ranks:
        assert item["allreduce_correct"] is True
        assert item["receipt_completed"] is True
        assert item["consumer_dependency_returned"] is True
        assert item["unrelated_done_after_enqueue"] is False
        assert item["completion_source"] == "cuda_event_query_after_backend_work_wait"


@pytest.mark.parametrize("policy", ("fifo", "static_fifo"))
@pytest.mark.skipif(os.environ.get("RUN_JOBPACER_RUNTIME_NCCL") != "1",
                    reason="set RUN_JOBPACER_RUNTIME_NCCL=1 to run the dual-GPU NCCL test")
def test_two_rank_phase3_collectives_have_matching_group_order(policy, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    output_path = tmp_path / f"phase3-{policy}.json"
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--dag", str(ROOT / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json"),
               "--backend", "nccl", "--world-size", "2", "--policy", policy,
               "--warmup-iterations", "1", "--setup-timeout", "60", "--timeout", "30",
               "--observation-mode", "diagnostic", "--output", str(output_path)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=90, check=False)
    assert result.returncode == 0, (
        f"phase3 NCCL replay failed ({result.returncode})\n"
        f"stdout:\n{result.stdout[-6000:]}\nstderr:\n{result.stderr[-6000:]}"
    )
    payload = json.loads(output_path.read_text())
    assert payload["validation"]["all_collectives_correct"] is True
    assert payload["validation"]["rank_count"] == 2
    assert len({rank["device_uuid"] for rank in payload["ranks"]}) == 2
    assert payload["ranks"][0]["launch_sequence"] == payload["ranks"][1]["launch_sequence"]
    assert payload["ranks"][0]["task_sequence"] == payload["ranks"][1]["task_sequence"]


@pytest.mark.skipif(os.environ.get("RUN_JOBPACER_RUNTIME_NCCL") != "1",
                    reason="set RUN_JOBPACER_RUNTIME_NCCL=1 to run the dual-GPU NCCL test")
def test_two_rank_bare_uses_common_layered_order_and_correct_collectives(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    env["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"
    output_path = tmp_path / "phase3-bare-multi-group.json"
    dag_path = ROOT / "benchmark/phase3/experiments/dag-semantics/smoke/multi-group.json"
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--dag", str(dag_path), "--comm-engine", "bare", "--policy", "bare",
               "--backend", "nccl", "--world-size", "2", "--warmup-iterations", "1",
               "--setup-timeout", "60", "--timeout", "30", "--output", str(output_path)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=90, check=False)
    assert result.returncode == 0, (
        f"bare NCCL replay failed ({result.returncode})\n"
        f"stdout:\n{result.stdout[-6000:]}\nstderr:\n{result.stderr[-6000:]}"
    )
    payload = json.loads(output_path.read_text())
    ranks = payload["ranks"]
    assert payload["validation"]["all_collectives_correct"] is True
    assert payload["validation"]["rank_count"] == 2
    assert len({rank["device_uuid"] for rank in ranks}) == 2
    for rank in ranks:
        assert rank["comm_engine"] == "bare"
        assert rank["bare_contract"] == "bare-ordered-v2-layered-round-robin"
        assert rank["nccl_launch_order_implicit"] == "1"
        assert rank["bare_launch_sequence"] == rank["bare_default_order"]
        assert rank["completion_source"] == "cuda_event_query_after_backend_work_wait"
    assert ranks[0]["bare_launch_sequence"] == ranks[1]["bare_launch_sequence"]


@pytest.mark.skipif(os.environ.get("RUN_JOBPACER_RUNTIME_NCCL") != "1",
                    reason="set RUN_JOBPACER_RUNTIME_NCCL=1 to run the dual-GPU NCCL test")
def test_two_rank_gpu_dag_waits_for_compute_event_before_dependent_comm(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
    output_path = tmp_path / "phase3-cuda-compute.json"
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3",
               "--dag", str(ROOT / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json"),
               "--backend", "nccl", "--world-size", "2", "--policy", "fifo",
               "--compute-mode", "cuda-matmul", "--compute-matrix-size", "128",
               "--compute-repeats", "1", "--warmup-iterations", "1",
               "--setup-timeout", "60", "--timeout", "30",
               "--observation-mode", "diagnostic", "--output", str(output_path)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=90, check=False)
    assert result.returncode == 0, (
        f"CUDA DAG replay failed ({result.returncode})\n"
        f"stdout:\n{result.stdout[-6000:]}\nstderr:\n{result.stderr[-6000:]}"
    )
    payload = json.loads(output_path.read_text())
    assert payload["validation"]["all_collectives_correct"] is True
    for rank in payload["ranks"]:
        assert rank["compute_mode"] == "cuda-matmul"
        assert rank["gpu_compute_program_count"] == 4
        completed = [event for event in rank["dag_events"] if event.get("kind") == "compute_completed"]
        assert len(completed) == 4
        assert all(event["completion_source"] == "cuda_compute_event_query" for event in completed)
        assert all(event["device_elapsed_ms"] > 0 for event in completed)
