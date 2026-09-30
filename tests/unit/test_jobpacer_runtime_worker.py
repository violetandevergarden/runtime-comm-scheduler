from __future__ import annotations

import sys
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.jobpacer.runtime import runtime_worker


def test_worker_drains_epoch_before_deferred_tensor_validation(monkeypatch):
    events = []

    class Runtime:
        def finish_epoch(self, timeout):
            assert timeout > 0
            events.append("drain")

    monkeypatch.setattr(
        runtime_worker.resource,
        "getrusage",
        lambda _who: events.append("drain-metrics") or "usage-snapshot",
    )
    monkeypatch.setattr(
        runtime_worker,
        "_validate_deferred",
        lambda _records: events.append("validation") or (100, 200),
    )

    result = runtime_worker._finish_epoch_then_validate(
        Runtime(), [], deadline=runtime_worker.time.monotonic() + 1.0)

    assert events == ["drain", "drain-metrics", "validation"]
    assert result[0] > 0
    assert result[2] == "usage-snapshot"
    assert result[3:] == (100, 200)


def test_job_failure_passes_replay_deadline_to_adapter_abort():
    observed = {}

    class Runtime:
        def abort(self, error, **context):
            observed.update(context)

    deadline = time.monotonic() + 2.0
    with pytest.raises(RuntimeError, match="injected"):
        runtime_worker._run_jobs(
            [SimpleNamespace(job_id="job-0")],
            lambda _job: (_ for _ in ()).throw(RuntimeError("injected")),
            runtime=Runtime(), deadline=deadline, stop_event=threading.Event(),
            thread_name_prefix="test",
        )

    assert observed["deadline"] == deadline


def test_runtime_worker_rejects_direct_nccl_linear_replay_before_setup(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    args = SimpleNamespace(
        comm_engine="new", timeout=1.0, setup_timeout=1.0,
        poll_interval=0.001, dag_poll_interval=0.001,
        backend="nccl", dag=None,
    )
    with pytest.raises(ValueError, match="accepts DAG inputs only"):
        runtime_worker.run_rank(args)


def test_runtime_worker_rejects_direct_nccl_schema1_dag_before_setup(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    root = Path(__file__).parents[2]
    args = SimpleNamespace(
        comm_engine="new", timeout=1.0, setup_timeout=1.0,
        poll_interval=0.001, dag_poll_interval=0.001,
        backend="nccl", dag=root / "benchmark/phase3/experiments/dag-semantics/smoke/linear.json",
        epoch=0, compute_jitter=0.0, comm_profile=None, compute_profile=None,
        profile_strict=True, wait_budget_s=0.02, policy="fifo",
    )
    with pytest.raises(ValueError, match="schema-v2 cuda-program DAG"):
        runtime_worker.run_rank(args)
