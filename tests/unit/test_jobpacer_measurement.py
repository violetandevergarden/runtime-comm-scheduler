"""Regression checks for JobPacer's event and experiment semantics."""

from __future__ import annotations

import json
import sys
import threading
import time
import hashlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[2] / "benchmark/phase1.2"))

from examples.jobpacer import replay_worker
from examples.jobpacer.plan_builder import ltf_score, planned_tasks, policy_diagnostics
from examples.jobpacer.visualize import timeline_data, trace_paths, validate_trace
from examples.jobpacer.workloads import CollectiveComm, Job, Workload
from runtime_comm_scheduler import (
    AdmissionScheduler,
    CommIntent,
    DirectLaunchExecutor,
    Plan,
    TaskKey,
)


class FakeWork:
    def __init__(self, done: bool = False):
        self.done = threading.Event()
        if done:
            self.done.set()

    def wait(self, timeout=None):
        seconds = timeout.total_seconds() if hasattr(timeout, "total_seconds") else timeout
        return self.done.wait(seconds)

    def is_completed(self):
        return self.done.is_set()


def _key(ordinal=0):
    return TaskKey(0, 0, "dp", "job", 0, 0, ordinal)


def test_scheduled_wait_separates_deferred_binding_from_backend_wait():
    key = _key()
    plan = Plan(0, 0, ((key, "all_reduce", 4),))
    launch_entered = threading.Event()
    release_launch = threading.Event()
    underlying = FakeWork()

    def launch():
        launch_entered.set()
        release_launch.wait()
        return underlying

    scheduler = AdmissionScheduler(
        plan,
        local_group_ids={"job"},
        executor=DirectLaunchExecutor(),
        completion_poll_interval_s=0.001,
    )
    work = scheduler.submit(
        CommIntent(key, "all_reduce", None, None, 4, launch_fn=launch)
    )
    assert launch_entered.wait(1)
    waiter = threading.Thread(target=work.wait)
    waiter.start()
    deadline = time.monotonic() + 1
    while work.timing.application_wait_start_ts is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert work.timing.application_wait_start_ts is not None
    assert work.timing.underlying_wait_start_ts is None
    release_launch.set()
    deadline = time.monotonic() + 1
    while work.timing.underlying_wait_start_ts is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert work.timing.underlying_wait_start_ts >= work.timing.application_wait_start_ts
    underlying.done.set()
    waiter.join(1)
    scheduler.finish_window(timeout=1)
    assert not waiter.is_alive()
    assert work.timing.wait_return_ts >= work.timing.underlying_wait_start_ts
    assert work.timing.deferred_binding_wait_us >= 0
    scheduler.close()


def test_bare_completion_observer_records_completion_once(monkeypatch):
    class Probe:
        def is_completed(self, work):
            return work.is_completed()

    monkeypatch.setattr(replay_worker, "WorkIsCompletedProbe", Probe)
    observer = replay_worker._CompletionObserver(0.001, lambda _error: None)
    work = FakeWork()
    record = observer.register("key", work, replay_worker._now_us())
    work.done.set()
    observer.finish(1)
    assert record["completion_observed_ts"] is not None
    assert record["actual_duration_us"] >= 0
    assert not observer._thread.is_alive()


def test_bare_completion_observer_timeout_is_bounded(monkeypatch):
    class Probe:
        def is_completed(self, _work):
            return False

    monkeypatch.setattr(replay_worker, "WorkIsCompletedProbe", Probe)
    observer = replay_worker._CompletionObserver(0.001, lambda _error: None)
    observer.register("key", FakeWork(), replay_worker._now_us())
    with pytest.raises(TimeoutError, match="timed out observing"):
        observer.finish(0.01)
    assert not observer._thread.is_alive()


def test_bare_completion_observer_propagates_probe_error(monkeypatch):
    class Probe:
        def is_completed(self, _work):
            raise RuntimeError("probe failed")

    observed = []
    monkeypatch.setattr(replay_worker, "WorkIsCompletedProbe", Probe)
    observer = replay_worker._CompletionObserver(0.001, observed.append)
    observer.register("key", FakeWork(), replay_worker._now_us())
    with pytest.raises(RuntimeError, match="probe failed"):
        observer.finish(1)
    assert observed and not observer._thread.is_alive()


@pytest.mark.parametrize(
    ("consumer", "expected"), ((0.005, 0.032), (0.02, 0.042))
)
def test_ltf_score_models_communication_compute_overlap(consumer, expected):
    workload = Workload(
        "score",
        (
            Job(
                "job",
                (
                    CollectiveComm(0, consumer_compute_s=consumer, estimated_comm_s=0.01),
                    CollectiveComm(
                        1,
                        producer_compute_s=0.002,
                        consumer_compute_s=0.02,
                        estimated_comm_s=0.003,
                    ),
                ),
            ),
        ),
    )
    task = planned_tasks(workload, "ltf")
    first, second = sorted(task.values(), key=lambda item: item.communication.id)
    scores = (first, second)
    assert ltf_score(tuple(scores), 0) == pytest.approx(expected)


def test_ltf_diagnostics_include_candidate_scores_and_stable_tie_break():
    workload = Workload(
        "tie",
        (
            Job("b", (CollectiveComm(0, estimated_comm_s=0.01),)),
            Job("a", (CollectiveComm(0, estimated_comm_s=0.01),)),
        ),
    )
    result = policy_diagnostics(workload, "ltf")
    assert result["score_definition"].startswith("zero-admission-delay")
    assert result["steps"][0]["selected_key"][3] == "a"
    assert [item["score"] for item in result["steps"][0]["candidates"]] == [0.01, 0.01]


def _valid_trace():
    key = [0, 0, "jobpacer", "job", 0, 0, 0]
    task = {
        "key": key,
        "ordinal": 0,
        "producer_compute_start_ts": 101,
        "ready_record_ts": 110,
        "submit_api_start_ts": 111,
        "submit_api_return_ts": 112,
        "admit_ts": 112,
        "collective_call_start_ts": 112,
        "collective_call_return_ts": 115,
        "completion_observed_ts": 120,
        "consumer_compute_start_ts": 112,
        "consumer_compute_end_ts": 114,
        "application_wait_start_ts": 116,
        "underlying_wait_start_ts": 116,
        "wait_return_ts": 117,
        "application_task_end_ts": 117,
        "validation_start_ts": 211,
        "validation_end_ts": 220,
    }
    rank = {
        "rank": 0,
        "trace_schema_version": 2,
        "application_release_ts": 100,
        "application_end_ts": 200,
        "communication_drain_end_ts": 210,
        "validation_end_ts": 240,
        "harness_start_ts": 240,
        "harness_end_ts": 250,
        "application_makespan_us": 100,
        "communication_drain_makespan_us": 110,
        "max_outstanding": 1,
        "selection": "runtime_arrival",
        "plan": {"keys": [key]},
        "control": {},
        "jobs": [{"job_id": "job", "tasks": [task]}],
    }
    return {"ranks": [rank]}


def test_timeline_validates_canonical_boundaries_and_uses_rank_origin():
    trace = _valid_trace()
    data = timeline_data(trace, 0)
    assert data["origin_ts"] == 100
    assert {item["kind"] for item in data["jobs"][0]["tasks"][0]["intervals"]} == {
        "producer_compute",
        "consumer_compute",
        "admission_wait",
        "communication",
        "application_binding_wait",
        "backend_wait",
        "validation",
    }


def test_timeline_fails_closed_for_missing_or_reversed_timestamps():
    trace = _valid_trace()
    del trace["ranks"][0]["jobs"][0]["tasks"][0]["underlying_wait_start_ts"]
    with pytest.raises(ValueError, match="underlying_wait_start_ts"):
        validate_trace(trace, "missing.json")
    trace = _valid_trace()
    trace["ranks"][0]["jobs"][0]["tasks"][0]["completion_observed_ts"] = 114
    with pytest.raises(ValueError, match="reversed"):
        validate_trace(trace, "reversed.json")


def test_trace_selection_uses_only_complete_manifest_entries(tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir()
    listed = batch / "raw/workload/fifo/run-00.json"
    listed.parent.mkdir(parents=True)
    listed.write_text("{}")
    (listed.parent / "run-99.json").write_text("stale")
    (batch / "manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "runs": [
                    {
                        "workload": "workload",
                        "scenario": "fifo",
                        "path": str(listed.relative_to(batch)),
                        "sha256": hashlib.sha256(listed.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    assert trace_paths(batch, "workload", "fifo") == [listed]
