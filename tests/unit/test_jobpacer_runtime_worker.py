from __future__ import annotations

import argparse
import threading
import time
from types import SimpleNamespace

import pytest

from examples.jobpacer import runtime_worker
from examples.jobpacer.workloads import built_workload


class _Runtime:
    def __init__(self):
        self.aborts = []

    def abort(self, error, **details):
        self.aborts.append((error, details))


def test_run_jobs_empty_and_preserves_input_order():
    runtime = _Runtime()
    jobs = [SimpleNamespace(job_id=name) for name in ("first", "second")]
    release_first = threading.Event()
    first_waiting = threading.Event()
    second_finished = threading.Event()

    def run_one(job):
        if job.job_id == "first":
            first_waiting.set()
            release_first.wait(1)
        else:
            second_finished.set()
        return {"job_id": job.job_id}

    assert runtime_worker._run_jobs([], run_one, runtime=runtime, deadline=time.monotonic() + 1,
                                    stop_event=threading.Event(), thread_name_prefix="test") == []
    outcome = {}

    def run_jobs():
        try:
            outcome["results"] = runtime_worker._run_jobs(
                jobs, run_one, runtime=runtime, deadline=time.monotonic() + 1,
                stop_event=threading.Event(), thread_name_prefix="test")
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run_jobs)
    thread.start()
    try:
        assert first_waiting.wait(1)
        assert second_finished.wait(1)
    finally:
        release_first.set()
    thread.join(1)
    assert not thread.is_alive()
    assert "error" not in outcome
    results = outcome["results"]
    assert [result["job_id"] for result in results] == ["first", "second"]
    assert runtime.aborts == []


def test_run_jobs_first_error_wakes_other_jobs_and_aborts_once():
    runtime = _Runtime()
    stop = threading.Event()
    jobs = [SimpleNamespace(job_id=name) for name in ("failed", "peer")]

    def run_one(job):
        if job.job_id == "failed":
            raise ValueError("root failure")
        stop.wait(1)
        return {"job_id": job.job_id}

    with pytest.raises(ValueError, match="root failure"):
        runtime_worker._run_jobs(jobs, run_one, runtime=runtime, deadline=time.monotonic() + 1,
                                 stop_event=stop, thread_name_prefix="test")
    assert stop.is_set()
    assert len(runtime.aborts) == 1
    assert runtime.aborts[0][1]["job_id"] == "failed"


def test_run_jobs_uses_one_join_deadline_for_all_threads(monkeypatch):
    runtime = _Runtime()
    jobs = [SimpleNamespace(job_id=name) for name in ("first", "second")]

    def run_one(job):
        return {"job_id": job.job_id}

    now = [100.0]
    monkeypatch.setattr(runtime_worker.time, "monotonic", lambda: now[0])
    join_budgets = []
    real_join = threading.Thread.join

    def controlled_join(thread, timeout=None):
        join_budgets.append(timeout)
        real_join(thread, timeout=1)
        if len(join_budgets) == 1:
            now[0] += 6

    monkeypatch.setattr(threading.Thread, "join", controlled_join)
    results = runtime_worker._run_jobs(
        jobs, run_one, runtime=runtime, deadline=now[0] + 10,
        stop_event=threading.Event(), thread_name_prefix="test")
    assert [result["job_id"] for result in results] == ["first", "second"]
    assert join_budgets == pytest.approx([10, 4])


def test_new_groups_destroys_partial_group_creation(monkeypatch):
    created = object()
    destroyed = []
    outcomes = iter((created, RuntimeError("group creation failed")))

    def new_group(_ranks):
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(runtime_worker.dist, "new_group", new_group)
    monkeypatch.setattr(runtime_worker.dist, "destroy_process_group", lambda group: destroyed.append(group))
    workload = built_workload("balanced")
    with pytest.raises(RuntimeError, match="group creation failed"):
        runtime_worker._new_groups(workload, 2)
    assert destroyed == [created]


def test_run_rank_closes_groups_when_initialization_fails(monkeypatch):
    groups = (object(), object())
    destroyed = []
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "12345")
    monkeypatch.setattr(runtime_worker.dist, "init_process_group", lambda *args, **kwargs: None)
    group_iter = iter(groups)
    monkeypatch.setattr(runtime_worker.dist, "new_group", lambda _ranks, **_kwargs: next(group_iter))
    def fail_barrier():
        raise RuntimeError("barrier failed")
    monkeypatch.setattr(runtime_worker.dist, "barrier", fail_barrier)
    monkeypatch.setattr(runtime_worker.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime_worker.dist, "destroy_process_group", lambda *args: destroyed.append(args))
    args = argparse.Namespace(timeout=1.0, setup_timeout=1.0, poll_interval=0.01, dag_poll_interval=0.01,
                              compute_jitter=0.0, dag=None, workload="balanced", static_order=None,
                              policy="fifo", backend="gloo", epoch=0, control_port=12345, fault="none")
    with pytest.raises(RuntimeError, match="barrier failed"):
        runtime_worker.run_rank(args)
    assert destroyed == [(groups[1],), (groups[0],), ()]


@pytest.mark.parametrize("job_fails", (False, True))
def test_run_rank_keeps_one_replay_deadline_through_finish_and_skips_finish_after_failure(
    monkeypatch, job_fails
):
    from runtime_comm_scheduler.runtime import EventLog
    from runtime_comm_scheduler.runtime import coordinator as coordinator_module
    from runtime_comm_scheduler.runtime import transport as transport_module

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "12345")
    groups = [object(), object()]
    monkeypatch.setattr(runtime_worker.dist, "init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_worker.dist, "new_group", lambda *args, **kwargs: groups.pop(0))
    monkeypatch.setattr(runtime_worker.dist, "barrier", lambda: None)
    monkeypatch.setattr(runtime_worker.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime_worker.dist, "destroy_process_group", lambda *args: None)

    now = [100.0]
    monkeypatch.setattr(runtime_worker.time, "monotonic", lambda: now[0])
    captured = {}

    class FakeCoordinator:
        def __init__(self, _endpoints, **kwargs):
            self.epoch_timeout_s = kwargs["epoch_timeout_s"]
            self.records = []
            captured["coordinator"] = self

    class FakeServer:
        def __init__(self, coordinator, *_args):
            self.coordinator = coordinator

        def start(self):
            pass

        def wait_ready(self, timeout):
            captured["server_ready_timeout"] = timeout

        def close(self):
            pass

    class FakeRuntime:
        def __init__(self, *_args, **_kwargs):
            self.failure = None
            self.event_log = EventLog("runtime", 0)
            self.grant_order = []
            self.launch_order = []
            self.finish_timeouts = []
            self.aborts = []
            captured["runtime"] = self

        def register_group(self, *_args):
            pass

        def start(self, timeout):
            captured["runtime_start_timeout"] = timeout
            now[0] += 0.25  # setup time must not consume the replay budget

        def finish_epoch(self, timeout):
            self.finish_timeouts.append(timeout)

        def abort(self, error, **details):
            self.aborts.append((error, details))
            self.failure = error

        def close(self):
            pass

    monkeypatch.setattr(coordinator_module, "CoordinatorState", FakeCoordinator)
    monkeypatch.setattr(transport_module, "CoordinatorServer", FakeServer)
    monkeypatch.setattr(runtime_worker, "ControlClient", lambda *args: object())
    monkeypatch.setattr(runtime_worker, "RankRuntime", FakeRuntime)

    def run_jobs(_jobs, _run_one, *, deadline, runtime, **_kwargs):
        captured["replay_deadline"] = deadline
        if job_fails:
            error = ValueError("job failed")
            runtime.abort(error, stage="job")
            raise error
        now[0] += 0.75
        return []

    monkeypatch.setattr(runtime_worker, "_run_jobs", run_jobs)
    args = argparse.Namespace(timeout=2.0, setup_timeout=1.0, poll_interval=0.01,
                              dag_poll_interval=0.01, compute_jitter=0.0, dag=None,
                              workload="balanced", static_order=None, policy="fifo",
                              backend="gloo", epoch=0, control_port=12345, fault="none")

    if job_fails:
        with pytest.raises(ValueError, match="job failed"):
            runtime_worker.run_rank(args)
    else:
        runtime_worker.run_rank(args)
    assert captured["coordinator"].epoch_timeout_s == pytest.approx(3.0)
    assert captured["runtime_start_timeout"] == pytest.approx(1.0)
    assert captured["server_ready_timeout"] == pytest.approx(0.75)
    assert captured["replay_deadline"] == pytest.approx(102.25)
    if job_fails:
        assert captured["runtime"].finish_timeouts == []
    else:
        # The runner consumed 0.75s; finish gets only the remainder of the same deadline.
        assert now[0] == pytest.approx(101.0)
        assert captured["runtime"].finish_timeouts == pytest.approx([1.25])
