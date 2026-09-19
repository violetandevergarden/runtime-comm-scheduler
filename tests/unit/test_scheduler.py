"""M4.5 worker-only scheduler, Work semantics, and failure tests."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from datetime import timedelta

import pytest

from runtime_comm_scheduler import (
    AdmissionScheduler,
    CommIntent,
    DirectLaunchExecutor,
    IntentState,
    Plan,
    SchedulerClosedError,
    SchedulerFailedError,
    TaskKey,
    WindowIncompleteError,
)
from runtime_comm_scheduler.validate import ValidationError


class FakeWork:
    """A physically-completing CPU-like Work used by the scheduler tests."""

    def __init__(self):
        self._done = threading.Event()
        self.wait_timeouts: list[float | None] = []

    def complete(self) -> None:
        self._done.set()

    def wait(self, timeout=None) -> bool:
        seconds = timeout.total_seconds() if isinstance(timeout, timedelta) else timeout
        self.wait_timeouts.append(seconds)
        return self._done.wait(seconds)

    def is_completed(self) -> bool:
        return self._done.is_set()


class StreamOrderedFakeWork(FakeWork):
    """Model NCCL wait: dependency insertion succeeds before physical completion."""

    def wait(self, timeout=None) -> bool:
        seconds = timeout.total_seconds() if isinstance(timeout, timedelta) else timeout
        self.wait_timeouts.append(seconds)
        return True


def _key(ordinal=0, process_group_id="dp"):
    return TaskKey(
        iteration=0,
        microbatch=0,
        parallelism="dp",
        process_group_id=process_group_id,
        layer_id=0,
        bucket_id=0,
        ordinal=ordinal,
    )


def _plan(keys, op="all_reduce", num_bytes=8):
    return Plan(version=0, window_id=0, entries=tuple((k, op, num_bytes) for k in keys))


def _intent(key, launch, op="all_reduce", num_bytes=8, ready_event=None):
    return CommIntent(
        key=key,
        op=op,
        tensor=None,
        process_group=None,
        num_bytes=num_bytes,
        launch_fn=launch,
        ready_event=ready_event,
    )


def _scheduler(plan, **kwargs):
    groups = kwargs.pop("local_group_ids", plan.group_ids())
    return AdmissionScheduler(
        plan,
        local_group_ids=groups,
        executor=DirectLaunchExecutor(),
        completion_poll_interval_s=0.001,
        **kwargs,
    )


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError(f"condition not met within {timeout}s")


def _make_work(created, works, key, work_type=FakeWork):
    work = work_type()
    created.append((key, threading.get_ident()))
    works[key] = work
    return work


def test_submit_never_launches_on_caller_thread_and_fifo_order():
    k0, k1 = _key(0), _key(1)
    sched = _scheduler(_plan([k0, k1]))
    caller = threading.get_ident()
    created, works = [], {}
    w0 = sched.submit(_intent(k0, lambda: _make_work(created, works, k0)))
    w1 = sched.submit(_intent(k1, lambda: _make_work(created, works, k1)))
    _wait_for(lambda: len(created) == 2)
    assert [key for key, _thread in created] == [k0, k1]
    assert {thread for _key_, thread in created} == {sched._thread.ident}
    assert sched._thread.ident != caller
    works[k0].complete()
    works[k1].complete()
    sched.finish_window(timeout=1)
    assert w0.is_completed() and w1.is_completed()
    assert sched.sequence_log() == [k0, k1]
    assert sched.group_sequence_log() == {"dp": [k0, k1]}
    sched.close()


def test_single_worker_never_overlaps_launcher_calls():
    keys = [_key(index) for index in range(4)]
    scheduler = _scheduler(_plan(keys))
    active = 0
    maximum_active = 0
    guard = threading.Lock()
    works = {}

    def launch(key):
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.005)
        work = FakeWork()
        works[key] = work
        with guard:
            active -= 1
        return work

    returned = [
        scheduler.submit(_intent(key, lambda key=key: launch(key)))
        for key in keys
    ]
    _wait_for(lambda: len(works) == len(keys))
    assert maximum_active == 1
    for underlying in works.values():
        underlying.complete()
    scheduler.finish_window(timeout=1)
    assert all(work.is_completed() for work in returned)
    scheduler.close()


def test_rank_wide_order_is_preserved_across_groups():
    a0, b0, a1 = _key(0, "a"), _key(0, "b"), _key(1, "a")
    sched = _scheduler(_plan([a0, b0, a1]))
    created, works = [], {}
    returned = [
        sched.submit(_intent(a1, lambda: _make_work(created, works, a1))),
        sched.submit(_intent(b0, lambda: _make_work(created, works, b0))),
        sched.submit(_intent(a0, lambda: _make_work(created, works, a0))),
    ]
    _wait_for(lambda: len(created) == 3)
    assert [key for key, _thread in created] == [a0, b0, a1]
    assert sched.sequence_log() == [a0, b0, a1]
    assert sched.group_sequence_log() == {"a": [a0, a1], "b": [b0]}
    for work in works.values():
        work.complete()
    sched.finish_window(timeout=1)
    assert all(work.is_completed() for work in returned)
    sched.close()


def test_local_projection_skips_non_member_groups_and_rejects_their_intents():
    dp, tp = _key(0, "dp"), _key(0, "tp")
    plan = _plan([dp, tp])
    sched = _scheduler(plan, local_group_ids={"dp"})
    work = FakeWork()
    scheduled = sched.submit(_intent(dp, lambda: work))
    with pytest.raises(ValidationError, match="not in this rank's plan projection"):
        sched.submit(_intent(tp, FakeWork))
    work.complete()
    sched.finish_window(timeout=1)
    assert scheduled.is_completed()
    assert sched.sequence_log() == [dp]
    sched.close()


def test_unknown_local_group_is_rejected_at_initialization():
    with pytest.raises(ValidationError, match="absent from plan"):
        _scheduler(_plan([_key()]), local_group_ids={"unknown"})


def test_blocking_launcher_does_not_hold_scheduler_state_lock():
    k0, k1 = _key(0), _key(1)
    entered = threading.Event()
    release = threading.Event()
    created, works = [], {}

    def slow_launch():
        entered.set()
        release.wait()
        return _make_work(created, works, k0)

    sched = _scheduler(_plan([k0, k1]))
    w0 = sched.submit(_intent(k0, slow_launch))
    assert entered.wait(1)
    started = time.monotonic()
    w1 = sched.submit(_intent(k1, lambda: _make_work(created, works, k1)))
    assert time.monotonic() - started < 0.1
    release.set()
    _wait_for(lambda: len(created) == 2)
    works[k0].complete()
    works[k1].complete()
    sched.finish_window(timeout=1)
    assert w0.is_completed() and w1.is_completed()
    sched.close()


def test_global_outstanding_slot_released_without_consumer_wait():
    a, b = _key(0, "a"), _key(0, "b")
    sched = _scheduler(_plan([a, b]), max_outstanding=1)
    created, works = [], {}
    wa = sched.submit(_intent(a, lambda: _make_work(created, works, a)))
    wb = sched.submit(_intent(b, lambda: _make_work(created, works, b)))
    _wait_for(lambda: len(created) == 1)
    assert [key for key, _thread in created] == [a]
    works[a].complete()
    _wait_for(lambda: len(created) == 2)
    assert wa.is_completed()
    works[b].complete()
    sched.finish_window(timeout=1)
    assert wb.is_completed()
    sched.close()


def test_cpu_ready_event_gates_rank_wide_head():
    k0, k1 = _key(0), _key(1)
    ready = threading.Event()
    created, works = [], {}
    sched = _scheduler(_plan([k0, k1]))
    sched.submit(
        _intent(k0, lambda: _make_work(created, works, k0), ready_event=ready)
    )
    sched.submit(_intent(k1, lambda: _make_work(created, works, k1)))
    time.sleep(0.02)
    assert created == []
    ready.set()
    sched.nudge()
    _wait_for(lambda: len(created) == 2)
    for work in works.values():
        work.complete()
    sched.finish_window(timeout=1)
    sched.close()


def test_wait_inserts_dependency_but_does_not_mark_physical_completion():
    key = _key()
    underlying = StreamOrderedFakeWork()
    sched = _scheduler(_plan([key]))
    work = sched.submit(_intent(key, lambda: underlying))
    _wait_for(lambda: work.is_bound)
    assert work.wait()
    assert work.intent.state is IntentState.SUBMITTED
    assert work.timing.first_wait_ts is not None
    assert work.timing.application_wait_start_ts == work.timing.first_wait_ts
    assert work.timing.underlying_wait_start_ts is not None
    assert work.timing.wait_return_ts is not None
    assert work.timing.complete_ts is None
    underlying.complete()
    sched.finish_window(timeout=1)
    assert work.intent.state is IntentState.COMPLETED
    assert work.timing.complete_ts is not None
    sched.close()


def test_wait_timeout_before_bind_is_bounded():
    k0, k1 = _key(0), _key(1)
    sched = _scheduler(_plan([k0, k1]))
    parked = sched.submit(_intent(k1, FakeWork))
    started = time.monotonic()
    assert parked.wait(timeout=0.03) is False
    assert 0.02 <= time.monotonic() - started < 0.2
    sched.close()
    with pytest.raises(SchedulerClosedError):
        parked.wait()


def test_wait_timeout_uses_one_deadline_across_binding_and_underlying_wait():
    key = _key()
    release_launch = threading.Event()
    underlying = FakeWork()

    def launch():
        release_launch.wait()
        return underlying

    sched = _scheduler(_plan([key]))
    work = sched.submit(_intent(key, launch))
    timer = threading.Timer(0.03, release_launch.set)
    timer.start()
    started = time.monotonic()
    assert work.wait(timeout=0.08) is False
    elapsed = time.monotonic() - started
    timer.join()
    assert 0.065 <= elapsed < 0.14
    assert underlying.wait_timeouts[0] < 0.065
    sched.close()


def test_zero_timeout_wait_succeeds_for_already_completed_bound_work():
    key = _key()
    underlying = FakeWork()
    scheduler = _scheduler(_plan([key]))
    work = scheduler.submit(_intent(key, lambda: underlying))
    _wait_for(lambda: work.is_bound)
    underlying.complete()
    assert work.wait(timeout=0) is True
    scheduler.finish_window(timeout=1)
    scheduler.close()


def test_launch_error_fails_current_and_pending_work():
    k0, k1 = _key(0), _key(1)
    entered = threading.Event()

    def boom():
        entered.set()
        raise RuntimeError("launch boom")

    sched = _scheduler(_plan([k0, k1]))
    w0 = sched.submit(_intent(k0, boom))
    w1 = sched.submit(_intent(k1, FakeWork))
    assert entered.wait(1)
    with pytest.raises(RuntimeError, match="launch boom"):
        w0.wait(timeout=1)
    with pytest.raises(SchedulerFailedError, match="scheduler failed during launch") as caught:
        w1.wait(timeout=1)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "launch boom"
    with pytest.raises(RuntimeError, match="launch boom"):
        sched.sequence_log()
    assert w0.intent.state is IntentState.FAILED
    assert w1.intent.state is IntentState.FAILED
    with pytest.raises(RuntimeError, match="launch boom"):
        sched.close()
    assert not sched._thread.is_alive()


def test_launch_error_wakes_every_waiter_on_unbound_work():
    key = _key()
    entered = threading.Event()
    release = threading.Event()
    observed = []

    def launch():
        entered.set()
        release.wait()
        raise RuntimeError("wake all")

    scheduler = _scheduler(_plan([key]))
    work = scheduler.submit(_intent(key, launch))
    assert entered.wait(1)

    def wait_for_failure():
        try:
            work.wait(timeout=1)
        except BaseException as exc:  # noqa: BLE001 - assertion captures exact error
            observed.append(exc)

    waiters = [threading.Thread(target=wait_for_failure) for _ in range(3)]
    for waiter in waiters:
        waiter.start()
    release.set()
    for waiter in waiters:
        waiter.join(timeout=1)
        assert not waiter.is_alive()
    assert len(observed) == 3
    assert all(isinstance(error, RuntimeError) for error in observed)
    assert all(str(error) == "wake all" for error in observed)
    with pytest.raises(RuntimeError, match="wake all"):
        scheduler.close()


def test_invalid_synchronous_launch_result_is_fail_stop():
    key = _key()
    sched = _scheduler(_plan([key]))
    work = sched.submit(_intent(key, lambda: None))
    with pytest.raises(TypeError, match="asynchronous Work"):
        work.wait(timeout=1)
    with pytest.raises(TypeError, match="asynchronous Work"):
        sched.close()


def test_duplicate_and_metadata_validation_are_fail_stop_before_launch():
    key = _key()
    sched = _scheduler(_plan([key]))
    gate = threading.Event()
    work = sched.submit(_intent(key, lambda: (gate.wait(), FakeWork())[1]))
    with pytest.raises(ValidationError, match="duplicate"):
        sched.submit(_intent(key, FakeWork))
    with pytest.raises(ValidationError, match="op mismatch"):
        sched.submit(_intent(key, FakeWork, op="all_gather"))
    gate.set()
    _wait_for(lambda: work.is_bound)
    sched.close()


def test_finish_window_reports_missing_keys():
    k0, k1 = _key(0), _key(1)
    sched = _scheduler(_plan([k0, k1]))
    sched.submit(_intent(k0, FakeWork))
    with pytest.raises(WindowIncompleteError, match="missing local intents"):
        sched.finish_window(timeout=0.1)
    sched.close()


def test_close_is_idempotent_and_fails_unbound_work():
    k0, k1 = _key(0), _key(1)
    sched = _scheduler(_plan([k0, k1]))
    parked = sched.submit(_intent(k1, FakeWork))
    sched.close()
    sched.close()
    assert not sched._thread.is_alive()
    with pytest.raises(SchedulerClosedError):
        parked.wait()


def test_keepalive_objects_survive_while_scheduled_work_exists():
    class Token:
        pass

    key = _key()
    ready = threading.Event()
    scheduler = _scheduler(_plan([key]))
    token = Token()
    token_ref = weakref.ref(token)
    intent = _intent(key, FakeWork, ready_event=ready)
    intent.keepalive = (token,)
    work = scheduler.submit(intent)
    del token, intent
    gc.collect()
    assert token_ref() is not None

    scheduler.close()
    del work
    gc.collect()
    assert token_ref() is None


def test_close_wakes_work_while_launcher_is_still_running():
    key = _key()
    entered = threading.Event()
    release = threading.Event()

    def launch():
        entered.set()
        release.wait()
        return FakeWork()

    scheduler = _scheduler(_plan([key]))
    work = scheduler.submit(_intent(key, launch))
    assert entered.wait(1)
    timer = threading.Timer(0.03, release.set)
    timer.start()
    scheduler.close()
    timer.join()
    with pytest.raises(SchedulerClosedError):
        work.wait()
    assert work.intent.state is IntentState.FAILED


def test_completion_probe_must_claim_physical_semantics():
    class NonPhysicalProbe:
        supports_physical_completion = False

        def is_completed(self, underlying):
            return underlying.is_completed()

    with pytest.raises(ValueError, match="physical-completion"):
        AdmissionScheduler(
            _plan([_key()]),
            local_group_ids={"dp"},
            executor=DirectLaunchExecutor(),
            completion_probe=NonPhysicalProbe(),
        )


def test_completion_error_fails_inflight_and_pending_work():
    class ErrorWork(FakeWork):
        def is_completed(self):
            raise RuntimeError("completion boom")

    k0, k1 = _key(0), _key(1)
    scheduler = _scheduler(_plan([k0, k1]), max_outstanding=1)
    w0 = scheduler.submit(_intent(k0, ErrorWork))
    w1 = scheduler.submit(_intent(k1, FakeWork))
    with pytest.raises(RuntimeError, match="completion boom"):
        w0.wait(timeout=1)
    with pytest.raises(SchedulerFailedError, match="scheduler failed during completion") as caught:
        w1.wait(timeout=1)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "completion boom"
    assert w0.timing.error_stage == "completion"
    assert w1.timing.error_stage == "completion"
    with pytest.raises(RuntimeError, match="completion boom"):
        scheduler.close()


def test_telemetry_is_monotonic_and_completion_is_autonomous():
    key = _key()
    underlying = FakeWork()
    sched = _scheduler(_plan([key]))
    work = sched.submit(_intent(key, lambda: underlying))
    _wait_for(lambda: work.is_bound)
    underlying.complete()
    sched.finish_window(timeout=1)
    (timing,) = sched.timings()
    assert timing.intent_ts <= timing.admit_ts
    assert timing.admit_ts <= timing.launch_start_ts
    assert timing.launch_start_ts <= timing.submit_ts
    assert timing.submit_ts <= timing.complete_ts
    assert timing.first_wait_ts is None
    assert timing.actual_duration_us is not None
    assert work.mark_completed() is False
    assert work.intent.state is IntentState.COMPLETED
    sched.close()
