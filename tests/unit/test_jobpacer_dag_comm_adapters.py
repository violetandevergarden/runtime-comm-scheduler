from __future__ import annotations

import sys
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.jobpacer.runtime import dag_comm_adapters as adapters
from runtime_comm_scheduler.runtime import (
    CollectiveSpec, EventLog, HandleState, LocalBinding, TaskSpec,
)
from runtime_comm_scheduler import TaskKey


class _Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


class _BoundWork:
    is_bound = True

    def __init__(self, clock: _Clock, timeouts: list[float]):
        self.clock, self.timeouts = clock, timeouts

    def wait(self, *, timeout):
        if isinstance(timeout, timedelta):
            timeout = timeout.total_seconds()
        self.timeouts.append(timeout)
        self.clock.now += min(0.2, timeout)


def test_old_adapter_uses_one_deadline_for_each_work_and_scheduler_close(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(adapters.time, "monotonic", clock.monotonic)
    work_timeouts: list[float] = []
    close_timeouts: list[float] = []

    class Scheduler:
        def close(self, *, timeout):
            close_timeouts.append(timeout)

    adapter = object.__new__(adapters.LegacySchedulerDagAdapter)
    adapter._handles = {
        f"task-{index}": SimpleNamespace(work=_BoundWork(clock, work_timeouts))
        for index in range(2)
    }
    adapter._handles_lock = threading.Lock()
    adapter._closed = False
    adapter.failure_signal = None
    adapter.failure_drain_timeout_s = 10.0
    adapter._replay_deadline = 100.6
    adapter._cleanup_deadline = None
    adapter.event_log = EventLog("runtime", 0)
    adapter.scheduler = Scheduler()

    adapter.abort(RuntimeError("injected"), deadline=100.6)

    assert work_timeouts == pytest.approx([0.6, 0.4])
    assert close_timeouts == pytest.approx([0.2])
    assert clock.now == pytest.approx(100.4)


def test_old_adapter_abort_includes_submit_accepted_while_failure_starts():
    entered_submit = threading.Event()
    release_submit = threading.Event()
    abort_started = threading.Event()
    waited: list[float] = []
    close_timeouts: list[float] = []
    deadline = adapters.time.monotonic() + 2.0
    key = TaskKey(0, 0, "test", "g", 0, 0, 0)
    spec = TaskSpec(0, "job", "job/comm", "g", 0,
                    CollectiveSpec("all_reduce", 1, 4, "float32", (1,)))

    class Work:
        is_bound = True

        def wait(self, *, timeout):
            waited.append(timeout)

    class Scheduler:
        def submit(self, _intent):
            entered_submit.set()
            assert release_submit.wait(2.0)
            return Work()

        def close(self, *, timeout):
            close_timeouts.append(timeout)

    adapter = object.__new__(adapters.LegacySchedulerDagAdapter)
    adapter.task_keys = {spec.task_id: key}
    adapter.plan = SimpleNamespace(metadata=lambda _key: ("all_reduce", 4))
    adapter.scheduler = Scheduler()
    adapter.probe = object()
    adapter._handles = {}
    adapter._handles_lock = threading.Lock()
    adapter._closed = False
    adapter.failure_signal = None
    adapter.failure_drain_timeout_s = 2.0
    adapter._replay_deadline = 101.0
    adapter._cleanup_deadline = None
    adapter.event_log = EventLog("runtime", 0)
    binding = LocalBinding(object(), object(), lambda: None, device="cuda:0")
    submit_error: list[BaseException] = []

    def submit():
        try:
            adapter.submit(spec, binding, SimpleNamespace())
        except BaseException as exc:
            submit_error.append(exc)

    def abort():
        abort_started.set()
        adapter.abort(RuntimeError("injected"), deadline=deadline)

    submit_thread = threading.Thread(target=submit)
    submit_thread.start()
    assert entered_submit.wait(2.0)
    abort_thread = threading.Thread(target=abort)
    abort_thread.start()
    assert abort_started.wait(2.0)
    release_submit.set()
    submit_thread.join(2.0)
    abort_thread.join(2.0)

    assert not submit_thread.is_alive()
    assert not abort_thread.is_alive()
    assert submit_error == []
    assert len(adapter._handles) == 1
    assert len(waited) == 1 and 1.9 < waited[0] <= 2.0
    assert len(close_timeouts) == 1
    assert 0 <= close_timeouts[0] <= waited[0]


def test_raw_adapter_joins_inflight_binding_then_drains_all_works_to_same_deadline(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(adapters.time, "monotonic", clock.monotonic)
    work_timeouts: list[float] = []

    class Handle:
        def __init__(self, receipt=None):
            self.state = HandleState.BOUND
            self.receipt = receipt

        def fail(self, _error):
            self.state = HandleState.FAILED

    def receipt():
        return SimpleNamespace(backend_work=_BoundWork(clock, work_timeouts))

    late_binding = Handle()

    class Dispatcher:
        def __init__(self):
            self.join_timeouts = []

        def join(self, timeout):
            self.join_timeouts.append(timeout)
            clock.now += min(0.2, timeout)
            # Model launch() returning and binding its Work while abort joins.
            late_binding.receipt = receipt()

    dispatcher = Dispatcher()
    adapter = object.__new__(adapters.RawOrderedDagAdapter)
    adapter.failure_signal = None
    adapter.failure_drain_timeout_s = 10.0
    adapter._replay_deadline = 100.7
    adapter._cleanup_deadline = None
    adapter._condition = threading.Condition()
    adapter._failure = None
    adapter._stopping = False
    adapter._thread = dispatcher
    adapter._handles = {
        "task-0": Handle(receipt()),
        "task-1": Handle(receipt()),
        "task-2-in-launch": late_binding,
    }
    adapter.event_log = EventLog("runtime", 0)

    adapter.abort(RuntimeError("injected"), deadline=100.7)

    assert dispatcher.join_timeouts == pytest.approx([0.7])
    assert work_timeouts == pytest.approx([0.5, 0.3, 0.1])
    assert clock.now == pytest.approx(100.7)
