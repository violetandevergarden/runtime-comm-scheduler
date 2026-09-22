from __future__ import annotations

import queue
import threading

import pytest

from runtime_comm_scheduler.runtime.coordinator import CoordinatorState
from runtime_comm_scheduler.runtime.model import CollectiveSpec, GroupSpec, TaskHint, TaskSpec
from runtime_comm_scheduler.runtime.policy import Idle, Wait
from runtime_comm_scheduler.runtime import transport as transport_module


def _task(group_id: str, group_seq: int) -> TaskSpec:
    return TaskSpec(
        0,
        group_id,
        f"{group_id}/comm-{group_seq}",
        group_id,
        group_seq,
        CollectiveSpec("all_reduce", 1, 4, "float32", (1,)),
    )


def _payload(task: TaskSpec) -> dict:
    return {"task": task.to_dict(), "hint": TaskHint(0.0, 0.001, 0.0).to_dict()}


def test_next_deadline_is_the_earliest_epoch_or_wait_deadline():
    coordinator = CoordinatorState((0, 1), epoch_timeout_s=10)
    coordinator.apply(0, "REGISTER_GROUP", 1, {"group": {"epoch": 0, "group_id": "g", "ranks": [0, 1]}}, 2.0)
    assert coordinator.next_deadline == 12.0


def test_terminal_coordinator_has_no_next_deadline():
    coordinator = CoordinatorState((0,), epoch_timeout_s=10)
    group = GroupSpec(0, "g", (0,))
    coordinator.apply(0, "REGISTER_GROUP", 1, {"group": group.to_dict()}, 2.0)
    assert coordinator.next_deadline == 12.0

    coordinator.fail("test_failure")

    assert coordinator.next_deadline is None


def test_five_ms_epoch_deadline_is_exposed_without_fixed_interval_rounding():
    coordinator = CoordinatorState((0,), epoch_timeout_s=0.005)
    group = GroupSpec(0, "g", (0,))
    coordinator.apply(0, "REGISTER_GROUP", 1, {"group": group.to_dict()}, 10.0)
    assert coordinator.next_deadline == pytest.approx(10.005, abs=1e-12)


def test_event_loop_checks_a_five_ms_deadline_before_the_50_ms_poll_cap(monkeypatch):
    class FakeCoordinator:
        done = False
        next_deadline = 100.005

        def tick(self, now):
            assert now == 100.0
            return []

    class EmptyQueue:
        def __init__(self, stop):
            self.stop = stop
            self.timeout = None

        def get(self, timeout=None):
            self.timeout = timeout
            self.stop.set()
            raise queue.Empty

    server = object.__new__(transport_module.CoordinatorServer)
    server.coordinator = FakeCoordinator()
    server._stop = threading.Event()
    server._inbound = EmptyQueue(server._stop)
    server._dispatch = lambda messages: None
    monkeypatch.setattr(transport_module.time, "monotonic", lambda: 100.0)

    server._event_loop()

    assert server._inbound.timeout == pytest.approx(0.005, abs=1e-12)


def test_writer_flushes_finished_even_if_stop_is_already_set():
    class FakeConnection:
        def __init__(self):
            self.messages = []

        def sendall(self, message):
            self.messages.append(message)

    server = object.__new__(transport_module.CoordinatorServer)
    server._stop = threading.Event()
    server._lock = threading.Lock()
    server._finish_sent = {0: threading.Event()}
    server._inbound = queue.Queue()
    messages = queue.Queue()
    connection = FakeConnection()
    finished = {"protocol": 1, "kind": "FINISHED", "epoch": 0, "endpoint": 0, "payload": {}}
    messages.put(finished)
    messages.put(None)
    server._stop.set()

    thread = threading.Thread(
        target=server._writer_loop,
        args=(0, connection, messages),
    )
    thread.start()
    thread.join(1)

    assert not thread.is_alive()
    assert connection.messages
    assert server._finish_sent[0].is_set()


def test_close_waits_for_finished_writer_before_closing_connections():
    class FinishedCoordinator:
        finished = True

    class FinishEvent:
        def __init__(self):
            self.entered = threading.Event()
            self.released = threading.Event()

        def wait(self, timeout):
            self.entered.set()
            return self.released.wait(timeout)

    server = object.__new__(transport_module.CoordinatorServer)
    server.coordinator = FinishedCoordinator()
    server._listener = None
    server._stop = threading.Event()
    server._lock = threading.Lock()
    server._finish_sent = {0: FinishEvent()}
    server._writers = {}
    server._connections = {}
    server._threads = []
    event = server._finish_sent[0]
    thread = threading.Thread(target=server.close, kwargs={"timeout": 1.0})
    thread.start()

    assert event.entered.wait(1)
    assert thread.is_alive()
    event.released.set()
    thread.join(1)
    assert not thread.is_alive()


def test_terminal_event_loop_waits_for_close_instead_of_polling():
    class Stop:
        def __init__(self):
            self.entered = threading.Event()
            self.released = threading.Event()

        def is_set(self):
            return self.released.is_set()

        def wait(self):
            self.entered.set()
            self.released.wait()

    class FinishedCoordinator:
        done = True

    server = object.__new__(transport_module.CoordinatorServer)
    server.coordinator = FinishedCoordinator()
    server._stop = Stop()
    server._inbound = None
    server._dispatch = lambda messages: None
    thread = threading.Thread(target=server._event_loop)
    thread.start()

    assert server._stop.entered.wait(1)
    server._stop.released.set()
    thread.join(1)
    assert not thread.is_alive()


class _AlwaysWait:
    def __init__(self):
        self.calls = []

    def decide(self, snapshot):
        self.calls.append(snapshot)
        if not snapshot.eligible:
            return Idle("NO_ELIGIBLE")
        return Wait("future", snapshot.now + 0.005)

    def missing_tasks(self, task_ids):
        return ()


def test_active_wait_deadline_does_not_refresh_on_continuous_messages():
    coordinator = CoordinatorState((0,), epoch_timeout_s=1.0)
    policy = _AlwaysWait()
    coordinator.policy = policy
    group = GroupSpec(0, "g", (0,))
    coordinator.apply(0, "REGISTER_GROUP", 1, {"group": group.to_dict()}, 0.0)
    current = _task("g", 0)
    future = _task("g", 1)

    coordinator.apply(0, "OFFER", 2, _payload(current), 0.001)
    assert coordinator.active_wait is not None
    original_deadline = coordinator.active_wait.deadline

    coordinator.apply(0, "DECLARE", 3, _payload(future), 0.002)

    assert len(policy.calls) >= 3
    assert coordinator.active_wait is not None
    assert coordinator.active_wait.deadline == original_deadline


def test_epoch_timeout_is_checked_between_continuous_messages_without_tick():
    coordinator = CoordinatorState((0,), epoch_timeout_s=0.005)
    group = GroupSpec(0, "g", (0,))
    coordinator.apply(0, "REGISTER_GROUP", 1, {"group": group.to_dict()}, 0.0)
    coordinator.apply(0, "OFFER", 2, _payload(_task("g", 0)), 0.004)

    out = coordinator.apply(0, "INPUT_CLOSED", 3, {}, 0.005)

    assert [item.kind for item in out] == ["FAILED"]
    assert coordinator.failed is not None
    assert coordinator.failed["reason"] == "epoch_timeout"
