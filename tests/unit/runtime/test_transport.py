from __future__ import annotations

import threading

from runtime_comm_scheduler.runtime.telemetry import EventLog
from runtime_comm_scheduler.runtime.transport import ControlClient


class _Writer:
    def __init__(self):
        self.payloads = []

    def write(self, payload):
        self.payloads.append(payload)

    def flush(self):
        pass


class _GatedLock:
    def __init__(self):
        self.attempted = threading.Event()
        self.release = threading.Event()

    def __enter__(self):
        self.attempted.set()
        if not self.release.wait(1):
            raise TimeoutError("test did not release the send lock")
        return self

    def __exit__(self, *_exc):
        return False


def test_control_send_records_one_rank_local_lock_and_write_interval():
    event_log = EventLog("runtime", 1, thread_sharded=True)
    client = ControlClient(1, 5, "127.0.0.1", 0, event_log=event_log)
    client._writer = _Writer()
    lock = _GatedLock()
    client._lock = lock
    failures = []

    def send():
        try:
            client.send("OFFER", {"task": {"task_id": "job-0/comm-2"}})
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)

    thread = threading.Thread(target=send)
    thread.start()
    assert lock.attempted.wait(1)
    lock.release.set()
    thread.join(1)

    assert not thread.is_alive()
    assert failures == []
    assert len(client._writer.payloads) == 1
    events = event_log.as_dict()["events"]
    assert len(events) == 1
    event = events[0]
    assert event["kind"] == "control_send_interval"
    assert event["message_kind"] == "OFFER"
    assert event["task_id"] == "job-0/comm-2"
    assert event["lock_wait_start_us"] <= event["lock_acquired_us"]
    assert event["lock_acquired_us"] <= event["socket_write_start_us"]
    assert event["socket_write_start_us"] <= event["socket_write_end_us"]
    assert event["lock_wait_us"] == event["lock_acquired_us"] - event["lock_wait_start_us"]
    assert event["socket_write_flush_us"] == (
        event["socket_write_end_us"] - event["socket_write_start_us"]
    )


def test_control_send_skips_timing_event_when_event_log_is_disabled():
    event_log = EventLog("runtime", 0, enabled=False)
    client = ControlClient(0, 5, "127.0.0.1", 0, event_log=event_log)
    client._writer = _Writer()

    client.send("DECLARE", {"task": {"task_id": "job-0/comm-0"}})

    assert event_log.as_dict()["events"] == []
    assert len(client._writer.payloads) == 1
