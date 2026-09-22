from __future__ import annotations

import threading
import time

import pytest

from runtime_comm_scheduler.runtime.handle import HandleState, RuntimeHandle


class Work:
    def wait(self, timeout=None):
        raise AssertionError("wait_host must not call Work.wait")


def test_wait_host_waits_for_runtime_completion_only():
    handle = RuntimeHandle("task")
    handle.grant(1)
    handle.bind(Work())
    result = []

    thread = threading.Thread(target=lambda: result.append(handle.wait_host(1)))
    thread.start()
    time.sleep(0.01)
    assert thread.is_alive()
    handle.mark_completed()
    thread.join(1)
    assert result == [True]


def test_wait_host_failure_wakes_with_original_error():
    handle = RuntimeHandle("task")
    handle.grant(1)
    handle.bind(Work())
    error = RuntimeError("boom")
    result = []

    def wait():
        try:
            handle.wait_host(1)
        except BaseException as exc:  # noqa: BLE001
            result.append(exc)

    thread = threading.Thread(target=wait)
    thread.start()
    time.sleep(0.01)
    handle.fail(error)
    thread.join(1)
    assert result == [error]
    assert handle.state is HandleState.FAILED
    assert handle.fail(RuntimeError("ignored")) is False


def test_wait_host_timeout_does_not_call_work_wait():
    handle = RuntimeHandle("task")
    handle.grant(1)
    handle.bind(Work())
    assert handle.wait_host(0) is False


def test_wait_host_rejects_negative_timeout():
    with pytest.raises(ValueError):
        RuntimeHandle("task").wait_host(-1)
