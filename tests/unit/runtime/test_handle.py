from __future__ import annotations

import threading
import time

import pytest

from runtime_comm_scheduler.runtime.handle import HandleState, RuntimeHandle


class Work:
    def wait(self, timeout=None):
        raise AssertionError("wait_host must not call Work.wait")


class Receipt:
    def __init__(self):
        self.waited_streams = []

    def is_completed(self):
        return True

    def wait_on(self, stream, timeout=None):
        self.waited_streams.append((stream, timeout))
        return True


class Stream:
    def wait_stream(self, _work):
        raise AssertionError("backend Work must not be passed to wait_stream")


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


def test_wait_on_waits_for_binding_then_adds_explicit_dependency():
    handle = RuntimeHandle("task")
    handle.grant(1)
    receipt = Receipt()
    stream = Stream()
    result = []
    entered = threading.Event()

    def wait():
        entered.set()
        result.append(handle.wait_on(stream, timeout=1.0))

    thread = threading.Thread(target=wait)
    thread.start()
    assert entered.wait(1)
    handle.bind(receipt)
    thread.join(1)
    assert not thread.is_alive()
    assert result == [True]
    assert receipt.waited_streams == [(stream, 1.0)]
    assert handle.state is HandleState.BOUND


def test_wait_on_receipt_remains_available_after_handle_completion():
    handle = RuntimeHandle("task")
    handle.grant(1)
    receipt = Receipt()
    handle.bind(receipt)
    handle.mark_completed()
    stream = Stream()
    assert handle.wait_on(stream, timeout=0.0) is True
    assert receipt.waited_streams == [(stream, 0.0)]


def test_wait_on_has_finite_binding_deadline_and_rejects_unsupported_receipt():
    handle = RuntimeHandle("task")
    handle.grant(1)
    assert handle.wait_on(Stream(), timeout=0.0) is False
    handle.bind(Work())
    with pytest.raises(NotImplementedError, match="stream dependencies"):
        handle.wait_on(Stream(), timeout=0.0)


def test_wait_on_propagates_failure_to_a_blocked_consumer():
    handle = RuntimeHandle("task")
    handle.grant(1)
    expected = RuntimeError("launch failed")
    result = []

    def wait():
        try:
            handle.wait_on(Stream(), timeout=1.0)
        except BaseException as exc:  # noqa: BLE001
            result.append(exc)

    thread = threading.Thread(target=wait)
    thread.start()
    handle.fail(expected)
    thread.join(1)
    assert result == [expected]
