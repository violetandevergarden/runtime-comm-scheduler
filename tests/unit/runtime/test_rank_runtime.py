from __future__ import annotations

import queue
import threading
import time

import pytest

from runtime_comm_scheduler.runtime import (
    CollectiveSpec,
    GroupSpec,
    LocalBinding,
    RankRuntime,
    RuntimeState,
    TaskHint,
    TaskSpec,
)


class FakeTransport:
    def __init__(self, *, block_kind: str | None = None):
        self.incoming = queue.Queue()
        self.sent = []
        self.block_kind = block_kind
        self.entered = threading.Event()
        self.release = threading.Event()

    def connect(self):
        self.incoming.put({"kind": "READY", "epoch": 0, "payload": {}})

    def send(self, kind, payload):
        if kind == self.block_kind:
            self.entered.set()
            self.release.wait(1)
        self.sent.append((kind, payload))

    def receive(self, timeout=None):
        return self.incoming.get(timeout=timeout)

    def close(self):
        pass


class FakeTensor:
    shape = (2,)
    dtype = "torch.float32"
    device = "cpu"

    def numel(self):
        return 2

    def element_size(self):
        return 4


class ImmediateWork:
    def wait(self, timeout=None):
        return True

    def is_completed(self):
        return True


class FakeExecutor:
    def __init__(self):
        self.launches = 0

    def launch(self, binding):
        self.launches += 1
        return ImmediateWork()


class FailFirstExecutor(FakeExecutor):
    def launch(self, binding):
        self.launches += 1
        if self.launches == 1:
            raise RuntimeError("launch failed")
        return ImmediateWork()


def _task(ordinal=0):
    return TaskSpec(
        0, "job", f"job/comm-{ordinal}", "job", ordinal,
        CollectiveSpec("all_reduce", 2, 8, "float32", (2,)),
    )


def _runtime(transport=None, executor=None):
    transport = transport or FakeTransport()
    runtime = RankRuntime(0, 0, transport, executor=executor or FakeExecutor(), completion_poll_interval_s=0.001)
    runtime.register_group(GroupSpec(0, "job", (0,)), object())
    runtime.start()
    return runtime, transport


def test_submitted_precedes_completed_when_work_is_already_complete():
    transport = FakeTransport(block_kind="SUBMITTED")
    runtime, _ = _runtime(transport)
    task = _task()
    handle = runtime.submit(
        task,
        LocalBinding(FakeTensor(), runtime._process_groups["job"], lambda: ImmediateWork()),
        TaskHint(0, 0.001, 0),
    )
    transport.incoming.put({
        "kind": "GRANT",
        "epoch": 0,
        "payload": {"task": task.to_dict(), "decision_seq": 1},
    })
    assert transport.entered.wait(1)
    time.sleep(0.02)
    assert [kind for kind, _ in transport.sent].count("COMPLETED") == 0
    transport.release.set()
    assert handle.wait_host(1)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and [kind for kind, _ in transport.sent].count("COMPLETED") == 0:
        time.sleep(0.001)
    assert [kind for kind, _ in transport.sent if kind in {"OFFER", "SUBMITTED", "COMPLETED"}] == [
        "OFFER", "SUBMITTED", "COMPLETED"
    ]
    assert runtime._tasks[task.task_id].binding is None
    runtime.close()


def test_invalid_grant_does_not_launch_backend():
    executor = FakeExecutor()
    runtime, transport = _runtime(executor=executor)
    task = _task()
    runtime.submit(
        task,
        LocalBinding(FakeTensor(), runtime._process_groups["job"], lambda: ImmediateWork()),
        TaskHint(0, 0.001, 0),
    )
    invalid = TaskSpec(0, "job", task.task_id, "job", 1, task.collective)
    transport.incoming.put({
        "kind": "GRANT",
        "epoch": 0,
        "payload": {"task": invalid.to_dict(), "decision_seq": 1},
    })
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and runtime.failure is None:
        time.sleep(0.001)
    assert isinstance(runtime.failure, BaseException)
    assert executor.launches == 0
    runtime.close()


def test_input_close_rejects_new_tasks():
    runtime, transport = _runtime()
    with pytest.raises(TimeoutError):
        runtime.finish_epoch(0)
    assert runtime.state is RuntimeState.FAILED
    with pytest.raises(BaseException):
        runtime.declare(_task(), TaskHint(0, 0.001, 0))
    runtime.close()


def test_abort_from_created_and_failed_states_is_idempotent():
    transport = FakeTransport()
    runtime = RankRuntime(0, 0, transport, executor=FakeExecutor())
    first = RuntimeError("application failed")
    runtime.abort(first, stage="test")
    assert runtime.state is RuntimeState.FAILED
    assert runtime.failure is first
    runtime.abort(RuntimeError("later failure"), stage="test")
    assert runtime.failure is first
    assert transport.sent == []
    runtime.close()


def test_abort_from_running_fails_submitted_handle_and_notifies_coordinator():
    runtime, transport = _runtime()
    task = _task()
    handle = runtime.submit(
        task,
        LocalBinding(FakeTensor(), runtime._process_groups["job"], lambda: ImmediateWork()),
        TaskHint(0, 0.001, 0),
    )
    error = RuntimeError("compute failed")
    runtime.abort(error, stage="application", job_id="job")
    assert runtime.state is RuntimeState.FAILED
    assert runtime.failure is error
    assert handle.state.value == "failed"
    assert any(kind == "FAILED" for kind, _ in transport.sent)
    runtime.close()


def test_abort_from_input_closed_unblocks_finish_epoch():
    runtime, transport = _runtime()
    errors = []

    def finish():
        try:
            runtime.finish_epoch(5)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=finish)
    thread.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not any(kind == "INPUT_CLOSED" for kind, _ in transport.sent):
        time.sleep(0.001)
    assert any(kind == "INPUT_CLOSED" for kind, _ in transport.sent)
    error = RuntimeError("runner failed after close")
    runtime.abort(error, stage="dag_runner")
    thread.join(1)
    assert not thread.is_alive()
    assert errors == [error]
    assert runtime.state is RuntimeState.FAILED
    runtime.close()


def test_submit_message_cannot_be_overtaken_by_finish_epoch():
    transport = FakeTransport(block_kind="OFFER")
    runtime, _ = _runtime(transport)
    task = _task()
    submit_done = threading.Event()
    finish_started = threading.Event()
    finish_errors = []

    def submit():
        runtime.submit(
            task,
            LocalBinding(FakeTensor(), runtime._process_groups["job"], lambda: ImmediateWork()),
            TaskHint(0, 0.001, 0),
        )
        submit_done.set()

    def finish():
        finish_started.set()
        try:
            runtime.finish_epoch(0.2)
        except BaseException as exc:  # noqa: BLE001
            finish_errors.append(exc)

    submit_thread = threading.Thread(target=submit)
    finish_thread = threading.Thread(target=finish)
    submit_thread.start()
    assert transport.entered.wait(1)
    finish_thread.start()
    assert finish_started.wait(1)
    time.sleep(0.01)
    assert not any(kind == "INPUT_CLOSED" for kind, _ in transport.sent)

    transport.release.set()
    submit_thread.join(1)
    finish_thread.join(1)
    assert submit_done.is_set()
    assert finish_errors
    assert [kind for kind, _ in transport.sent if kind in {"OFFER", "INPUT_CLOSED"}][:2] == [
        "OFFER", "INPUT_CLOSED"
    ]
    runtime.close()


def test_failure_stops_queued_tasks_from_reaching_executor():
    executor = FailFirstExecutor()
    runtime, transport = _runtime(executor=executor)
    tasks = (_task(0), _task(1))
    for task in tasks:
        runtime.submit(
            task,
            LocalBinding(FakeTensor(), runtime._process_groups["job"], lambda: ImmediateWork()),
            TaskHint(0, 0.001, 0),
        )
    for decision_seq, task in enumerate(tasks, 1):
        transport.incoming.put({
            "kind": "GRANT",
            "epoch": 0,
            "payload": {"task": task.to_dict(), "decision_seq": decision_seq},
        })

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and runtime.failure is None:
        time.sleep(0.001)
    assert isinstance(runtime.failure, RuntimeError)
    assert executor.launches == 1
    runtime.close()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("shape", "tensor shape mismatch"),
        ("numel", "tensor numel mismatch"),
        ("element_size", "tensor bytes mismatch"),
        ("dtype", "tensor dtype mismatch"),
        ("device", "tensor device mismatch"),
    ],
)
def test_binding_rejects_each_tensor_metadata_mismatch(field, message):
    runtime, transport = _runtime()
    tensor = FakeTensor()
    if field == "shape":
        tensor.shape = (1,)
    elif field == "numel":
        tensor.numel = lambda: 1
    elif field == "element_size":
        tensor.element_size = lambda: 2
    elif field == "dtype":
        tensor.dtype = "int32"
    elif field == "device":
        tensor.device = "cuda"

    with pytest.raises(ValueError, match=message):
        runtime.submit(
            _task(),
            LocalBinding(tensor, runtime._process_groups["job"], lambda: ImmediateWork(), device="cpu"),
            TaskHint(0, 0.001, 0),
        )
    assert not any(kind == "OFFER" for kind, _ in transport.sent)
    runtime.close()


def test_duplicate_grant_fails_runtime_and_cannot_launch_twice():
    executor = FakeExecutor()
    runtime, transport = _runtime(executor=executor)
    task = _task()
    runtime.submit(
        task,
        LocalBinding(FakeTensor(), runtime._process_groups["job"], lambda: ImmediateWork()),
        TaskHint(0, 0.001, 0),
    )
    grant = {
        "kind": "GRANT",
        "epoch": 0,
        "payload": {"task": task.to_dict(), "decision_seq": 1},
    }
    transport.incoming.put(grant)
    transport.incoming.put(grant)

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and runtime.failure is None:
        time.sleep(0.001)
    assert isinstance(runtime.failure, RuntimeError)
    assert runtime.grant_order == [task.task_id]
    assert executor.launches <= 1
    runtime.close()
