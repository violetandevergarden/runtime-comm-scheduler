from __future__ import annotations

from contextlib import contextmanager
import subprocess
import sys
from pathlib import Path

import pytest

from runtime_comm_scheduler.runtime.executor import (
    CudaCollectiveExecutor,
    CudaCollectiveWork,
    DirectExecutor,
)
from runtime_comm_scheduler.runtime.model import LocalBinding


class Device:
    type = "cuda"

    def __init__(self, index):
        self.index = index

    def __str__(self):
        return f"cuda:{self.index}"

    def __eq__(self, other):
        return isinstance(other, Device) and self.index == other.index


class Event:
    def __init__(self, device):
        self.device = device
        self.recorded_stream = None
        self.complete = False

    def record(self, stream):
        self.recorded_stream = stream

    def query(self):
        return self.complete


class Stream:
    def __init__(self, device):
        self.device = device
        self.waited_events = []

    def wait_event(self, event):
        self.waited_events.append(event)


class Work:
    def __init__(self, actions):
        self.actions = actions
        self.success = True

    def wait(self):
        self.actions.append("work.wait")

    def is_completed(self):
        raise AssertionError("CUDA receipt must use its event as physical completion evidence")

    def is_success(self):
        raise AssertionError("CUDA receipt must not call deprecated Work.is_success()")


class FakeCuda:
    def __init__(self):
        self.streams = []
        self.events = []
        self.current_stream = None

    def is_available(self):
        return True

    def device_count(self):
        return 2

    def Stream(self, device):
        result = Stream(device)
        self.streams.append(result)
        return result

    def Event(self, **_kwargs):
        result = Event(Device(0))
        self.events.append(result)
        return result

    @contextmanager
    def device(self, _device):
        yield

    @contextmanager
    def stream(self, stream):
        old = self.current_stream
        self.current_stream = stream
        try:
            yield
        finally:
            self.current_stream = old


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()

    @staticmethod
    def device(value):
        if isinstance(value, Device):
            return value
        if value == "cuda:0":
            return Device(0)
        if value == "cuda:1":
            return Device(1)
        raise ValueError(value)


def test_cuda_executor_bridges_producer_and_backend_work_to_done_event():
    fake_torch = FakeTorch()
    executor = CudaCollectiveExecutor("cuda:0", torch_module=fake_torch)
    actions = []
    process_group = object()
    producer = Event(Device(0))
    tensor = type("Tensor", (), {"device": Device(0)})()

    def launch():
        assert fake_torch.cuda.current_stream is fake_torch.cuda.streams[0]
        assert fake_torch.cuda.streams[0].waited_events == [producer]
        actions.append("launch")
        return Work(actions)

    receipt = executor.launch(LocalBinding(tensor, process_group, launch,
                                           producer_event=producer, device="cuda:0"))
    gate = fake_torch.cuda.streams[0]
    done = fake_torch.cuda.events[0]
    assert actions == ["launch", "work.wait"]
    assert done.recorded_stream is gate
    assert receipt.is_completed() is False
    done.complete = True
    assert receipt.is_completed() is True
    consumer = Stream(Device(0))
    assert receipt.wait_on(consumer) is True
    assert consumer.waited_events == [done]
    with pytest.raises(ValueError, match="consumer stream device"):
        receipt.wait_on(Stream(Device(1)))


def test_cuda_executor_rejects_device_mismatch_before_launch():
    executor = CudaCollectiveExecutor("cuda:0", torch_module=FakeTorch())
    called = []
    tensor = type("Tensor", (), {"device": Device(1)})()
    binding = LocalBinding(tensor, object(), lambda: called.append(True), device="cuda:1")
    with pytest.raises(ValueError, match="does not match"):
        executor.launch(binding)
    assert called == []


def test_direct_executor_fails_closed_for_producer_event():
    tensor = type("Tensor", (), {"device": "cpu"})()
    binding = LocalBinding(tensor, object(), lambda: None, producer_event=object())
    with pytest.raises(ValueError, match="cannot honor"):
        DirectExecutor().launch(binding)


def test_importing_executor_does_not_import_torch():
    root = Path(__file__).resolve().parents[3]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [sys.executable, "-c", "import sys; import runtime_comm_scheduler.runtime.executor; assert 'torch' not in sys.modules"],
        cwd=root, env=env, check=True, timeout=10,
    )
