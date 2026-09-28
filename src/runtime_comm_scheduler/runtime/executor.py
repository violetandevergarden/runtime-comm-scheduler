"""Rank-local launch and completion boundaries."""

from __future__ import annotations

from typing import Any

from .model import LocalBinding


def _validate_work(work: Any) -> None:
    if work is None or not callable(getattr(work, "wait", None)):
        raise TypeError("launch must return an asynchronous Work-like object")
    if not callable(getattr(work, "is_completed", None)):
        raise TypeError("launch result must provide is_completed()")


class DirectExecutor:
    def launch(self, binding: LocalBinding) -> Any:
        if binding.producer_event is not None:
            raise ValueError("DirectExecutor cannot honor a CUDA producer event")
        _timing(binding, "collective_api_start")
        try:
            work = binding.launch()
        finally:
            _timing(binding, "collective_api_return")
        _validate_work(work)
        return work


class CudaCollectiveWork:
    """Local receipt joining backend work to an explicit CUDA completion event."""

    def __init__(self, work: Any, event: Any, device: Any, keepalive: tuple[Any, ...]):
        self.backend_work = work
        self.comm_done = event
        self.device = device
        self.keepalive = keepalive
        self.completion_source = "cuda_event_query_after_backend_work_wait"

    def is_completed(self) -> bool:
        # The explicit event is the physical-completion contract. Some PyTorch
        # NCCL Work versions expose is_success() only as a deprecated stub.
        return bool(self.comm_done.query())

    def wait_on(self, stream: Any, timeout: float | None = None) -> bool:
        del timeout  # Stream dependency insertion is asynchronous and has no host wait.
        stream_device = getattr(stream, "device", None)
        if stream_device is None or str(stream_device) != str(self.device):
            raise ValueError(f"consumer stream device {stream_device!r} does not match {self.device}")
        wait_event = getattr(stream, "wait_event", None)
        if not callable(wait_event):
            raise TypeError("consumer stream must provide wait_event(event)")
        wait_event(self.comm_done)
        return True


class CudaCollectiveExecutor:
    """Launch CUDA collectives through a per-process-group gate stream.

    PyTorch and CUDA are imported only when this CUDA-specific executor is
    instantiated, leaving CPU-only imports independent from CUDA availability.
    """

    supports_producer_dependency = True

    def __init__(self, device: Any, *, torch_module: Any | None = None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = torch_module.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("CudaCollectiveExecutor requires an explicit cuda:N device")
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch_module.cuda.device_count() <= self.device.index:
            raise ValueError(f"CUDA device {self.device} is not visible")
        if _env_enabled("TORCH_NCCL_BLOCKING_WAIT"):
            raise RuntimeError("CUDA stream completion bridge is unsupported with TORCH_NCCL_BLOCKING_WAIT")
        self._streams: dict[int, tuple[Any, Any]] = {}

    def _gate_stream(self, process_group: Any) -> Any:
        key = id(process_group)
        found = self._streams.get(key)
        if found is not None:
            group_ref, stream = found
            if group_ref is process_group:
                return stream
        stream = self.torch.cuda.Stream(device=self.device)
        self._streams[key] = (process_group, stream)
        return stream

    def launch(self, binding: LocalBinding) -> CudaCollectiveWork:
        torch = self.torch
        tensor = binding.tensor
        if getattr(tensor, "device", None) != self.device:
            raise ValueError(f"tensor device {getattr(tensor, 'device', None)} does not match {self.device}")
        producer_event = binding.producer_event
        if producer_event is not None:
            event_device = getattr(producer_event, "device", None)
            if event_device is None or event_device != self.device:
                raise ValueError(f"producer event device {event_device!r} does not match {self.device}")
        gate = self._gate_stream(binding.process_group)
        with torch.cuda.device(self.device), torch.cuda.stream(gate):
            if producer_event is not None:
                gate.wait_event(producer_event)
            _timing(binding, "collective_api_start")
            try:
                work = binding.launch()
            finally:
                _timing(binding, "collective_api_return")
            _validate_work(work)
            # For NCCL, wait() on this stream enqueues the communicator's
            # completion dependency without waiting for the device on the host.
            work.wait()
            done = torch.cuda.Event(blocking=False, interprocess=False)
            done.record(gate)
        return CudaCollectiveWork(work, done, self.device,
                                  (binding.tensor, binding.producer_event, work, done,
                                   binding.keepalive))


def _env_enabled(name: str) -> bool:
    import os
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _timing(binding: LocalBinding, kind: str) -> None:
    hook = binding.timing_hook
    if hook is not None:
        hook(kind)


class WorkIsCompletedProbe:
    supports_physical_completion = True

    def is_completed(self, work: Any) -> bool:
        return bool(work.is_completed())
