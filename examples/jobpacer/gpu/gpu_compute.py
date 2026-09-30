"""Small preallocated CUDA compute programs for phase 3 overlap experiments."""

from __future__ import annotations

from typing import Any



class CudaComputeReceipt:
    """Completion token for a fixed CUDA compute segment."""

    completion_source = "cuda_compute_event_query"

    def __init__(self, program: "CudaMatmulProgram", start_event: Any, done_event: Any):
        self.device = program.device
        self.start_event = start_event
        self.done_event = done_event
        self.keepalive = (program, program.left, program.right, program.output,
                          start_event, done_event)
        self._program = program

    def is_completed(self) -> bool:
        done = bool(self.done_event.query())
        if done:
            self._program._inflight = False
        return done

    def elapsed_ms(self) -> float:
        return float(self.start_event.elapsed_time(self.done_event))


class CudaMatmulProgram:
    """Preallocated fixed-count matmul work; no duration-chasing loop at replay time."""

    def __init__(self, device: Any, *, matrix_size: int, repeats: int, seed: int,
                 torch_module: Any | None = None):
        if matrix_size < 16:
            raise ValueError("matrix_size must be at least 16")
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = torch_module.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("CudaMatmulProgram requires an explicit cuda:N device")
        if not torch_module.cuda.is_available() or self.device.index >= torch_module.cuda.device_count():
            raise RuntimeError(f"CUDA device {self.device} is not available")
        self.matrix_size = int(matrix_size)
        self.repeats = int(repeats)
        self.seed = int(seed)
        self.stream = torch_module.cuda.Stream(device=self.device)
        generator = torch_module.Generator(device=self.device)
        generator.manual_seed(self.seed)
        with torch_module.cuda.device(self.device):
            self.left = torch_module.randn((matrix_size, matrix_size), device=self.device,
                                           dtype=torch_module.float32, generator=generator)
            self.right = torch_module.randn((matrix_size, matrix_size), device=self.device,
                                            dtype=torch_module.float32, generator=generator)
            self.output = torch_module.empty((matrix_size, matrix_size), device=self.device,
                                             dtype=torch_module.float32)
        self._inflight = False
        self._warmed = False
        self.warmup_count = 0

    def warmup(self, iterations: int = 1) -> None:
        """Warm the exact operator before application release."""
        if iterations < 0:
            raise ValueError("warmup iterations must be non-negative")
        torch = self.torch
        if iterations:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                for _ in range(iterations):
                    torch.mm(self.left, self.right, out=self.output)
            self.stream.synchronize()
        self.warmup_count = iterations
        self._warmed = True

    def submit(self) -> CudaComputeReceipt:
        if not self._warmed:
            raise RuntimeError("CUDA compute program must be warmed before replay")
        if self._inflight:
            raise RuntimeError("CUDA compute program cannot overlap its own unfinished reuse")
        torch = self.torch
        start = torch.cuda.Event(enable_timing=True, blocking=False)
        done = torch.cuda.Event(enable_timing=True, blocking=False)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            start.record(self.stream)
            for _ in range(self.repeats):
                torch.mm(self.left, self.right, out=self.output)
            done.record(self.stream)
        self._inflight = True
        return CudaComputeReceipt(self, start, done)
