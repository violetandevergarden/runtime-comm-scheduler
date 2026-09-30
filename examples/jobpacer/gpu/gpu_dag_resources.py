"""Rank-local CUDA buffers and fixed DAG node programs for schema-v2 inputs."""

from __future__ import annotations

import hashlib
import time
from graphlib import TopologicalSorter
from typing import Any, Mapping

from runtime_comm_scheduler.dag import CommNode, ComputeNode, DagGraph, DagJob
from runtime_comm_scheduler.runtime import LocalBinding


SEED_DERIVATION_VERSION = "sha256-torch-generator-v1"


def derive_tensor_seed(seed: int, sample_id: str, job_id: str, object_id: str,
                       rank: int, stage: str) -> int:
    key = "\0".join((str(seed), sample_id, job_id, object_id, str(rank), stage)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & ((1 << 63) - 1)


class GpuDagReceipt:
    completion_source = "cuda_event_query"

    def __init__(self, owner: "GpuDagResources", node_id: str, start_event: Any, done_event: Any):
        self.owner = owner
        self.node_id = node_id
        self.start_event = start_event
        self.done_event = done_event

    def is_completed(self) -> bool:
        return bool(self.done_event.query())

    def is_success(self) -> bool:
        return True

    def elapsed_ms(self) -> float | None:
        if not self.is_completed():
            return None
        return float(self.start_event.elapsed_time(self.done_event))


class _GpuDagNodeProgram:
    def __init__(self, owner: "GpuDagResources", node_id: str):
        self.owner, self.node_id = owner, node_id
        self.device = owner.device
        self.op = owner.programs[node_id]["op"]
        self.seed = owner.node_seeds[node_id]

    def submit(self) -> GpuDagReceipt:
        return self.owner.submit_compute(self.node_id)


class GpuDagResources:
    """Own buffers, exact node programs, CPU references, and their CUDA receipts."""

    def __init__(self, graph: DagGraph, job: DagJob, *, execution, seed: int,
                 device: str, rank: int, group_ranks: Mapping[str, tuple[int, ...]],
                 warmup_iterations: int = 0, matmul_precision: str = "highest",
                 torch_module=None):
        if torch_module is None:
            import torch as torch_module
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        self.torch = torch_module
        self.graph, self.job, self.execution = graph, job, execution
        self.seed, self.sample_id, self.rank = seed, execution.sample_id, rank
        self.device = torch_module.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("schema-v2 GPU DAG resources require an explicit cuda:N device")
        if not torch_module.cuda.is_available() or self.device.index >= torch_module.cuda.device_count():
            raise RuntimeError(f"CUDA device {self.device} is not available")
        if matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("matmul_precision must be highest, high, or medium")
        torch_module.set_float32_matmul_precision(matmul_precision)
        torch_module.backends.cuda.matmul.allow_tf32 = False

        self.buffers_spec = dict(execution.buffers[job.job_id])
        self.programs = {
            node.node_id: dict(execution.compute_programs[f"{job.job_id}/{node.node_id}"])
            for node in job.nodes if isinstance(node, ComputeNode)
        }
        self.comm_bindings = {
            node.node_id: execution.comm_bindings[f"{job.job_id}/{node.node_id}"]["buffer"]
            for node in job.nodes if isinstance(node, CommNode)
        }
        self.buffers: dict[str, Any] = {}
        self.init_stream = torch_module.cuda.Stream(device=self.device)
        self.compute_stream = torch_module.cuda.Stream(device=self.device)
        self.init_done = torch_module.cuda.Event(enable_timing=False, blocking=False)
        self._start_events: dict[str, Any] = {}
        self._done_events: dict[str, Any] = {}
        self.node_seeds = {
            node_id: derive_tensor_seed(seed, self.sample_id, job.job_id, node_id, rank, "compute")
            for node_id in self.programs
        }
        self.input_buffers: dict[str, tuple[Any, ...]] = {}
        self._allocate_buffers()
        self.expected_by_rank, self.expected_comm_by_node = self._build_cpu_references(group_ranks)
        self._prepare_program_inputs()
        self._warmup_and_reset(warmup_iterations)
        self.node_programs = {node_id: _GpuDagNodeProgram(self, node_id) for node_id in self.programs}

    def _seeded_tensor(self, buffer_id: str, rank: int):
        torch = self.torch
        spec = self.buffers_spec[buffer_id]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_tensor_seed(self.seed, self.sample_id, self.job.job_id,
                                                buffer_id, rank, "buffer-init"))
        return torch.randn(tuple(spec["shape"]), generator=generator, dtype=torch.float32)

    def _allocate_buffers(self) -> None:
        torch = self.torch
        with torch.cuda.device(self.device), torch.cuda.stream(self.init_stream):
            for buffer_id, spec in self.buffers_spec.items():
                shape = tuple(spec["shape"])
                tensor = torch.empty(shape, dtype=torch.float32, device=self.device)
                self.buffers[buffer_id] = tensor
                if spec["init"] == "seeded-random":
                    tensor.copy_(self._seeded_tensor(buffer_id, self.rank), non_blocking=True)
                elif spec["init"] == "zeros":
                    tensor.zero_()
            self.init_done.record(self.init_stream)
        self.init_stream.synchronize()

    def _prepare_program_inputs(self) -> None:
        torch = self.torch
        with torch.cuda.device(self.device), torch.cuda.stream(self.init_stream):
            for node_id, program in self.programs.items():
                if program["op"] != "matmul":
                    continue
                tensors = []
                for input_id in program["inputs"]:
                    # Input buffers are shared outputs in the workload DAG.  The
                    # compute program reads those buffers directly at execution.
                    tensors.append(self.buffers[input_id])
                self.input_buffers[node_id] = tuple(tensors)
            self.init_done.record(self.init_stream)
        self.init_stream.synchronize()

    def _ordered_nodes(self):
        predecessors = {node.node_id: set(node.deps) for node in self.job.nodes}
        return tuple(TopologicalSorter(predecessors).static_order())

    def _build_cpu_references(self, group_ranks: Mapping[str, tuple[int, ...]]):
        torch = self.torch
        ranks = tuple(group_ranks[next(node.group_id for node in self.job.nodes
                                       if isinstance(node, CommNode))])
        references: dict[int, dict[str, Any]] = {}
        for member in ranks:
            references[member] = {}
            for buffer_id, spec in self.buffers_spec.items():
                if spec["init"] == "seeded-random":
                    references[member][buffer_id] = self._seeded_tensor(buffer_id, member)
                elif spec["init"] == "zeros":
                    references[member][buffer_id] = torch.zeros(tuple(spec["shape"]), dtype=torch.float32)
        comm_expected: dict[str, Any] = {}
        for node_id in self._ordered_nodes():
            node = next(item for item in self.job.nodes if item.node_id == node_id)
            if isinstance(node, ComputeNode):
                program = self.programs[node_id]
                for member in ranks:
                    if program["op"] == "fill":
                        value = _fill_value(self.node_seeds_for_rank(node_id, member))
                        references[member][program["output"]] = torch.full(
                            tuple(self.buffers_spec[program["output"]]["shape"]), value,
                            dtype=torch.float32)
                    elif program["op"] == "matmul":
                        left, right = (references[member][name] for name in program["inputs"])
                        result = None
                        for _ in range(program["repeats"]):
                            result = torch.mm(left, right)
                        references[member][program["output"]] = result
                    else:
                        total = sum((torch.sum(references[member][name])
                                     for name in program["inputs"]), torch.tensor(0.0))
                        references[member][program["output"]] = total.reshape(())
            else:
                buffer_id = self.comm_bindings[node_id]
                reduced = sum((references[member][buffer_id] for member in ranks),
                              torch.zeros(tuple(self.buffers_spec[buffer_id]["shape"]), dtype=torch.float32))
                comm_expected[node_id] = reduced.clone()
                for member in ranks:
                    references[member][buffer_id] = reduced.clone()
        return references[self.rank], comm_expected

    def node_seeds_for_rank(self, node_id: str, rank: int) -> int:
        return derive_tensor_seed(self.seed, self.sample_id, self.job.job_id, node_id, rank, "compute")

    def _warmup_and_reset(self, iterations: int) -> None:
        torch = self.torch
        for _ in range(iterations):
            for node_id in self._ordered_nodes():
                if node_id not in self.programs:
                    continue
                receipt = self.submit_compute(node_id)
                receipt.done_event.synchronize()
        with torch.cuda.device(self.device), torch.cuda.stream(self.init_stream):
            for buffer_id, spec in self.buffers_spec.items():
                if spec["init"] == "seeded-random":
                    self.buffers[buffer_id].copy_(self._seeded_tensor(buffer_id, self.rank), non_blocking=True)
                elif spec["init"] == "zeros":
                    self.buffers[buffer_id].zero_()
            self.init_done.record(self.init_stream)
        self.init_stream.synchronize()

    def submit_compute(self, node_id: str) -> GpuDagReceipt:
        torch = self.torch
        program = self.programs[node_id]
        start = torch.cuda.Event(enable_timing=True, blocking=False)
        done = torch.cuda.Event(enable_timing=True, blocking=False)
        with torch.cuda.device(self.device), torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(self.init_done)
            start.record(self.compute_stream)
            if program["op"] == "fill":
                self.buffers[program["output"]].fill_(_fill_value(self.node_seeds[node_id]))
            elif program["op"] == "matmul":
                left, right = (self.buffers[name] for name in program["inputs"])
                output = self.buffers[program["output"]]
                for _ in range(program["repeats"]):
                    torch.mm(left, right, out=output)
            else:
                output = self.buffers[program["output"]]
                output.zero_()
                for input_id in program["inputs"]:
                    value = torch.sum(self.buffers[input_id])
                    output.add_(value)
            done.record(self.compute_stream)
        self._start_events[node_id], self._done_events[node_id] = start, done
        return GpuDagReceipt(self, node_id, start, done)

    def make_binding(self, comm: CommNode, process_group: Any) -> LocalBinding:
        import torch.distributed as dist
        tensor = self.buffers[self.comm_bindings[comm.node_id]]

        def launch():
            return dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group, async_op=True)

        return LocalBinding(tensor, process_group, launch, device=str(self.device),
                            keepalive=(self, tensor))

    def validate(self) -> dict[str, bool]:
        torch = self.torch
        actual = {key: tensor.detach().cpu() for key, tensor in self.buffers.items()}
        results = {}
        for buffer_id, expected in self.expected_by_rank.items():
            if buffer_id not in actual:
                continue
            results[buffer_id] = bool(torch.allclose(actual[buffer_id], expected, rtol=1e-4, atol=1e-4))
        for node_id, expected in self.expected_comm_by_node.items():
            buffer_id = self.comm_bindings[node_id]
            results[f"comm:{node_id}"] = bool(torch.allclose(actual[buffer_id], expected,
                                                               rtol=1e-4, atol=1e-4))
        return results


def _fill_value(seed: int) -> float:
    return float((seed % 997 + 1) / 128.0)
