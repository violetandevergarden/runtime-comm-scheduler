"""Immutable control-plane model for the Stage 3.1 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "float32": 4,
    "int32": 4,
    "float64": 8,
    "int64": 8,
}


@dataclass(frozen=True)
class GroupSpec:
    """某个调度轮次中，一个逻辑通信组的成员定义"""

    epoch: int   # 调度运行轮次，不是training中的epoch
    group_id: str
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.epoch < 0 or not self.group_id:
            raise ValueError("group epoch and group_id must be valid")
        if not self.ranks or any(not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 for rank in self.ranks):
            raise ValueError("group ranks must be non-negative integers")
        ranks = tuple(sorted(self.ranks))
        if len(set(ranks)) != len(ranks):
            raise ValueError("group ranks must be unique")
        object.__setattr__(self, "ranks", ranks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "group_id": self.group_id,
            "ranks": list(self.ranks),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GroupSpec":
        return cls(value["epoch"], value["group_id"], tuple(int(rank) for rank in value["ranks"]))


@dataclass(frozen=True)
class CollectiveSpec:
    """一个集合通信"""

    op: Literal["all_reduce"]
    numel: int      # tensor中的数据总数  
    num_bytes: int      # tensor的大小
    dtype: str
    shape: tuple[int, ...]
    reduction: str = "sum"

    def __post_init__(self) -> None:
        if self.op != "all_reduce":
            raise ValueError("Stage 3.1 supports only all_reduce")
        if not isinstance(self.numel, int) or isinstance(self.numel, bool) or self.numel <= 0:
            raise ValueError("numel must be a positive integer")
        if not isinstance(self.num_bytes, int) or isinstance(self.num_bytes, bool) or self.num_bytes <= 0:
            raise ValueError("num_bytes must be a positive integer")
        if not self.shape or any(not isinstance(dim, int) or dim <= 0 for dim in self.shape):
            raise ValueError("shape must contain positive dimensions")
        product = 1
        for dim in self.shape:
            product *= dim
        if product != self.numel:
            raise ValueError(f"shape product {product} does not equal numel {self.numel}")
        dtype_bytes = _DTYPE_BYTES.get(self.dtype)
        if dtype_bytes is None or dtype_bytes * self.numel != self.num_bytes:
            raise ValueError("num_bytes does not match shape, numel, and dtype")
        if not self.reduction:
            raise ValueError("reduction must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "numel": self.numel,
            "num_bytes": self.num_bytes,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "reduction": self.reduction,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CollectiveSpec":
        return cls(
            value["op"], int(value["numel"]), int(value["num_bytes"]),
            value["dtype"], tuple(int(item) for item in value["shape"]),
            value.get("reduction", "sum"),
        )


@dataclass(frozen=True)
class TaskSpec:
    """把一个集合通信包装成真正的任务，附加各种属性"""

    epoch: int
    job_id: str
    task_id: str
    group_id: str
    group_seq: int      # 此任务在通信组内的顺序，为了保证所有 rank 以相同顺序发起 collective
    collective: CollectiveSpec

    def __post_init__(self) -> None:
        if self.epoch < 0 or not self.job_id or not self.task_id or not self.group_id:
            raise ValueError("task identity fields must be non-empty")
        if self.group_seq < 0:
            raise ValueError("group_seq must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "job_id": self.job_id,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "group_seq": self.group_seq,
            "collective": self.collective.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskSpec":
        return cls(
            int(value["epoch"]), value["job_id"], value["task_id"],
            value["group_id"], int(value["group_seq"]),
            CollectiveSpec.from_dict(value["collective"]),
        )


@dataclass(frozen=True)
class TaskHint:
    ready_after_s: float | None     # 预计从协调器收到 DECLARE 起，再过多少秒能够 SUBMIT。
    estimated_comm_s: float     # 预计该 collective 自发起到真正完成需要的时间
    remaining_tail_s: float     # 预计当前通信之后，该作业还剩多少执行时间

    def __post_init__(self) -> None:
        if self.ready_after_s is not None and self.ready_after_s < 0:
            raise ValueError("ready_after_s must be non-negative or None")
        if self.estimated_comm_s < 0 or self.remaining_tail_s < 0:
            raise ValueError("task estimates must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_after_s": self.ready_after_s,
            "estimated_comm_s": self.estimated_comm_s,
            "remaining_tail_s": self.remaining_tail_s,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskHint":
        return cls(value.get("ready_after_s"), float(value["estimated_comm_s"]), float(value["remaining_tail_s"]))


@dataclass
class LocalBinding:
    """保存通信任务在当前 rank 上真正执行时所需的本地资源"""
    tensor: Any
    process_group: Any
    launch: Callable[[], Any]
    producer_event: Any = None
    device: Any = None
    keepalive: tuple[Any, ...] = ()
