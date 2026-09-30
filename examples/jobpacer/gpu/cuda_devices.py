"""Helpers for validating inherited CUDA visibility before spawning ranks."""

from __future__ import annotations

import os


def validate_visible_cuda_devices(world_size: int) -> list[int]:
    """Return the logical per-rank ordinals without rewriting visibility."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = [item.strip() for item in visible.split(",")]
        if not tokens or any(not item for item in tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES is empty or contains an empty device token")
        normalized = [item.casefold() for item in tokens]
        if len(set(normalized)) != len(normalized):
            raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate devices")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("NCCL launch requested but CUDA is unavailable")
    count = torch.cuda.device_count()
    if count < world_size:
        raise ValueError(f"NCCL launch needs {world_size} visible GPUs, found {count}")
    return list(range(world_size))


def visible_cuda_uuids(world_size: int) -> tuple[str, ...]:
    """Return UUIDs in the inherited logical CUDA namespace without changing visibility."""
    ordinals = validate_visible_cuda_devices(world_size)
    import torch

    values = []
    for ordinal in ordinals:
        uuid = str(getattr(torch.cuda.get_device_properties(ordinal), "uuid", ""))
        if not uuid:
            raise RuntimeError(f"PyTorch did not report a UUID for cuda:{ordinal}")
        values.append(uuid)
    if len(set(values)) != len(values):
        raise RuntimeError(f"visible CUDA UUIDs are duplicated: {values}")
    return tuple(values)


def compute_profile_software(matmul_precision: str) -> dict[str, object]:
    import torch

    return {"pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "unavailable",
            "matmul_precision": matmul_precision,
            "allow_tf32": False}
