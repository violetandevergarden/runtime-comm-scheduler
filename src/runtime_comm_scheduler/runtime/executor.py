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
        work = binding.launch()
        _validate_work(work)
        return work


class WorkIsCompletedProbe:
    supports_physical_completion = True

    def is_completed(self, work: Any) -> bool:
        return bool(work.is_completed())
