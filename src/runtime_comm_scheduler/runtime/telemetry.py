"""JSON-safe event collection for replay traces."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any


def monotonic_us() -> int:
    return time.perf_counter_ns() // 1000


@dataclass
class EventLog:
    source: str
    endpoint: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def record(self, kind: str, *, task_id: str | None = None, seq: int | None = None, **fields: Any) -> None:
        event = {"source": self.source, "kind": kind, "time_us": monotonic_us(), **fields}
        if self.endpoint is not None:
            event["endpoint"] = self.endpoint
        if task_id is not None:
            event["task_id"] = task_id
        if seq is not None:
            event["seq"] = seq
        with self._lock:
            self.events.append(event)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            events = list(self.events)
        return {"source": self.source, "endpoint": self.endpoint, "events": events}

    def dumps(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
