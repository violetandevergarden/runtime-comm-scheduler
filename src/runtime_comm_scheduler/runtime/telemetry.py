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
    enabled: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)
    thread_sharded: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _local: threading.local = field(default_factory=threading.local, init=False, repr=False,
                                   compare=False)
    _shards: dict[int, list[dict[str, Any]]] = field(default_factory=dict, init=False,
                                                      repr=False, compare=False)

    def record(self, kind: str, *, task_id: str | None = None, seq: int | None = None, **fields: Any) -> None:
        if not self.enabled:
            return
        event = {"source": self.source, "kind": kind, "time_us": monotonic_us(), **fields}
        if self.endpoint is not None:
            event["endpoint"] = self.endpoint
        if task_id is not None:
            event["task_id"] = task_id
        if seq is not None:
            event["seq"] = seq
        self._append(event)

    def record_at(self, kind: str, time_us: int, *, task_id: str | None = None,
                  seq: int | None = None, **fields: Any) -> None:
        if not self.enabled:
            return
        event = {"source": self.source, "kind": kind, "time_us": time_us, **fields}
        if self.endpoint is not None:
            event["endpoint"] = self.endpoint
        if task_id is not None:
            event["task_id"] = task_id
        if seq is not None:
            event["seq"] = seq
        self._append(event)

    def _append(self, event: dict[str, Any]) -> None:
        if not self.thread_sharded:
            with self._lock:
                self.events.append(event)
            return
        shard = getattr(self._local, "events", None)
        if shard is None:
            thread_id = threading.get_ident()
            with self._lock:
                shard = self._shards.get(thread_id)
                if shard is None:
                    shard = []
                    self._shards[thread_id] = shard
            self._local.events = shard
        shard.append(event)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            events = list(self.events)
            for shard in self._shards.values():
                events.extend(shard)
        if self.thread_sharded:
            events.sort(key=lambda item: item["time_us"])
        return {"source": self.source, "endpoint": self.endpoint, "events": events}

    def dumps(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
