"""通信事件时序 schema 与单调时钟。

M2 起 scheduler 在 submit/admit/submit-complete 边界记录微秒时间戳；
``now_us`` 是单调时钟（不受系统时间调整影响），供调度与实验对比使用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .intent import TaskKey


def now_us() -> int:
    """返回单调时钟的微秒时间戳（跨 rank 不可比，仅本 rank 相对比较）。"""
    return time.perf_counter_ns() // 1000


@dataclass
class CommTiming:
    key: TaskKey
    intent_ts: Optional[int] = None
    ready_record_ts: Optional[int] = None
    producer_compute_start_ts: Optional[int] = None
    submit_api_start_ts: Optional[int] = None
    submit_api_return_ts: Optional[int] = None
    admit_ts: Optional[int] = None
    collective_call_start_ts: Optional[int] = None
    collective_call_return_ts: Optional[int] = None
    consumer_compute_start_ts: Optional[int] = None
    consumer_compute_end_ts: Optional[int] = None
    application_wait_start_ts: Optional[int] = None
    underlying_wait_start_ts: Optional[int] = None
    wait_return_ts: Optional[int] = None
    application_task_end_ts: Optional[int] = None
    completion_observed_ts: Optional[int] = None
    validation_start_ts: Optional[int] = None
    validation_end_ts: Optional[int] = None
    launch_start_ts: Optional[int] = None
    submit_ts: Optional[int] = None
    first_wait_ts: Optional[int] = None
    complete_ts: Optional[int] = None
    error_ts: Optional[int] = None
    error_stage: Optional[str] = None
    predicted_duration_us: Optional[float] = None
    submit_api_duration_us: Optional[float] = None
    collective_call_duration_us: Optional[float] = None
    post_return_completion_observation_us: Optional[float] = None
    call_to_completion_observation_us: Optional[float] = None
    deferred_binding_wait_us: Optional[float] = None
    underlying_wait_duration_us: Optional[float] = None
    validation_duration_us: Optional[float] = None
    actual_duration_us: Optional[float] = None

    def set_completion_observed(self, timestamp: int) -> None:
        """Set canonical completion telemetry and its compatibility aliases."""
        self.completion_observed_ts = timestamp
        self.complete_ts = timestamp
        if self.collective_call_start_ts is not None:
            self.call_to_completion_observation_us = float(
                timestamp - self.collective_call_start_ts
            )
        if self.collective_call_return_ts is not None:
            self.post_return_completion_observation_us = float(
                timestamp - self.collective_call_return_ts
            )
        # Kept for one compatibility period. It is not a physical service time.
        start = self.collective_call_start_ts or self.submit_ts
        if start is not None:
            self.actual_duration_us = float(timestamp - start)

    def set_wait_start(self, timestamp: int) -> None:
        """Record the first application-side wait and its old alias."""
        if self.application_wait_start_ts is None:
            self.application_wait_start_ts = timestamp
            self.first_wait_ts = timestamp

    def set_underlying_wait_start(self, timestamp: int) -> None:
        if self.underlying_wait_start_ts is None:
            self.underlying_wait_start_ts = timestamp
            if self.application_wait_start_ts is not None:
                self.deferred_binding_wait_us = float(
                    timestamp - self.application_wait_start_ts
                )

    def set_wait_return(self, timestamp: int) -> None:
        if self.wait_return_ts is None:
            self.wait_return_ts = timestamp
        if self.underlying_wait_start_ts is not None:
            self.underlying_wait_duration_us = float(
                timestamp - self.underlying_wait_start_ts
            )
