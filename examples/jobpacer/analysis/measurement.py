"""Derived, rank-local capacity measurements for JobPacer traces."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def occupancy_metrics(rank_trace: dict[str, Any]) -> dict[str, Any]:
    """Measure half-open admission/launch intervals in the closed run window."""
    window_start = int(rank_trace["application_release_ts"])
    window_end = int(rank_trace["communication_drain_end_ts"])
    if window_end < window_start:
        raise ValueError("communication drain precedes application release")

    tasks = [task for job in rank_trace.get("jobs", []) for task in job.get("tasks", [])]
    intervals = {
        "admission": [],
        "launched": [],
        "pending_launch": [],
    }
    boundaries = {window_start, window_end}
    for task in tasks:
        start = int(task["admit_ts"])
        launch = int(task["collective_call_start_ts"])
        complete = int(task["completion_observed_ts"])
        if not start <= launch <= complete:
            raise ValueError(f"invalid task occupancy boundaries: {task.get('key')}")
        intervals["admission"].append((start, complete))
        intervals["launched"].append((launch, complete))
        intervals["pending_launch"].append((start, launch))
        boundaries.update((start, launch, complete))

    ordered = sorted(min(window_end, max(window_start, point)) for point in boundaries)
    duration_by_count = {name: defaultdict(int) for name in intervals}
    peaks = {name: 0 for name in intervals}
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        for name, spans in intervals.items():
            count = sum(left <= start < right for left, right in spans)
            duration_by_count[name][count] += end - start
            peaks[name] = max(peaks[name], count)

    def distribution(name: str) -> dict[str, int]:
        return {
            str(count): duration
            for count, duration in sorted(duration_by_count[name].items())
        }

    weighted = {
        name: sum(count * duration for count, duration in values.items())
        for name, values in duration_by_count.items()
    }
    configured_value = rank_trace.get("max_outstanding")
    configured = int(configured_value) if configured_value is not None else None
    return {
        "configured_max_outstanding": configured,
        "peak_admission_occupancy": peaks["admission"],
        "peak_launched_inflight": peaks["launched"],
        "admission_occupancy_time_us": weighted["admission"],
        "launched_inflight_time_us": weighted["launched"],
        "pending_launch_time_us": weighted["pending_launch"],
        "admission_occupancy_by_count_us": distribution("admission"),
        "launched_inflight_by_count_us": distribution("launched"),
        "pending_launch_by_count_us": distribution("pending_launch"),
        "observation_window_us": window_end - window_start,
        "finite_capacity_ok": (
            configured is None or configured == 0 or peaks["admission"] <= configured
        ),
    }
