"""Plot saved Phase 3 makespans and rank-local execution timelines.

The plots consume existing batch summaries and raw result JSON. They never
launch replays or recompute experiment statistics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from examples.jobpacer.analysis.benchmark_paths import REPOSITORY_ROOT, repository_path


RESULTS_ROOT = REPOSITORY_ROOT / "benchmark/phase3/results"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "suites/20260923-compact/figures"
DEFAULT_TIMELINE_BATCH = (
    RESULTS_ROOT
    / "readiness/L1-head-misalignment/20260923-compact-main"
)
ARMS = (
    "old-bare-fifo",
    "old-fifo",
    "old-ltf",
    "new-static_fifo",
    "new-static_ltf",
    "new-fifo",
    "new-ltf",
    "new-lookahead",
    "new-ltf-on-ready",
    "new-ltf-precreate",
    "new-ltf-poll-1ms",
    "new-ltf-poll-0.2ms",
)
ARM_COLORS = {
    "old-bare-fifo": "#777777",
    "old-fifo": "#4c78a8",
    "old-ltf": "#f58518",
    "new-static_fifo": "#54a24b",
    "new-static_ltf": "#eeca3b",
    "new-fifo": "#b279a2",
    "new-ltf": "#e45756",
    "new-lookahead": "#72b7b2",
    "linear": "#4c78a8",
    "dag": "#f58518",
    "A": "#4c78a8",
    "B": "#f58518",
    "new-ltf-on-ready": "#f58518",
    "new-ltf-precreate": "#4c78a8",
    "new-ltf-poll-1ms": "#4c78a8",
    "new-ltf-poll-0.2ms": "#e45756",
    "before-producer": "#4c78a8",
    "on-submit": "#e45756",
}
LANES = ("compute", "preparation", "admission", "communication", "application_wait")
LANE_COLORS = {
    "compute": "#8c8c8c",
    "producer": "#8c8c8c",
    "consumer": "#55a868",
    "preparation": "#9467bd",
    "admission": "#e17c05",
    "communication": "#4c78a8",
    "application_wait": "#c44e52",
    "validation": "#8172b2",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(record: dict[str, Any], field: str) -> float | None:
    value = record.get(field)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _arm(record: dict[str, Any]) -> str:
    if record.get("arm"):
        return str(record["arm"])
    if record.get("_plot_arm"):
        return str(record["_plot_arm"])
    group = record.get("group", "")
    policy = record.get("policy", "")
    return f"{group}-{policy}" if group else str(policy)


def _arm_order(arm: str) -> tuple[int, str]:
    return (ARMS.index(arm) if arm in ARMS else len(ARMS), arm)


def _batch_label(batch_dir: str | Path) -> str:
    batch = repository_path(batch_dir).resolve()
    try:
        return batch.relative_to(RESULTS_ROOT.resolve()).as_posix()
    except ValueError:
        return batch.name


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def collect_main_batches(results_root: str | Path) -> list[dict[str, Any]]:
    """Collect successful runs from saved ``*-main`` result batches."""
    root = repository_path(results_root).resolve()
    batches = []
    for summary_path in sorted(root.rglob("summary.csv")):
        if summary_path.parent.name == "tables":
            continue
        if not summary_path.parent.name.endswith("-main"):
            continue
        rows = _read_csv(summary_path)
        successful = [
            row for row in rows
            if row.get("status") == "ok" and _number(row, "makespan_s") is not None
        ]
        if not successful:
            continue
        relative = summary_path.relative_to(root)
        scene = relative.parts[1].split("-", 1)[0]
        batches.append({
            "label": scene,
            "path": summary_path.parent,
            "rows": successful,
            "total": len(rows),
        })
    return batches


def collect_comparable_batches(results_root: str | Path) -> list[dict[str, Any]]:
    """Find strategy batches and explicit multi-source comparisons with raw runs."""
    root = repository_path(results_root).resolve()
    batches = []
    for summary_path in sorted(root.rglob("summary.csv")):
        if summary_path.parent.name == "tables":
            continue
        rows = _read_csv(summary_path)
        successful = [
            row for row in rows
            if row.get("status") == "ok"
            and _number(row, "makespan_s") is not None
        ]
        arms = {_arm(row) for row in successful}
        if len(arms) < 2:
            continue
        batch_path = summary_path.parent.resolve()
        batches.append({
            "label": batch_path.relative_to(root).as_posix(),
            "path": batch_path,
            "rows": successful,
            "total": len(rows),
        })

    # The bridge compares the linear and DAG adapters under one shared policy;
    # the treatment arm is the input mode, not a fabricated policy difference.
    for batch_path in sorted((root / "bridge").glob("*/*")):
        rows = []
        total = 0
        for input_mode in ("linear", "dag"):
            summary_path = batch_path / input_mode / "summary.csv"
            if not summary_path.is_file():
                continue
            source_rows = _read_csv(summary_path)
            total += len(source_rows)
            for row in source_rows:
                if row.get("status") != "ok" or _number(row, "makespan_s") is None:
                    continue
                row["_plot_arm"] = input_mode
                row["_raw_path"] = str(
                    summary_path.parent / "raw" / f"{row['run_id']}.json"
                )
                rows.append(row)
        if len({_arm(row) for row in rows}) >= 2:
            batches.append({
                "label": batch_path.relative_to(root).as_posix(),
                "path": batch_path,
                "rows": rows,
                "total": total,
            })

    # Noise runs are stored as seed/A and seed/B leaves. Combine those leaves
    # at the noise batch level so the real A/B treatment is plotted together.
    noise_batches: dict[Path, list[dict[str, Any]]] = {}
    noise_totals: dict[Path, int] = {}
    for summary_path in sorted((root / "noise").rglob("summary.csv")):
        if summary_path.parent.name not in {"A", "B"}:
            continue
        batch_path = summary_path.parent.parent.parent.resolve()
        source_rows = _read_csv(summary_path)
        noise_totals[batch_path] = noise_totals.get(batch_path, 0) + len(source_rows)
        for row in source_rows:
            if row.get("status") != "ok" or _number(row, "makespan_s") is None:
                continue
            row["_plot_arm"] = summary_path.parent.name
            row["_raw_path"] = str(
                summary_path.parent / "raw" / f"{row['run_id']}.json"
            )
            noise_batches.setdefault(batch_path, []).append(row)
    for batch_path, rows in sorted(noise_batches.items()):
        if len({_arm(row) for row in rows}) >= 2:
            batches.append({
                "label": batch_path.relative_to(root).as_posix(),
                "path": batch_path,
                "rows": rows,
                "total": noise_totals[batch_path],
            })

    batches.sort(key=lambda batch: batch["label"])
    return batches


def select_representative_runs(
    batch_dir: str | Path, rows: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Pick one successful, median-makespan raw trace for each arm."""
    batch = repository_path(batch_dir).resolve()
    if rows is None:
        rows = [
            row for row in _read_csv(batch / "summary.csv")
            if row.get("status") == "ok" and _number(row, "makespan_s") is not None
        ]
    selected = []
    arms = sorted({_arm(row) for row in rows}, key=_arm_order)
    for arm in arms:
        arm_rows = [row for row in rows if _arm(row) == arm]
        median = _percentile([_number(row, "makespan_s") for row in arm_rows], 0.5)
        arm_rows.sort(
            key=lambda row: (abs(_number(row, "makespan_s") - median), row["run_id"])
        )
        run = arm_rows[0]
        raw_path = Path(run.get("_raw_path", batch / "raw" / f"{run['run_id']}.json"))
        if not raw_path.is_file():
            raise FileNotFoundError(f"summary-listed trace is missing: {raw_path}")
        trace = json.loads(raw_path.read_text(encoding="utf-8"))
        if trace.get("validation", {}).get("status") != "ok":
            raise ValueError(f"representative trace did not validate: {raw_path}")
        selected.append({
            "arm": arm,
            "run_id": run["run_id"],
            "makespan_ms": float(run["makespan_s"]) * 1000,
            "path": raw_path,
            "trace": trace,
        })
    if not selected:
        raise ValueError(f"no successful runs found in {batch}")
    return selected


def _add_interval(
    output: list[dict[str, Any]], lane: str, kind: str,
    start: Any, end: Any, origin: float,
) -> None:
    start_value = _number({"value": start}, "value")
    end_value = _number({"value": end}, "value")
    if start_value is None or end_value is None:
        raise ValueError(f"missing timestamp for timeline interval {kind}")
    if end_value < start_value:
        raise ValueError(f"reversed timestamps for timeline interval {kind}")
    output.append({
        "lane": lane,
        "kind": kind,
        "start_ms": (start_value - origin) / 1000,
        "end_ms": (end_value - origin) / 1000,
    })


def phase3_timeline_data(trace: dict[str, Any], rank: int) -> dict[str, Any]:
    """Build Phase 3 intervals on one rank's monotonic clock, relative to release."""
    rank_record = next(
        (item for item in trace.get("ranks", []) if item.get("rank") == rank), None
    )
    if rank_record is None:
        raise ValueError(f"trace has no rank {rank}")
    origin = _number(rank_record, "application_release_ts")
    if origin is None:
        raise ValueError(f"rank {rank} trace has no application_release_ts")

    runtime_events: dict[str, dict[str, float]] = {}
    for event in rank_record.get("runtime_events", []):
        task_id = event.get("task_id")
        timestamp = _number(event, "time_us")
        if task_id is not None and timestamp is not None:
            runtime_events.setdefault(str(task_id), {})[str(event.get("kind"))] = timestamp

    jobs_by_id: dict[str, dict[str, Any]] = {}
    is_dag = trace.get("input_mode") == "dag" or bool(rank_record.get("dag_events"))

    # Linear Phase 3 tasks and the preserved Phase 1/2 baseline traces use
    # different field names but share a clock unit and the same plot lanes.
    if not is_dag:
        for job in rank_record.get("jobs", []):
            job_id = str(job["job_id"])
            job_data = jobs_by_id.setdefault(job_id, {"job_id": job_id, "intervals": [], "markers": []})
            for task in job.get("tasks", []):
                ordinal = task.get("ordinal", task.get("key", "?"))
                task_id = str(task.get("task_id", f"{job_id}/{ordinal}"))
                events = runtime_events.get(task_id, {})
                is_new_runtime = "producer_start_ts" in task
                if is_new_runtime:
                    producer_start, ready = task.get("producer_start_ts"), task.get("ready_ts")
                    admit = events.get("grant_received")
                    submit_call = task.get("submit_call_ts")
                    submit_return = task.get("submit_return_ts")
                    collective_start = events.get("collective_call_start", events.get("launch_start"))
                    collective_return = events.get("collective_call_return")
                    complete = events.get("completion_observed")
                    consumer_start, consumer_end = task.get("consumer_start_ts"), task.get("first_wait_ts")
                    wait_start, wait_end = task.get("first_wait_ts"), task.get("consumer_end_ts")
                    prepare_start = task.get("binding_create_start_ts")
                    prepare_end = task.get("binding_create_end_ts")
                else:
                    producer_start = task.get("producer_compute_start_ts")
                    ready = task.get("ready_record_ts")
                    admit = task.get("admit_ts")
                    submit_call = task.get("submit_api_start_ts")
                    submit_return = task.get("submit_api_return_ts")
                    collective_start = task.get("collective_call_start_ts")
                    collective_return = task.get("collective_call_return_ts")
                    complete = task.get("completion_observed_ts")
                    consumer_start = task.get("consumer_compute_start_ts")
                    consumer_end = task.get("consumer_compute_end_ts")
                    wait_start, wait_end = task.get("application_wait_start_ts"), task.get("wait_return_ts")
                    prepare_start = task.get("tensor_create_start_ts")
                    prepare_end = task.get("tensor_create_end_ts")

                intervals = job_data["intervals"]
                if is_new_runtime:
                    declare_start = events.get("declare_call_start")
                    declare_end = events.get("declare_call_end")
                    if declare_start is not None and declare_end is not None:
                        _add_interval(intervals, "preparation", "DECLARE call",
                                      declare_start, declare_end, origin)
                _add_interval(intervals, "compute", "producer", producer_start, ready, origin)
                if prepare_start is not None and prepare_end is not None:
                    _add_interval(intervals, "preparation", "tensor/binding creation",
                                  prepare_start, prepare_end, origin)
                if ready is not None and submit_call is not None and submit_call >= ready:
                    _add_interval(intervals, "preparation", "producer ready → submit call",
                                  ready, submit_call, origin)
                if submit_call is not None and submit_return is not None:
                    _add_interval(intervals, "preparation", "submit API call",
                                  submit_call, submit_return, origin)
                if submit_call is not None and admit is not None:
                    _add_interval(intervals, "admission", "submit call → grant/admit",
                                  submit_call, admit, origin)
                if admit is not None and collective_start is not None:
                    _add_interval(intervals, "admission", "grant/admit → collective start",
                                  admit, collective_start, origin)
                if collective_start is not None and complete is not None:
                    _add_interval(intervals, "communication", "collective start → completion observation",
                                  collective_start, complete, origin)
                _add_interval(intervals, "compute", "consumer", consumer_start, consumer_end, origin)
                _add_interval(intervals, "application_wait", "application/backend wait", wait_start, wait_end, origin)
                marker = {"ready_ms": (float(ready) - origin) / 1000}
                for name, timestamp in (("submit_call_ms", submit_call), ("submit_return_ms", submit_return),
                                        ("admit_ms", admit), ("collective_start_ms", collective_start),
                                        ("collective_return_ms", collective_return),
                                        ("complete_ms", complete), ("wait_return_ms", wait_end)):
                    if timestamp is not None:
                        marker[name] = (float(timestamp) - origin) / 1000
                if is_new_runtime:
                    for name, event_name in (
                        ("declare_call_start_ms", "declare_call_start"),
                        ("declare_call_end_ms", "declare_call_end"),
                        ("submit_runtime_start_ms", "submit_call_start"),
                        ("offer_send_start_ms", "offer_send_start"),
                        ("offer_send_end_ms", "offer_send_end"),
                        ("grant_control_dequeued_ms", "grant_control_loop_dequeued"),
                        ("launch_queue_put_start_ms", "launch_queue_put_start"),
                        ("launch_queue_put_end_ms", "launch_queue_put_end"),
                        ("launch_worker_dequeued_ms", "launch_worker_dequeued"),
                    ):
                        timestamp = events.get(event_name)
                        if timestamp is not None:
                            marker[name] = (float(timestamp) - origin) / 1000
                    for name, event_name in (("submitted_send_start_ms", "submitted_send_start"),
                                             ("submitted_send_end_ms", "submitted_send_end")):
                        timestamp = events.get(event_name)
                        if timestamp is not None:
                            marker[name] = (float(timestamp) - origin) / 1000
                job_data["markers"].append(marker)

    # DAG traces expose compute and communication node events separately.
    # Use the same job-level lanes, without comparing clocks across ranks.
    if is_dag and rank_record.get("dag_events"):
        dag_nodes: dict[tuple[str, str], dict[str, Any]] = {}
        for event in rank_record["dag_events"]:
            job_id, node_id = event.get("job_id"), event.get("node_id")
            if job_id is not None and node_id is not None:
                dag_nodes.setdefault((str(job_id), str(node_id)), {})[
                    str(event.get("kind"))
                ] = event
        for (job_id, node_id), events in sorted(dag_nodes.items()):
            job_data = jobs_by_id.setdefault(job_id, {"job_id": job_id, "intervals": [], "markers": []})
            task_id = f"{job_id}/{node_id}"
            runtime = runtime_events.get(task_id, {})
            if "compute_started" in events:
                _add_interval(
                    job_data["intervals"], "compute", "compute",
                    events["compute_started"].get("time_us"),
                    events.get("compute_completed", {}).get("time_us"), origin,
                )
            if "node_ready" in events and runtime.get("grant_received") is not None:
                ready = events["node_ready"].get("time_us")
                admit = runtime["grant_received"]
                _add_interval(job_data["intervals"], "admission", "ready → grant/admit", ready, admit, origin)
                job_data["markers"].append({"ready_ms": (float(ready) - origin) / 1000,
                                             "admit_ms": (admit - origin) / 1000})
            submit = runtime.get("launch_start")
            complete = runtime.get("completion_observed")
            if submit is not None and complete is not None:
                _add_interval(job_data["intervals"], "communication", "collective → completion observed",
                              submit, complete, origin)
                job_data["markers"].append({"submit_ms": (submit - origin) / 1000,
                                             "complete_ms": (complete - origin) / 1000})
            wait_start = events.get("comm_submit_return", {}).get("time_us")
            wait_end = events.get("comm_completed_observed", {}).get("time_us")
            if wait_start is not None and wait_end is not None:
                _add_interval(job_data["intervals"], "application_wait", "application/backend wait",
                              wait_start, wait_end, origin)

    if not jobs_by_id:
        raise ValueError("trace has no linear-task or DAG timeline events")

    validation_start = _number(rank_record, "validation_start_ts")
    validation_end = _number(rank_record, "validation_end_ts")
    validation = None
    if validation_start is not None and validation_end is not None:
        validation = {
            "start_ms": (validation_start - origin) / 1000,
            "end_ms": (validation_end - origin) / 1000,
        }
    makespan_us = _number(rank_record, "application_makespan_us")
    run_makespan_us = _number(trace.get("performance", {}), "workload_makespan_us")
    if run_makespan_us is None:
        rank_durations = [_number(item, "application_makespan_us") for item in trace.get("ranks", [])]
        run_makespan_us = max((item for item in rank_durations if item is not None), default=None)
    return {
        "rank": rank,
        "jobs": [jobs_by_id[key] for key in sorted(jobs_by_id)],
        "validation": validation,
        "makespan_ms": makespan_us / 1000 if makespan_us is not None else None,
        "workload_makespan_ms": run_makespan_us / 1000 if run_makespan_us is not None else None,
    }


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "visualization requires Matplotlib; install with "
            "python -m pip install -e '.[visualization]'"
        ) from exc
    return plt, Line2D, Patch


def _save(figure: Any, output: str | Path) -> Path:
    destination = repository_path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg")
    return destination


def render_makespan_comparison(results_root: str | Path, output: str | Path) -> Path:
    """Compare run-level makespans for all saved main workload batches."""
    plt, _Line2D, _Patch = _matplotlib()
    batches = collect_main_batches(results_root)
    if not batches:
        raise ValueError(f"no successful main batches found under {results_root}")
    batches.sort(key=lambda batch: batch["label"])
    figure, axes = plt.subplots(
        len(batches), 1, figsize=(13, max(4.2, 3.7 * len(batches))), squeeze=False
    )
    for axis, batch in zip(axes.flat, batches):
        arms = sorted({_arm(row) for row in batch["rows"]}, key=_arm_order)
        for x, arm in enumerate(arms):
            values = [float(row["makespan_s"]) * 1000 for row in batch["rows"] if _arm(row) == arm]
            color = ARM_COLORS.get(arm, "#4c78a8")
            jitter = [((index % 9) - 4) * 0.025 for index in range(len(values))]
            axis.scatter([x + offset for offset in jitter], values, color=color, alpha=0.4, s=16)
            axis.vlines(x, _percentile(values, 0.1), _percentile(values, 0.9), color=color, linewidth=2.2)
            median = _percentile(values, 0.5)
            axis.plot([x - 0.16, x + 0.16], [median, median], color="black", linewidth=1.4)
        axis.set_title(f"{batch['label']} · successful runs {len(batch['rows'])}/{batch['total']}", loc="left")
        axis.set_ylabel("workload makespan (ms)")
        axis.set_xticks(range(len(arms)))
        axis.set_xticklabels(arms, rotation=18, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Phase 3 main batches · overall workload makespan\n"
        "dots = runs; colored bars = descriptive P10–P90; black bar = median. "
        "Old/new arms compare whole execution paths.",
        y=0.995, fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2)
    path = _save(figure, output)
    plt.close(figure)
    return path


def render_batch_makespan_comparison(
    batch_dir: str | Path,
    output: str | Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    label: str | None = None,
) -> Path:
    """Compare all successful arms within one result batch."""
    plt, _Line2D, _Patch = _matplotlib()
    batch_path = repository_path(batch_dir).resolve()
    if rows is None:
        rows = [
            row for row in _read_csv(batch_path / "summary.csv")
            if row.get("status") == "ok" and _number(row, "makespan_s") is not None
        ]
    arms = sorted({_arm(row) for row in rows}, key=_arm_order)
    if len(arms) < 2:
        raise ValueError(f"batch has fewer than two successful arms: {batch_path}")

    figure, axis = plt.subplots(1, 1, figsize=(max(8, 1.55 * len(arms) + 3), 5.4))
    for x, arm in enumerate(arms):
        values = [float(row["makespan_s"]) * 1000 for row in rows if _arm(row) == arm]
        color = ARM_COLORS.get(arm, "#4c78a8")
        jitter = [((index % 9) - 4) * 0.025 for index in range(len(values))]
        axis.scatter([x + offset for offset in jitter], values, color=color, alpha=0.4, s=18)
        axis.vlines(x, _percentile(values, 0.1), _percentile(values, 0.9), color=color, linewidth=2.2)
        axis.plot([x - 0.16, x + 0.16], [_percentile(values, 0.5)] * 2, color="black", linewidth=1.4)
    axis.set_ylabel("workload makespan (ms)")
    axis.set_xticks(range(len(arms)))
    axis.set_xticklabels(arms, rotation=18, ha="right", fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"{label or _batch_label(batch_path)}\n"
        "dots = successful runs; colored bars = descriptive P10–P90; black bar = median",
        y=0.99, fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    path = _save(figure, output)
    plt.close(figure)
    return path


def render_timeline_comparison(
    batch_dir: str | Path,
    rank: int,
    output: str | Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    label: str | None = None,
    x_min_ms: float | None = None,
    x_max_ms: float | None = None,
) -> Path:
    """Render median-makespan representative traces as Phase 1.2-style panels."""
    plt, Line2D, Patch = _matplotlib()
    selected = select_representative_runs(batch_dir, rows)
    data = [phase3_timeline_data(item["trace"], rank) for item in selected]
    job_count = max(len(item["jobs"]) for item in data)
    lane_order = {lane: index for index, lane in enumerate(LANES)}
    job_stride = len(LANES) + 1
    panel_height = max(5.2, 0.25 * (job_count * job_stride + 2) + 1.2)
    figure, axes = plt.subplots(
        len(selected), 1,
        figsize=(16, panel_height * len(selected) + 2.0),
        sharex=True, squeeze=False,
    )
    for axis, item, panel in zip(axes.flat, selected, data):
        labels = []
        positions = []
        for job_index, job in enumerate(panel["jobs"]):
            base_y = job_index * job_stride
            for lane in LANES:
                positions.append(base_y + lane_order[lane])
                labels.append(f"{job['job_id']} {lane.replace('_', ' ')}")
            for interval in job["intervals"]:
                lane_y = base_y + lane_order[interval["lane"]]
                color = LANE_COLORS.get(interval["kind"], LANE_COLORS[interval["lane"]])
                axis.barh(
                    lane_y,
                    interval["end_ms"] - interval["start_ms"],
                    left=interval["start_ms"],
                    height=0.48,
                    color=color,
                    alpha=0.85,
                )
            for marker in job["markers"]:
                ready = marker.get("ready_ms")
                if ready is not None:
                    axis.plot(ready, base_y + 0.28, "^", color="black", ms=5)
                for key, lane in (("submit_call_ms", "preparation"), ("submit_return_ms", "preparation"),
                                  ("admit_ms", "admission"), ("submitted_send_start_ms", "admission"),
                                  ("submitted_send_end_ms", "admission"),
                                  ("collective_start_ms", "communication"),
                                  ("collective_return_ms", "communication"),
                                  ("complete_ms", "communication"), ("wait_return_ms", "application_wait")):
                    value = marker.get(key)
                    if value is not None:
                        y = base_y + lane_order[lane]
                        axis.vlines(value, y - 0.28, y + 0.28, color="black", linewidth=0.8)
        validation_y = len(panel["jobs"]) * job_stride
        positions.append(validation_y)
        labels.append("harness validation")
        validation = panel["validation"]
        if validation is not None:
            axis.barh(
                validation_y,
                validation["end_ms"] - validation["start_ms"],
                left=validation["start_ms"], height=0.48,
                color=LANE_COLORS["validation"], alpha=0.85,
            )
        axis.set_title(
            f"{item['arm']} · representative {item['run_id']} · "
            f"makespan {item['makespan_ms']:.3f} ms",
            fontsize=10,
        )
        axis.set_xlabel("time from this rank's application release (ms)")
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.tick_params(axis="y", labelsize=9, pad=8)
        axis.grid(axis="x", alpha=0.25)
        if x_min_ms is not None or x_max_ms is not None:
            axis.set_xlim(x_min_ms, x_max_ms)

    interval_legend = [
        Patch(color=LANE_COLORS["producer"], label="producer compute"),
        Patch(color=LANE_COLORS["consumer"], label="consumer compute"),
        Patch(color=LANE_COLORS["preparation"], label="local preparation / submit API"),
        Patch(color=LANE_COLORS["admission"], label="admission and local emission waits"),
        Patch(color=LANE_COLORS["communication"], label="collective start → completion observation"),
        Patch(color=LANE_COLORS["application_wait"], label="application/backend wait"),
        Patch(color=LANE_COLORS["validation"], label="harness validation"),
        Line2D([], [], marker="^", color="black", linestyle="None", label="ready"),
        Line2D([], [], marker="|", color="black", linestyle="None", label="grant/admit · submit · complete"),
    ]
    figure.legend(handles=interval_legend, loc="lower center", ncol=4, fontsize=8, frameon=False)
    figure.suptitle(
        f"Phase 3 timeline comparison · {label or _batch_label(batch_dir)} · rank {rank}\n"
        "independent representative runs nearest each arm's median, not paired samples; "
        "rank-local clocks are rebased independently",
        y=0.995, fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.94), h_pad=1.8)
    path = _save(figure, output)
    plt.close(figure)
    return path


def select_paired_runs(batch_dir: str | Path, rows: list[dict[str, Any]]) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    """Use the lexicographically first seed/repeat block with 2+ successful arms."""
    batch = repository_path(batch_dir).resolve()
    by_block: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or _number(row, "makespan_s") is None:
            continue
        block = (int(row["epoch"]), int(row["repeat"]))
        arm = _arm(row)
        if arm in by_block.setdefault(block, {}):
            raise ValueError(f"duplicate arm in paired block {block}: {arm}")
        by_block[block][arm] = row
    selected_block = next((block for block in sorted(by_block) if len(by_block[block]) >= 2), None)
    if selected_block is None:
        raise ValueError(f"no seed/repeat block has two successful arms in {batch}")
    selected = []
    for arm, row in sorted(by_block[selected_block].items(), key=lambda item: _arm_order(item[0])):
        raw_path = Path(row.get("_raw_path", batch / "raw" / f"{row['run_id']}.json"))
        if not raw_path.is_file():
            raise FileNotFoundError(f"paired trace is missing: {raw_path}")
        trace = json.loads(raw_path.read_text(encoding="utf-8"))
        if trace.get("validation", {}).get("status") != "ok":
            raise ValueError(f"paired trace did not validate: {raw_path}")
        selected.append({"arm": arm, "run_id": row["run_id"],
                         "makespan_ms": float(row["makespan_s"]) * 1000,
                         "path": raw_path, "trace": trace})
    return selected_block, selected


def render_paired_timeline_comparison(
    batch_dir: str | Path, output: str | Path, *,
    rows: list[dict[str, Any]] | None = None, label: str | None = None,
) -> Path:
    """Plot the same seed/repeat for every arm, with independent rank clocks."""
    plt, Line2D, Patch = _matplotlib()
    batch = repository_path(batch_dir).resolve()
    if rows is None:
        rows = _read_csv(batch / "summary.csv")
    block, selected = select_paired_runs(batch, rows)
    panel_data = [
        (item, [phase3_timeline_data(item["trace"], rank) for rank in (0, 1)])
        for item in selected
    ]
    job_count = max(len(panel["jobs"]) for _item, ranks in panel_data for panel in ranks)
    stride = len(LANES) + 1
    height = max(5.2, 0.23 * (job_count * stride + 2) + 1.2)
    figure, axes = plt.subplots(
        len(selected), 2, figsize=(18, height * len(selected) + 1.8),
        sharex=True, squeeze=False,
    )
    for row_index, (item, ranks) in enumerate(panel_data):
        for rank, panel in enumerate(ranks):
            axis = axes[row_index, rank]
            labels = []
            positions = []
            for job_index, job in enumerate(panel["jobs"]):
                base_y = job_index * stride
                for lane_index, lane in enumerate(LANES):
                    positions.append(base_y + lane_index)
                    labels.append(f"{job['job_id']} {lane.replace('_', ' ')}")
                for interval in job["intervals"]:
                    lane_y = base_y + LANES.index(interval["lane"])
                    axis.barh(lane_y, interval["end_ms"] - interval["start_ms"],
                              left=interval["start_ms"], height=0.48,
                              color=LANE_COLORS.get(interval["kind"],
                                                    LANE_COLORS[interval["lane"]]), alpha=0.85)
                for marker in job["markers"]:
                    ready = marker.get("ready_ms")
                    if ready is not None:
                        axis.plot(ready, base_y + 0.28, "^", color="black", ms=4)
                    marker_lanes = (
                        ("submit_call_ms", "preparation"), ("submit_return_ms", "preparation"),
                        ("admit_ms", "admission"), ("submitted_send_start_ms", "admission"),
                        ("submitted_send_end_ms", "admission"),
                        ("collective_start_ms", "communication"),
                        ("collective_return_ms", "communication"), ("complete_ms", "communication"),
                        ("wait_return_ms", "application_wait"),
                    )
                    for name, lane in marker_lanes:
                        value = marker.get(name)
                        if value is not None:
                            y = base_y + LANES.index(lane)
                            axis.vlines(value, y - 0.25, y + 0.25, color="black", linewidth=0.7)
            validation_y = len(panel["jobs"]) * stride
            positions.append(validation_y)
            labels.append("harness validation")
            validation = panel["validation"]
            if validation is not None:
                axis.barh(validation_y, validation["end_ms"] - validation["start_ms"],
                          left=validation["start_ms"], height=0.48,
                          color=LANE_COLORS["validation"], alpha=0.85)
            axis.set_title(
                f"{item['arm']} · rank {rank} · local application {panel['makespan_ms']:.3f} ms\n"
                f"run {item['run_id']} · workload makespan {panel['workload_makespan_ms']:.3f} ms",
                fontsize=9,
            )
            axis.set_xlabel("time from this rank's application release (ms)")
            axis.set_yticks(positions)
            axis.set_yticklabels(labels, fontsize=8)
            axis.grid(axis="x", alpha=0.25)
    legend = [
        Patch(color=LANE_COLORS["producer"], label="producer compute"),
        Patch(color=LANE_COLORS["preparation"], label="local preparation / submit API"),
        Patch(color=LANE_COLORS["admission"], label="admission / launch wait"),
        Patch(color=LANE_COLORS["communication"], label="collective start → completion observation"),
        Patch(color=LANE_COLORS["consumer"], label="consumer compute"),
        Patch(color=LANE_COLORS["application_wait"], label="application wait"),
        Patch(color=LANE_COLORS["validation"], label="harness validation"),
        Line2D([], [], marker="^", color="black", linestyle="None", label="producer ready"),
        Line2D([], [], marker="|", color="black", linestyle="None",
               label="submit / grant / SUBMITTED / call / return / completion / wait return"),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=3, fontsize=8, frameon=False)
    figure.suptitle(
        f"Paired Phase 3 timeline diagnostic · {label or _batch_label(batch)}\n"
        f"predeclared selection: first available seed {block[0]}, repeat {block[1]}; "
        "rank clocks are rebased independently",
        y=0.995, fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=1.2, w_pad=1.0)
    path = _save(figure, output)
    plt.close(figure)
    return path


def render_rank_pair_timeline(
    trace: dict[str, Any] | str | Path, output: str | Path, *, label: str,
) -> Path:
    """Render one run's two rank-local timelines without cross-rank subtraction."""
    plt, Line2D, Patch = _matplotlib()
    if isinstance(trace, (str, Path)):
        trace = json.loads(Path(trace).read_text(encoding="utf-8"))
    panels = [phase3_timeline_data(trace, rank) for rank in (0, 1)]
    job_count = max(len(panel["jobs"]) for panel in panels)
    stride = len(LANES) + 1
    height = max(6.0, 0.25 * (job_count * stride + 2) + 1.2)
    figure, axes = plt.subplots(1, 2, figsize=(18, height), sharex=True, squeeze=False)
    marker_lanes = (
        ("declare_call_start_ms", "preparation"), ("declare_call_end_ms", "preparation"),
        ("submit_call_ms", "preparation"), ("submit_runtime_start_ms", "preparation"),
        ("offer_send_start_ms", "admission"), ("offer_send_end_ms", "admission"),
        ("submit_return_ms", "preparation"), ("admit_ms", "admission"),
        ("grant_control_dequeued_ms", "admission"), ("launch_queue_put_start_ms", "admission"),
        ("launch_queue_put_end_ms", "admission"), ("launch_worker_dequeued_ms", "admission"),
        ("collective_start_ms", "communication"), ("collective_return_ms", "communication"),
        ("complete_ms", "communication"), ("wait_return_ms", "application_wait"),
    )
    for rank, (axis, panel) in enumerate(zip(axes.flat, panels)):
        positions, labels = [], []
        for job_index, job in enumerate(panel["jobs"]):
            base_y = job_index * stride
            for lane_index, lane in enumerate(LANES):
                positions.append(base_y + lane_index)
                labels.append(f"{job['job_id']} {lane.replace('_', ' ')}")
            for interval in job["intervals"]:
                lane_y = base_y + LANES.index(interval["lane"])
                axis.barh(lane_y, interval["end_ms"] - interval["start_ms"],
                          left=interval["start_ms"], height=0.48,
                          color=LANE_COLORS.get(interval["kind"],
                                                LANE_COLORS[interval["lane"]]), alpha=0.82)
            for marker in job["markers"]:
                ready = marker.get("ready_ms")
                if ready is not None:
                    axis.plot(ready, base_y + 0.28, "^", color="black", ms=4)
                for name, lane in marker_lanes:
                    value = marker.get(name)
                    if value is not None:
                        y = base_y + LANES.index(lane)
                        axis.vlines(value, y - 0.24, y + 0.24,
                                    color="#222222", linewidth=0.55, alpha=0.8)
        validation_y = len(panel["jobs"]) * stride
        positions.append(validation_y)
        labels.append("harness validation")
        validation = panel["validation"]
        if validation is not None:
            axis.barh(validation_y, validation["end_ms"] - validation["start_ms"],
                      left=validation["start_ms"], height=0.48,
                      color=LANE_COLORS["validation"], alpha=0.82)
        axis.set_title(
            f"rank {rank} · local application {panel['makespan_ms']:.3f} ms\n"
            f"workload makespan {panel['workload_makespan_ms']:.3f} ms",
            fontsize=10,
        )
        axis.set_xlabel("time from this rank's application release (ms)")
        axis.set_yticks(positions)
        axis.set_yticklabels(labels, fontsize=8)
        axis.grid(axis="x", alpha=0.25)
    figure.legend(handles=[
        Patch(color=LANE_COLORS["producer"], label="producer"),
        Patch(color=LANE_COLORS["preparation"], label="DECLARE / local preparation"),
        Patch(color=LANE_COLORS["admission"], label="OFFER / grant / launch handoff"),
        Patch(color=LANE_COLORS["communication"], label="collective → completion observation"),
        Patch(color=LANE_COLORS["application_wait"], label="application wait"),
        Line2D([], [], marker="|", color="#222222", linestyle="None",
               label="local request, grant, launch and completion boundaries"),
    ], loc="lower center", ncol=3, fontsize=8, frameon=False)
    figure.suptitle(
        f"Paired rank-local control-path timeline · {label}\n"
        "The two panels use independent local clocks; absolute rank timestamps are not compared.",
        y=0.995, fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95), w_pad=1.2)
    path = _save(figure, output)
    plt.close(figure)
    return path


def write_overhead_diagnostics(batch_dir: str | Path) -> list[Path]:
    """Export rank-local polling diagnostics and a concise batch report."""
    batch = repository_path(batch_dir).resolve()
    summary_rows = _read_csv(batch / "summary.csv")
    diagnostics: list[dict[str, Any]] = []
    for row in summary_rows:
        raw_path = batch / "raw" / f"{row['run_id']}.json"
        if not raw_path.is_file():
            continue
        trace = json.loads(raw_path.read_text(encoding="utf-8"))
        rank_timings = trace.get("metrics", {}).get("rank_task_timings", {})
        for rank_record in trace.get("ranks", []):
            rank = str(rank_record.get("rank"))
            task_map = rank_timings.get(rank, {})
            task_values = list(task_map.values())

            def median_task_metric(name: str) -> float | None:
                values = [float(value[name]) for value in task_values
                          if isinstance(value.get(name), (int, float))]
                return statistics.median(values) if values else None

            probes = [float(value["completion_probe_count"]) for value in task_values
                      if isinstance(value.get("completion_probe_count"), (int, float))]
            diagnostics.append({
                "run_id": row["run_id"],
                "arm": _arm(row),
                "status": row.get("status", ""),
                "epoch": row.get("epoch", ""),
                "repeat": row.get("repeat", ""),
                "rank": rank,
                "poll_interval_s": row.get("poll_interval_s", ""),
                "application_makespan_ms": _scale(rank_record.get("application_makespan_us"), 1 / 1000),
                "workload_makespan_ms": _scale(_number(row, "makespan_s"), 1000),
                "communication_drain_makespan_ms": _scale(rank_record.get("communication_drain_makespan_us"), 1 / 1000),
                "preparation_total_ms": _scale(rank_record.get("preparation_total_us"), 1 / 1000),
                "binding_creation_total_ms": _scale(rank_record.get("binding_creation_total_us"), 1 / 1000),
                "process_cpu_ms": _scale(rank_record.get("process_cpu_time_s"), 1000),
                "voluntary_context_switches": rank_record.get("voluntary_context_switches"),
                "involuntary_context_switches": rank_record.get("involuntary_context_switches"),
                "task_count": len(task_values),
                "median_application_wait_ms": _scale(median_task_metric("application_wait_s"), 1000),
                "median_completion_to_continue_ms": _scale(
                    median_task_metric("completion_observed_to_application_continue_s"), 1000
                ),
                "median_call_return_to_observation_ms": _scale(
                    median_task_metric("call_return_to_completion_observation_s"), 1000
                ),
                "completion_probe_count": sum(probes) if probes else None,
            })

    diagnostic_fields = [
        "run_id", "arm", "status", "epoch", "repeat", "rank", "poll_interval_s",
        "application_makespan_ms", "workload_makespan_ms", "communication_drain_makespan_ms",
        "preparation_total_ms", "binding_creation_total_ms", "process_cpu_ms",
        "voluntary_context_switches", "involuntary_context_switches", "task_count",
        "median_application_wait_ms", "median_completion_to_continue_ms",
        "median_call_return_to_observation_ms", "completion_probe_count",
    ]
    table_dir = batch / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = batch / "process-diagnostics.csv"
    for path in (diagnostics_path, table_dir / diagnostics_path.name):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=diagnostic_fields)
            writer.writeheader()
            writer.writerows(diagnostics)

    coordinator_diagnostics = _collect_coordinator_diagnostics(batch, summary_rows)
    coordinator_fields = [
        "run_id", "arm", "status", "epoch", "repeat", "task_id", "next_task_id",
        "grant_to_all_submitted_ms", "all_submitted_to_all_completed_ms",
        "all_completed_to_next_grant_ms", "eligible_to_grant_ms",
        "legal_candidate_present_during_gap",
    ]
    coordinator_path = batch / "coordinator-diagnostics.csv"
    for path in (coordinator_path, table_dir / coordinator_path.name):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=coordinator_fields)
            writer.writeheader()
            writer.writerows(coordinator_diagnostics)

    report_lines = [
        f"# Runtime overhead batch: {batch.name}",
        "",
        "Generated from the preserved run summaries and raw rank traces. Intervals and CPU/context-switch values are rank-local; no cross-rank timestamps are subtracted.",
        "",
        f"Runs: {len(summary_rows)} total, {sum(row.get('status') == 'ok' for row in summary_rows)} successful, {sum(row.get('status') != 'ok' for row in summary_rows)} failed.",
        f"Offline analysis source SHA-256: `{hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()}`.",
        "",
        "| Arm | successful runs | workload makespan median (ms) | rank app makespan median (ms) | process CPU median (ms/rank) | app wait median (ms/task) | probes median (/rank-run) | voluntary switches median (/rank-run) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    arms = sorted({_arm(row) for row in summary_rows}, key=_arm_order)
    for arm in arms:
        rows = [row for row in summary_rows if _arm(row) == arm and row.get("status") == "ok"]
        rank_rows = [record for record in diagnostics if record["arm"] == arm and record["status"] == "ok"]

        def median_column(records: list[dict[str, Any]], key: str) -> str:
            values = [float(record[key]) for record in records if record.get(key) not in (None, "")]
            return f"{statistics.median(values):.3f}" if values else "n/a"

        makespan_values = [float(row["makespan_s"]) * 1000 for row in rows]
        makespan_median = f"{statistics.median(makespan_values):.3f}" if makespan_values else "n/a"
        report_lines.append(
            f"| {arm} | {len(rows)} | {makespan_median} | "
            f"{median_column(rank_rows, 'application_makespan_ms')} | {median_column(rank_rows, 'process_cpu_ms')} | "
            f"{median_column(rank_rows, 'median_application_wait_ms')} | {median_column(rank_rows, 'completion_probe_count')} | "
            f"{median_column(rank_rows, 'voluntary_context_switches')} |"
        )
    report_lines.extend([
        "",
        "| Arm | grant → all SUBMITTED median (ms) | all SUBMITTED → all COMPLETED median (ms) | all COMPLETED → next grant median (ms) | gaps with a legal candidate |",
        "|---|---:|---:|---:|---:|",
    ])
    for arm in arms:
        central = [row for row in coordinator_diagnostics if row["arm"] == arm]

        def median_central(key: str) -> str:
            values = [float(row[key]) for row in central if row.get(key) not in (None, "")]
            return f"{statistics.median(values):.3f}" if values else "n/a"

        gap_rows = [row for row in central if row.get("all_completed_to_next_grant_ms") not in (None, "")]
        legal_gaps = sum(row.get("legal_candidate_present_during_gap") is True for row in gap_rows)
        report_lines.append(
            f"| {arm} | {median_central('grant_to_all_submitted_ms')} | "
            f"{median_central('all_submitted_to_all_completed_ms')} | "
            f"{median_central('all_completed_to_next_grant_ms')} | {legal_gaps}/{len(gap_rows)} |"
        )
    report_lines.extend([
        "",
        "The per-run paired estimates are in `paired-summary.csv`; paired per-job estimates are in `job-paired-summary.csv`; run/rank diagnostics are in `process-diagnostics.csv`; coordinator-clock intervals are in `coordinator-diagnostics.csv`. The figures directory contains overall makespan, independent representative timelines, and the predeclared paired timeline.",
        "",
        "These descriptive medians do not isolate causal components: timing intervals can overlap, completion observation is not the physical completion instant, and preparation plus application duration is not end-to-end process duration.",
        "",
    ])
    report_path = batch / "analysis.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    job_paired_paths = write_job_paired_summary(batch)
    coordinator_paths = [coordinator_path, table_dir / coordinator_path.name]
    return [diagnostics_path, table_dir / diagnostics_path.name,
            *job_paired_paths, *coordinator_paths, report_path]


def _collect_coordinator_diagnostics(
    batch: Path, summary_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    diagnostics = []
    for summary in summary_rows:
        raw_path = batch / "raw" / f"{summary['run_id']}.json"
        if not raw_path.is_file():
            continue
        trace = json.loads(raw_path.read_text(encoding="utf-8"))
        timings = trace.get("metrics", {}).get("coordinator_task_timings", {})
        for task_id, item in timings.items():
            diagnostics.append({
                "run_id": summary["run_id"], "arm": _arm(summary),
                "status": summary.get("status", ""),
                "epoch": summary.get("epoch", ""), "repeat": summary.get("repeat", ""),
                "task_id": task_id, "next_task_id": item.get("next_task_id"),
                "grant_to_all_submitted_ms": _scale(item.get("grant_to_all_submitted_s"), 1000),
                "all_submitted_to_all_completed_ms": _scale(item.get("all_submitted_to_all_completed_s"), 1000),
                "all_completed_to_next_grant_ms": _scale(item.get("all_completed_to_next_grant_s"), 1000),
                "eligible_to_grant_ms": _scale(item.get("eligible_to_grant_s"), 1000),
                "legal_candidate_present_during_gap": item.get("legal_candidate_present_during_gap"),
            })
    return diagnostics


def write_job_paired_summary(batch_dir: str | Path) -> list[Path]:
    """Pair per-job JCT by seed after taking each seed's common-repeat median."""
    batch = repository_path(batch_dir).resolve()
    jobs_path = batch / "jobs.csv"
    if not jobs_path.is_file():
        return []
    job_rows = _read_csv(jobs_path)
    by_block: dict[tuple[str, str, int, int], float] = {}
    for row in job_rows:
        if row.get("status") != "ok" or _number(row, "jct_s") is None:
            continue
        key = (row["job_id"], _arm(row), int(row["epoch"]), int(row["repeat"]))
        if key in by_block:
            raise ValueError(f"duplicate job JCT block {key}")
        by_block[key] = float(row["jct_s"])

    metadata_path = batch / "analysis.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    bootstrap_samples = int(metadata.get("bootstrap_samples", 0))
    tie_threshold = float(metadata.get("tie_threshold", 0.0))
    seeds = sorted({seed for _job, _arm_name, seed, _repeat in by_block})
    arms = sorted({arm for _job, arm, _seed, _repeat in by_block}, key=_arm_order)
    jobs = sorted({job for job, _arm_name, _seed, _repeat in by_block})
    fields = [
        "job_id", "baseline", "candidate", "paired_seeds", "missing_seed_blocks",
        "baseline_median_jct_ms", "candidate_median_jct_ms", "median_difference_ms",
        "median_speedup", "p10_speedup", "p90_speedup", "wins", "ties", "losses",
        "bootstrap_speedup_low", "bootstrap_speedup_high",
    ]
    output: list[dict[str, Any]] = []
    for job in jobs:
        for first_index, baseline in enumerate(arms):
            for candidate in arms[first_index + 1:]:
                paired = []
                for seed in seeds:
                    repeats = sorted({
                        repeat for job_id, arm, block_seed, repeat in by_block
                        if job_id == job and block_seed == seed and arm in {baseline, candidate}
                    })
                    common = [repeat for repeat in repeats
                              if (job, baseline, seed, repeat) in by_block
                              and (job, candidate, seed, repeat) in by_block]
                    if common:
                        paired.append((
                            statistics.median(by_block[(job, baseline, seed, repeat)] for repeat in common),
                            statistics.median(by_block[(job, candidate, seed, repeat)] for repeat in common),
                        ))
                ratios = [base / cand for base, cand in paired]
                differences = [cand - base for base, cand in paired]
                bootstrap = []
                if paired and bootstrap_samples:
                    rng = random.Random(f"job:{job}:{baseline}:{candidate}")
                    for _ in range(bootstrap_samples):
                        sample = [ratios[rng.randrange(len(ratios))] for _index in range(len(ratios))]
                        bootstrap.append(statistics.median(sample))
                baseline_values = [value[0] for value in paired]
                candidate_values = [value[1] for value in paired]
                output.append({
                    "job_id": job, "baseline": baseline, "candidate": candidate,
                    "paired_seeds": len(paired), "missing_seed_blocks": len(seeds) - len(paired),
                    "baseline_median_jct_ms": statistics.median(baseline_values) * 1000 if paired else None,
                    "candidate_median_jct_ms": statistics.median(candidate_values) * 1000 if paired else None,
                    "median_difference_ms": statistics.median(differences) * 1000 if paired else None,
                    "median_speedup": statistics.median(ratios) if ratios else None,
                    "p10_speedup": _percentile(ratios, 0.1) if ratios else None,
                    "p90_speedup": _percentile(ratios, 0.9) if ratios else None,
                    "wins": sum(ratio > 1 + tie_threshold for ratio in ratios),
                    "ties": sum(abs(ratio - 1) <= tie_threshold for ratio in ratios),
                    "losses": sum(ratio < 1 - tie_threshold for ratio in ratios),
                    "bootstrap_speedup_low": _percentile(bootstrap, 0.025) if bootstrap else None,
                    "bootstrap_speedup_high": _percentile(bootstrap, 0.975) if bootstrap else None,
                })
    paths = [batch / "job-paired-summary.csv", batch / "tables" / "job-paired-summary.csv"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output)
    return paths


def _scale(value: Any, factor: float) -> float | None:
    number = _number({"value": value}, "value")
    return number * factor if number is not None else None


def render_suite(
    results_root: str | Path,
    timeline_batch_dir: str | Path,
    output_dir: str | Path,
    rank: int = 0,
) -> list[Path]:
    output = repository_path(output_dir).resolve()
    paths = [
        render_makespan_comparison(results_root, output / "makespan-comparison.svg"),
        render_timeline_comparison(
            timeline_batch_dir, rank, output / "timeline-comparison.svg"
        ),
        render_paired_timeline_comparison(
            timeline_batch_dir, output / "paired-timeline-diagnostic.svg"
        ),
    ]
    for batch in collect_comparable_batches(results_root):
        figure_dir = batch["path"] / "figures"
        paths.append(
            render_batch_makespan_comparison(
                batch["path"], figure_dir / "makespan-comparison.svg",
                rows=batch["rows"], label=batch["label"],
            )
        )
        paths.append(
            render_timeline_comparison(
                batch["path"], rank, figure_dir / "timeline-comparison.svg",
                rows=batch["rows"], label=batch["label"],
            )
        )
        paths.append(
            render_paired_timeline_comparison(
                batch["path"], figure_dir / "paired-timeline-diagnostic.svg",
                rows=batch["rows"], label=batch["label"],
            )
        )
    return paths


def render_control_path_comparison(batch_dir: str | Path, rank: int = 0) -> list[Path]:
    """Render each fixed-workload block separately for control-path batches."""
    batch = repository_path(batch_dir).resolve()
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") not in {"E2", "E3", "F1", "F2", "F3", "F4"}:
        raise ValueError(f"not a supported control-path batch: {batch}")
    rows = _read_csv(batch / "summary.csv")
    workloads = sorted({row.get("workload", "") for row in rows if row.get("workload")})
    paths: list[Path] = []
    for workload in workloads:
        selected_rows = [row for row in rows
                         if row.get("workload") == workload and row.get("status") == "ok"
                         and _number(row, "makespan_s") is not None]
        if len({_arm(row) for row in selected_rows}) < 2:
            continue
        figure_dir = batch / "figures" / workload
        figure_dir.mkdir(parents=True, exist_ok=True)
        label = f"{manifest['stage']} · {workload} · {batch.name}"
        if manifest.get("observation_mode") == "minimal":
            paths.append(render_batch_makespan_comparison(
                batch, figure_dir / "makespan-comparison.svg",
                rows=selected_rows, label=label,
            ))
            continue
        paths.extend((
            render_batch_makespan_comparison(
                batch, figure_dir / "makespan-comparison.svg",
                rows=selected_rows, label=label,
            ),
            render_timeline_comparison(
                batch, rank, figure_dir / f"timeline-rank-{rank}.svg",
                rows=selected_rows, label=label,
            ),
            render_paired_timeline_comparison(
                batch, figure_dir / "paired-timeline-diagnostic.svg",
                rows=selected_rows, label=label,
            ),
        ))
    if not paths:
        raise ValueError(f"no comparable successful workload blocks in {batch}")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    suite = subparsers.add_parser(
        "suite", help="render comparison plots for the suite and each comparable batch"
    )
    suite.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    suite.add_argument("--timeline-batch-dir", type=Path, default=DEFAULT_TIMELINE_BATCH)
    suite.add_argument("--rank", type=int, default=0)
    suite.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    timeline = subparsers.add_parser("timeline", help="render one batch's strategy comparison")
    timeline.add_argument("--batch-dir", type=Path, required=True)
    timeline.add_argument("--rank", type=int, default=0)
    timeline.add_argument("--output", type=Path, required=True)
    timeline.add_argument("--x-min-ms", type=float)
    timeline.add_argument("--x-max-ms", type=float)

    batch_parser = subparsers.add_parser(
        "batch", help="render makespan and independent/paired timelines for one batch"
    )
    batch_parser.add_argument("--batch-dir", type=Path, required=True)
    batch_parser.add_argument("--rank", type=int, default=0)

    control_path = subparsers.add_parser(
        "control-path", help="render makespan and rank-local timelines for a control-path batch"
    )
    control_path.add_argument("--batch-dir", type=Path, required=True)
    control_path.add_argument("--rank", type=int, default=0)

    args = parser.parse_args(argv)
    command = args.command or "suite"
    try:
        if command == "suite":
            paths = render_suite(
                getattr(args, "results_root", RESULTS_ROOT),
                getattr(args, "timeline_batch_dir", DEFAULT_TIMELINE_BATCH),
                getattr(args, "output_dir", DEFAULT_OUTPUT_DIR),
                getattr(args, "rank", 0),
            )
        elif command == "control-path":
            paths = render_control_path_comparison(args.batch_dir, args.rank)
        elif command == "batch":
            batch_dir = repository_path(args.batch_dir).resolve()
            figure_dir = batch_dir / "figures"
            figure_dir.mkdir(parents=True, exist_ok=True)
            rows = [row for row in _read_csv(batch_dir / "summary.csv")
                    if row.get("status") == "ok" and _number(row, "makespan_s") is not None]
            paths = [
                render_batch_makespan_comparison(
                    batch_dir, figure_dir / "makespan-comparison.svg", rows=rows,
                ),
                render_timeline_comparison(
                    batch_dir, args.rank, figure_dir / "timeline-comparison.svg", rows=rows,
                ),
                render_paired_timeline_comparison(
                    batch_dir, figure_dir / "paired-timeline-diagnostic.svg", rows=rows,
                ),
            ]
            paths.extend(write_overhead_diagnostics(batch_dir))
        else:
            paths = [render_timeline_comparison(
                args.batch_dir, args.rank, args.output,
                x_min_ms=args.x_min_ms, x_max_ms=args.x_max_ms,
            )]
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        print(f"Phase 3 visualization failed: {exc}", file=sys.stderr)
        return 2
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
