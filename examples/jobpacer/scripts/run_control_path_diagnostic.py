"""Run interleaved Phase 3 control-path diagnostic batches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.jobpacer.analysis.benchmark_paths import repository_path
from examples.jobpacer.scripts import run_experiments as batch


ROOT = Path(__file__).resolve().parents[3]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        if fields:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _plan(seed: int, repeats: int, order_seed: int,
          workloads: dict[str, Path], stage: str = "E2",
          seed_blocks: tuple[int, ...] | None = None) -> list[dict[str, Any]]:
    plan = []
    if stage == "E2":
        arms = ("old-fifo", "new-static_fifo")
    elif stage == "E3":
        arms = ("new-poll", "new-wakeup")
    else:
        arms = ("before-producer", "on-submit")
    observation_mode = {"F1": "minimal", "F2": "diagnostic", "F3": "minimal",
                        "F4": "minimal"}.get(
        stage, "diagnostic")
    if stage == "F4":
        blocks = [(name, block_seed, repeat)
                  for name in workloads
                  for block_seed in (seed_blocks or (seed,))
                  for repeat in range(repeats)]
        random.Random(order_seed).shuffle(blocks)
        for workload_name, block_seed, repeat in blocks:
            mode_order = ["before-producer", "on-submit"]
            random.Random(order_seed + block_seed * 1_000_003 + repeat * 31
                          + sum(map(ord, workload_name))).shuffle(mode_order)
            for mode in mode_order:
                plan.append({
                    "run_id": (f"{len(plan) + 1:04d}-{workload_name}-{mode}"
                               f"-e{block_seed}-r{repeat}"),
                    "workload_name": workload_name,
                    "workload": str(workloads[workload_name]),
                    "arm": mode, "completion_wakeup_on_submit": False,
                    "declaration_mode": mode, "observation_mode": observation_mode,
                    "repeat": repeat, "seed": block_seed,
                    "block_order": [f"{workload_name}:{candidate}" for candidate in mode_order],
                })
        return plan
    for repeat in range(repeats):
        block = [(name, arm) for name in workloads for arm in arms]
        random.Random(order_seed + seed * 1_000_003 + repeat).shuffle(block)
        for workload_name, arm in block:
            plan.append({
                "run_id": f"{len(plan) + 1:04d}-{workload_name}-{arm}-e{seed}-r{repeat}",
                "workload_name": workload_name,
                "workload": str(workloads[workload_name]),
                "arm": arm,
                "completion_wakeup_on_submit": arm == "new-wakeup",
                "declaration_mode": arm if stage in {"F1", "F2", "F3"} else "before-producer",
                "observation_mode": observation_mode,
                "repeat": repeat,
                "seed": seed,
                "block_order": [f"{name}:{candidate}" for name, candidate in block],
            })
    return plan


def _summary_row(record: dict[str, Any], result: dict[str, Any] | None,
                 raw_path: Path) -> dict[str, Any]:
    perf = (result or {}).get("performance", {})
    result_metrics = (result or {}).get("metrics", {})
    rank_application_makespans = perf.get("rank_application_makespans_us", ())
    makespan_us = (max(rank_application_makespans)
                   if rank_application_makespans else perf.get("application_makespan_us"))
    if makespan_us is None:
        makespan_us = perf.get("workload_makespan_us")
    ranks = {int(item.get("rank", -1)): item for item in (result or {}).get("ranks", [])}
    rank_app_us = {rank: item.get("application_makespan_us") for rank, item in ranks.items()}
    rank_drain_us = {rank: item.get("communication_drain_makespan_us") for rank, item in ranks.items()}
    task_count_by_rank = {
        rank: sum(len(job.get("tasks", [])) for job in item.get("jobs", []))
        for rank, item in ranks.items()
    }
    protocol_counts = next((item.get("protocol_transition_counts", {}) for item in ranks.values()
                            if item.get("protocol_transition_counts")), {})
    send_counts = [item.get("protocol_send_call_counts", {}) for item in ranks.values()]
    task_messages = sum(task_count_by_rank.values())
    declaration_mode = record.get("declaration_mode", "before-producer")
    rank_drain_gap_us = {
        rank: drain - rank_app_us[rank]
        for rank, drain in rank_drain_us.items()
        if drain is not None and rank_app_us.get(rank) is not None
    }
    return {
        "run_id": record["run_id"], "workload": record["workload_name"],
        "arm": record["arm"], "epoch": record["seed"],
        "repeat": record["repeat"], "seed": record["seed"],
        "observation_mode": record.get("observation_mode", "diagnostic"),
        "declaration_mode": record.get("declaration_mode", "before-producer"),
        "completion_wakeup_on_submit": record.get("completion_wakeup_on_submit", False),
        "makespan_boundary": "maximum_rank_application_release_to_end_duration",
        "status": ("process_failed" if record.get("returncode") != 0 else
                   (result or {}).get("validation", {}).get("status", "missing_result")),
        "makespan_s": makespan_us / 1_000_000 if makespan_us is not None else None,
        "rank0_application_s": rank_app_us.get(0) / 1_000_000 if rank_app_us.get(0) is not None else None,
        "rank1_application_s": rank_app_us.get(1) / 1_000_000 if rank_app_us.get(1) is not None else None,
        "application_to_drain_s": (
            perf["communication_drain_makespan_us"] / 1_000_000 - makespan_us / 1_000_000
            if makespan_us is not None and perf.get("communication_drain_makespan_us") is not None
            else None),
        "rank0_application_to_drain_us": rank_drain_gap_us.get(0),
        "rank1_application_to_drain_us": rank_drain_gap_us.get(1),
        "task_count_per_rank": json.dumps(task_count_by_rank, sort_keys=True),
        "offer_send_calls": sum(item.get("offer", 0) for item in send_counts),
        "declare_send_calls": sum(item.get("declare", 0) for item in send_counts),
        "offer_messages_expected": task_messages,
        "declare_messages_expected": 0 if declaration_mode == "on-submit" else task_messages,
        "submitted_reports": protocol_counts.get("submitted_reports"),
        "completed_reports": protocol_counts.get("completed_reports"),
        "all_collectives_correct": (result or {}).get("validation", {}).get("all_collectives_correct"),
        "process_cpu_time_s": sum(result_metrics.get("rank_process_cpu_time_s", {}).values())
        if result_metrics.get("rank_process_cpu_time_s") else None,
        "voluntary_context_switches": sum(item.get("voluntary", 0) for item in
                                           result_metrics.get("rank_context_switches", {}).values())
        if result_metrics.get("rank_context_switches") else None,
        "involuntary_context_switches": sum(item.get("involuntary", 0) for item in
                                             result_metrics.get("rank_context_switches", {}).values())
        if result_metrics.get("rank_context_switches") else None,
        "mean_job_jct_s": statistics.mean(
            item["makespan_us"] / 1_000_000 for item in perf.get("job_makespans", [])
        ) if perf.get("job_makespans") else None,
        "returncode": record.get("returncode"),
        "wall_time_s": record.get("wall_time_s"),
        "result_path": str(raw_path),
        "error": record.get("error"),
    }


def _distribution(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    ordered = sorted(values)
    p90_index = max(0, (9 * len(ordered) + 9) // 10 - 1)
    return statistics.median(ordered), ordered[p90_index], ordered[-1]


def _declaration_segment_rows(task_rows: list[dict[str, Any]],
                              coordinator_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize intervals within each replay before comparing repeats."""
    task_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in task_rows:
        if row.get("rank") is not None:
            task_groups.setdefault((row["run_id"], int(row["rank"])), []).append(row)
    coordinator_groups: dict[str, list[dict[str, Any]]] = {}
    for row in coordinator_rows:
        coordinator_groups.setdefault(row["run_id"], []).append(row)
    rank_metrics = (
        "application_wait_return_to_next_declare_start_us",
        "previous_wait_return_to_next_submit_call_us",
        "declare_call_us", "declare_control_send_lock_wait_us",
        "declare_control_send_total_us",
        "offer_control_send_lock_wait_us", "completed_control_send_lock_wait_us",
        "offer_control_send_total_us", "completed_socket_write_flush_us",
        "completed_control_send_total_us", "submit_call_to_grant_us",
        "grant_to_collective_start_us", "collective_return_to_submitted_send_end_us",
        "completion_observed_to_application_continue_us",
    )
    output: list[dict[str, Any]] = []
    for (run_id, rank), rows in sorted(task_groups.items()):
        entry: dict[str, Any] = {"run_id": run_id, "rank": rank, "scope": "rank_local"}
        for name in rank_metrics:
            values = [float(row[name]) for row in rows if row.get(name) is not None]
            median, p90, maximum = _distribution(values)
            entry[f"{name}_median_us"] = median
            entry[f"{name}_p90_us"] = p90
            entry[f"{name}_max_us"] = maximum
            entry[f"{name}_observations"] = len(values)
        output.append(entry)
    coordinator_metrics = (
        "capacity_release_to_last_offer_enqueued_us",
        "candidate_ready_after_capacity_release_us",
        "capacity_held_after_candidate_ready_us",
        "both_conditions_to_decision_start_us", "decision_processing_us",
        "post_decision_to_queue_us", "writer_queue_to_sendall_end_us",
    )
    for run_id, rows in sorted(coordinator_groups.items()):
        entry = {"run_id": run_id, "rank": "coordinator", "scope": "coordinator_clock"}
        for name in coordinator_metrics:
            values = [float(row[name]) for row in rows if row.get(name) is not None]
            median, p90, maximum = _distribution(values)
            entry[f"{name}_median_us"] = median
            entry[f"{name}_p90_us"] = p90
            entry[f"{name}_max_us"] = maximum
            entry[f"{name}_observations"] = len(values)
        output.append(entry)
    return output


def _paired_declaration_segment_rows(
    segment_rows: list[dict[str, Any]], run_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair run-reduced diagnostic segments by repeat and same rank/clock scope."""
    records = {item["run_id"]: item for item in run_records}
    segments = {(row["run_id"], row["scope"], str(row["rank"])): row
                for row in segment_rows}
    repeat_keys = sorted({(item["repeat"], row["scope"], str(row["rank"]))
                          for row in segment_rows
                          for item in (records[row["run_id"]],)})
    paired: list[dict[str, Any]] = []
    metric_names = sorted({name[:-len("_median_us")] for row in segment_rows
                           for name in row if name.endswith("_median_us")})
    for repeat, scope, rank in repeat_keys:
        arms = {}
        for item in run_records:
            if item["repeat"] != repeat:
                continue
            row = segments.get((item["run_id"], scope, rank))
            if row is not None:
                arms[item["arm"]] = row
        before = arms.get("before-producer", {})
        direct = arms.get("on-submit", {})
        for metric in metric_names:
            baseline = before.get(f"{metric}_median_us")
            candidate = direct.get(f"{metric}_median_us")
            if baseline is None and candidate is None:
                continue
            baseline_p90 = before.get(f"{metric}_p90_us")
            candidate_p90 = direct.get(f"{metric}_p90_us")
            paired.append({
                "repeat": repeat, "scope": scope, "rank": rank, "metric": metric,
                "before_producer_median_us": baseline,
                "on_submit_median_us": candidate,
                "on_submit_minus_before_median_us": (
                    candidate - baseline if candidate is not None and baseline is not None else None),
                "before_producer_p90_us": baseline_p90,
                "on_submit_p90_us": candidate_p90,
                "on_submit_minus_before_p90_us": (
                    candidate_p90 - baseline_p90
                    if candidate_p90 is not None and baseline_p90 is not None else None),
            })
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in paired:
        delta = row["on_submit_minus_before_median_us"]
        if delta is not None:
            key = (row["scope"], row["rank"], row["metric"])
            grouped.setdefault(key, []).append(float(delta))
    statistics_rows = []
    for (scope, rank, metric), deltas in sorted(grouped.items()):
        statistics_rows.append({
            "scope": scope, "rank": rank, "metric": metric,
            "paired_repeats": len(deltas), "median_delta_us": statistics.median(deltas),
            "min_delta_us": min(deltas), "max_delta_us": max(deltas),
            "on_submit_lower_repeats": sum(value < 0 for value in deltas),
            "on_submit_higher_repeats": sum(value > 0 for value in deltas),
        })
    return paired, statistics_rows


def _paired_declaration_rows(
    summary_rows: list[dict[str, Any]], workload_names: tuple[str, ...],
    seed_values: tuple[int, ...], repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paired_rows = []
    seed_rows = []
    for workload_name in workload_names:
        for seed in seed_values:
            for repeat in range(repeats):
                block = {row["arm"]: row for row in summary_rows
                         if row["workload"] == workload_name and row["seed"] == seed
                         and row["repeat"] == repeat and row["status"] == "ok"}
                before, direct = block.get("before-producer"), block.get("on-submit")
                paired_rows.append({
                    "workload": workload_name, "seed": seed, "repeat": repeat,
                    "baseline_arm": "before-producer", "candidate_arm": "on-submit",
                    "baseline_makespan_s": before["makespan_s"] if before else None,
                    "candidate_makespan_s": direct["makespan_s"] if direct else None,
                    "candidate_minus_baseline_ms": (
                        (direct["makespan_s"] - before["makespan_s"]) * 1000
                        if before and direct else None),
                    "baseline_over_candidate_speedup": (
                        before["makespan_s"] / direct["makespan_s"]
                        if before and direct and direct["makespan_s"] else None),
                })
            successful = [row for row in paired_rows
                          if row["workload"] == workload_name and row["seed"] == seed
                          and row["candidate_minus_baseline_ms"] is not None]
            seed_rows.append({
                "workload": workload_name, "seed": seed,
                "attempted_repeats": repeats, "successful_repeats": len(successful),
                "baseline_median_s": statistics.median(
                    row["baseline_makespan_s"] for row in successful) if successful else None,
                "candidate_median_s": statistics.median(
                    row["candidate_makespan_s"] for row in successful) if successful else None,
                "candidate_minus_baseline_median_ms": statistics.median(
                    row["candidate_minus_baseline_ms"] for row in successful) if successful else None,
                "baseline_over_candidate_speedup": statistics.median(
                    row["baseline_over_candidate_speedup"] for row in successful) if successful else None,
            })
    statistics_rows = []
    for workload_name in workload_names:
        rows = [row for row in paired_rows if row["workload"] == workload_name]
        deltas = [row["candidate_minus_baseline_ms"] for row in rows
                  if row["candidate_minus_baseline_ms"] is not None]
        ratios = [row["baseline_over_candidate_speedup"] for row in rows
                  if row["baseline_over_candidate_speedup"] is not None]
        seed_deltas = [row["candidate_minus_baseline_median_ms"] for row in seed_rows
                       if row["workload"] == workload_name
                       and row["candidate_minus_baseline_median_ms"] is not None]
        statistics_rows.append({
            "workload": workload_name, "baseline_arm": "before-producer",
            "candidate_arm": "on-submit", "attempted_repeat_pairs": len(rows),
            "successful_repeat_pairs": len(deltas),
            "candidate_minus_baseline_median_ms": statistics.median(deltas) if deltas else None,
            "candidate_minus_baseline_min_ms": min(deltas) if deltas else None,
            "candidate_minus_baseline_max_ms": max(deltas) if deltas else None,
            "baseline_over_candidate_speedup_median": statistics.median(ratios) if ratios else None,
            "baseline_over_candidate_speedup_min": min(ratios) if ratios else None,
            "baseline_over_candidate_speedup_max": max(ratios) if ratios else None,
            "candidate_faster_repeat_pairs": sum(value < 0 for value in deltas),
            "candidate_slower_repeat_pairs": sum(value > 0 for value in deltas),
            "seed_blocks": len(seed_deltas),
            "seed_median_delta_min_ms": min(seed_deltas) if seed_deltas else None,
            "seed_median_delta_max_ms": max(seed_deltas) if seed_deltas else None,
            "on_submit_faster_seed_blocks": sum(value < 0 for value in seed_deltas),
        })
    return paired_rows, statistics_rows, seed_rows


def _job_jct_rows(loaded: list[tuple[dict[str, Any], dict[str, Any] | None, Path]]) -> list[dict[str, Any]]:
    rows = []
    for record, result, _path in loaded:
        for item in (result or {}).get("performance", {}).get("job_makespans", []):
            rows.append({
                "run_id": record["run_id"], "workload": record["workload_name"],
                "seed": record["seed"], "repeat": record["repeat"],
                "arm": record["arm"], "job_id": item["job_id"],
                "jct_s": item["makespan_us"] / 1_000_000,
            })
    return rows


def _compute_sample_pair_rows(
    loaded: list[tuple[dict[str, Any], dict[str, Any] | None, Path]],
    expected_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    blocks: dict[tuple[str, int, int], dict[str, dict[tuple[int, str, str], float]]] = {}
    invalid: set[tuple[str, int, int]] = set()
    for record, result, _path in loaded:
        key = (record["workload_name"], int(record["seed"]), int(record["repeat"]))
        if result is None:
            invalid.add(key)
            continue
        samples: dict[tuple[int, str, str], float] = {}
        for rank_data in result.get("ranks", []):
            rank = int(rank_data["rank"])
            for job in rank_data.get("jobs", []):
                for sample_key, value in job.get("compute_samples_s", {}).items():
                    samples[(rank, job["job_id"], sample_key)] = float(value)
        blocks.setdefault(key, {})[record["arm"]] = samples
    rows = []
    for workload, seed, repeat in sorted(set(blocks) | invalid):
        arms = blocks.get((workload, seed, repeat), {})
        before, direct = arms.get("before-producer"), arms.get("on-submit")
        if before is None or direct is None or (workload, seed, repeat) in invalid:
            rows.append({"workload": workload, "seed": seed, "repeat": repeat,
                         "sample_count": None, "mismatch_count": None,
                         "expected_sample_count": (expected_counts or {}).get(workload),
                         "sample_count_matches_expected": None,
                         "samples_match": None, "status": "incomplete_pair"})
            continue
        keys = set(before) | set(direct)
        mismatch = sum(key not in before or key not in direct or before.get(key) != direct.get(key)
                       for key in keys)
        expected = (expected_counts or {}).get(workload)
        rows.append({"workload": workload, "seed": seed, "repeat": repeat,
                     "sample_count": len(keys), "mismatch_count": mismatch,
                     "expected_sample_count": expected,
                     "sample_count_matches_expected": (len(keys) == expected
                                                       if expected is not None else None),
                     "samples_match": mismatch == 0, "status": "ok"})
    return rows


def _expected_compute_sample_counts(workloads: dict[str, Path]) -> dict[str, int]:
    expected = {}
    for name, path in workloads.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        communication_count = sum(len(job.get("communications", []))
                                  for job in data.get("jobs", []))
        expected[name] = communication_count * 4
    return expected


def _job_seed_pair_rows(job_rows: list[dict[str, Any]], seed_values: tuple[int, ...],
                        repeats: int) -> list[dict[str, Any]]:
    index = {(row["workload"], row["seed"], row["repeat"], row["arm"], row["job_id"]): row["jct_s"]
             for row in job_rows}
    scenarios = sorted({(row["workload"], row["job_id"]) for row in job_rows})
    output = []
    for workload, job_id in scenarios:
        for seed in seed_values:
            paired = [(index[(workload, seed, repeat, "before-producer", job_id)],
                       index[(workload, seed, repeat, "on-submit", job_id)])
                      for repeat in range(repeats)
                      if (workload, seed, repeat, "before-producer", job_id) in index
                      and (workload, seed, repeat, "on-submit", job_id) in index]
            baseline = statistics.median(pair[0] for pair in paired) if paired else None
            candidate = statistics.median(pair[1] for pair in paired) if paired else None
            output.append({
                "workload": workload, "job_id": job_id, "seed": seed,
                "paired_repeats": len(paired), "baseline_median_jct_s": baseline,
                "candidate_median_jct_s": candidate,
                "candidate_minus_baseline_median_jct_ms": (
                    (candidate - baseline) * 1000 if baseline is not None and candidate is not None else None),
                "baseline_over_candidate_jct_speedup": (
                    baseline / candidate if baseline is not None and candidate else None),
            })
    return output


def _paired_resource_rows(summary_rows: list[dict[str, Any]], workload_names: tuple[str, ...],
                          seed_values: tuple[int, ...], repeats: int
                          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = (
        "makespan_s", "rank0_application_s", "rank1_application_s",
        "process_cpu_time_s", "voluntary_context_switches",
        "involuntary_context_switches", "rank0_application_to_drain_us",
        "rank1_application_to_drain_us",
    )
    paired = []
    for workload in workload_names:
        for seed in seed_values:
            for repeat in range(repeats):
                block = {row["arm"]: row for row in summary_rows
                         if row["workload"] == workload and row["seed"] == seed
                         and row["repeat"] == repeat and row["status"] == "ok"}
                before, direct = block.get("before-producer"), block.get("on-submit")
                row: dict[str, Any] = {"workload": workload, "seed": seed, "repeat": repeat}
                for metric in metrics:
                    base = before.get(metric) if before else None
                    candidate = direct.get(metric) if direct else None
                    row[f"before_producer_{metric}"] = base
                    row[f"on_submit_{metric}"] = candidate
                    row[f"on_submit_minus_before_{metric}"] = (
                        candidate - base if candidate is not None and base is not None else None)
                paired.append(row)
    statistics_rows = []
    for workload in workload_names:
        for metric in metrics:
            deltas = [row[f"on_submit_minus_before_{metric}"] for row in paired
                      if row["workload"] == workload
                      and row[f"on_submit_minus_before_{metric}"] is not None]
            statistics_rows.append({
                "workload": workload, "metric": metric, "paired_runs": len(deltas),
                "median_candidate_minus_before": statistics.median(deltas) if deltas else None,
                "min_candidate_minus_before": min(deltas) if deltas else None,
                "max_candidate_minus_before": max(deltas) if deltas else None,
                "on_submit_lower_runs": sum(value < 0 for value in deltas),
                "on_submit_higher_runs": sum(value > 0 for value in deltas),
            })
    return paired, statistics_rows


def _rebuild_derived(batch_dir: Path) -> dict[str, Any]:
    batch_dir = batch_dir.resolve(strict=True)
    manifest_path = batch_dir / "manifest.json"
    runs_path = batch_dir / "runs.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = manifest.get("stage")
    if stage not in {"E2", "E3", "F1", "F2", "F3", "F4"}:
        raise ValueError(f"unsupported control-path batch stage: {stage!r}")
    raw_dir = (batch_dir / "raw").resolve(strict=True)
    records = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    seed_values = tuple(manifest.get("seed_blocks", (manifest.get("seed", 0),)))
    expected_runs = len(manifest.get("workloads", {})) * int(
        manifest.get("repeats_per_workload_arm", 0)) * 2
    if stage == "F4":
        expected_runs *= len(seed_values)
    if expected_runs <= 0 or len(records) != expected_runs:
        raise ValueError(f"run ledger is incomplete: expected {expected_runs}, found {len(records)}")
    if len({record.get("run_id") for record in records}) != len(records):
        raise ValueError("run ledger contains duplicate run ids")
    loaded = []
    for record in records:
        raw_path = Path(record["result_path"]).resolve()
        if raw_path.parent != raw_dir or raw_path.suffix != ".json":
            raise ValueError(f"raw result path escapes the batch raw/: {raw_path}")
        result = None
        if raw_path.is_file():
            if _sha256(raw_path) != record.get("result_sha256"):
                raise ValueError(f"raw result SHA-256 mismatch: {raw_path}")
            result = json.loads(raw_path.read_text(encoding="utf-8"))
        loaded.append((record, result, raw_path))

    summary_rows = [_summary_row(record, result, path)
                    for record, result, path in loaded]
    task_rows = []
    coordinator_task_rows = []
    coordinator_event_rows = []
    for record, result, _path in loaded:
        old = record["arm"] == "old-fifo"
        common_config = {
            "epoch": record["seed"], "repeat": record["repeat"],
            "workload": record["workload"],
            "compute_jitter": float(manifest.get("compute_jitter", 0.0)),
            "binding_preparation": "legacy_tensor_precreated" if old else "precreate",
            "poll_interval_s": 0.001,
            "observation_mode": record.get("observation_mode", "diagnostic"),
            "declaration_mode": record.get("declaration_mode", "before-producer"),
            "completion_wakeup_on_submit": record.get("completion_wakeup_on_submit", False),
        }
        task_rows.extend(batch._task_timing_rows({
            "run_id": record["run_id"], "arm": record["arm"],
            "group": "old" if old else "new", "config": common_config,
            "returncode": record.get("returncode"),
        }, result))
        task_diag, event_diag = batch._coordinator_diagnostic_rows({
            "run_id": record["run_id"], "arm": record["arm"],
            "config": common_config, "returncode": record.get("returncode"),
        }, result)
        coordinator_task_rows.extend(task_diag)
        coordinator_event_rows.extend(event_diag)

    if stage == "E2":
        arms = ("old-fifo", "new-static_fifo")
    elif stage == "E3":
        arms = ("new-poll", "new-wakeup")
    else:
        arms = ("before-producer", "on-submit")
    workload_names = tuple(manifest.get("workloads", {}))
    repeats = int(manifest["repeats_per_workload_arm"])
    if stage == "F4":
        paired_rows, paired_statistics, seed_summary_rows = _paired_declaration_rows(
            summary_rows, workload_names, seed_values, repeats)
        job_rows = _job_jct_rows(loaded)
        job_seed_rows = _job_seed_pair_rows(job_rows, seed_values, repeats)
        resource_rows, resource_statistics = _paired_resource_rows(
            summary_rows, workload_names, seed_values, repeats)
        frozen_workloads = {
            name: Path(info["frozen_path"])
            for name, info in manifest.get("workloads", {}).items()
        }
        sample_pair_rows = _compute_sample_pair_rows(
            loaded, _expected_compute_sample_counts(frozen_workloads))
    else:
        paired_rows = []
        for workload_name in workload_names:
            for repeat in range(repeats):
                block = {row["arm"]: row for row in summary_rows
                         if row["workload"] == workload_name and row["repeat"] == repeat
                         and row["status"] == "ok"}
                baseline, candidate = (block.get(arms[0]), block.get(arms[1]))
                paired_rows.append({
                    "workload": workload_name, "seed": manifest["seed"], "repeat": repeat,
                    "baseline_arm": arms[0], "candidate_arm": arms[1],
                    "baseline_makespan_s": baseline["makespan_s"] if baseline else None,
                    "candidate_makespan_s": candidate["makespan_s"] if candidate else None,
                    "candidate_minus_baseline_ms": (
                        (candidate["makespan_s"] - baseline["makespan_s"]) * 1000
                        if baseline and candidate else None),
                    "baseline_over_candidate_speedup": (
                        baseline["makespan_s"] / candidate["makespan_s"]
                        if baseline and candidate and candidate["makespan_s"] else None),
                })
        paired_statistics = []
        for workload_name in workload_names:
            rows = [row for row in paired_rows if row["workload"] == workload_name]
            deltas = [row["candidate_minus_baseline_ms"] for row in rows
                      if row["candidate_minus_baseline_ms"] is not None]
            speedups = [row["baseline_over_candidate_speedup"] for row in rows
                        if row["baseline_over_candidate_speedup"] is not None]
            paired_statistics.append({
                "workload": workload_name, "baseline_arm": arms[0], "candidate_arm": arms[1],
                "attempted_pairs": len(rows), "successful_pairs": len(deltas),
                "candidate_minus_baseline_median_ms": statistics.median(deltas) if deltas else None,
                "candidate_minus_baseline_min_ms": min(deltas) if deltas else None,
                "candidate_minus_baseline_max_ms": max(deltas) if deltas else None,
                "baseline_over_candidate_speedup_median": statistics.median(speedups) if speedups else None,
                "baseline_over_candidate_speedup_min": min(speedups) if speedups else None,
                "baseline_over_candidate_speedup_max": max(speedups) if speedups else None,
                "candidate_faster_pairs": sum(value < 0 for value in deltas),
                "candidate_slower_pairs": sum(value > 0 for value in deltas),
            })
    segment_rows = (_declaration_segment_rows(task_rows, coordinator_task_rows)
                    if stage == "F2" else [])
    paired_segment_rows, paired_segment_statistics = (
        _paired_declaration_segment_rows(segment_rows, records)
        if stage == "F2" else ([], []))

    derived = {
        "summary.csv": summary_rows, "task-timings.csv": task_rows,
        "coordinator-task-timings.csv": coordinator_task_rows,
        "coordinator-events.csv": coordinator_event_rows,
        "paired-summary.csv": paired_rows,
        "paired-summary-statistics.csv": paired_statistics,
    }
    if stage == "F2":
        derived["run-segments.csv"] = segment_rows
        derived["paired-segments.csv"] = paired_segment_rows
        derived["paired-segment-statistics.csv"] = paired_segment_statistics
    if stage == "F4":
        derived["job-jct.csv"] = job_rows
        derived["job-seed-paired-summary.csv"] = job_seed_rows
        derived["compute-sample-pairs.csv"] = sample_pair_rows
        derived["seed-paired-summary.csv"] = seed_summary_rows
        derived["paired-resource-summary.csv"] = resource_rows
        derived["paired-resource-statistics.csv"] = resource_statistics
    tables_dir = batch_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    for name, rows in derived.items():
        batch._write_csv(batch_dir / name, rows)
        batch._write_csv(tables_dir / name, rows)
    raw_digest = _sha256(runs_path)
    analysis = [
        f"# {stage} control-path diagnostic",
        "",
        f"Derived tables rebuilt from {len(loaded)} SHA-256-verified raw records. "
        "This operation did not modify manifest.json, runs.jsonl, or raw/.",
        f"runs.jsonl SHA-256: `{raw_digest}`.",
        "Completion-probe intervals are computed within each rank-local clock; coordinator "
        "intervals use only the coordinator clock.",
    ]
    if stage == "F4":
        analysis.extend((
            "F4 uses five predeclared seed blocks with three paired repeats per L0/L1 workload. "
            "Seeds are paired by workload and repeat; repeats are summarized within seed before "
            "the seed-block comparison.",
            f"Exact paired execution-sample matches: "
            f"{sum(row.get('samples_match') is True for row in sample_pair_rows)}"
            f"/{len(sample_pair_rows)} workload/seed/repeat blocks.",
            "Review `seed-paired-summary.csv`, `job-seed-paired-summary.csv`, and "
            "`compute-sample-pairs.csv`; these results do not by themselves establish "
            "generalization beyond the fixed L0/L1 inputs and five seed blocks.",
        ))
    (batch_dir / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    return {"batch_dir": str(batch_dir), "runs": len(loaded),
            "failed": sum(row["status"] != "ok" for row in summary_rows),
            "derived_tables": len(derived), "runs_sha256": raw_digest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workload-4k", type=Path)
    parser.add_argument("--workload-1m", type=Path)
    parser.add_argument("--workload-l0", type=Path)
    parser.add_argument("--workload-l1", type=Path)
    parser.add_argument("--comm-profile", type=Path)
    parser.add_argument("--stage", choices=("E2", "E3", "F1", "F2", "F3", "F4"), default="E2")
    parser.add_argument("--rebuild-derived", action="store_true",
                        help="rebuild CSV/analysis from SHA-verified raw files only")
    parser.add_argument("--seed", type=int, default=6200)
    parser.add_argument("--seeds", default="5300,5301,5302,5303,5304",
                        help="five seed blocks for F4")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compute-jitter", type=float, default=0.0)
    parser.add_argument("--order-seed", type=int, default=20260924)
    parser.add_argument("--timeout", type=float, default=35.0)
    args = parser.parse_args(argv)
    if args.stage == "F4":
        if args.repeats != 3:
            parser.error("F4 is specified as three repeats per seed/workload/arm")
    elif args.repeats != 5:
        parser.error(f"{args.stage} is specified as five repeats per workload/arm")
    if not 0 <= args.compute_jitter < 1:
        parser.error("compute-jitter must be in [0, 1)")
    try:
        seed_values = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    except ValueError:
        parser.error("--seeds must be comma-separated integers")
    if args.stage == "F4" and (len(seed_values) != 5 or len(set(seed_values)) != 5):
        parser.error("F4 requires exactly five distinct seed blocks")
    if args.stage != "F4" and args.compute_jitter != 0:
        parser.error("compute-jitter is fixed at zero outside F4")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    output_dir = repository_path(args.output_dir).resolve()
    if args.rebuild_derived:
        print(json.dumps(_rebuild_derived(output_dir), sort_keys=True))
        return 0
    if args.comm_profile is None:
        parser.error("--comm-profile is required for new experiment runs")
    workloads: dict[str, Path] = {}
    if args.stage == "F4":
        if args.workload_l0 is None or args.workload_l1 is None:
            parser.error("F4 requires --workload-l0 and --workload-l1")
        workloads = {
            "L0-balanced": repository_path(args.workload_l0).resolve(strict=True),
            "L1-head-misalignment": repository_path(args.workload_l1).resolve(strict=True),
        }
    elif args.stage == "F3":
        if args.workload_1m is None:
            parser.error("F3 requires --workload-1m")
        workloads = {"1MiB-32x": repository_path(args.workload_1m).resolve(strict=True)}
    elif args.stage in {"E2", "E3"}:
        if args.workload_1m is None:
            parser.error("E2/E3 require --workload-1m")
        if args.workload_4k is None:
            parser.error("E2/E3 require --workload-4k")
        workloads = {"4KiB-32x": repository_path(args.workload_4k).resolve(strict=True)}
        workloads["1MiB-32x"] = repository_path(args.workload_1m).resolve(strict=True)
    else:
        if args.workload_4k is None:
            parser.error("F1/F2 require --workload-4k")
        workloads = {"4KiB-32x": repository_path(args.workload_4k).resolve(strict=True)}
    profile = repository_path(args.comm_profile).resolve(strict=True)
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw"
    inputs_dir = output_dir / "inputs"
    tables_dir = output_dir / "tables"
    for directory in (raw_dir, inputs_dir, tables_dir):
        directory.mkdir()
    frozen_workloads = {}
    for name, path in workloads.items():
        frozen = inputs_dir / path.name
        frozen.write_bytes(path.read_bytes())
        frozen_workloads[name] = {"path": str(path), "frozen_path": str(frozen),
                                  "sha256": _sha256(path)}
    frozen_profile = inputs_dir / "comm-profile.json"
    frozen_profile.write_bytes(profile.read_bytes())
    source_snapshot = batch._source_snapshot()
    environment = {
        "python": sys.version, "platform": platform.platform(), "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "GLOO_SOCKET_IFNAME")},
        "torch": __import__("torch").__version__,
    }
    manifest = {
        "schema_version": 1, "batch_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage, "backend": "gloo", "world_size": 2,
        "max_inflight": 1, "binding_preparation": "precreate",
        "makespan_boundary": "maximum_rank_application_release_to_end_duration",
        "observation_mode": {"F1": "minimal", "F3": "minimal", "F4": "minimal"}.get(args.stage, "diagnostic"),
        "declaration_modes": (["before-producer", "on-submit"]
                              if args.stage in {"F1", "F2", "F3", "F4"} else None),
        "completion_poll_interval_s": 0.001,
        "warmup_iterations": 1, "wait_budget_s": 0.02,
        "compute_jitter": args.compute_jitter,
        "seed": seed_values[0] if args.stage == "F4" else args.seed,
        "seed_blocks": list(seed_values) if args.stage == "F4" else [args.seed],
        "repeats_per_workload_arm": args.repeats,
        "order_seed": args.order_seed, "workloads": frozen_workloads,
        "profile": {"path": str(profile), "frozen_path": str(frozen_profile),
                    "sha256": _sha256(profile)},
        "environment": environment,
        "git_head": batch._git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(batch._git("status", "--porcelain")),
        "source_snapshot": source_snapshot,
        "control_path_note": "Rank-local and coordinator timestamps are never cross-subtracted.",
        "policy": "fifo" if args.stage == "F4" else "static_fifo",
        "comparison": ("old-fifo vs new-static_fifo" if args.stage == "E2" else
                       "new-static_fifo polling vs submission-woken completion probe"
                       if args.stage == "E3" else
                       "before-producer DECLARE vs direct on-submit OFFER"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (inputs_dir / "batch-config.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    args.output_dir = output_dir
    plan = _plan(args.seed, args.repeats, args.order_seed, workloads, args.stage,
                 seed_values if args.stage == "F4" else None)
    runs_path = output_dir / "runs.jsonl"
    run_records = []
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", ""))),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    })
    args.workload = ""
    args.dag = None
    args.backend = "gloo"
    args.world_size = 2
    args.compute_jitter = args.compute_jitter
    args.wait_budget_s = 0.02
    args.poll_interval = 0.001
    args.binding_preparation = "precreate"
    args.warmup_iterations = 1
    args.declaration_mode = "before-producer"
    args.comm_profile = profile
    args.static_ltf_order = None
    for item in plan:
        result_path = raw_dir / f"{item['run_id']}.json"
        record_path = raw_dir / f"{item['run_id']}"
        args.workload = item["workload"]
        old = item["arm"] == "old-fifo"
        wake_completion = item["completion_wakeup_on_submit"]
        args.declaration_mode = item["declaration_mode"]
        policy = "fifo" if old or args.stage == "F4" else "static_fifo"
        command = batch._command(
            args, policy, item["seed"], result_path,
            old=old, binding_preparation="precreate", poll_interval=0.001,
            observation_mode=item["observation_mode"], completion_wakeup=wake_completion,
            declaration_mode=item["declaration_mode"]
            if args.stage in {"F1", "F2", "F3", "F4"} else None)
        start = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                       text=True, timeout=args.timeout + 10)
            stdout, stderr = completed.stdout, completed.stderr
            returncode = completed.returncode
            error = None if returncode == 0 else stderr[-4000:]
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = str(exc.stdout or ""), str(exc.stderr or "")
            returncode, error = None, f"timeout: {stderr[-3000:]}"
        (record_path.with_suffix(".stdout")).write_text(stdout)
        (record_path.with_suffix(".stderr")).write_text(stderr)
        elapsed = time.monotonic() - start
        try:
            result = json.loads(result_path.read_text()) if result_path.is_file() else None
        except json.JSONDecodeError:
            result = None
            error = "invalid result JSON"
        run = {
            **item, "command": command, "started_at": started_at,
            "wall_time_s": elapsed, "returncode": returncode,
            "result_path": str(result_path),
            "result_sha256": _sha256(result_path) if result_path.is_file() else None,
            "observation_mode": item["observation_mode"],
            "declaration_mode": item["declaration_mode"],
            "error": error,
        }
        with runs_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(run, sort_keys=True) + "\n")
        run_records.append((run, result))

    summary_rows = [_summary_row(record, result, Path(record["result_path"]))
                    for record, result in run_records]
    task_rows = [row for record, result in run_records
                 for row in batch._task_timing_rows({
                     "run_id": record["run_id"], "arm": record["arm"],
                     "group": "old" if record["arm"] == "old-fifo" else "new",
                     "config": {"epoch": record["seed"], "repeat": record["repeat"],
                                "workload": record["workload"], "compute_jitter": args.compute_jitter,
                                "binding_preparation": "precreate",
                                "poll_interval_s": 0.001,
                                "observation_mode": record["observation_mode"],
                                "declaration_mode": record["declaration_mode"],
                                "completion_wakeup_on_submit": record[
                                    "completion_wakeup_on_submit"]},
                     "returncode": record["returncode"],
                 }, result)]
    coordinator_task_rows = []
    coordinator_event_rows = []
    for record, result in run_records:
        task_diag, event_diag = batch._coordinator_diagnostic_rows({
            "run_id": record["run_id"], "arm": record["arm"],
            "config": {"epoch": record["seed"], "repeat": record["repeat"],
                       "observation_mode": record["observation_mode"],
                       "declaration_mode": record["declaration_mode"],
                       "completion_wakeup_on_submit": record[
                           "completion_wakeup_on_submit"]},
            "returncode": record["returncode"],
        }, result)
        coordinator_task_rows.extend(task_diag)
        coordinator_event_rows.extend(event_diag)
    if args.stage == "E2":
        baseline_arm, candidate_arm = "old-fifo", "new-static_fifo"
    elif args.stage == "E3":
        baseline_arm, candidate_arm = "new-poll", "new-wakeup"
    else:
        baseline_arm, candidate_arm = "before-producer", "on-submit"
    if args.stage == "F4":
        paired_rows, paired_statistics, seed_summary_rows = _paired_declaration_rows(
            summary_rows, tuple(workloads), seed_values, args.repeats)
        job_rows = _job_jct_rows([(record, result, Path(record["result_path"]))
                                  for record, result in run_records])
        job_seed_rows = _job_seed_pair_rows(job_rows, seed_values, args.repeats)
        resource_rows, resource_statistics = _paired_resource_rows(
            summary_rows, tuple(workloads), seed_values, args.repeats)
        sample_pair_rows = _compute_sample_pair_rows(
            [(record, result, Path(record["result_path"])) for record, result in run_records],
            _expected_compute_sample_counts(workloads))
    else:
        paired_rows = []
        for workload_name in workloads:
            for repeat in range(args.repeats):
                block = {row["arm"]: row for row in summary_rows
                         if row["workload"] == workload_name and row["repeat"] == repeat
                         and row["status"] == "ok"}
                old_row, new_row = block.get(baseline_arm), block.get(candidate_arm)
                paired_rows.append({
                    "workload": workload_name, "seed": args.seed, "repeat": repeat,
                    "baseline_arm": baseline_arm, "candidate_arm": candidate_arm,
                    "baseline_makespan_s": old_row["makespan_s"] if old_row else None,
                    "candidate_makespan_s": new_row["makespan_s"] if new_row else None,
                    "candidate_minus_baseline_ms": (
                        (new_row["makespan_s"] - old_row["makespan_s"]) * 1000
                        if old_row and new_row else None),
                    "baseline_over_candidate_speedup": (
                        old_row["makespan_s"] / new_row["makespan_s"]
                        if old_row and new_row and new_row["makespan_s"] else None),
                })
        paired_statistics = []
        for workload_name in workloads:
            rows = [row for row in paired_rows if row["workload"] == workload_name]
            deltas = [row["candidate_minus_baseline_ms"] for row in rows
                      if row["candidate_minus_baseline_ms"] is not None]
            speedups = [row["baseline_over_candidate_speedup"] for row in rows
                        if row["baseline_over_candidate_speedup"] is not None]
            paired_statistics.append({
                "workload": workload_name, "baseline_arm": baseline_arm,
                "candidate_arm": candidate_arm, "attempted_pairs": len(rows),
                "successful_pairs": len(deltas),
                "candidate_minus_baseline_median_ms": statistics.median(deltas) if deltas else None,
                "candidate_minus_baseline_min_ms": min(deltas) if deltas else None,
                "candidate_minus_baseline_max_ms": max(deltas) if deltas else None,
                "baseline_over_candidate_speedup_median": statistics.median(speedups) if speedups else None,
                "baseline_over_candidate_speedup_min": min(speedups) if speedups else None,
                "baseline_over_candidate_speedup_max": max(speedups) if speedups else None,
                "candidate_faster_pairs": sum(value < 0 for value in deltas),
                "candidate_slower_pairs": sum(value > 0 for value in deltas),
            })
    segment_rows = (_declaration_segment_rows(task_rows, coordinator_task_rows)
                    if args.stage == "F2" else [])
    paired_segment_rows, paired_segment_statistics = (
        _paired_declaration_segment_rows(segment_rows, [record for record, _ in run_records])
        if args.stage == "F2" else ([], []))
    artifacts = [
        ("summary.csv", summary_rows), ("task-timings.csv", task_rows),
        ("coordinator-task-timings.csv", coordinator_task_rows),
        ("coordinator-events.csv", coordinator_event_rows), ("paired-summary.csv", paired_rows),
        ("paired-summary-statistics.csv", paired_statistics),
    ]
    if args.stage == "F2":
        artifacts.append(("run-segments.csv", segment_rows))
        artifacts.append(("paired-segments.csv", paired_segment_rows))
        artifacts.append(("paired-segment-statistics.csv", paired_segment_statistics))
    if args.stage == "F4":
        artifacts.extend((
            ("job-jct.csv", job_rows),
            ("job-seed-paired-summary.csv", job_seed_rows),
            ("compute-sample-pairs.csv", sample_pair_rows),
            ("seed-paired-summary.csv", seed_summary_rows),
            ("paired-resource-summary.csv", resource_rows),
            ("paired-resource-statistics.csv", resource_statistics),
        ))
    for filename, rows in artifacts:
        _write_csv(output_dir / filename, rows)
        _write_csv(tables_dir / filename, rows)
    success = all(row["status"] == "ok" for row in summary_rows)
    if args.stage == "F4":
        success = (success
                   and all(row["all_collectives_correct"] is True for row in summary_rows)
                   and all(row["offer_send_calls"] == row["offer_messages_expected"]
                           and row["declare_send_calls"] == row["declare_messages_expected"]
                           and row["submitted_reports"] == row["offer_messages_expected"]
                           and row["completed_reports"] == row["offer_messages_expected"]
                           for row in summary_rows)
                   and all(row["samples_match"] is True
                           and row["sample_count_matches_expected"] is True
                           for row in sample_pair_rows)
                   and len(sample_pair_rows) == len(workloads) * len(seed_values) * args.repeats)
    analysis = [
        f"# {args.stage} control-path diagnostic",
        "",
        (f"F4 application regression: two ranks, CPU/Gloo, new FIFO, `max_inflight=1`, precreated "
         "bindings, 1 ms completion poll, one warmup, compute jitter 0.3, five seed blocks × three "
         "repeats per L0/L1 workload/arm. Modes were randomized within each workload/seed/repeat pair; "
         "all runs were serial." if args.stage == "F4" else
         f"Fixed-input {args.stage} batch: two ranks, CPU/Gloo, `max_inflight=1`, precreated bindings, "
         "1 ms completion poll, one warmup, five repeats per workload/arm. Configurations were "
         "randomized within each repeat block and executed serially."),
        "",
        ("The five seed blocks are compared only after taking the median of their three paired repeats. "
         "The 32 collectives in a replay are not treated as independent experimental samples." if args.stage == "F4" else
         "The five repeats are fixed-input system-noise observations, not five independent seed blocks. "
         "The 32 collectives in a replay are not treated as independent experimental samples."),
        "Makespan uses the common application-release to application-end boundary on both paths. "
        "Coordinator timestamps are differenced only within the coordinator; rank timestamps only "
        "within a rank. Socket send completion is not remote receipt, and completion observation is "
        "not physical completion time.",
        "",
        f"Successful runs: {sum(row['status'] == 'ok' for row in summary_rows)}/{len(summary_rows)}.",
        "Use `task-timings.csv`, `coordinator-task-timings.csv`, and `coordinator-events.csv` "
        "to identify recurring intervals before proposing any implementation change.",
    ]
    if args.stage in {"F1", "F2", "F3", "F4"}:
        analysis.extend((
            "Declaration-mode comparison is paired by repeat. Negative candidate-minus-baseline "
            "means `on-submit` was faster; speedup is before-producer/on-submit. Failed attempts "
            "remain in the run ledger and are not silently replaced.",
            ("F4 repeats are nested within the five seed blocks; they are not counted as additional seeds."
             if args.stage == "F4" else
             "The five repeats are fixed-input observations, not independent seed blocks."),
        ))
    if args.stage == "F4":
        analysis.extend((
            "For each seed block, the three paired repeats are reduced to within-arm medians before "
            "comparing seeds. `seed-paired-summary.csv` is the primary makespan summary; "
            "`job-seed-paired-summary.csv` retains per-job JCT comparisons.",
            "`paired-resource-statistics.csv` reports paired CPU time, context switches, rank application "
            "time, and application-to-drain differences.",
            f"Exact compute-sample matches: {sum(row['samples_match'] is True for row in sample_pair_rows)}"
            f"/{len(sample_pair_rows)} workload/seed/repeat pairs; expected sample-count matches: "
            f"{sum(row['sample_count_matches_expected'] is True for row in sample_pair_rows)}"
            f"/{len(sample_pair_rows)}. F4 status is successful only if "
            "all replay validation, protocol-count, and sample-equality checks pass.",
        ))
    if args.stage == "F2":
        analysis.append(
            "`run-segments.csv` summarizes intervals within each replay (median, P90 and maximum), "
            "separately per rank and coordinator clock; repeats are compared only after this reduction."
        )
        analysis.append(
            "`paired-segments.csv` and `paired-segment-statistics.csv` compare those run-level "
            "summaries within the same repeat and rank/clock domain; they do not treat the 32 "
            "communications as independent runs."
        )
    (output_dir / "analysis.md").write_text("\n".join(analysis) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "runs": len(summary_rows),
                      "failed": sum(row["status"] != "ok" for row in summary_rows)}, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
