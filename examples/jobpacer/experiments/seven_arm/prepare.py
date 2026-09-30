"""Prepare, profile-freeze, and order the Phase 3 GPU seven-arm input suite."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from examples.jobpacer.experiments.seven_arm.suite import (
    ARMS,
    FORMAL_WORKLOAD_SEEDS,
    ORDER_SEED,
    PILOT_WORKLOAD_SEEDS,
    freeze_suite,
    make_order_table,
    prepare_suite,
    verify_order_table,
)


def _csv_history(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        source = path / "summary.csv" if path.is_dir() else path
        if source.name == "runs.jsonl":
            for line in source.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("arm") and isinstance(row.get("wall_time_s"), (float, int)):
                    result.setdefault(row["arm"], []).append(row)
            continue
        if not source.is_file():
            raise ValueError(f"history summary does not exist: {source}")
        with source.open(newline="") as stream:
            for row in csv.DictReader(stream):
                arm = row.get("arm")
                if not arm:
                    continue
                try:
                    row["wall_time_s"] = float(row["wall_time_s"])
                except (KeyError, TypeError, ValueError):
                    continue
                result.setdefault(arm, []).append(row)
    return result


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _preview(order: dict[str, Any], history_paths: list[Path]) -> dict[str, Any]:
    verify_order_table(order)
    history = _csv_history(history_paths)
    counts = {arm: 0 for arm in order["arms"]}
    for block in order["blocks"]:
        for run in block["arms"]:
            counts[run["arm"]] += 1
    per_arm = {}
    estimated_p50 = estimated_p90 = disk_bytes = 0.0
    missing = []
    missing_disk = []
    for arm, run_count in counts.items():
        rows = history.get(arm, [])
        walls = [float(row["wall_time_s"]) for row in rows
                 if isinstance(row.get("wall_time_s"), (float, int)) and row["wall_time_s"] >= 0]
        sizes = [int(row["raw_bytes"]) for row in rows if str(row.get("raw_bytes", "")).isdigit()]
        if not walls:
            missing.append(arm)
            missing_disk.append(arm)
            per_arm[arm] = {"planned_runs": run_count, "history_runs": len(rows),
                            "p50_wall_s": None, "p90_wall_s": None}
            continue
        p50, p90 = statistics.median(walls), _percentile(walls, 0.9)
        estimated_p50 += p50 * run_count
        estimated_p90 += p90 * run_count
        size = int(statistics.median(sizes)) if sizes else 0
        if not sizes:
            missing_disk.append(arm)
        disk_bytes += size * run_count
        per_arm[arm] = {"planned_runs": run_count, "history_runs": len(rows),
                        "p50_wall_s": p50, "p90_wall_s": p90,
                        "median_raw_bytes": size or None}
    return {
        "block_count": order["block_count"], "run_count": order["run_count"],
        "scenario_count": len({block["scenario"] for block in order["blocks"]}),
        "seed_count": len({block["workload_seed"] for block in order["blocks"]}),
        "arms": order["arms"], "runs_per_arm": counts, "history_by_arm": per_arm,
        "history_coverage": f"{len(order['arms']) - len(missing)}/{len(order['arms'])}",
        "estimated_wall_p50_s": None if missing else estimated_p50,
        "estimated_wall_p90_s": None if missing else estimated_p90,
        "estimated_disk_bytes": disk_bytes if not missing and not missing_disk else None,
        "disk_history_coverage": f"{len(order['arms']) - len(missing_disk)}/{len(order['arms'])}",
        "estimates_include": ["process launch", "group setup", "allocation", "warmup",
                               "application", "validation", "drain"] if not missing else None,
        "missing_history_arms": missing,
    }


def prepare_candidates(output_dir: Path, *, scope: str = "formal",
                       workload_seeds: str | None = None) -> dict[str, Any]:
    if scope not in {"formal", "pilot"}:
        raise ValueError("scope must be formal or pilot")
    raw_seeds = workload_seeds or ",".join(map(
        str, FORMAL_WORKLOAD_SEEDS if scope == "formal" else PILOT_WORKLOAD_SEEDS))
    seeds = tuple(int(part.strip()) for part in raw_seeds.split(",") if part.strip())
    expected_count = 5 if scope == "formal" else 3
    if len(seeds) != expected_count:
        raise ValueError(f"{scope} scope requires exactly {expected_count} workload seed(s)")
    suite = prepare_suite(output_dir.resolve(), seeds=seeds)
    return {"output_dir": str(output_dir.resolve()),
            "scenarios": len(suite["scenarios"]),
            "samples": sum(len(row["samples"]) for row in suite["scenarios"].values()),
            "status": suite["status"]}


def freeze_candidates(input_dir: Path, compute_profile: Path, comm_profile: Path, *,
                      calibration_input_dir: Path | None = None, world_size: int = 2) -> dict[str, Any]:
    manifest = freeze_suite(input_dir, compute_profile, comm_profile,
                            world_size=world_size,
                            calibration_input_dir=calibration_input_dir)
    return {"suite_manifest": str(input_dir.resolve() / "suite-manifest.json"),
            "samples": len(manifest["samples"]), "status": manifest["status"]}


def write_order(suite_manifest: Path, output: Path, *, order_seed: int = ORDER_SEED,
                repeats: int = 5, arms: str = ",".join(ARMS)) -> dict[str, Any]:
    manifest = json.loads(suite_manifest.read_text())
    selected_arms = tuple(part.strip() for part in arms.split(",") if part.strip())
    table = make_order_table(manifest, order_seed=order_seed,
                             repeats=repeats, arms=selected_arms)
    verify_order_table(table)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(table, sort_keys=True, indent=2) + "\n")
    return {"order": str(output.resolve()), "blocks": table["block_count"],
            "runs": table["run_count"], "arms": table["arms"]}


def preview_order(order_path: Path, history_paths: list[Path]) -> dict[str, Any]:
    order_table = json.loads(order_path.read_text())
    return _preview(order_table, history_paths)
