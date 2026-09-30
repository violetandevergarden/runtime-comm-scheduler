"""Prepare, profile-freeze, and order the Phase 3 GPU seven-arm input suite."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from examples.jobpacer.scripts.gpu_seven_arm_suite import (
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="write six templates and five frozen workload samples")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--scope", choices=("formal", "pilot"), default="formal")
    prepare.add_argument("--workload-seeds",
                         help="override the default seeds for formal or pilot input generation")
    freeze = commands.add_parser("freeze", help="apply calibrated profiles and save shared static orders")
    freeze.add_argument("--input-dir", type=Path, required=True)
    freeze.add_argument("--compute-profile", type=Path, required=True)
    freeze.add_argument("--comm-profile", type=Path, required=True)
    freeze.add_argument("--calibration-input-dir", type=Path,
                        help="formal five-seed suite used by the compute profiler; defaults to --input-dir")
    freeze.add_argument("--world-size", type=int, default=2)
    order = commands.add_parser("order", help="write a deterministic block/arm order table")
    order.add_argument("--suite-manifest", type=Path, required=True)
    order.add_argument("--output", type=Path, required=True)
    order.add_argument("--order-seed", type=int, default=ORDER_SEED)
    order.add_argument("--repeats", type=int, default=5)
    order.add_argument("--arms", default=",".join(ARMS))
    preview = commands.add_parser("preview", help="estimate full batch wall time from prior runs")
    preview.add_argument("--order", type=Path, required=True)
    preview.add_argument("--history-summary", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    if args.command == "prepare":
        try:
            raw_seeds = args.workload_seeds or ",".join(map(
                str, FORMAL_WORKLOAD_SEEDS if args.scope == "formal" else PILOT_WORKLOAD_SEEDS))
            seeds = tuple(int(part.strip()) for part in raw_seeds.split(",") if part.strip())
            expected_count = 5 if args.scope == "formal" else 3
            if len(seeds) != expected_count:
                raise ValueError(f"{args.scope} scope requires exactly {expected_count} workload seed(s)")
            suite = prepare_suite(args.output_dir.resolve(), seeds=seeds)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                          "scenarios": len(suite["scenarios"]),
                          "samples": sum(len(row["samples"]) for row in suite["scenarios"].values()),
                          "status": suite["status"]}, sort_keys=True))
        return 0
    if args.command == "freeze":
        try:
            manifest = freeze_suite(args.input_dir, args.compute_profile, args.comm_profile,
                                    world_size=args.world_size,
                                    calibration_input_dir=args.calibration_input_dir)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({"suite_manifest": str(args.input_dir.resolve() / "suite-manifest.json"),
                          "samples": len(manifest["samples"]), "status": manifest["status"]},
                         sort_keys=True))
        return 0
    if args.command == "order":
        try:
            manifest = json.loads(args.suite_manifest.read_text())
            arms = tuple(part.strip() for part in args.arms.split(",") if part.strip())
            table = make_order_table(manifest, order_seed=args.order_seed,
                                     repeats=args.repeats, arms=arms)
            verify_order_table(table)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(table, sort_keys=True, indent=2) + "\n")
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        print(json.dumps({"order": str(args.output.resolve()), "blocks": table["block_count"],
                          "runs": table["run_count"], "arms": table["arms"]}, sort_keys=True))
        return 0
    try:
        order_table = json.loads(args.order.read_text())
        result = _preview(order_table, args.history_summary)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
