"""Preview or run the fixed, reduced Phase 3 CPU/Gloo experiment suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from examples.jobpacer.comm_profile import apply_profile, load_profile
from examples.jobpacer.analysis.benchmark_paths import resolve_migrated_path
from examples.jobpacer.runtime.runtime_adapter import apply_dag_profile, load_dag
from examples.jobpacer.scripts import run_experiments as batch
from examples.jobpacer.workloads import load_workload


ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = ROOT / "benchmark/phase3/experiments"
RESULTS_ROOT = ROOT / "benchmark/phase3/results"
DEFAULT_SUITE = INPUT_ROOT / "suites/compact.json"
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
CASE_RESULT_PATHS = {
    "L0": "baseline/L0-balanced",
    "L1": "readiness/L1-head-misalignment",
    "L2": "priority/L2-tail-asymmetry",
    "G2": "priority/G2-multi-frontier",
    "L3": "lookahead/L3-lookahead",
    "L4": "lookahead/L4-late-prediction",
    "G4": "lookahead/G4-prediction-frontier",
    "G1": "dag-semantics/G1-diamond",
    "G3": "dag-semantics/G3-group-order",
    "G0-linear": "bridge/G0-linear-bridge/linear",
    "G0-dag": "bridge/G0-linear-bridge/dag",
    "L5": "readiness/L5-member-skew",
}


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _case_batch_dir(case_id: str, suite_id: str, phase: str) -> Path:
    try:
        relative = CASE_RESULT_PATHS[case_id]
    except KeyError as exc:
        raise ValueError(f"no semantic result category for case {case_id!r}") from exc
    category, _, _ = relative.partition("/")
    # The case path may contain a category plus a scenario and, for G0, a runner subdir.
    parts = Path(relative).parts
    scenario_path = Path(*parts[1:])
    batch_dir = RESULTS_ROOT / category / scenario_path / f"{suite_id}-{phase}"
    if len(parts) > 2:
        # G0 results share one scenario batch but retain separate linear/DAG runner outputs.
        scenario = parts[1]
        runner = parts[2]
        batch_dir = RESULTS_ROOT / category / scenario / f"{suite_id}-{phase}" / runner
    return batch_dir


def _suite(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1 or not isinstance(document.get("sections"), dict):
        raise ValueError("compact suite schema_version must be 1 with sections")
    counts = {}
    for section_name in ("main", "mechanism", "lookahead"):
        section = document["sections"][section_name]
        counts[section_name] = len(section["seeds"]) * section["repeats"] * sum(
            len(case["arms"]) for case in section["cases"])
    isolated = document["sections"]["isolated"]
    counts["isolated"] = len(isolated["seeds"]) * isolated["repeats"] * len(
        isolated["conditions"]) * len(isolated["arms"])
    expected = document["expected_replays"]
    expected_by_section = {**expected, "lookahead": expected["lookahead_optional"]}
    if (any(counts[key] != expected_by_section[key] for key in counts)
            or sum(counts[key] for key in ("main", "mechanism", "isolated")) != expected["base_total"]
            or sum(counts.values()) != expected["with_optional"]):
        raise ValueError(f"suite budget mismatch: {counts}")
    for name in ("main", "mechanism", "lookahead"):
        for case in document["sections"][name]["cases"]:
            arms = case["arms"]
            if not arms or len(arms) != len(set(arms)) or any(arm not in batch.ARM_SPECS for arm in arms):
                raise ValueError(f"invalid arms for {name}/{case['id']}")
            if case["baseline"] not in arms or any(arm not in arms for arm in case.get("secondary_baselines", [])):
                raise ValueError(f"invalid baselines for {name}/{case['id']}")
            for first, second in document.get("primary_comparisons", {}).get(case["id"], []):
                if name != "mechanism" and (first not in arms or second not in arms):
                    raise ValueError(f"primary comparison outside selected arms: {name}/{case['id']}")
    return document


def _cpu_set(value: str) -> set[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        bounds = part.strip().split("-", 1)
        if not all(bound.isdigit() for bound in bounds):
            raise ValueError(f"invalid CPU affinity range: {part!r}")
        start, end = int(bounds[0]), int(bounds[-1])
        if end < start:
            raise ValueError(f"invalid CPU affinity range: {part!r}")
        cpus.update(range(start, end + 1))
    if not cpus:
        raise ValueError("CPU affinity must not be empty")
    return cpus


def _validate_profile(profile_path: Path, cases: list[dict[str, Any]], affinity: set[int],
                      threads: dict[str, str]) -> None:
    profile = load_profile(profile_path)
    environment = profile.environment
    if environment.get("hostname") != socket.gethostname():
        raise ValueError("profile hostname differs from this machine; recalibrate")
    if environment.get("torch_version") != __import__("torch").__version__:
        raise ValueError("profile PyTorch version differs; recalibrate")
    if environment.get("cpu_affinity") != sorted(affinity):
        raise ValueError("profile CPU affinity differs or is unrecorded; recalibrate under this affinity")
    expected_threads = {key: os.environ.get(key) for key in (*THREAD_KEYS, "GLOO_SOCKET_IFNAME")}
    if environment.get("thread_environment") != expected_threads:
        raise ValueError("profile thread environment differs or is unrecorded; recalibrate under these threads")
    if any(os.environ.get(key) != value for key, value in threads.items()):
        raise ValueError("suite thread environment was not applied")
    replay_environment = {"backend": "gloo", "device_type": "cpu", "world_size": 2}
    for case in cases:
        if "dag" in case:
            dag = load_dag(INPUT_ROOT / case["dag"], world_size=2)
            apply_dag_profile(dag, profile, replay_environment, strict=True)
        else:
            workload = load_workload(INPUT_ROOT / case["workload"])
            apply_profile(workload, profile, replay_environment, strict=True)


def _case_command(case: dict[str, Any], section: dict[str, Any], common: dict[str, Any],
                  output: Path, profile: Path, tie_threshold: float) -> list[str]:
    command = [sys.executable, "-m", "examples.jobpacer.scripts.run_experiments",
               "--output-dir", str(output),
               "--seeds", ",".join(map(str, section["seeds"])),
               "--repeats", str(section["repeats"]),
               "--arms", ",".join(case["arms"]),
               "--compute-jitter", str(section["compute_jitter"]),
               "--wait-budget-s", str(case.get("wait_budget_s", 0.02)),
               "--poll-interval", str(common["poll_interval_s"]),
               "--warmup-iterations", str(common["warmup_iterations"]),
               "--timeout", str(common["timeout_s"]),
               "--order-seed", str(common["order_seed"]),
               "--bootstrap-samples", str(common["bootstrap_samples"]),
               "--tie-threshold", str(tie_threshold),
               "--baseline", case["baseline"],
               "--comm-profile", str(profile)]
    source_key = "dag" if "dag" in case else "workload"
    command.extend((f"--{source_key}", str(INPUT_ROOT / case[source_key])))
    for baseline in case.get("secondary_baselines", []):
        command.extend(("--secondary-baseline", baseline))
    if case.get("static_ltf_order"):
        command.extend(("--static-ltf-order", str(INPUT_ROOT / case["static_ltf_order"])))
    return command


def _preview(document: dict[str, Any], phase: str, selected: list[dict[str, Any]],
             history_paths: list[Path], *, include_isolated: bool) -> dict[str, Any]:
    sections = document["sections"]
    rows = []
    for name, case in selected:
        section = sections[name]
        count = len(section["seeds"]) * section["repeats"] * len(case["arms"])
        source = case.get("workload", case.get("dag"))
        walls = batch._historical_walls(history_paths, Path(source).stem, tuple(case["arms"]))
        per_block = sum(walls.get(arm, 0.0) for arm in case["arms"])
        covered = all(arm in walls for arm in case["arms"])
        rows.append({"section": name, "case": case["id"], "arms": case["arms"],
                     "group_count": len(case["arms"]),
                     "seeds": len(section["seeds"]), "repeats": section["repeats"],
                     "replays": count,
                     "estimated_wall_time_s": per_block * len(section["seeds"]) * section["repeats"]
                     if covered else None})
    if include_isolated:
        rows.append({"section": "isolated", "case": "L0-interleaved", "arms": ["new-fifo", "new-ltf"],
                     "group_count": 6,
                     "seeds": 1, "repeats": 5, "replays": 30, "estimated_wall_time_s": None})
    estimates = [row["estimated_wall_time_s"] for row in rows]
    return {"suite": document["name"], "phase": phase, "cases": rows,
            "total_replays": sum(row["replays"] for row in rows),
            "estimated_wall_time_s": sum(estimates) if all(value is not None for value in estimates) else None,
            "estimated_from": [str(path) for path in history_paths]}


def _check_mechanism_evidence(suite_id: str, case: dict[str, Any]) -> None:
    required = case.get("requires_mechanism_case")
    if not required:
        return
    path = _case_batch_dir(required, suite_id, "mechanism") / "mechanisms.csv"
    if not path.exists():
        raise ValueError(f"{case['id']} main batch requires mechanism records: {path}")
    import csv
    with path.open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    for arm in ("fifo", "ltf"):
        arm_rows = [row for row in rows if f"new-{arm}" in row["run_id"]]
        count = sum(row["status"] == "ok" and row["mechanism_triggered"] == "True"
                    for row in arm_rows)
        minimum = int(case["minimum_triggered_per_arm"])
        if len(arm_rows) != 5 or count < minimum:
            raise ValueError(f"{case['id']} {arm} verified candidate competition in {count}/5; require {minimum}/5")


def _check_lookahead_evidence(suite_id: str, case_id: str, minimum: int) -> None:
    path = _case_batch_dir(case_id, suite_id, "mechanism") / "mechanisms.csv"
    if not path.exists():
        raise ValueError(f"Lookahead performance requires mechanism pilot: {path}")
    import csv
    with path.open(newline="") as input_file:
        rows = [row for row in csv.DictReader(input_file) if "new-lookahead" in row["run_id"]]
    count = sum(row["status"] == "ok" and row["mechanism_triggered"] == "True" for row in rows)
    if len(rows) != 5 or count < minimum:
        raise ValueError(f"{case_id} Lookahead mechanism reached {count}/5; require {minimum}/5")


def _check_g0_bridge(suite_id: str) -> None:
    linear_path = _case_batch_dir("G0-linear", suite_id, "mechanism") / "runs.jsonl"
    dag_path = _case_batch_dir("G0-dag", suite_id, "mechanism") / "runs.jsonl"
    if not linear_path.exists() or not dag_path.exists():
        return
    linear = batch._latest_records(linear_path)
    dag = batch._latest_records(dag_path)
    if len(linear) != 5 or len(dag) != 5:
        raise ValueError("G0 bridge requires five successful repeats from each runner")
    linear_results = {record["config"]["repeat"]: batch._load(Path(record["result_path"]))
                      for record in linear.values()}
    dag_results = {record["config"]["repeat"]: batch._load(Path(record["result_path"]))
                   for record in dag.values()}
    expected_order = ["job-0/comm-0", "job-1/comm-0", "job-0/comm-1", "job-1/comm-1"]
    checked = []
    for repeat in range(5):
        left, right = linear_results.get(repeat), dag_results.get(repeat)
        if not left or not right or any(item["validation"]["status"] != "ok" for item in (left, right)):
            raise ValueError(f"G0 bridge repeat {repeat} is missing or failed")
        for rank in (0, 1):
            left_rank = next(item for item in left["ranks"] if item["rank"] == rank)
            right_rank = next(item for item in right["ranks"] if item["rank"] == rank)
            if (left_rank["launch_sequence"] != expected_order
                    or right_rank["launch_sequence"] != expected_order):
                raise ValueError(f"G0 bridge repeat {repeat} rank {rank} launch order differs")
            left_jobs = {job["job_id"]: job for job in left_rank["jobs"]}
            right_jobs = {job["job_id"]: job for job in right_rank["jobs"]}
            for job_id in ("job-0", "job-1"):
                for index in (0, 1):
                    a = left_jobs[job_id]["compute_samples_s"]
                    b = right_jobs[job_id]["compute_samples_s"]
                    if a[f"comm-{index}/producer"] != b[f"c{index}"] or a[f"comm-{index}/consumer"] != 0:
                        raise ValueError(f"G0 bridge repeat {repeat} rank {rank} compute samples differ")
            checked.append({"repeat": repeat, "rank": rank, "launch_order": expected_order,
                            "compute_samples_equal": True, "linear_consumer_overlap_s": 0})
    path = RESULTS_ROOT / "bridge/G0-linear-bridge" / f"{suite_id}-mechanism" / "G0-bridge-check.json"
    path.write_text(json.dumps({"status": "ok", "checks": checked}, indent=2, sort_keys=True) + "\n")


def _write_primary_comparisons(batch_output: Path, pairs: list[list[str]]) -> None:
    import csv
    source = batch_output / "paired-summary.csv"
    with source.open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    selected = [row for row in rows if [row["baseline"], row["candidate"]] in pairs]
    batch._write_csv(batch_output / "primary-paired.csv", selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "main", "mechanism", "isolated", "lookahead"), default="all")
    parser.add_argument("--case", action="append", default=[], help="run or preview only named case IDs")
    parser.add_argument("--execute", action="store_true", help="run selected phase; default is read-only preview")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--comm-profile", type=Path)
    parser.add_argument("--cpu-affinity", help="CPU list/ranges, e.g. 0-7,12-15; required for execution")
    parser.add_argument("--tie-threshold", type=float,
                        help="pilot-justified fractional tie threshold, fixed before main/Lookahead runs")
    parser.add_argument("--history-summary", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    try:
        suite_path = _repo_path(args.suite).resolve(strict=True)
        document = _suite(suite_path)
        selected = [(name, case) for name in ("mechanism", "main", "lookahead")
                    if args.phase in ("all", name)
                    for case in document["sections"][name]["cases"]
                    if not args.case or case["id"] in args.case]
        known_ids = {case["id"] for _name, case in selected}
        if args.case and (set(args.case) - known_ids or args.phase == "isolated"):
            raise ValueError("--case must name a case in the selected phase")
        output_argument = _repo_path(args.output_dir)
        history_sources = ([_repo_path(path) for path in args.history_summary]
                           if args.history_summary else ([output_argument] if output_argument.exists() else []))
        history_sources = [resolve_migrated_path(path, RESULTS_ROOT / "migration-map.json")
                           for path in history_sources]
        report = _preview(document, args.phase, selected, history_sources,
                          include_isolated=args.phase == "isolated" or (args.phase == "all" and not args.case))
        if not args.execute:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.phase == "all":
            raise ValueError("execution requires one explicit phase; Lookahead is optional")
        if args.resume and not output_argument.exists():
            raise ValueError("--resume requires an existing suite output directory")
        if not args.comm_profile or not args.cpu_affinity:
            raise ValueError("execution requires --comm-profile and --cpu-affinity")
        if args.phase in {"main", "lookahead"} and args.tie_threshold is None:
            raise ValueError("main/Lookahead execution requires a pilot-justified --tie-threshold")
        if args.tie_threshold is not None and not 0 <= args.tie_threshold < 1:
            raise ValueError("tie threshold must be in [0, 1)")
        affinity = _cpu_set(args.cpu_affinity)
        available = os.sched_getaffinity(0)
        if not affinity.issubset(available):
            raise ValueError(f"CPU affinity {sorted(affinity)} is not within available {sorted(available)}")
        os.sched_setaffinity(0, affinity)
        threads = document["common"]["thread_environment"]
        os.environ.update(threads)
        profile_path = resolve_migrated_path(
            _repo_path(args.comm_profile), RESULTS_ROOT / "migration-map.json").resolve(strict=True)
        _validate_profile(profile_path, [case for _name, case in selected] if selected else [
            {"workload": path} for path in document["sections"]["isolated"]["conditions"]
        ], affinity, threads)
        output_root = output_argument.resolve()
        suites_root = (RESULTS_ROOT / "suites").resolve()
        if not output_root.is_relative_to(suites_root):
            raise ValueError("executed suite indexes must be stored under benchmark/phase3/results/suites/<batch-id>")
        suite_id = output_root.name
        if args.phase == "main":
            for _name, case in selected:
                _check_mechanism_evidence(suite_id, case)
        if args.phase == "lookahead":
            for required in document["sections"]["lookahead"]["requires_mechanism_cases"]:
                _check_lookahead_evidence(suite_id, required,
                                          document["sections"]["lookahead"]["minimum_triggered_lookahead"])
        suite_manifest = {"schema_version": 1, "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                          "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                          "cpu_affinity": sorted(affinity), "thread_environment": threads,
                          "source_snapshot_digest": batch._source_snapshot()["digest"],
                          "result_layout": "semantic-categories-v1",
                          "case_outputs": {
                              f"{name}/{case['id']}": str(_case_batch_dir(case["id"], suite_id, name).relative_to(ROOT))
                              for name in ("mechanism", "main", "lookahead")
                              for case in document["sections"][name]["cases"]
                          },
                          "isolated_output": str((RESULTS_ROOT / "isolated/L0-balanced" /
                                                   f"{suite_id}-diagnostic").relative_to(ROOT))}
        suite_manifest_path = output_root / "compact-manifest.json"
        if suite_manifest_path.exists():
            if json.loads(suite_manifest_path.read_text()) != suite_manifest:
                raise ValueError("suite resume rejected: suite/profile/affinity/threads/source changed")
            if not args.resume:
                raise ValueError("suite output exists; use --resume")
        else:
            if args.resume or output_root.exists():
                raise ValueError("new suite requires a fresh output directory")
            output_root.mkdir(parents=True)
            suite_manifest_path.write_text(json.dumps(suite_manifest, indent=2, sort_keys=True) + "\n")
        if args.phase == "isolated":
            isolated_output = RESULTS_ROOT / "isolated/L0-balanced" / f"{suite_id}-diagnostic"
            command = [sys.executable, "-m", document["sections"]["isolated"]["runner"],
                       "--output-dir", str(isolated_output),
                       "--comm-profile", str(profile_path)]
            if args.resume:
                command.append("--resume")
            return subprocess.run(command, cwd=ROOT, check=False).returncode
        for name, case in selected:
            batch_output = _case_batch_dir(case["id"], suite_id, name)
            batch_output.parent.mkdir(parents=True, exist_ok=True)
            command = _case_command(case, document["sections"][name], document["common"],
                                    batch_output, profile_path, args.tie_threshold or 0.0)
            if batch_output.exists():
                if not args.resume:
                    raise ValueError(f"result batch already exists; choose a new suite ID or pass --resume: {batch_output}")
                command.append("--resume")
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                return completed.returncode
            _write_primary_comparisons(batch_output,
                                       document.get("primary_comparisons", {}).get(case["id"], []))
        if args.phase == "mechanism":
            _check_g0_bridge(suite_id)
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
