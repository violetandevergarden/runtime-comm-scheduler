"""Plan/run paired A/A and minimal/full diagnostics, separate from performance blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

from examples.jobpacer.scripts.gpu_seven_arm_suite import BARE_ARM, CONTRACT_VERSION
from examples.jobpacer.scripts.run_gpu_seven_arm import (
    ARM_CONFIG, ROOT, _canonical, _cuda_environment, _reserve_run_attempt,
    _run_child, _verify_frozen_inputs, _write_json, source_snapshot,
)

REPRESENTATIVES = (BARE_ARM, "old-static-fifo", "new-dynamic-ltf")


def plan(suite_path: Path, output: Path, *, pairs: int = 5, timeout: float = 60) -> dict:
    if pairs < 5 or timeout <= 0:
        raise ValueError("measurement diagnostics require at least five pairs and positive timeout")
    suite_path = suite_path.resolve(strict=True)
    suite = json.loads(suite_path.read_text())
    if suite.get("contract_version") != CONTRACT_VERSION or suite.get("scope") != "pilot":
        raise ValueError("measurement diagnostics require profiled v2 pilot inputs")
    _verify_frozen_inputs(suite_path.parent, suite)
    if output.exists():
        raise FileExistsError(output)
    sample = next(row for row in suite["samples"] if row["scenario"] == "D1-asymmetric-frontiers")
    runs = []
    for kind in ("aa", "observation"):
        for pair in range(pairs):
            for arm in REPRESENTATIVES:
                labels = ["A", "B"] if pair % 2 == 0 else ["B", "A"]
                for label in labels:
                    runs.append({"run_id": f"{kind}-{arm}-{pair}-{label}",
                                 "block_id": f"{kind}-{arm}-{pair}", "kind": kind,
                                 "pair": pair, "arm": arm, "label": label,
                                 "mode": "full" if kind == "observation" and label == "B" else "minimal"})
    output.mkdir(parents=True)
    for directory in ("raw", "logs", "source"):
        (output / directory).mkdir()
    manifest = {"schema": "jobpacer-seven-arm-measurement-v1", "suite_path": str(suite_path),
                "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                "sample": sample, "timeout": timeout, "runs": runs,
                "environment": _cuda_environment(), "source_snapshot": source_snapshot(output / "source")}
    _write_json(output / "plan.json", manifest)
    return manifest


def analyze(directory: Path) -> dict:
    manifest = json.loads((directory / "plan.json").read_text())
    suite = json.loads(Path(manifest["suite_path"]).read_text())
    records = json.loads((directory / "records.json").read_text())
    grouped = {}
    errors = []
    artifacts = []
    suite_path = Path(manifest["suite_path"])
    if hashlib.sha256(suite_path.read_bytes()).hexdigest() != manifest["suite_sha256"]:
        errors.append("suite changed after measurement plan")
    try:
        _verify_frozen_inputs(suite_path.parent, suite)
    except (ValueError, OSError) as exc:
        errors.append(str(exc))
    planned = {row["run_id"]: row for row in manifest["runs"]}
    seen = set()
    for row in records:
        intended = planned.get(row["run_id"])
        if intended is None or row["run_id"] in seen or any(row.get(key) != value for key, value in intended.items()):
            errors.append(f"unplanned, duplicate or changed run: {row['run_id']}")
            continue
        seen.add(row["run_id"])
        raw_path = directory / row["raw_path"]
        if row.get("returncode") != 0 or not row.get("group_exited") or row.get("timed_out"):
            errors.append(f"unsuccessful run: {row['run_id']}")
            continue
        if not raw_path.is_file() or hashlib.sha256(raw_path.read_bytes()).hexdigest() != row.get("raw_sha256"):
            errors.append(f"raw changed/missing: {row['run_id']}")
            continue
        raw = json.loads(raw_path.read_text())
        ranks = raw.get("ranks", [])
        from examples.jobpacer.analysis.runtime_results import validate_results, expected_dag_results
        from examples.jobpacer.runtime.runtime_adapter import load_dag
        dag = load_dag(Path(manifest["suite_path"]).parent / manifest["sample"]["path"], world_size=2)
        expected = expected_dag_results(dag.graph, 2)
        check = validate_results(ranks, 2, expected=expected["expected"], expected_nodes=expected["expected_nodes"])
        engine, policy, _ = ARM_CONFIG[row["arm"]]
        if (check.get("status") != "ok" or len(ranks) != 2 or any(
                rank.get("comm_engine") != engine or rank.get("policy") != policy
                or rank.get("observation_mode") != row["mode"]
                or not rank.get("gpu_dag_validation")
                or not all(value for checks in rank["gpu_dag_validation"].values() for value in checks.values())
                for rank in ranks)):
            errors.append(f"semantic/identity validation failed: {row['run_id']}")
            continue
        value = {"time_us": raw["performance"]["workload_makespan_us"],
                 "cpu_s": sum(rank["process_cpu_time_s"] for rank in ranks),
                 "events": sum(len(rank.get("runtime_events", [])) + len(rank.get("dag_events", [])) for rank in ranks),
                 "context_switches": sum(rank["voluntary_context_switches"] + rank["involuntary_context_switches"] for rank in ranks),
                 "mechanism": next((rank.get("mechanism_counts") for rank in ranks if rank["rank"] == 0), None)}
        grouped.setdefault((row["kind"], row["arm"], row["pair"]), {})[row["label"]] = value
        artifacts.append({"path": str(raw_path.resolve()), "sha256": row["raw_sha256"]})
    pairs = []
    for (kind, arm, pair), values in sorted(grouped.items()):
        if set(values) != {"A", "B"}:
            errors.append(f"incomplete pair: {kind}/{arm}/{pair}")
            continue
        pairs.append({"kind": kind, "arm": arm, "pair": pair, **values,
                      "delta_us": values["B"]["time_us"] - values["A"]["time_us"],
                      "ratio": values["B"]["time_us"] / values["A"]["time_us"]})
    coverage = all(sum(row["kind"] == kind and row["arm"] == arm for row in pairs) >= 5
                   for kind in ("aa", "observation") for arm in REPRESENTATIVES)
    minimal_competition = [row["A"]["mechanism"] for row in pairs
                           if row["kind"] == "observation" and row["arm"] == "new-dynamic-ltf"]
    full_competition = [row["B"]["mechanism"] for row in pairs
                        if row["kind"] == "observation" and row["arm"] == "new-dynamic-ltf"]
    mechanism_preserved = bool(minimal_competition) and all(minimal_competition + full_competition) and sum(
        item["fifo_ltf_disagreements"] > 0 for item in minimal_competition) >= 2
    checks = {"all_planned_runs": len(records) == len(manifest["runs"]), "semantic_checks": not errors,
              "five_pairs_per_path_and_kind": coverage, "minimal_mechanism_preserved": bool(mechanism_preserved)}
    report = {"gate": "measurement", "passed": all(checks.values()), "checks": checks,
              "source_snapshot_sha256": manifest["source_snapshot"]["digest"],
              "generator_version": suite["generator_version"],
              "profiles": {key: value["sha256"] for key, value in suite["profiles"].items()},
              "artifacts": artifacts, "pairs": pairs, "errors": errors,
              "noise": {arm: {"median_absolute_aa_delta_us": statistics.median(
                  abs(row["delta_us"]) for row in pairs if row["arm"] == arm and row["kind"] == "aa")}
                  for arm in REPRESENTATIVES if any(row["arm"] == arm and row["kind"] == "aa" for row in pairs)},
              "interpretation": "small noise diagnostic; no log-cost subtraction or significance stopping"}
    _write_json(directory / "measurement-audit.json", report)
    return report


def run(directory: Path) -> dict:
    directory = directory.resolve(strict=True)
    manifest = json.loads((directory / "plan.json").read_text())
    suite_path = Path(manifest["suite_path"])
    if hashlib.sha256(suite_path.read_bytes()).hexdigest() != manifest["suite_sha256"]:
        raise ValueError("measurement suite changed after plan")
    if source_snapshot()["digest"] != manifest["source_snapshot"]["digest"] or _cuda_environment() != manifest["environment"]:
        raise ValueError("measurement source/environment changed after plan")
    suite = json.loads(suite_path.read_text())
    _verify_frozen_inputs(suite_path.parent, suite)
    if (directory / "records.json").exists() or any((directory / "logs").iterdir()):
        raise ValueError("diagnostic attempts already exist; preserve them and plan a new directory")
    records = []
    for row in manifest["runs"]:
        reservation = _reserve_run_attempt(row, directory, 1)
        engine, policy, order_key = ARM_CONFIG[row["arm"]]
        sample = manifest["sample"]
        command = [sys.executable, "-m", "examples.jobpacer.scripts.run_phase3", "--comm-engine", engine,
                   "--policy", policy, "--backend", "nccl", "--world-size", "2", "--startup-attempts", "1",
                   "--dag", str(suite_path.parent / sample["path"]), "--warmup-iterations", "5",
                   "--timeout", str(manifest["timeout"]), "--setup-timeout", str(manifest["timeout"]),
                   "--observation-mode", row["mode"], "--output", str(directory / reservation["raw_path"])]
        for kind, flag in (("compute", "--compute-profile"), ("communication", "--comm-profile")):
            command += [flag, str(suite_path.parent / suite["profiles"][kind]["path"])]
        if order_key:
            command += ["--static-order", str(suite_path.parent / sample[order_key])]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")))
        code, stdout, stderr, timed_out, exited = _run_child(command, cwd=ROOT, env=env,
            timeout_s=2 * manifest["timeout"] + 5, batch_dir=directory, reservation=reservation)
        (directory / reservation["stdout_path"]).write_text(stdout)
        (directory / reservation["stderr_path"]).write_text(stderr)
        raw_path = directory / reservation["raw_path"]
        records.append({**row, "command": command, "returncode": code, "timed_out": timed_out,
                        "group_exited": exited, "raw_path": reservation["raw_path"],
                        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest() if raw_path.is_file() else None})
        _write_json(directory / "records.json", records)
        if code != 0 or timed_out or not exited:
            break
    return analyze(directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        if args.suite is None:
            parser.error("plan requires --suite")
        result = plan(args.suite, args.output)
    else:
        result = run(args.output) if args.command == "run" else analyze(args.output)
    print(json.dumps(result, sort_keys=True))
    return 1 if result.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
