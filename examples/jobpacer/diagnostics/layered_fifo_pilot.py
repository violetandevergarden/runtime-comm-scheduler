"""Run a small NCCL pilot for layered, round-robin static communication order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from examples.jobpacer.runtime.runtime_adapter import parse_dag
from examples.jobpacer.experiments.phase3.batch import _cuda_environment, source_snapshot
from runtime_comm_scheduler.dag import build_layered_fifo_order, validate_static_order


ROOT = Path(__file__).resolve().parents[3]
PILOT_SCENARIOS = ("L1-skew-tail", "D1-asymmetric-frontiers")
ARMS = ("bare-ordered", "old-static-fifo", "new-static-fifo", "new-dynamic-fifo")
ORDER_VARIANTS = ("job-0-first", "job-1-first")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha(payload)


def _prepare_inputs(suite_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    suite_root = suite_manifest_path.resolve(strict=True).parent
    suite = json.loads(suite_manifest_path.read_text())
    if suite.get("scope") != "pilot" or not suite.get("profiles"):
        raise ValueError("layered FIFO pilot requires a frozen, profiled pilot suite")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite layered FIFO pilot directory {output_dir}")
    output_dir.mkdir(parents=True)
    input_root = output_dir / "inputs"
    profile_root = input_root / "profiles"
    profile_root.mkdir(parents=True)
    profile_paths = {}
    for name, filename in (("communication", "nccl-communication-profile.json"),
                           ("compute", "gpu-compute-profile.json")):
        source = suite_root / suite["profiles"][name]["path"]
        target = profile_root / filename
        target.write_bytes(source.read_bytes())
        profile_paths[name] = target.relative_to(output_dir).as_posix()

    sample_map = {row["scenario"]: row for row in suite.get("samples", [])
                  if row.get("workload_seed") == 9101}
    prepared = []
    for scenario in PILOT_SCENARIOS:
        sample = sample_map.get(scenario)
        if sample is None:
            raise ValueError(f"pilot suite is missing {scenario} seed 9101")
        source = suite_root / sample["path"]
        original = json.loads(source.read_text())
        original_input_sha = _sha(source.read_bytes())
        for variant in ORDER_VARIANTS:
            document = json.loads(json.dumps(original))
            if variant == "job-1-first":
                document["jobs"] = list(reversed(document["jobs"]))
            dag = parse_dag(document, world_size=2)
            job_order = tuple(job.job_id for job in dag.graph.jobs)
            original_job_order = tuple(job["job_id"] for job in original["jobs"])
            expected_job_order = original_job_order if variant == "job-0-first" else tuple(reversed(original_job_order))
            if job_order != expected_job_order:
                raise ValueError(f"unexpected job tie order for {scenario}/{variant}: {job_order}")
            sequence = build_layered_fifo_order(
                dag.graph, extra_predecessors=dag.execution.submit_after,
                job_order=job_order,
            )
            validate_static_order(sequence, dag.graph)
            destination = input_root / scenario / variant / "workload-9101.json"
            input_sha = _write_json(destination, document)
            relative = destination.relative_to(output_dir).as_posix()
            order_path = destination.with_name("fifo-order.json")
            order_sha = _write_json(order_path, list(sequence))
            prepared.append({
                "scenario": scenario, "variant": variant, "job_order": list(job_order),
                "source_input_path": sample["path"],
                "source_input_sha256": original_input_sha,
                "input_path": relative, "input_sha256": input_sha,
                "input_hash": dag.input_hash,
                "fifo_order_path": order_path.relative_to(output_dir).as_posix(),
                "fifo_order_sha256": order_sha, "fifo_sequence": list(sequence),
                "profiled_execution_sample_hash": sample.get("execution_sample_hash"),
            })
    return {
        "schema": "jobpacer-layered-fifo-pilot", "schema_version": 1,
        "status": "prepared", "source_suite_manifest": str(suite_manifest_path.resolve()),
        "source_suite_manifest_sha256": _sha(suite_manifest_path.read_bytes()),
        "source_profile_sha256": {
            name: suite["profiles"][name]["sha256"] for name in ("communication", "compute")
        },
        "profile_paths": profile_paths,
        "rule": "topological-layer-job-round-robin-v1",
        "bare_contract": "bare-ordered-v2-layered-round-robin",
        "arms": list(ARMS), "scenarios": list(PILOT_SCENARIOS),
        "seed": 9101, "order_variants": list(ORDER_VARIANTS),
        "samples": prepared,
    }


def _head_wait(raw: dict[str, Any], arm: str) -> dict[str, Any]:
    ranks = raw.get("ranks", [])
    if arm == "bare-ordered":
        rows = []
        for rank in ranks:
            releases = [event for event in rank.get("runtime_events", [])
                        if event.get("kind") == "bare_order_head_released"]
            rows.append({"rank": rank.get("rank"),
                         "seconds": sum(float(event.get("order_wait_s", 0.0))
                                        for event in releases),
                         "released_heads": len(releases), "measured": True})
        return {"source": "bare_order_head_released.order_wait_s", "by_rank": rows,
                "measured": True}
    if arm == "old-static-fifo":
        return {"source": "old plan adapter does not expose static-head wait", "seconds": None,
                "measured": False}
    if arm == "new-static-fifo":
        rank0 = next((rank for rank in ranks if rank.get("rank") == 0), {})
        decisions = rank0.get("decision_records", [])
        if not decisions:
            return {"source": "rank0 decision_records missing", "seconds": None,
                    "measured": False}
        intervals = [row for row in decisions
                     if row.get("kind") == "idle_interval"
                     and row.get("reason") == "STATIC_HEAD_BLOCKED"]
        return {"source": "rank0 decision_records STATIC_HEAD_BLOCKED intervals",
                "seconds": sum(float(row.get("duration", 0.0)) for row in intervals),
                "interval_count": len(intervals), "measured": True}
    return {"source": "dynamic FIFO has no fixed head", "seconds": None}


def _command(arm: str, sample: dict[str, Any], paths: dict[str, Any], output: Path,
             *, timeout_s: float, setup_timeout_s: float, warmup: int) -> list[str]:
    if arm == "bare-ordered":
        policy, engine = "bare", "bare"
    elif arm == "old-static-fifo":
        policy, engine = "static_fifo", "old"
    elif arm == "new-static-fifo":
        policy, engine = "static_fifo", "new"
    else:
        policy, engine = "fifo", "new"
    command = [
        sys.executable, "-m", "examples.jobpacer.runtime.replay_launcher",
        "--policy", policy, "--comm-engine", engine,
        "--dag", str(paths["root"] / sample["input_path"]),
        "--backend", "nccl", "--world-size", "2",
        "--timeout", str(timeout_s), "--setup-timeout", str(setup_timeout_s),
        "--warmup-iterations", str(warmup), "--matmul-precision", "highest",
        "--observation-mode", "full",
        "--comm-profile", str(paths["root"] / paths["profiles"]["communication"]),
        "--compute-profile", str(paths["root"] / paths["profiles"]["compute"]),
        "--output", str(output),
    ]
    if arm in {"old-static-fifo", "new-static-fifo"}:
        command.extend(("--static-order", str(paths["root"] / sample["fifo_order_path"])))
    return command


def run_pilot(suite_manifest_path: Path, output_dir: Path, *, timeout_s: float = 90.0,
              setup_timeout_s: float = 30.0, warmup: int = 5) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest = _prepare_inputs(suite_manifest_path, output_dir)
    manifest["environment"] = _cuda_environment()
    manifest["source_snapshot"] = source_snapshot(output_dir / "source" / "repository")
    _write_json(output_dir / "pilot-manifest.json", manifest)
    paths = {"root": output_dir, "profiles": manifest["profile_paths"]}
    run_records = []
    result_rows = []
    for sample in manifest["samples"]:
        for arm in ARMS:
            run_id = f"{sample['scenario']}-{sample['variant']}-{arm}"
            raw_path = output_dir / "raw" / f"{run_id}.json"
            stdout_path = output_dir / "logs" / f"{run_id}.stdout.log"
            stderr_path = output_dir / "logs" / f"{run_id}.stderr.log"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            command = _command(arm, sample, paths, raw_path, timeout_s=timeout_s,
                               setup_timeout_s=setup_timeout_s, warmup=warmup)
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                item for item in ("src", ".", environment.get("PYTHONPATH", "")) if item
            )
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command, cwd=ROOT, env=environment, text=True, capture_output=True,
                    timeout=setup_timeout_s + timeout_s + 10.0, check=False,
                )
                stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                returncode = None
            duration_s = time.monotonic() - started
            stdout_path.write_text(stdout)
            stderr_path.write_text(stderr)
            record = {
                "run_id": run_id, "scenario": sample["scenario"], "variant": sample["variant"],
                "arm": arm, "command": command, "returncode": returncode,
                "duration_s": duration_s,
                "raw_path": raw_path.relative_to(output_dir).as_posix(),
                "stdout_path": stdout_path.relative_to(output_dir).as_posix(),
                "stderr_path": stderr_path.relative_to(output_dir).as_posix(),
                "raw_sha256": _sha(raw_path.read_bytes()) if raw_path.is_file() else None,
            }
            run_records.append(record)
            if returncode != 0 or not raw_path.is_file():
                manifest["status"] = "failed"
                manifest["runs"] = run_records
                _write_json(output_dir / "pilot-manifest.json", manifest)
                raise RuntimeError(f"pilot run failed ({run_id}); see {stderr_path}")
            raw = json.loads(raw_path.read_text())
            ranks = raw.get("ranks", [])
            rank_sequences = [
                tuple(row.get("bare_launch_sequence", row.get("launch_sequence",
                                                               row.get("shared_plan_sequence", ()))))
                for row in ranks
            ]
            checks = {
                "validation_ok": raw.get("validation", {}).get("status") == "ok",
                "two_rank_results": len(ranks) == 2 and {row.get("rank") for row in ranks} == {0, 1},
                "common_rank_launch_sequence": len(rank_sequences) == 2
                and rank_sequences[0] == rank_sequences[1],
            }
            if arm in {"bare-ordered", "old-static-fifo", "new-static-fifo"}:
                checks["frozen_fifo_sequence"] = all(
                    sequence == tuple(sample["fifo_sequence"]) for sequence in rank_sequences
                )
            if arm == "bare-ordered":
                checks["bare_order_contract"] = all(
                    row.get("bare_contract") == "bare-ordered-v2-layered-round-robin"
                    and tuple(row.get("bare_default_order", ())) == tuple(sample["fifo_sequence"])
                    and tuple(row.get("bare_job_tie_order", ())) == tuple(sample["job_order"])
                    for row in ranks
                )
            result_rows.append({
                "run_id": run_id, "scenario": sample["scenario"], "variant": sample["variant"],
                "job_order": sample["job_order"], "arm": arm,
                "checks": checks,
                "validation": raw.get("validation"),
                "head_wait": _head_wait(raw, arm),
                "performance": raw.get("performance"),
                "rank_sequences": [
                    {"rank": row.get("rank"),
                     "sequence": row.get("bare_launch_sequence", row.get("launch_sequence",
                                                                          row.get("shared_plan_sequence"))),
                     "bare_default_order": row.get("bare_default_order"),
                     "peak_inflight": row.get("bare_peak_inflight", row.get("peak_inflight"))}
                    for row in raw.get("ranks", [])
                ],
                "duration_s": duration_s,
            })
            manifest["runs"] = list(run_records)
            _write_json(output_dir / "pilot-manifest.json", manifest)
    manifest["status"] = "complete"
    manifest["runs"] = run_records
    manifest["summary"] = result_rows
    manifest["checks_passed"] = all(
        all(row["checks"].values()) for row in result_rows
    )
    if not manifest["checks_passed"]:
        manifest["status"] = "validation-failed"
    _write_json(output_dir / "pilot-manifest.json", manifest)
    analyze_pilot(output_dir)
    return json.loads((output_dir / "pilot-manifest.json").read_text())


def analyze_pilot(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve(strict=True)
    manifest_path = output_dir / "pilot-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    sample_map = {(row["scenario"], row["variant"]): row for row in manifest["samples"]}
    summary = []
    for attempt in manifest.get("runs", []):
        raw_path = output_dir / attempt["raw_path"]
        raw = json.loads(raw_path.read_text()) if raw_path.is_file() else {}
        sample = sample_map[(attempt["scenario"], attempt["variant"])]
        ranks = raw.get("ranks", [])
        rank_sequences = [
            tuple(row.get("bare_launch_sequence", row.get("launch_sequence",
                                                           row.get("shared_plan_sequence", ()))))
            for row in ranks
        ]
        checks = {
            "raw_present": raw_path.is_file(),
            "returncode_zero": attempt.get("returncode") == 0,
            "validation_ok": raw.get("validation", {}).get("status") == "ok",
            "all_collectives_correct": raw.get("validation", {}).get("all_collectives_correct") is True,
            "two_rank_results": len(ranks) == 2 and {row.get("rank") for row in ranks} == {0, 1},
            "common_rank_launch_sequence": len(rank_sequences) == 2
            and rank_sequences[0] == rank_sequences[1],
        }
        if attempt["arm"] in {"bare-ordered", "old-static-fifo", "new-static-fifo"}:
            checks["frozen_fifo_sequence"] = all(
                sequence == tuple(sample["fifo_sequence"]) for sequence in rank_sequences
            )
        if attempt["arm"] == "bare-ordered":
            checks["bare_order_contract"] = all(
                row.get("bare_contract") == manifest["bare_contract"]
                and tuple(row.get("bare_default_order", ())) == tuple(sample["fifo_sequence"])
                and tuple(row.get("bare_job_tie_order", ())) == tuple(sample["job_order"])
                for row in ranks
            )
        summary.append({
            "run_id": attempt["run_id"], "scenario": attempt["scenario"],
            "variant": attempt["variant"], "job_order": sample["job_order"],
            "arm": attempt["arm"], "checks": checks,
            "validation": raw.get("validation"),
            "head_wait": _head_wait(raw, attempt["arm"]),
            "performance": raw.get("performance"),
            "rank_sequences": [
                {"rank": row.get("rank"), "sequence": list(rank_sequences[index]),
                 "bare_default_order": row.get("bare_default_order"),
                 "peak_inflight": row.get("bare_peak_inflight", row.get("peak_inflight"))}
                for index, row in enumerate(ranks)
            ],
            "duration_s": attempt.get("duration_s"),
            "raw_sha256": _sha(raw_path.read_bytes()) if raw_path.is_file() else None,
        })
    passed = len(summary) == len(manifest.get("runs", [])) and all(
        all(row["checks"].values()) for row in summary
    )
    analysis = {
        "schema": "jobpacer-layered-fifo-pilot-analysis", "schema_version": 2,
        "source_manifest_sha256": _sha(manifest_bytes),
        "analysis_source_sha256": _sha(Path(__file__).read_bytes()),
        "status": "ok" if passed else "validation-failed",
        "checks_passed": passed, "run_count": len(summary), "summary": summary,
        "head_wait_correction": (
            "old static FIFO wait is unmeasured; its adapter emits no static-head timer, "
            "so missing events are represented as null rather than zero"
        ),
    }
    analysis_path = output_dir / "analysis.json"
    _write_json(analysis_path, analysis)
    if manifest.get("summary") and not manifest.get("analysis_revision"):
        _write_json(output_dir / "analysis-prior-summary.json", manifest["summary"])
    manifest["analysis_revision"] = "head-wait-null-for-uninstrumented-old-adapter-v2"
    manifest["summary"] = summary
    manifest["checks_passed"] = passed
    manifest["status"] = "complete" if passed else "validation-failed"
    manifest["analysis_sha256"] = _sha(analysis_path.read_bytes())
    _write_json(manifest_path, manifest)
    return analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true",
                        help="rebuild the analysis from an existing pilot's immutable raw outputs")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--setup-timeout", type=float, default=30.0)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    args = parser.parse_args(argv)
    if args.analyze_only:
        result = analyze_pilot(args.output_dir)
    else:
        if args.suite_manifest is None:
            parser.error("--suite-manifest is required unless --analyze-only is used")
        result = run_pilot(args.suite_manifest, args.output_dir, timeout_s=args.timeout,
                           setup_timeout_s=args.setup_timeout, warmup=args.warmup_iterations)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "status": result["status"],
                      "runs": result.get("run_count", len(result.get("runs", [])))}, sort_keys=True))
    return 0 if result.get("checks_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
