"""Validation and statistical analysis of frozen Phase 3 GPU batches."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from examples.jobpacer.experiments.phase3.batch import (
    ARM_CONFIG, _accepted_blocks, _canonical, _read_ledger, _sha,
    _verify_frozen_inputs, _write_json,
)
from examples.jobpacer.experiments.phase3.suite import (
    ARMS, BARE_ARM, CONTRACT_VERSION, FORMAL_WORKLOAD_SEEDS, SCENARIOS,
)


def _bootstrap_median(values: list[float], *, samples: int, seed: int) -> tuple[float, float] | None:
    if not values or not samples:
        return None
    import random
    rng = random.Random(seed)
    boot = [statistics.median(values[rng.randrange(len(values))] for _ in values)
            for _ in range(samples)]
    boot.sort()
    return boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]


def _rank0_decisions(result: dict[str, Any]) -> list[dict[str, Any]]:
    rank = next((item for item in result.get("ranks", []) if item.get("rank") == 0), {})
    return rank.get("decision_records", [])


def _static_head_hol(order: list[str], decisions: list[dict[str, Any]]) -> bool:
    snapshots = [row for row in decisions if row.get("kind") == "policy_snapshot"]
    dispatches = sorted((row for row in decisions
                         if row.get("kind") == "decision" and row.get("decision") == "dispatch"),
                        key=lambda row: row.get("now", 0.0))
    for idle in decisions:
        if idle.get("kind") != "idle_interval" or idle.get("reason") != "STATIC_HEAD_BLOCKED":
            continue
        if float(idle.get("duration", 0.0)) <= 0:
            continue
        prior = [row for row in dispatches if row.get("now", 0.0) < idle.get("start", 0.0)]
        cursor = len(prior)
        if cursor >= len(order):
            continue
        head = order[cursor]
        for snapshot in snapshots:
            now = snapshot.get("now", -1.0)
            if idle.get("start", 0.0) <= now <= idle.get("end", 0.0):
                eligible = {item.get("task_id") for item in snapshot.get("eligible", [])}
                if eligible and head not in eligible:
                    return True
    return False


def _dynamic_fifo_bypass(order: list[str], decisions: list[dict[str, Any]]) -> bool:
    dispatched: set[str] = set()
    for row in sorted((item for item in decisions
                       if item.get("kind") == "decision" and item.get("decision") == "dispatch"),
                      key=lambda item: item.get("now", 0.0)):
        head = next((task_id for task_id in order if task_id not in dispatched), None)
        eligible = set(row.get("eligible", []))
        selected = row.get("task_id")
        if head is not None and head not in eligible and selected in eligible and selected != head:
            return True
        if isinstance(selected, str):
            dispatched.add(selected)
    return False


def _ltf_frontier_choice(decisions: list[dict[str, Any]]) -> tuple[bool, bool]:
    snapshots = [row for row in decisions if row.get("kind") == "policy_snapshot"]
    saw_distinct = False
    selected_highest = False
    for decision in decisions:
        if decision.get("kind") != "decision" or decision.get("decision") != "dispatch":
            continue
        eligible = set(decision.get("eligible", []))
        if len(eligible) < 2:
            continue
        now = decision.get("now", 0.0)
        matching = [row for row in snapshots
                    if row.get("now", float("inf")) <= now
                    and {item.get("task_id") for item in row.get("eligible", [])} == eligible]
        if not matching:
            continue
        snapshot = max(matching, key=lambda row: row.get("now", 0.0))
        scores = {item["task_id"]: float(item["estimated_comm_s"])
                  + float(item["remaining_tail_s"])
                  for item in snapshot.get("eligible", [])}
        if len(set(scores.values())) < 2:
            continue
        saw_distinct = True
        highest = max(scores.values())
        if scores.get(decision.get("task_id")) == highest:
            selected_highest = True
    return saw_distinct, selected_highest


def _formal_matrix_checks(manifest: dict[str, Any], suite: dict[str, Any],
                          order: dict[str, Any], *, accepted_blocks: int,
                          accepted_runs: int, validation_errors: list[str]) -> dict[str, bool]:
    expected_blocks = {
        (scenario, seed, repeat)
        for scenario in SCENARIOS for seed in FORMAL_WORKLOAD_SEEDS for repeat in range(5)
    }
    actual_blocks = {
        (row.get("scenario"), row.get("workload_seed"), row.get("repeat"))
        for row in order.get("blocks", [])
    }
    expected_samples = {(scenario, seed) for scenario in SCENARIOS for seed in FORMAL_WORKLOAD_SEEDS}
    actual_samples = {
        (row.get("scenario"), row.get("workload_seed")) for row in suite.get("samples", [])
    }
    arms = list(order.get("arms", []))
    qualification = suite.get("bare_qualification", {})
    evidence = qualification.get("evidence") or {}
    qualified_bare = (qualification.get("arm") == BARE_ARM
                      and qualification.get("status") == "qualified"
                      and evidence.get("contract_version") == "bare-ordered-v2-layered-round-robin")
    exact_original_arms = arms == list(ARMS)
    order_matches_manifest = arms == manifest.get("arms")
    return {
        "formal_stage": manifest.get("stage") == "formal",
        "formal_scope": manifest.get("scope") == "formal" and suite.get("scope") == "formal",
        "six_scenarios": set(row[0] for row in actual_blocks) == set(SCENARIOS),
        "five_formal_seeds": set(row[1] for row in actual_blocks) == set(FORMAL_WORKLOAD_SEEDS),
        "five_repeats": manifest.get("repeats_per_sample") == 5,
        "exact_150_block_grid": len(order.get("blocks", [])) == 150 and actual_blocks == expected_blocks,
        "exact_1050_planned_runs": (len(order.get("arms", [])) == 7
                                     and order.get("run_count") == 1050
                                     and manifest.get("planned_blocks") == 150
                                     and manifest.get("planned_runs") == 1050),
        "exact_30_formal_samples": len(suite.get("samples", [])) == 30 and actual_samples == expected_samples,
        "bare_qualified": qualified_bare,
        "contract_version": suite.get("contract_version") == CONTRACT_VERSION,
        "exact_frozen_arm_ids": order_matches_manifest and exact_original_arms,
        "all_150_blocks_independently_accepted": accepted_blocks == 150,
        "all_1050_runs_independently_accepted": accepted_runs == 1050,
        "all_independent_validation_passed": not validation_errors,
    }


def analyze_batch(batch_dir: Path, *, bootstrap_samples: int = 2000, analysis_seed: int = 20260928) -> dict[str, Any]:
    batch_dir = batch_dir.resolve(strict=True)
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    order = json.loads((batch_dir / "order.json").read_text())
    ledger = _read_ledger(batch_dir / "runs.jsonl")
    validation_errors = []
    suite_for_check = {}
    bad_blocks: set[str] = set()
    try:
        if _sha((batch_dir / "order.json").read_bytes()) != manifest["order_sha256"]:
            raise ValueError("order table hash differs from batch manifest")
        suite_for_check = json.loads((batch_dir / "inputs/suite-manifest.json").read_text())
        if _sha((batch_dir / "inputs/suite-manifest.json").read_bytes()) != manifest["suite_manifest_sha256"]:
            raise ValueError("suite manifest hash differs from batch manifest")
        _verify_frozen_inputs(batch_dir / "inputs", suite_for_check)
        archived_root = batch_dir / "source/repository"
        for source in manifest["source_snapshot"]["files"]:
            path = archived_root / source["path"]
            if not path.is_file() or _sha(path.read_bytes()) != source["sha256"]:
                raise ValueError(f"archived source bytes changed: {path}")
        recorded_reservations = {row.get("reservation_path") for row in ledger
                                 if row.get("reservation_path")}
        orphaned = sorted(path.relative_to(batch_dir).as_posix()
                          for path in (batch_dir / "logs" / "reservations").glob("*.json")
                          if path.relative_to(batch_dir).as_posix() not in recorded_reservations)
        if orphaned:
            raise ValueError("attempt reservations lack completion ledger records: " + ", ".join(orphaned))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        validation_errors.append(f"frozen batch integrity failure: {exc}")
        bad_blocks.update(block["block_id"] for block in order["blocks"])
    planned = {run["run_id"]: run for block in order["blocks"] for run in block["arms"]}
    expected_contract_hash = _sha(_canonical({
        "execution_contract": manifest["execution_contract"],
        "replay_settings": manifest["replay_settings"],
    }))
    for record in ledger:
        run = planned.get(record.get("run_id"))
        if run is None:
            validation_errors.append(f"ledger has an unplanned run: {record.get('run_id')}")
            continue
        record_errors = []
        artifacts = [("raw_path", "raw_sha256"), ("stdout_path", "stdout_sha256"),
                     ("stderr_path", "stderr_sha256")]
        if record.get("reservation_path"):
            artifacts.append(("reservation_path", "reservation_sha256"))
        for path_key, hash_key in artifacts:
            artifact = batch_dir / record[path_key]
            expected_hash = record.get(hash_key)
            if expected_hash is not None and (not artifact.is_file() or _sha(artifact.read_bytes()) != expected_hash):
                record_errors.append(f"attempt artifact hash mismatch: {artifact}")
        if record.get("contract_sha256") != expected_contract_hash:
            record_errors.append(f"contract hash mismatch: {record.get('run_id')}")
        for field in ("input_sha256", "execution_sample_hash", "estimate_view_hash",
                      "compute_profile_sha256", "comm_profile_sha256"):
            if record.get(field) != run.get(field):
                record_errors.append(f"attempt {field} mismatch: {record.get('run_id')}")
        if record.get("source_snapshot_sha256") != manifest["source_snapshot"]["digest"]:
            record_errors.append(f"attempt source hash mismatch: {record.get('run_id')}")
        validation_errors.extend(record_errors)
        if record_errors:
            bad_blocks.add(record["block_id"])
    by_block_attempt: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in ledger:
        by_block_attempt.setdefault((row["block_id"], int(row["block_attempt"])), []).append(row)
    accepted: dict[str, tuple[int, dict[str, Any]]] = {}
    for block in order["blocks"]:
        if block["block_id"] in bad_blocks:
            continue
        options = []
        for attempt in sorted({key[1] for key in by_block_attempt if key[0] == block["block_id"]}):
            rows = by_block_attempt[(block["block_id"], attempt)]
            if (len(rows) == len(order["arms"]) and {row["arm"] for row in rows} == set(order["arms"])
                    and all(row.get("validation_status") == "ok"
                            and row.get("failure_class") == "none"
                            and row.get("returncode") == 0 for row in rows)):
                options.append((attempt, {row["arm"]: row for row in rows}))
        if len(options) > 1:
            validation_errors.append(f"multiple complete attempts for {block['block_id']}")
        if options:
            accepted[block["block_id"]] = options[-1]
    rows_by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in order["arms"]}
    point_rows = []
    job_rows = []
    pilot_evidence: dict[str, dict[str, Any]] = {}
    mechanism_rows = []
    dynamic_diagnostics = []
    accepted_attempts = []
    for block in order["blocks"]:
        selected = accepted.get(block["block_id"])
        if not selected:
            continue
        block_error_start = len(validation_errors)
        mechanism = {"scenario": block["scenario"], "seed": block["workload_seed"],
                     "repeat": block["repeat"], "competition": False, "hol": False, "bypass": False}
        block_attempt, arms = selected
        accepted_attempts.append({"block_id": block["block_id"], "block_attempt": block_attempt})
        if any(row.get("input_sha256") != block.get("input_sha256")
               or row.get("execution_sample_hash") != block.get("execution_sample_hash")
               or row.get("estimate_view_hash") != block.get("estimate_view_hash")
               or row.get("compute_profile_sha256") != block.get("compute_profile_sha256")
               or row.get("comm_profile_sha256") != block.get("comm_profile_sha256")
               for row in arms.values()):
            validation_errors.append(f"paired input/profile hashes differ in {block['block_id']}")
        for arm, row in arms.items():
            path = batch_dir / row["raw_path"]
            result = json.loads(path.read_text())
            if row.get("raw_sha256") != _sha(path.read_bytes()):
                validation_errors.append(f"raw hash mismatch: {path}")
                bad_blocks.add(block["block_id"])
            decisions = _rank0_decisions(result)
            if arm in {"new-dynamic-fifo", "new-dynamic-ltf"}:
                from examples.jobpacer.experiments.phase3.gates import decision_evidence
                diagnostic = decision_evidence(decisions, "fifo" if arm.endswith("fifo") else "ltf")
                rank0 = next((rank for rank in result.get("ranks", []) if rank.get("rank") == 0), {})
                dynamic_diagnostics.append({"block_id": block["block_id"], "arm": arm,
                                            "scenario": block["scenario"], **diagnostic,
                                            "minimal_counts": rank0.get("mechanism_counts")})
                if manifest["replay_settings"].get("observation_mode") == "full":
                    validation_errors.extend(diagnostic["errors"])
                if arm.endswith("ltf"):
                    mechanism["competition"] = diagnostic["competition_observed"]
            evidence = pilot_evidence.setdefault(block["scenario"], {
                "runs": 0, "validation_ok_runs": 0,
                "static_fifo_hol": 0, "dynamic_fifo_bypass": 0,
                "ltf_distinct_frontier": 0, "ltf_selected_highest": 0,
            })
            evidence["runs"] += 1
            if result.get("validation", {}).get("status") == "ok":
                evidence["validation_ok_runs"] += 1
            if arm == "new-static-fifo" and block.get("fifo_sequence"):
                mechanism["hol"] = _static_head_hol(block["fifo_sequence"], decisions)
                evidence["static_fifo_hol"] += int(mechanism["hol"])
            if arm == "new-dynamic-fifo" and block.get("fifo_sequence"):
                mechanism["bypass"] = _dynamic_fifo_bypass(block["fifo_sequence"], decisions)
                evidence["dynamic_fifo_bypass"] += int(mechanism["bypass"])
            if arm == "new-dynamic-ltf":
                distinct, highest = _ltf_frontier_choice(decisions)
                evidence["ltf_distinct_frontier"] += int(distinct)
                evidence["ltf_selected_highest"] += int(distinct and highest)
            try:
                from examples.jobpacer.analysis.runtime_results import expected_dag_results, validate_results
                from examples.jobpacer.runtime.runtime_adapter import load_dag
                dag_path = batch_dir / "inputs" / block["sample_path"]
                dag = load_dag(dag_path, world_size=2)
                expected_data = expected_dag_results(dag.graph, 2)
                replay_check = validate_results(
                    result.get("ranks", []), 2,
                    expected=expected_data["expected"],
                    expected_nodes=expected_data["expected_nodes"],
                    digests={dag.manifest_digest},
                    expected_config={"comm_engine": ARM_CONFIG[arm][0], "policy": ARM_CONFIG[arm][1],
                                     "max_inflight": None if arm == BARE_ARM else 1,
                                     "observation_mode": manifest["replay_settings"].get("observation_mode", "full"),
                                     "estimate_view_hash": block["estimate_view_hash"], "input_hash": dag.input_hash},
                )
                if replay_check.get("status") != "ok":
                    validation_errors.append(
                        f"launch/task projection failed for {block['block_id']}/{arm}: "
                        + "; ".join(replay_check.get("errors", [])))
                expected_static = block.get("fifo_sequence") if arm.endswith("static-fifo") or arm == BARE_ARM else (
                    block.get("ltf_sequence") if arm.endswith("static-ltf") else None)
                if expected_static is not None:
                    for rank_result in result.get("ranks", []):
                        if rank_result.get("launch_sequence") != expected_static:
                            validation_errors.append(
                                f"static sequence mismatch for {block['block_id']}/{arm}/"
                                f"rank-{rank_result.get('rank')}")
                if result.get("validation", {}).get("status") != "ok":
                    validation_errors.append(f"replay validation not ok: {block['block_id']}/{arm}")
                rank_uuids = [item.get("device_uuid") for item in sorted(
                    result.get("ranks", []), key=lambda value: value.get("rank", -1))]
                expected_uuids = manifest.get("environment_at_plan", {}).get("device_uuids", [])
                if expected_uuids and rank_uuids != expected_uuids:
                    validation_errors.append(f"rank-to-GPU UUID mapping mismatch: {block['block_id']}/{arm}")
                gpu_checks = [job.get("gpu_buffer_validation", {})
                              for rank_result in result.get("ranks", [])
                              for job in rank_result.get("jobs", [])]
                if any(not checks or not all(checks.values()) for checks in gpu_checks):
                    validation_errors.append(f"GPU buffer validation failed: {block['block_id']}/{arm}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                validation_errors.append(f"could not independently validate {block['block_id']}/{arm}: {exc}")
            performance = result.get("performance", {})
            makespan = performance.get("workload_makespan_us")
            jcts = [item.get("makespan_us") for item in performance.get("job_makespans", [])]
            mean_jct = statistics.mean(jcts) if jcts else None
            measurement = {"block_id": block["block_id"], "scenario": block["scenario"],
                           "workload_seed": block["workload_seed"], "repeat": block["repeat"],
                           "arm": arm, "block_attempt": block_attempt,
                           "makespan_s": makespan / 1_000_000 if makespan is not None else None,
                           "mean_jct_s": mean_jct / 1_000_000 if mean_jct is not None else None,
                           "job_jct_s": {item["job_id"]: item["makespan_us"] / 1_000_000
                                          for item in performance.get("job_makespans", [])}}
            rows_by_arm.setdefault(arm, []).append(measurement)
            point_rows.append(measurement)
            job_rows.extend({
                "block_id": block["block_id"], "scenario": block["scenario"],
                "workload_seed": block["workload_seed"], "repeat": block["repeat"],
                "arm": arm, "job_id": job_id, "jct_s": jct_s,
            } for job_id, jct_s in measurement["job_jct_s"].items())
        if len(validation_errors) == block_error_start:
            mechanism["hol_and_bypass"] = mechanism["hol"] and mechanism["bypass"]
            mechanism_rows.append(mechanism)
        if len(validation_errors) > block_error_start:
            accepted.pop(block["block_id"], None)
            accepted_attempts = [item for item in accepted_attempts
                                 if item["block_id"] != block["block_id"]]
            point_rows = [item for item in point_rows if item["block_id"] != block["block_id"]]
            job_rows = [item for item in job_rows if item["block_id"] != block["block_id"]]
            for arm in rows_by_arm:
                rows_by_arm[arm] = [item for item in rows_by_arm[arm]
                                    if item["block_id"] != block["block_id"]]
    primary = (("new-static-fifo", "new-dynamic-fifo"), ("new-static-ltf", "new-dynamic-ltf"))
    paired: dict[str, Any] = {}
    secondary = (("old-static-fifo", "new-dynamic-fifo"), ("old-static-ltf", "new-dynamic-ltf"),
                 ("new-dynamic-fifo", "new-dynamic-ltf")) + tuple((BARE_ARM, arm) for arm in ARMS if arm != BARE_ARM)
    for baseline, candidate in primary + secondary:
        if baseline not in order["arms"] or candidate not in order["arms"]:
            continue
        candidate_points = {row["block_id"]: row for row in rows_by_arm[candidate]}
        baseline_points = {row["block_id"]: row for row in rows_by_arm[baseline]}
        comparison: dict[str, Any] = {"baseline": baseline, "candidate": candidate,
                                     "role": "primary" if (baseline, candidate) in primary else "secondary"}
        for metric in ("makespan_s", "mean_jct_s"):
            by_seed: dict[tuple[str, int], list[tuple[float, float]]] = {}
            repeat_points = []
            for block_id in sorted(set(candidate_points) & set(baseline_points)):
                base = baseline_points[block_id]
                cand = candidate_points[block_id]
                if base[metric] is None or cand[metric] is None or cand[metric] == 0:
                    continue
                delta = cand[metric] - base[metric]
                ratio = base[metric] / cand[metric]
                key = (base["scenario"], int(base["workload_seed"]))
                by_seed.setdefault(key, []).append((delta, ratio))
                repeat_points.append({"block_id": block_id, "delta": delta, "ratio": ratio})
            seed_rows = [{"scenario": scenario, "workload_seed": workload_seed,
                          "median_delta": statistics.median(item[0] for item in values),
                          "median_ratio": statistics.median(item[1] for item in values),
                          "paired_repeats": len(values)}
                         for (scenario, workload_seed), values in sorted(by_seed.items())]
            scenarios = sorted({row["scenario"] for row in seed_rows})
            scenario_deltas = {
                scenario: [row["median_delta"] for row in seed_rows if row["scenario"] == scenario]
                for scenario in scenarios}
            comparison[metric] = {
                "complete_repeat_pairs": len(repeat_points),
                "seed_block_medians": seed_rows,
                "scenario_median_delta": {scenario: statistics.median(values)
                                           for scenario, values in scenario_deltas.items()},
                "scenario_bootstrap_95pct_delta": {
                    scenario: _bootstrap_median(values, samples=bootstrap_samples,
                                                seed=analysis_seed + index)
                    for index, (scenario, values) in enumerate(scenario_deltas.items())},
                "paired_repeat_points": repeat_points,
            }
        paired[f"{baseline}__to__{candidate}"] = comparison
    failure_counts = {}
    for row in ledger:
        current = failure_counts.setdefault(row["arm"], {"attempts": 0, "failures": 0, "classes": {}})
        current["attempts"] += 1
        if row.get("failure_class") != "none":
            current["failures"] += 1
            kind = row.get("failure_class", "unknown")
            current["classes"][kind] = current["classes"].get(kind, 0) + 1
    required_runs = len(order["blocks"]) * len(order["arms"])
    pilot_gates = {}
    for scenario in ("L0-balanced", "L1-skew-tail", "D0-fork-join",
                     "D1-asymmetric-frontiers", "D2-cross-job-skew", "D3-order-and-sinks"):
        evidence = pilot_evidence.get(scenario, {"runs": 0, "validation_ok_runs": 0})
        if scenario == "L0-balanced":
            passed = evidence.get("runs", 0) > 0 and evidence.get("validation_ok_runs", 0) == evidence.get("runs")
            criterion = "all arms validate; no performance direction required"
        elif scenario in {"L1-skew-tail", "D2-cross-job-skew"}:
            passed = (evidence.get("static_fifo_hol", 0) > 0
                      and evidence.get("dynamic_fifo_bypass", 0) > 0)
            criterion = "static FIFO head HOL with another eligible task; dynamic FIFO dispatches a non-head candidate"
        elif scenario == "D1-asymmetric-frontiers":
            passed = (evidence.get("ltf_distinct_frontier", 0) > 0
                      and evidence.get("ltf_selected_highest", 0) > 0)
            criterion = "dynamic LTF observes distinct comm+tail scores among candidates and selects a highest-scoring task"
        else:
            passed = evidence.get("runs", 0) > 0 and evidence.get("validation_ok_runs", 0) == evidence.get("runs")
            criterion = "all arms validate the DAG, GPU buffers, communication projection, and terminal completion"
        pilot_gates[scenario] = {"passed": bool(passed), "criterion": criterion, **evidence}
    from examples.jobpacer.experiments.phase3.gates import mechanism_gates
    pilot_gates.update(mechanism_gates(mechanism_rows))
    # D2 opportunity is descriptive, never a forced benefit/selection gate.
    d2 = pilot_evidence.get("D2-cross-job-skew", {})
    pilot_gates["D2-cross-job-skew"]["passed"] = bool(d2.get("runs")) and d2.get("runs") == d2.get("validation_ok_runs")
    pilot_scope = manifest.get("scope") == "pilot"
    pilot_gates_passed = pilot_scope and all(row["passed"] for row in pilot_gates.values())
    formal_completion_checks = _formal_matrix_checks(
        manifest, json.loads((batch_dir / "inputs/suite-manifest.json").read_text()), order,
        accepted_blocks=len(accepted), accepted_runs=len(point_rows),
        validation_errors=validation_errors,
    )
    decomposition = []
    points = {(row["block_id"], row["arm"]): row for row in point_rows}
    for block in order["blocks"]:
        for policy in ("fifo", "ltf"):
            triple = [points.get((block["block_id"], arm)) for arm in
                      (f"old-static-{policy}", f"new-static-{policy}", f"new-dynamic-{policy}")]
            if not all(triple):
                continue
            old, static, dynamic = triple
            for metric in ("makespan_s", "mean_jct_s"):
                if any(row[metric] is None for row in triple):
                    continue
                decomposition.append({"block_id": block["block_id"], "policy": policy, "metric": metric,
                    "old_minus_dynamic": old[metric] - dynamic[metric],
                    "static_minus_dynamic": static[metric] - dynamic[metric],
                    "static_minus_old": static[metric] - old[metric]})
    readiness_passed = False
    try:
        from examples.jobpacer.experiments.phase3.gates import verify_readiness
        verify_readiness(suite_for_check, manifest["source_snapshot"]["digest"])
        readiness_passed = True
    except (ValueError, OSError, KeyError):
        pass
    arm_seed_summaries = {}
    for arm, points_for_arm in rows_by_arm.items():
        groups = {}
        for point in points_for_arm:
            groups.setdefault((point["scenario"], point["workload_seed"]), []).append(point)
        arm_seed_summaries[arm] = [
            {"scenario": scenario, "workload_seed": seed, "repeats": len(points),
             **{metric: statistics.median(point[metric] for point in points)
                for metric in ("makespan_s", "mean_jct_s") if all(point[metric] is not None for point in points)}}
            for (scenario, seed), points in sorted(groups.items())]
    analysis = {
        "schema": "jobpacer-gpu-seven-arm-analysis", "schema_version": 1,
        "batch_id": manifest["batch_id"], "planned_blocks": len(order["blocks"]),
        "planned_runs": required_runs, "accepted_complete_blocks": len(accepted),
        "accepted_runs": len(point_rows), "missing_blocks": sorted(
            set(block["block_id"] for block in order["blocks"]) - set(accepted)),
        "attempted_runs": len(ledger), "failure_counts_by_arm": failure_counts,
        "accepted_attempts": accepted_attempts, "paired_comparisons": paired,
        "pilot_scope": pilot_scope,
        "dynamic_dispatch_diagnostics": dynamic_diagnostics,
        "paired_cost_identity": decomposition,
        "arm_raw_distributions": rows_by_arm,
        "matrix_complete": len(accepted) == len(order["blocks"]) and len(point_rows) == required_runs,
        "semantic_checks_passed": bool(point_rows) and not validation_errors,
        "mechanism_gates_passed": pilot_gates_passed,
        "measurement_checks_passed": readiness_passed,
        "arm_seed_summaries": arm_seed_summaries,
        "net_performance_result": "see paired comparisons; intervals are exploratory, no multiplicity correction",
        "pilot_mechanism_gates": pilot_gates,
        "pilot_gates_passed": pilot_gates_passed,
        "validation_errors": validation_errors,
        "formal_completion_checks": formal_completion_checks,
        "complete_formal_matrix": all(formal_completion_checks.values()),
        "interpretation": "paired seed-block medians; limited to these frozen workload samples",
    }
    gate_base = {"source_snapshot_sha256": manifest["source_snapshot"]["digest"],
                 "generator_version": suite_for_check.get("generator_version"),
                 "profiles": {key: value["sha256"] for key, value in suite_for_check.get("profiles", {}).items()},
                 "artifacts": [{"path": str((batch_dir / row["raw_path"]).resolve()), "sha256": row["raw_sha256"]}
                               for row in ledger if row.get("raw_sha256")]}
    for gate, checks in (
        ("mechanism", {"seed_thresholds": pilot_gates_passed,
                       "semantics": analysis["semantic_checks_passed"], "matrix": analysis["matrix_complete"]}),
        ("rehearsal", {"seven_arms": set(order["arms"]) == set(ARMS),
                       "all_scenarios": {row["scenario"] for row in point_rows} == set(SCENARIOS),
                       "semantics": analysis["semantic_checks_passed"], "matrix": analysis["matrix_complete"]}),
    ):
        _write_json(batch_dir / f"{gate}-audit.json", {**gate_base, "gate": gate,
                    "checks": checks, "passed": all(checks.values())})
    _write_json(batch_dir / "analysis.json", analysis)
    _write_json(batch_dir / "tables" / "analysis.json", analysis)
    if point_rows:
        fields = ("block_id", "scenario", "workload_seed", "repeat", "arm", "block_attempt",
                  "makespan_s", "mean_jct_s")
        with (batch_dir / "tables" / "accepted-points.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in point_rows:
                writer.writerow({key: row.get(key) for key in fields})
    if job_rows:
        fields = ("block_id", "scenario", "workload_seed", "repeat", "arm", "job_id", "jct_s")
        with (batch_dir / "tables" / "job-jct.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(job_rows)
    checksums = []
    for path in sorted(item for item in batch_dir.rglob("*")
                       if item.is_file() and item.name != "SHA256SUMS.txt"):
        checksums.append(f"{_sha(path.read_bytes())}  {path.relative_to(batch_dir).as_posix()}")
    (batch_dir / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    return analysis
