from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

from examples.jobpacer.runtime.runtime_adapter import load_dag
from examples.jobpacer.runtime.dag_comm_adapters import build_bare_order
from examples.jobpacer.experiments.seven_arm.suite import (
    ARMS,
    CONTRACT_VERSION,
    BARE_ARM,
    FORMAL_WORKLOAD_SEEDS,
    MECHANISM_PILOT_ARMS,
    SCENARIOS,
    _execution_hash,
    expand_sample,
    freeze_suite,
    make_order_table,
    make_template,
    prepare_suite,
    topology_hash,
    PILOT_WORKLOAD_SEEDS,
    verify_compute_profile_inputs,
    verify_order_table,
)
from examples.jobpacer.scripts.run_gpu_compute_profile import collect_calibration_cases
from examples.jobpacer.experiments.seven_arm.batch import (
    _accepted_blocks,
    _classify_failure,
    _dynamic_fifo_bypass,
    _formal_matrix_checks,
    _ltf_frontier_choice,
    _limit_order_table,
    _recover_orphaned_attempts,
    _reserve_run_attempt,
    _start_replay_process,
    _static_head_hol,
    _verify_frozen_inputs,
    _run_command,
    analyze_batch,
    create_batch,
    run_batch,
    source_snapshot,
)
from examples.jobpacer.experiments.seven_arm.prepare import _preview
from runtime_comm_scheduler.dag import (
    CommNode, build_layered_fifo_order, build_static_order, validate_static_order,
)
from runtime_comm_scheduler.dag.model import compute_tails


def test_six_scenario_templates_are_valid_cuda_dags_with_real_workload_structure():
    for scenario in SCENARIOS:
        dag = load_dag_from_document(make_template(scenario))
        assert dag.execution.schema_version == 2
        assert dag.execution.mode == "cuda-program"
        assert len(dag.graph.jobs) >= 2
        assert all(len({node.group_id for node in job.nodes if isinstance(node, CommNode)}) >= 1
                   for job in dag.graph.jobs)
        assert all(node.collective.op == "all_reduce" and node.collective.num_bytes >= 256 * 256 * 4
                   for job in dag.graph.jobs for node in job.nodes if isinstance(node, CommNode))
        assert all(program["op"] in {"fill", "matmul", "sum_join"}
                   for program in dag.execution.compute_programs.values())


def test_layered_fifo_round_robins_jobs_and_accepts_swapped_tie_order(tmp_path):
    suite = prepare_suite(tmp_path, seeds=(9101,))
    for scenario in SCENARIOS:
        sample = next(row for row in suite["scenarios"][scenario]["samples"]
                      if row["workload_seed"] == 9101)
        dag = load_dag(tmp_path / sample["path"], world_size=2)
        bare_order = build_bare_order(
            dag.graph, submit_after=dag.execution.submit_after, input_hash=dag.input_hash,
        )
        assert bare_order.sequence == tuple(sample["fifo_sequence"])
        assert bare_order.job_order == tuple(job.job_id for job in dag.graph.jobs)
        assert build_static_order(
            dag.graph, "static_fifo", extra_predecessors=dag.execution.submit_after,
        ) == bare_order.sequence

    sample = next(row for row in suite["scenarios"]["L0-balanced"]["samples"]
                  if row["workload_seed"] == 9101)
    dag = load_dag(tmp_path / sample["path"], world_size=2)
    default_order = tuple(sample["fifo_sequence"])
    expected = tuple(
        f"job-{job}/comm-s{layer}"
        for layer in range(3) for job in (0, 1)
    )
    assert default_order == expected
    assert json.loads((tmp_path / sample["fifo_order_path"]).read_text()) == list(expected)
    validate_static_order(default_order, dag.graph)

    swapped = build_layered_fifo_order(
        dag.graph, extra_predecessors=dag.execution.submit_after,
        job_order=("job-1", "job-0"),
    )
    assert swapped == tuple(
        f"job-{job}/comm-s{layer}"
        for layer in range(3) for job in (1, 0)
    )
    validate_static_order(swapped, dag.graph)


def load_dag_from_document(document):
    from examples.jobpacer.runtime.runtime_adapter import parse_dag
    return parse_dag(document, world_size=2)


def test_preparation_writes_five_distinct_seeded_samples_per_scenario(tmp_path):
    suite = prepare_suite(tmp_path)
    assert len(suite["scenarios"]) == 6
    for scenario, record in suite["scenarios"].items():
        assert len(record["samples"]) == 5
        sample_hashes = set()
        input_hashes = set()
        execution_signatures = set()
        for item in record["samples"]:
            path = tmp_path / item["path"]
            dag = load_dag(path, world_size=2)
            assert dag.execution.sample_id == item["sample_id"]
            assert dag.input_hash == item["input_hash"]
            assert item["fifo_order_sha256"]
            sample_hashes.add(item["execution_sample_hash"])
            input_hashes.add(item["input_sha256"])
            programs = dag.execution.compute_programs
            execution_signatures.add(tuple(sorted(
                (key, value["repeats"]) for key, value in programs.items()
                if value["op"] == "matmul")))
        assert len(sample_hashes) == len(input_hashes) == 5
        assert len(execution_signatures) == 5


def test_compute_profiler_discovers_all_shape_and_repeat_signatures(tmp_path):
    suite = prepare_suite(tmp_path)
    paths = [tmp_path / item["path"] for scenario in suite["scenarios"].values()
             for item in scenario["samples"]]
    workloads, cases = collect_calibration_cases(paths, world_size=2)
    assert len(workloads) == 30
    signatures = [item["signature"] for item in cases.values()]
    assert {item["op"] for item in signatures} == {"fill", "matmul", "sum_join"}
    assert {item["repeats"] for item in signatures if item["op"] == "matmul"} >= {6, 8, 10, 12, 16, 24, 32, 96, 160}
    assert len({json.dumps(item, sort_keys=True) for item in signatures}) == len(signatures)


def test_sample_expansion_is_deterministic_and_repeat_independent():
    template = make_template("L0-balanced")
    first = expand_sample(template, "L0-balanced", 8101, 0)
    second = expand_sample(template, "L0-balanced", 8101, 0)
    assert first == second
    assert _execution_hash(first) == _execution_hash(second)
    assert first["execution"]["sample_id"] == second["execution"]["sample_id"]
    assert first["seed"] == second["seed"]


def test_d1_tail_frontiers_are_asymmetric_and_d3_has_multiple_sinks():
    d1_template = make_template("D1-asymmetric-frontiers")
    d1 = load_dag_from_document(d1_template)
    tails = compute_tails(d1.graph)
    comms = {
        f"{job['job_id']}/{node['node_id']}": node
        for job in d1_template["jobs"] for node in job["nodes"]
        if node["kind"] == "comm"
    }
    first_frontier = ("job-0/frontier", "job-1/frontier")
    assert all(comms[task_id]["deps"] == ["producer"] for task_id in first_frontier)
    assert all(comms[task_id]["group_seq"] == 0 for task_id in first_frontier)
    assert comms[first_frontier[0]]["group_id"] != comms[first_frontier[1]]["group_id"]
    assert tails[first_frontier[1]] > tails[first_frontier[0]]
    d3 = load_dag_from_document(make_template("D3-order-and-sinks"))
    job0 = next(job for job in d3.graph.jobs if job.job_id == "job-0")
    successors = {dep for node in job0.nodes for dep in node.deps}
    sinks = {node.node_id for node in job0.nodes if node.node_id not in successors}
    assert {"sink-a", "sink-b"}.issubset(sinks)


def test_formal_order_is_1050_runs_with_balanced_arm_positions(tmp_path):
    suite = prepare_suite(tmp_path)
    manifest = {"contract_version": CONTRACT_VERSION, "workload_seeds": list(FORMAL_WORKLOAD_SEEDS),
                "formal_plan_amendment": {"historical": True},
                "replacement_arm_qualification": {
                    "arm": "raw-ordered-static-fifo", "status": "qualified", "evidence": {
                        "evidence_path": "raw-g1.json", "evidence_sha256": "a" * 64,
                        "source_snapshot_sha256": "b" * 64, "profile_sha256": "c" * 64,
                        "device_uuids": ["GPU0", "GPU1"], "contract_version": "raw-v1",
                    }}, "samples": []}
    for scenario, record in suite["scenarios"].items():
        for sample in record["samples"]:
            manifest["samples"].append({
                **sample, "scenario": scenario, "compute_profile_sha256": "c" * 64,
                "comm_profile_sha256": "d" * 64,
            })
    order = make_order_table(manifest, repeats=5)
    verify_order_table(order)
    assert order["block_count"] == 150
    assert order["run_count"] == 1050
    assert set(order["arms"]) == set(ARMS)
    for counts in order["arm_position_counts"].values():
        assert max(counts) - min(counts) <= 1


def test_order_rejects_legacy_contract_and_raw_substitution(tmp_path):
    suite = prepare_suite(tmp_path)
    manifest = {"contract_version": CONTRACT_VERSION, "workload_seeds": list(FORMAL_WORKLOAD_SEEDS), "samples": []}
    for scenario, record in suite["scenarios"].items():
        for sample in record["samples"]:
            manifest["samples"].append({**sample, "scenario": scenario,
                                        "compute_profile_sha256": "c" * 64,
                                        "comm_profile_sha256": "d" * 64})
    with pytest.raises(ValueError, match="contract version"):
        make_order_table({**manifest, "contract_version": "legacy"})
    with pytest.raises(ValueError, match="invalid.*arm"):
        make_order_table(manifest, arms=("raw-ordered-static-fifo",) + ARMS[1:])
    assert make_order_table(manifest)["run_count"] == 1050


def test_e2_pilot_is_separately_seeded_and_limited_to_324_initial_runs(tmp_path):
    suite = prepare_suite(tmp_path, seeds=PILOT_WORKLOAD_SEEDS)
    assert suite["scope"] == "pilot"
    manifest = {"contract_version": CONTRACT_VERSION, "workload_seeds": list(PILOT_WORKLOAD_SEEDS),
                "bare": {"status": "unqualified"}, "samples": []}
    for scenario, record in suite["scenarios"].items():
        for sample in record["samples"]:
            manifest["samples"].append({**sample, "scenario": scenario,
                                        "compute_profile_sha256": "c" * 64,
                                        "comm_profile_sha256": "d" * 64})
    order = make_order_table(manifest, repeats=3, arms=MECHANISM_PILOT_ARMS)
    assert order["block_count"] == 54
    assert order["run_count"] == 324


def test_pilot_and_formal_inputs_keep_identical_scenario_topologies(tmp_path):
    formal_dir = tmp_path / "formal"
    pilot_dir = tmp_path / "pilot"
    formal = prepare_suite(formal_dir)
    pilot = prepare_suite(pilot_dir, seeds=PILOT_WORKLOAD_SEEDS)
    for scenario in SCENARIOS:
        formal_path = formal_dir / formal["scenarios"][scenario]["samples"][0]["path"]
        pilot_path = pilot_dir / pilot["scenarios"][scenario]["samples"][0]["path"]
        formal_input = json.loads(formal_path.read_text())
        pilot_input = json.loads(pilot_path.read_text())
        assert topology_hash(formal_input) == topology_hash(pilot_input)


def test_attempt_recovery_never_splices_arms_across_block_attempts():
    arms = ("old-static-fifo", "new-static-fifo")
    partial = [
        {"block_id": "b0", "block_attempt": 1, "arm": arms[0], "validation_status": "ok",
         "failure_class": "none", "returncode": 0},
        {"block_id": "b0", "block_attempt": 2, "arm": arms[1], "validation_status": "ok",
         "failure_class": "none", "returncode": 0},
    ]
    assert _accepted_blocks(partial, arms) == {}
    complete_retry = partial + [
        {"block_id": "b0", "block_attempt": 2, "arm": arms[0], "validation_status": "ok",
         "failure_class": "none", "returncode": 0},
    ]
    assert _accepted_blocks(complete_retry, arms) == {"b0": 2}
    failed_exit = [{**row, "returncode": 1} for row in complete_retry[1:]]
    assert _accepted_blocks(failed_exit, arms) == {}


def test_smoke_order_limit_keeps_complete_arm_blocks_and_marks_the_subset():
    order = {
        "schema": "jobpacer-gpu-seven-arm-order", "schema_version": 1,
        "arms": ["old-static-fifo", "new-static-fifo"], "block_count": 2, "run_count": 4,
        "blocks": [
            {"block_id": "b0", "arms": [{"run_id": "b0-a", "arm": "old-static-fifo", "arm_position": 0},
                                          {"run_id": "b0-b", "arm": "new-static-fifo", "arm_position": 1}]},
            {"block_id": "b1", "arms": [{"run_id": "b1-b", "arm": "new-static-fifo", "arm_position": 0},
                                          {"run_id": "b1-a", "arm": "old-static-fifo", "arm_position": 1}]},
        ],
    }
    limited = _limit_order_table(order, 1)
    assert limited["block_count"] == 1
    assert limited["run_count"] == 2
    assert limited["planned_block_limit"] == 1
    assert limited["full_block_count_before_limit"] == 2
    assert {row["arm"] for row in limited["blocks"][0]["arms"]} == {"old-static-fifo", "new-static-fifo"}
    with pytest.raises(ValueError, match="max_blocks"):
        _limit_order_table(order, True)


def test_attempt_reservation_never_reuses_paths_after_an_unledgered_launch(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "logs").mkdir()
    run = {"run_id": "L0-balanced-8101-r0-new-dynamic-fifo", "block_id": "block-0"}
    first = _reserve_run_attempt(run, tmp_path, block_attempt=1)
    second = _reserve_run_attempt(run, tmp_path, block_attempt=2)
    assert (first["attempt"], second["attempt"]) == (1, 2)
    assert first["raw_path"] != second["raw_path"]
    assert first["reservation_path"] != second["reservation_path"]
    assert (tmp_path / first["reservation_path"]).is_file()
    assert (tmp_path / second["reservation_path"]).is_file()


def test_recovery_refuses_live_rank_child_then_records_orphan_after_it_exits(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "logs").mkdir()
    run = {"run_id": "block-0-arm-a", "block_id": "block-0", "arm": "old-static-fifo",
           "scenario": "L0-balanced", "workload_seed": 9101, "repeat": 0, "epoch": 0,
           "input_sha256": "i", "execution_sample_hash": "e", "estimate_view_hash": "v",
           "compute_profile_sha256": "c", "comm_profile_sha256": "m"}
    reservation = _reserve_run_attempt(run, tmp_path, block_attempt=1)
    child_pid_path = tmp_path / "rank-child.pid"
    child_code = (
        "import os, pathlib, signal, sys; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "signal.signal(signal.SIGUSR1, lambda *_: sys.exit(0)); signal.pause()"
    )
    parent_code = "\n".join((
        "import pathlib, subprocess, sys, time",
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)",
        "deadline=time.monotonic()+5",
        "while not pathlib.Path(sys.argv[1]).exists() and time.monotonic()<deadline:",
        "    time.sleep(.005)",
    ))
    child = _start_replay_process(
        [sys.executable, "-c", parent_code, str(child_pid_path), child_code],
        cwd=tmp_path, env=dict(os.environ), batch_dir=tmp_path, reservation=reservation,
    )
    try:
        child.communicate(timeout=5)
        deadline = time.monotonic() + 5
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(.005)
        assert child_pid_path.exists(), "controlled rank child did not reach its ready barrier"
        rank_pid = int(child_pid_path.read_text())
        manifest = {"execution_contract": {}, "replay_settings": {},
                    "source_snapshot": {"digest": "source-test"}}
        order = {"blocks": [{"arms": [run]}]}
        with pytest.raises(ValueError, match="still running"):
            _recover_orphaned_attempts(tmp_path, manifest, order, [])
        ledger_path = tmp_path / "runs.jsonl"
        assert not ledger_path.exists() or not ledger_path.read_text()

        os.kill(rank_pid, signal.SIGUSR1)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(rank_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.005)
        else:
            raise AssertionError("controlled rank child did not exit after release")
        rows = _recover_orphaned_attempts(tmp_path, manifest, order, [])
        assert len(rows) == 1
        assert rows[0]["failure_class"] == "parent_interrupted_before_ledger"
        assert rows[0]["child_identity"]["pid"] == child.pid
        assert rows[0]["reservation_path"] == reservation["reservation_path"]
    finally:
        if child_pid_path.exists():
            try:
                os.kill(int(child_pid_path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait(timeout=5)


def test_formal_complete_requires_exact_formal_grid_and_bare_evidence():
    order = {"arms": list(ARMS), "run_count": 1050, "blocks": [
        {"scenario": scenario, "workload_seed": seed, "repeat": repeat}
        for scenario in SCENARIOS for seed in FORMAL_WORKLOAD_SEEDS for repeat in range(5)]}
    suite = {"contract_version": CONTRACT_VERSION, "scope": "formal", "samples": [
        {"scenario": scenario, "workload_seed": seed}
        for scenario in SCENARIOS for seed in FORMAL_WORKLOAD_SEEDS],
        "bare_qualification": {"arm": BARE_ARM, "status": "qualified",
                               "evidence": {"contract_version": "bare-ordered-v2-layered-round-robin"}}}
    manifest = {"stage": "formal", "scope": "formal", "arms": list(ARMS),
                "repeats_per_sample": 5, "planned_blocks": 150, "planned_runs": 1050}
    checks = _formal_matrix_checks(manifest, suite, order, accepted_blocks=150,
                                   accepted_runs=1050, validation_errors=[])
    assert all(checks.values())
    assert not _formal_matrix_checks(manifest, {**suite, "bare_qualification": {}}, order,
        accepted_blocks=150, accepted_runs=1050, validation_errors=[])["bare_qualified"]
    assert not _formal_matrix_checks(manifest, suite, order,
        accepted_blocks=149, accepted_runs=1043, validation_errors=[])["all_150_blocks_independently_accepted"]


def _controlled_runner_batch(tmp_path, monkeypatch):
    import examples.jobpacer.experiments.seven_arm.batch as runner

    batch_dir = tmp_path / "controlled-batch"
    for directory in ("inputs", "logs/reservations", "raw", "source/repository", "tables", "figures"):
        (batch_dir / directory).mkdir(parents=True, exist_ok=True)
    samples = []
    for index, scenario in enumerate(SCENARIOS):
        samples.append({
            "scenario": scenario, "workload_seed": 9101,
            "path": f"inputs/{scenario}/workload-9101.json",
            "fifo_order_path": f"inputs/{scenario}/workload-9101.fifo-order.json",
            "fifo_sequence": [f"job-{index}/comm-0"], "ltf_order_path": None,
            "input_sha256": f"input-{index}", "execution_sample_hash": f"execution-{index}",
            "estimate_view_hash": f"estimate-{index}",
            "compute_profile_sha256": "compute-profile", "comm_profile_sha256": "comm-profile",
        })
    suite = {"contract_version": CONTRACT_VERSION, "schema": "jobpacer-gpu-seven-arm-suite", "schema_version": 1,
             "scope": "pilot", "status": "frozen-inputs-pilot-awaiting-batch",
             "workload_seeds": [9101], "samples": samples,
             "bare": {"status": "unqualified"}, "profiles": {},
             "execution_contract": {"backend": "nccl", "world_size": 2}}
    suite_path = batch_dir / "inputs/suite-manifest.json"
    suite_bytes = (json.dumps(suite, sort_keys=True, indent=2) + "\n").encode()
    suite_path.write_bytes(suite_bytes)
    order = make_order_table(suite, order_seed=3, repeats=1, arms=MECHANISM_PILOT_ARMS)
    order_path = batch_dir / "order.json"
    order_bytes = (json.dumps(order, sort_keys=True, indent=2) + "\n").encode()
    order_path.write_bytes(order_bytes)
    environment = {"visible_device_count": 2, "device_uuids": ["GPU0", "GPU1"],
                   "python": "test", "thread_environment": {}}
    manifest = {
        "schema": "jobpacer-gpu-seven-arm-batch", "schema_version": 1,
        "batch_id": batch_dir.name, "stage": "pilot-block-smoke", "scope": "pilot",
        "planned_blocks": order["block_count"], "planned_runs": order["run_count"],
        "arms": list(MECHANISM_PILOT_ARMS), "order_seed": 3, "repeats_per_sample": 1,
        "suite_manifest_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "order_sha256": hashlib.sha256(order_bytes).hexdigest(),
        "source_snapshot": {"digest": "source-test", "files": []},
        "environment_at_plan": environment,
        "execution_contract": suite["execution_contract"],
        "replay_settings": {"timeout_s": 30, "warmup_iterations": 5},
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    monkeypatch.setattr(runner, "_verify_frozen_inputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_cuda_environment", lambda: environment)
    monkeypatch.setattr(runner, "source_snapshot", lambda *_args, **_kwargs: {
        "digest": "source-test", "files": []})
    monkeypatch.setattr(runner, "_check_resume", lambda directory, *_args: runner._read_ledger(
        directory / "runs.jsonl"))
    return runner, batch_dir, order


def _stub_attempt_writer(runner, batch_dir, behavior, calls, monkeypatch):
    def execute(run, _suite, _directory, _timeout, _warmup, block_attempt, ledger_file):
        calls.append((run["block_id"], block_attempt, run["arm"]))
        failure = behavior(run, block_attempt, len(calls))
        record = {"run_id": run["run_id"], "block_id": run["block_id"],
                  "block_attempt": block_attempt, "arm": run["arm"],
                  "attempt": block_attempt,
                  "validation_status": "ok" if failure == "none" else "missing",
                  "failure_class": failure, "returncode": 0 if failure == "none" else 1}
        ledger_file.write(json.dumps(record, sort_keys=True) + "\n")
        ledger_file.flush()
        return record
    monkeypatch.setattr(runner, "_execute_one", execute)


def test_batch_retries_whole_block_once_after_second_environment_failure(tmp_path, monkeypatch):
    runner, batch_dir, order = _controlled_runner_batch(tmp_path, monkeypatch)
    calls = []
    _stub_attempt_writer(runner, batch_dir, lambda *_: "environment_port_conflict", calls, monkeypatch)
    with pytest.raises(RuntimeError, match="single whole-block environment retry"):
        run_batch(batch_dir, timeout=30, warmup=5, allow_pilot=True, resume=False)
    ledger = [json.loads(line) for line in (batch_dir / "runs.jsonl").read_text().splitlines()]
    first_block = order["blocks"][0]["block_id"]
    assert len(calls) == 2 * len(MECHANISM_PILOT_ARMS)
    assert {row["block_attempt"] for row in ledger} == {1, 2}
    assert {row["block_id"] for row in ledger} == {first_block}


def test_batch_retries_mixed_success_and_environment_failure_as_whole_block(tmp_path, monkeypatch):
    runner, batch_dir, order = _controlled_runner_batch(tmp_path, monkeypatch)
    calls = []
    first_block = order["blocks"][0]["block_id"]

    def fail_one_first_attempt(run, block_attempt, _call_number):
        if run["block_id"] == first_block and block_attempt == 1 and run["arm"] == MECHANISM_PILOT_ARMS[0]:
            return "environment_port_conflict"
        return "none"

    _stub_attempt_writer(runner, batch_dir, fail_one_first_attempt, calls, monkeypatch)
    assert run_batch(batch_dir, timeout=30, warmup=5, allow_pilot=True, resume=False) == 0
    ledger = [json.loads(line) for line in (batch_dir / "runs.jsonl").read_text().splitlines()]
    assert len(calls) == order["run_count"] + len(MECHANISM_PILOT_ARMS)
    first_rows = [row for row in ledger if row["block_id"] == first_block]
    assert {row["block_attempt"] for row in first_rows} == {1, 2}
    assert len([row for row in first_rows if row["block_attempt"] == 1]) == len(MECHANISM_PILOT_ARMS)
    assert len([row for row in first_rows if row["block_attempt"] == 2]) == len(MECHANISM_PILOT_ARMS)
    assert sum(row["failure_class"] == "environment_port_conflict" for row in first_rows) == 1
    assert all(row["failure_class"] == "none" for row in first_rows if row["block_attempt"] == 2)
    assert _accepted_blocks(ledger, tuple(MECHANISM_PILOT_ARMS)) == {
        block["block_id"]: (2 if block["block_id"] == first_block else 1)
        for block in order["blocks"]
    }


def test_batch_program_failure_is_not_retried_and_successful_blocks_are_skipped(tmp_path, monkeypatch):
    runner, batch_dir, _order = _controlled_runner_batch(tmp_path, monkeypatch)
    calls = []
    _stub_attempt_writer(runner, batch_dir, lambda *_: "application_or_correctness_failure", calls, monkeypatch)
    with pytest.raises(RuntimeError, match="application_or_correctness_failure"):
        run_batch(batch_dir, timeout=30, warmup=5, allow_pilot=True, resume=False)
    with pytest.raises(RuntimeError, match="non-retryable prior failure"):
        run_batch(batch_dir, timeout=30, warmup=5, allow_pilot=True, resume=True)
    assert len(calls) == 1

    runner, batch_dir, order = _controlled_runner_batch(tmp_path / "success", monkeypatch)
    calls = []
    _stub_attempt_writer(runner, batch_dir, lambda *_: "none", calls, monkeypatch)
    assert run_batch(batch_dir, timeout=30, warmup=5, allow_pilot=True, resume=False) == 0
    call_count = len(calls)
    assert call_count == order["run_count"]
    assert run_batch(batch_dir, timeout=30, warmup=5, allow_pilot=True, resume=True) == 0
    assert len(calls) == call_count


def test_preview_with_partial_history_does_not_claim_a_complete_disk_budget(tmp_path):
    order = {
        "schema": "jobpacer-gpu-seven-arm-order", "schema_version": 1,
        "arms": ["old-static-fifo", "new-static-fifo"], "block_count": 1, "run_count": 2,
        "blocks": [{"block_id": "b0", "scenario": "L0-balanced", "workload_seed": 1,
                    "arms": [{"run_id": "b0-arm-a", "arm": "old-static-fifo"},
                             {"run_id": "b0-arm-b", "arm": "new-static-fifo"}]}],
    }
    history = tmp_path / "history.csv"
    history.write_text("arm,wall_time_s,raw_bytes\nold-static-fifo,5.0,1024\n")
    preview = _preview(order, [history])
    assert preview["history_coverage"] == "1/2"
    assert preview["estimated_wall_p50_s"] is None
    assert preview["estimated_disk_bytes"] is None


def test_retry_classifier_limits_automatic_retry_to_known_port_conflicts():
    assert _classify_failure("OSError: [Errno 98] Address already in use", 1, False) == "environment_port_conflict"
    assert _classify_failure("collective task mismatch", 1, False) == "application_or_correctness_failure"
    assert _classify_failure("", None, True) == "replay_timeout"


def test_pilot_mechanism_checks_distinguish_head_hol_bypass_and_ltf_score():
    static_records = [
        {"kind": "decision", "decision": "dispatch", "task_id": "a", "now": 0.5},
        {"kind": "idle_interval", "reason": "STATIC_HEAD_BLOCKED", "start": 1.0,
         "end": 2.0, "duration": 1.0},
        {"kind": "policy_snapshot", "now": 1.5,
         "eligible": [{"task_id": "c", "estimated_comm_s": 0.1, "remaining_tail_s": 0.2}]},
    ]
    assert _static_head_hol(["a", "b", "c"], static_records)
    dynamic_records = [{"kind": "decision", "decision": "dispatch", "task_id": "c",
                       "eligible": ["c"], "now": 1.5}]
    assert _dynamic_fifo_bypass(["a", "b", "c"], dynamic_records)
    ltf_records = [
        {"kind": "policy_snapshot", "now": 3.0, "eligible": [
            {"task_id": "long", "estimated_comm_s": 0.2, "remaining_tail_s": 1.0},
            {"task_id": "short", "estimated_comm_s": 0.5, "remaining_tail_s": 0.1},
        ]},
        {"kind": "decision", "decision": "dispatch", "task_id": "long",
         "eligible": ["long", "short"], "now": 3.0},
    ]
    assert _ltf_frontier_choice(ltf_records) == (True, True)


def test_compute_profile_provenance_must_match_every_formal_sample_hash(tmp_path):
    suite_dir = tmp_path / "formal"
    suite = prepare_suite(suite_dir)
    rows = [sample for scenario in suite["scenarios"].values() for sample in scenario["samples"]]
    provenance = {"inputs": [{
        "path": str((suite_dir / sample["path"]).resolve()),
        "file_sha256": sample["input_sha256"],
        "input_hash": sample["input_hash"],
        "estimate_view_hash": sample["estimate_view_hash"],
    } for sample in rows]}
    assert len(rows) == 30
    assert verify_compute_profile_inputs(provenance, suite_dir)
    provenance["inputs"][0]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="complete formal sample set|calibration input is missing or changed"):
        verify_compute_profile_inputs(provenance, suite_dir)


def test_pilot_profile_freeze_uses_the_verified_formal_calibration_suite(tmp_path, monkeypatch):
    import examples.jobpacer.comm_profile as comm_profile_module
    import examples.jobpacer.gpu.gpu_compute_profile as compute_profile_module
    import examples.jobpacer.experiments.seven_arm.suite as suite_module

    formal_dir = tmp_path / "formal"
    pilot_dir = tmp_path / "pilot"
    formal_suite = prepare_suite(formal_dir)
    prepare_suite(pilot_dir, seeds=PILOT_WORKLOAD_SEEDS)
    compute_path = tmp_path / "profiles/compute-profile.json"
    comm_path = tmp_path / "profiles/comm-profile.json"
    compute_path.parent.mkdir(parents=True)
    compute_bytes = b'{"synthetic_compute_profile": true}\n'
    compute_path.write_bytes(compute_bytes)
    compute_sha = hashlib.sha256(compute_bytes).hexdigest()
    calibration_inputs = [sample for scenario in formal_suite["scenarios"].values()
                          for sample in scenario["samples"]]
    sidecar = {
        "profile_sha256": compute_sha,
        "device_uuids": ["GPU0", "GPU1"],
        "settings": {"warmup": 5, "iterations": 30,
                     "matmul_precision": "highest", "allow_tf32": False},
        "inputs": [{
            "path": str((formal_dir / sample["path"]).resolve()),
            "file_sha256": sample["input_sha256"],
            "input_hash": sample["input_hash"],
            "estimate_view_hash": sample["estimate_view_hash"],
        } for sample in calibration_inputs],
    }
    Path(str(compute_path) + ".manifest.json").write_text(json.dumps(sidecar))
    comm_path.write_text('{"synthetic_comm_profile": true}\n')

    compute_profile = SimpleNamespace(
        devices={"GPU0": {}, "GPU1": {}},
        raw={"schema_version": 1, "software": {"matmul_precision": "highest"}},
        software={"matmul_precision": "highest", "allow_tf32": False},
    )
    comm_profile = SimpleNamespace(
        environment={"world_size": 2, "backend": "nccl", "device_uuids": ["GPU0", "GPU1"]},
        settings={"warmup": 5, "iterations": 30}, schema_version=1,
    )
    monkeypatch.setattr(compute_profile_module, "load_gpu_compute_profile", lambda path: compute_profile)
    monkeypatch.setattr(comm_profile_module, "load_profile", lambda path: comm_profile)
    monkeypatch.setattr(suite_module, "apply_dag_profile", lambda dag, *args, **kwargs: dag)
    monkeypatch.setattr(suite_module, "apply_dag_compute_profile", lambda dag, *args, **kwargs:
                        SimpleNamespace(graph=dag.graph, estimate_view_hash="profiled-estimate"))

    frozen = freeze_suite(pilot_dir, compute_path, comm_path,
                          calibration_input_dir=formal_dir, world_size=2)
    assert frozen["status"] == "frozen-inputs-pilot-awaiting-batch"
    assert frozen["profiles"]["compute"]["calibration_input_suite_sha256"]
    assert len(frozen["samples"]) == len(SCENARIOS) * len(PILOT_WORKLOAD_SEEDS)
    _verify_frozen_inputs(pilot_dir, frozen)


def test_source_snapshot_covers_legacy_static_scheduler_modules():
    tracked_paths = {row["path"] for row in source_snapshot()["files"]}
    assert {
        "src/runtime_comm_scheduler/plan.py",
        "src/runtime_comm_scheduler/scheduler.py",
        "src/runtime_comm_scheduler/work.py",
    }.issubset(tracked_paths)


def test_plan_interruption_recovery_and_analysis_use_whole_paired_attempts(tmp_path, monkeypatch):
    import examples.jobpacer.analysis.runtime_results as runtime_results
    import examples.jobpacer.runtime.runtime_adapter as runtime_adapter
    import examples.jobpacer.experiments.seven_arm.batch as runner

    suite_root = tmp_path / "suite"
    (suite_root / "inputs/L0-balanced").mkdir(parents=True)
    (suite_root / "profiles").mkdir()
    (suite_root / "templates").mkdir()

    def write_json(path: Path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value, sort_keys=True, indent=2) + "\n"
        path.write_text(data)
        return hashlib.sha256(data.encode()).hexdigest()

    input_path = suite_root / "inputs/L0-balanced/workload-9101.json"
    input_sha = write_json(input_path, {"sample": "synthetic"})
    fifo_path = suite_root / "inputs/L0-balanced/workload-9101.fifo-order.json"
    ltf_path = suite_root / "inputs/L0-balanced/workload-9101.ltf-order.json"
    fifo_sequence, ltf_sequence = ["g0/comm-0", "g1/comm-0"], ["g1/comm-0", "g0/comm-0"]
    fifo_sha = write_json(fifo_path, fifo_sequence)
    ltf_sha = write_json(ltf_path, ltf_sequence)
    compute_profile = suite_root / "profiles/compute.json"
    compute_sha = write_json(compute_profile, {"devices": [
        {"device_uuid": "GPU0"}, {"device_uuid": "GPU1"}]})
    comm_profile = suite_root / "profiles/comm.json"
    comm_sha = write_json(comm_profile, {"synthetic_profile": True})
    sample = {
        "scenario": "L0-balanced", "workload_seed": 9101,
        "path": "inputs/L0-balanced/workload-9101.json", "input_sha256": input_sha,
        "execution_sample_hash": "e" * 64, "estimate_view_hash": "f" * 64,
        "fifo_order_path": "inputs/L0-balanced/workload-9101.fifo-order.json",
        "fifo_order_sha256": fifo_sha, "ltf_order_path": "inputs/L0-balanced/workload-9101.ltf-order.json",
        "ltf_order_sha256": ltf_sha, "fifo_sequence": fifo_sequence,
        "compute_profile_sha256": compute_sha, "comm_profile_sha256": comm_sha,
    }
    suite_manifest = {
        "contract_version": CONTRACT_VERSION,
        "schema": "jobpacer-gpu-seven-arm-suite", "schema_version": 1,
        "scope": "pilot", "status": "frozen-inputs-pilot-awaiting-batch",
        "workload_seeds": [9101], "samples": [sample],
        "bare": {"status": "unqualified"},
        "execution_contract": {"backend": "nccl", "world_size": 2},
        "profiles": {
            "compute": {"path": "profiles/compute.json", "sha256": compute_sha,
                        "calibration": {"settings": {"warmup": 5, "iterations": 30}}},
            "communication": {"path": "profiles/comm.json", "sha256": comm_sha,
                              "settings": {"warmup": 5, "iterations": 30}},
        },
    }
    write_json(suite_root / "suite-inputs.json", {"prepared": True})
    suite_manifest_path = suite_root / "suite-manifest.json"
    write_json(suite_manifest_path, suite_manifest)
    arms = list(MECHANISM_PILOT_ARMS)
    block_id = "L0-balanced-w9101-r0"
    common = {
        "block_id": block_id, "scenario": "L0-balanced", "workload_seed": 9101,
        "repeat": 0, "epoch": 0, "sample_path": sample["path"],
        "fifo_order_path": sample["fifo_order_path"], "ltf_order_path": sample["ltf_order_path"],
        "fifo_sequence": fifo_sequence, "ltf_sequence": ltf_sequence,
        "input_sha256": input_sha, "execution_sample_hash": sample["execution_sample_hash"],
        "estimate_view_hash": sample["estimate_view_hash"],
        "compute_profile_sha256": compute_sha, "comm_profile_sha256": comm_sha,
    }
    rows = [{**common, "arm": arm, "arm_position": index,
             "run_id": f"{block_id}-{arm}"} for index, arm in enumerate(arms)]
    order = {"schema": "jobpacer-gpu-seven-arm-order", "schema_version": 1,
             "arms": arms, "block_count": 1, "run_count": len(arms),
             "arm_position_counts": {},
             "blocks": [{**common, "block_id": block_id, "arms": rows}]}
    environment = {"visible_device_count": 2, "device_uuids": ["GPU0", "GPU1"],
                   "python": "test", "thread_environment": {}}
    monkeypatch.setattr(runner, "_cuda_environment", lambda: environment)
    monkeypatch.setattr(runner, "source_snapshot", lambda destination=None: {
        "digest": "source-test", "files": []})
    monkeypatch.setattr(runner, "make_order_table", lambda *args, **kwargs: order)
    batch_dir = tmp_path / "batch"
    batch_manifest = create_batch(
        suite_manifest_path, batch_dir, order_seed=1, repeats=1,
        arms=tuple(arms), timeout_s=30, max_blocks=1,
    )
    assert batch_manifest["stage"] == "pilot-block-smoke"
    assert batch_manifest["planned_blocks"] == 1
    archived_payload = batch_dir / "inputs"
    _verify_frozen_inputs(suite_root, suite_manifest)
    _verify_frozen_inputs(archived_payload, suite_manifest)

    call_state = {"count": 0, "interrupt_second": True}

    def fake_run_child(command, **kwargs):
        call_state["count"] += 1
        output = Path(command[command.index("--output") + 1])
        assert output.name.endswith(".json")
        assert Path(command[command.index("--comm-profile") + 1]).name == "comm.json"
        assert Path(command[command.index("--compute-profile") + 1]).name == "compute.json"
        policy = command[command.index("--policy") + 1]
        if policy in {"static_fifo", "static_ltf"}:
            static_path = Path(command[command.index("--static-order") + 1])
            assert static_path.name.endswith("fifo-order.json" if policy == "static_fifo" else "ltf-order.json")
        else:
            assert "--static-order" not in command
        if call_state["interrupt_second"] and call_state["count"] == 2:
            raise KeyboardInterrupt("simulated stop after reservation")
        child = _start_replay_process(
            [sys.executable, "-c", "pass"], cwd=kwargs["cwd"], env=kwargs["env"],
            batch_dir=batch_dir, reservation=kwargs["reservation"],
        )
        child.communicate(timeout=5)
        runner._persist_reservation_state(
            batch_dir, kwargs["reservation"], "child-exited",
            child_returncode=child.returncode,
        )
        launch_sequence = ltf_sequence if policy == "static_ltf" else fifo_sequence
        result = {
            "validation": {"status": "ok"},
            "ranks": [{"rank": rank, "device_uuid": f"GPU{rank}",
                       "launch_sequence": launch_sequence,
                       "jobs": [{"gpu_buffer_validation": {"checksum": True}}],
                       "decision_records": []}
                      for rank in range(2)],
            "performance": {"workload_makespan_us": 1000,
                            "job_makespans": [{"job_id": "job-0", "makespan_us": 1000}]},
        }
        output.write_text(json.dumps(result))
        return 0, "", "", False, True

    monkeypatch.setattr(runner, "_run_child", fake_run_child)
    with pytest.raises(KeyboardInterrupt, match="simulated stop"):
        run_batch(batch_dir, timeout=30, warmup=5, allow_pilot=True, resume=False)
    interrupted_reservation = list((batch_dir / "logs/reservations").glob("*.json"))
    assert len(interrupted_reservation) == 2

    call_state["interrupt_second"] = False
    assert run_batch(batch_dir, timeout=30, warmup=5, allow_pilot=True, resume=True) == 0
    ledger = [json.loads(line) for line in (batch_dir / "runs.jsonl").read_text().splitlines()]
    assert any(row["failure_class"] == "parent_interrupted_before_ledger" for row in ledger)
    assert _accepted_blocks(ledger, tuple(arms)) == {block_id: 2}

    monkeypatch.setattr(runtime_adapter, "load_dag", lambda *args, **kwargs: SimpleNamespace(graph=object(), manifest_digest="synthetic", input_hash="synthetic"))
    monkeypatch.setattr(runtime_results, "expected_dag_results", lambda *args, **kwargs: {
        "expected": {}, "expected_nodes": []})
    monkeypatch.setattr(runtime_results, "validate_results", lambda *args, **kwargs: {
        "status": "ok", "errors": []})
    analysis = analyze_batch(batch_dir, bootstrap_samples=0)
    assert analysis["accepted_complete_blocks"] == 1
    assert analysis["attempted_runs"] == 8
    assert analysis["validation_errors"] == []
