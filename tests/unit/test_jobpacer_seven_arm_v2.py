from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from examples.jobpacer.runtime.runtime_adapter import parse_dag, apply_dag_compute_profile
from examples.jobpacer.scripts.gpu_seven_arm_suite import (
    ARMS, BARE_ARM, EXECUTION_CONTRACT, PILOT_WORKLOAD_SEEDS,
    expand_sample, make_template, prepare_suite, topology_hash,
)
from examples.jobpacer.scripts.gpu_seven_arm_gates import decision_evidence, mechanism_gates, verify_readiness
from examples.jobpacer.scripts.run_gpu_seven_arm import _verified_pre_task_port_conflict


def test_v2_has_distinct_six_graphs_two_real_chains_and_explicit_capacities():
    assert ARMS[0] == BARE_ARM and "raw-ordered-static-fifo" not in ARMS
    assert EXECUTION_CONTRACT["capacity_by_engine"] == {"bare": None, "old": "local-one", "new": "global-one"}
    scenarios = ("L0-balanced", "L1-skew-tail", "D0-fork-join", "D1-asymmetric-frontiers",
                 "D2-cross-job-skew", "D3-order-and-sinks")
    # L0/L1 intentionally share chain topology, differing by execution skew.
    assert len({topology_hash(make_template(name)) for name in scenarios}) == 5
    for scenario in scenarios[:2]:
        for job in make_template(scenario)["jobs"]:
            for index, node in enumerate(job["nodes"]):
                assert node["deps"] == ([] if index == 0 else [job["nodes"][index - 1]["node_id"]])
    d1 = make_template(scenarios[3])
    occupancy = d1["jobs"][0]["nodes"][0]
    assert occupancy["kind"] == "comm" and occupancy["collective"]["num_bytes"] == 64 * 1024 * 1024


def test_profile_estimates_do_not_leak_execution_perturbations():
    template = make_template("L1-skew-tail")
    first = parse_dag(expand_sample(template, "L1-skew-tail", 8101, 0), world_size=2)
    second = parse_dag(expand_sample(template, "L1-skew-tail", 8102, 4), world_size=2)
    assert first.execution.compute_programs != second.execution.compute_programs
    class Profile:
        raw = {"schema_version": 3}
        def record(self, *, spec, **kwargs):
            return SimpleNamespace(device_event_p50_s=spec.get("repeats", 1) * 0.001)
    profiled = [apply_dag_compute_profile(dag, Profile(), device_uuids=("a", "b"), software={})
                for dag in (first, second)]
    assert profiled[0].estimate_view_hash == profiled[1].estimate_view_hash
    nominal = first.execution.profiles["nominal_compute_repeats"]
    assert nominal["job-0/producer-s0"] == nominal["job-1/producer-s0"] == 8


def _records(selected="long", *, same_winner=False):
    eligible = [
        {"task_id": "short", "estimated_comm_s": 1, "remaining_tail_s": 1, "eligible_seq": 1},
        {"task_id": "long", "estimated_comm_s": 1, "remaining_tail_s": 9, "eligible_seq": 0 if same_winner else 2},
    ]
    return [{"kind": "policy_snapshot", "now": 10, "eligible": eligible},
            {"kind": "decision", "decision": "dispatch", "now": 10, "decision_seq": 2,
             "eligible": ["short", "long"], "task_id": selected}]


def test_counterfactual_requires_same_snapshot_disagreement_and_correct_choice():
    assert decision_evidence(_records(), "ltf")["competition_observed"]
    assert not decision_evidence(_records(same_winner=True), "ltf")["competition_observed"]
    assert decision_evidence(_records("short"), "ltf")["errors"]
    stale = _records()
    stale[0]["now"] = 9
    assert decision_evidence(stale, "ltf")["errors"]


def test_pilot_gate_cannot_pool_seeds_or_count_one_lucky_replay():
    rows = [{"scenario": scenario, "seed": seed, "repeat": repeat,
             "competition": seed in PILOT_WORKLOAD_SEEDS[:2] and repeat < 2,
             "hol_and_bypass": seed in PILOT_WORKLOAD_SEEDS[:2] and repeat < 2}
            for scenario in ("L1-skew-tail", "D1-asymmetric-frontiers")
            for seed in PILOT_WORKLOAD_SEEDS for repeat in range(3)]
    assert all(gate["passed"] for gate in mechanism_gates(rows).values())
    assert not any(gate["passed"] for gate in mechanism_gates(
        [row for row in rows if row["seed"] != 9103]).values())
    for row in rows:
        row["competition"] = row["repeat"] == 0
    assert not mechanism_gates(rows)["D1-asymmetric-frontiers"]["passed"]


def test_formal_run_cannot_be_unlocked_with_boolean_gate_claims():
    with pytest.raises(ValueError, match="software.*pending"):
        verify_readiness({"readiness": {"software": True}}, "source")


def test_pre_task_port_conflict_uses_structured_eaddrinuse_evidence():
    raw = {
        "config": {"rendezvous_startup_attempts": [{"errors": [
            'rank 0 exited 1: {"error": "DistNetworkError: server socket failed; '
            'code: -98, name: EADDRINUSE, message: address already in use"}',
            "rank 1 exited -9: ",
        ]}]},
        "ranks": [],
        "validation": {"status": "failed"},
    }
    assert _verified_pre_task_port_conflict(
        raw, returncode=1, timed_out=False, process_group_exited=True)


@pytest.mark.parametrize("overrides", [
    {"ranks": [{"rank": 0}]},
    {"config": {"rendezvous_startup_attempts": [{"errors": ["application EADDRINUSE"]}]}},
])
def test_port_conflict_is_not_retryable_without_pre_task_startup_proof(overrides):
    raw = {
        "config": {"rendezvous_startup_attempts": [{"errors": [
            "DistNetworkError: code: -98, name: EADDRINUSE",
        ]}]},
        "ranks": [],
        "validation": {"status": "failed"},
        **overrides,
    }
    assert not _verified_pre_task_port_conflict(
        raw, returncode=1, timed_out=False, process_group_exited=True)
    assert not _verified_pre_task_port_conflict(
        raw, returncode=1, timed_out=False, process_group_exited=False)
    assert not _verified_pre_task_port_conflict(
        raw, returncode=1, timed_out=True, process_group_exited=True)


def test_generated_manifest_discloses_actual_and_nominal_work(tmp_path):
    suite = prepare_suite(tmp_path, seeds=PILOT_WORKLOAD_SEEDS)
    for scenario in suite["scenarios"].values():
        assert len({json.dumps(sample["actual_matmul_repeats"], sort_keys=True)
                    for sample in scenario["samples"]}) == 3
        assert len({json.dumps(sample["nominal_matmul_repeats"], sort_keys=True)
                    for sample in scenario["samples"]}) == 1


@pytest.mark.parametrize("overrides", [
    {"nccl_version": (2, 25, 0)}, {"cuda_version": "12.2"}, {"cuda_version": None},
    {"implicit": "0"}, {"blocking_wait": "1"},
])
def test_bare_rejects_unsupported_backend_before_launch(overrides):
    from examples.jobpacer.runtime.dag_comm_adapters import validate_bare_backend
    settings = dict(nccl_version=(2, 29, 3), cuda_version="12.6", implicit="1", blocking_wait="0")
    validate_bare_backend(**settings)
    with pytest.raises(ValueError):
        validate_bare_backend(**{**settings, **overrides})
