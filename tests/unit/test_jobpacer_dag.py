from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from runtime_comm_scheduler.dag import (
    CommNode,
    ComputeNode,
    DagGraph,
    DagJob,
    DagRunner,
    NodeState,
    build_static_order,
    dag_task_hint,
    dag_task_spec,
    validate_graph,
    validate_static_order,
)
from runtime_comm_scheduler.runtime import CollectiveSpec, CoordinatorState, GroupSpec, TaskHint
from runtime_comm_scheduler.dag.model import compute_tails
from runtime_comm_scheduler.dag import runner as runner_module
from examples.jobpacer.runtime_adapter import load_dag, parse_dag, sample_compute_duration


ROOT = Path(__file__).resolve().parents[2]
DAGS = ROOT / "benchmark/phase3"


def _collective():
    return {"op": "all_reduce", "numel": 2, "num_bytes": 8,
            "dtype": "float32", "shape": [2], "reduction": "sum"}


def _payload(nodes, *, name="test", groups=None, jobs=None, execution=None):
    return {
        "schema_version": 1,
        "name": name,
        "seed": 9,
        "groups": groups or [{"group_id": "g", "ranks": [0, 1]}],
        "execution": {"compute_duration_s": execution or {
            f"job/{node['node_id']}": node["estimated_duration_s"]
            for node in nodes if node["kind"] == "compute"
        }},
        "jobs": jobs or [{"job_id": "job", "nodes": nodes}],
    }


def _compute(node_id, deps=(), seconds=0.0):
    return {"node_id": node_id, "kind": "compute", "deps": list(deps), "estimated_duration_s": seconds}


def _comm(node_id, seq, deps=(), *, group="g", estimated=0.001):
    return {"node_id": node_id, "kind": "comm", "deps": list(deps), "group_id": group,
            "group_seq": seq, "estimated_comm_s": estimated, "collective": _collective()}


def test_examples_parse_with_stable_digest_and_contiguous_group_sequences():
    for name in ("linear", "diamond", "multi-group"):
        dag = load_dag(DAGS / f"{name}.json", world_size=2)
        assert len(dag.manifest_digest) == 64
        assert len(dag.expected_task_ids) > 0
        doc = json.loads(dag.canonical_json)
        assert parse_dag(doc).manifest_digest == dag.manifest_digest


def test_diamond_tail_uses_longest_branch_not_sum_or_execution_samples():
    nodes = [
        _comm("comm", 0, ["root"], estimated=0.005),
        _compute("root", (), 0.0),
        _compute("short", ["comm"], 0.003),
        _compute("long", ["comm"], 0.007),
        _comm("end", 1, ["short", "long"], estimated=0.011),
    ]
    raw = _payload(nodes, execution={"job/root": 0.0, "job/short": 0.9,
                                     "job/long": 0.1})
    dag = parse_dag(raw)
    tails = compute_tails(dag.graph)
    assert tails["job/comm"] == pytest.approx(0.018)
    assert tails["job/end"] == 0


def test_tail_uses_successor_duration_and_excludes_current_communication():
    dag = parse_dag(_payload([
        _comm("first", 0, ["root"], estimated=0.005),
        _compute("root"),
        _compute("middle", ["first"], 0.003),
        _comm("last", 1, ["middle"], estimated=0.011),
    ]))
    tails = compute_tails(dag.graph)
    assert tails["job/first"] == pytest.approx(0.014)
    assert tails["job/middle"] == pytest.approx(0.011)
    assert tails["job/last"] == 0


@pytest.mark.parametrize(("policy", "expected"), [
    ("fifo", "job-0/comm-b0"),
    ("ltf", "job-0/comm-a0"),
])
def test_multigroup_dag_tail_changes_coordinator_choice(policy, expected):
    dag = load_dag(DAGS / "multi-group.json", world_size=2)
    tails = compute_tails(dag.graph)
    assert tails["job-0/comm-a0"] == pytest.approx(0.021)
    assert tails["job-0/comm-b0"] == pytest.approx(0.003)
    nodes = {
        f"{job.job_id}/{node.node_id}": (job, node)
        for job in dag.graph.jobs for node in job.nodes
        if node.node_id in {"comm-a0", "comm-b0", "comm-c0"}
    }
    coordinator = CoordinatorState((0, 1), policy=policy)
    for endpoint in (0, 1):
        for seq, group in enumerate(dag.graph.groups, 1):
            coordinator.apply(endpoint, "REGISTER_GROUP", seq,
                              {"group": group.to_dict()}, 0.0)

    def offer(task_id, endpoint, event_seq, now):
        job, node = nodes[task_id]
        spec = dag_task_spec(job, node, epoch=0)
        hint = dag_task_hint(node, tail_s=tails[task_id])
        return coordinator.apply(endpoint, "OFFER", event_seq,
                                 {"task": spec.to_dict(), "hint": hint.to_dict()}, now)

    incumbent = "job-1/comm-c0"
    grants = offer(incumbent, 0, 4, 0.1) + offer(incumbent, 1, 4, 0.1)
    assert [item.payload["task"]["task_id"] for item in grants] == [incumbent, incumbent]
    for endpoint in (0, 1):
        offer("job-0/comm-b0", endpoint, 5, 0.2)
        offer("job-0/comm-a0", endpoint, 6, 0.3)
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "SUBMITTED", 7,
                          {"task_id": incumbent, "decision_seq": 1}, 0.4)
    coordinator.apply(0, "COMPLETED", 8,
                      {"task_id": incumbent, "decision_seq": 1}, 0.5)
    grants = coordinator.apply(1, "COMPLETED", 8,
                               {"task_id": incumbent, "decision_seq": 1}, 0.6)
    assert {item.payload["task"]["task_id"] for item in grants} == {expected}


def test_multigroup_lookahead_waits_for_its_declared_frontier_until_offer():
    dag = load_dag(DAGS / "multi-group.json", world_size=2)
    tails = compute_tails(dag.graph)
    comms = {
        f"{job.job_id}/{node.node_id}": (job, node)
        for job in dag.graph.jobs for node in job.nodes
        if node.node_id in {"comm-a0", "comm-b0", "comm-c0"}
    }
    coordinator = CoordinatorState((0, 1), policy="lookahead", wait_budget_s=0.02)
    for endpoint in (0, 1):
        for seq, group in enumerate(dag.graph.groups, 1):
            coordinator.apply(endpoint, "REGISTER_GROUP", seq,
                              {"group": group.to_dict()}, 0.0)

    def payload(task_id, ready_after_s=0.0):
        job, node = comms[task_id]
        spec = dag_task_spec(job, node, epoch=0)
        hint = dag_task_hint(node, tail_s=tails[task_id])
        hint = TaskHint(ready_after_s, hint.estimated_comm_s, hint.remaining_tail_s)
        return {"task": spec.to_dict(), "hint": hint.to_dict()}

    target = "job-1/comm-c0"
    target_payload = payload(target, ready_after_s=0.008)
    for endpoint in (0, 1):
        coordinator.apply(endpoint, "DECLARE", 4, target_payload, 0.001)
    for endpoint, (b_time, a_time) in enumerate(((0.002, 0.003), (0.003, 0.004))):
        coordinator.apply(endpoint, "OFFER", 5, payload("job-0/comm-b0"), b_time)
        coordinator.apply(endpoint, "OFFER", 6, payload("job-0/comm-a0"), a_time)

    assert coordinator.active_wait is not None
    assert coordinator.active_wait.target_task_id == target
    coordinator.apply(0, "OFFER", 7, target_payload, 0.007)
    grants = coordinator.apply(1, "OFFER", 7, target_payload, 0.008)
    assert {item.payload["task"]["task_id"] for item in grants} == {target}
    assert coordinator.active_wait is None
    assert any(item.get("kind") == "decision" and item.get("decision") == "wait"
               and item.get("target") == target for item in coordinator.records)
    assert any(item.get("kind") == "idle_interval" and item.get("reason") == "ACTIVE_LOOKAHEAD"
               and item.get("duration", 0) > 0 for item in coordinator.records)
    assert not any(item.get("kind") == "lookahead_deadline" and item.get("target") == target
                   for item in coordinator.records)


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["jobs"][0]["nodes"][0].update(deps=["missing"]), "missing deps"),
    (lambda p: p["jobs"][0]["nodes"][0].update(deps=["root", "root"]), "duplicate dependencies"),
    (lambda p: p["jobs"][0]["nodes"][0].update(estimated_duration_s=float("nan")), "finite non-negative"),
    (lambda p: p["jobs"][0]["nodes"][0].update(estimated_duration_s=True), "finite non-negative"),
    (lambda p: p["jobs"][0]["nodes"][0].update(extra=True), "unknown fields"),
])
def test_rejects_invalid_dag_inputs_before_runtime(mutate, message):
    raw = _payload([_compute("root"), _comm("comm", 0, ["root"])])
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        parse_dag(raw)


def test_rejects_local_cycles_group_sequence_gaps_and_static_order_errors():
    with pytest.raises(ValueError, match="dependency cycle"):
        parse_dag(_payload([_compute("a", ["b"]), _comm("b", 0, ["a"])]))

    with pytest.raises(ValueError, match="contiguous"):
        parse_dag(_payload([_compute("root"), _comm("a", 0, ["root"]),
                            _comm("b", 2, ["root"])]))

    dag = load_dag(DAGS / "diamond.json")
    with pytest.raises(ValueError, match="cover all communication"):
        validate_static_order(["job-0/comm-a"], dag.graph)
    with pytest.raises(ValueError, match="dependency"):
        validate_static_order(["job-0/comm-c", "job-0/comm-a"], dag.graph)
    assert set(build_static_order(dag.graph, "static_fifo")) == set(dag.expected_task_ids)


def test_rejects_cross_job_group_order_cycle():
    groups = [{"group_id": "g", "ranks": [0, 1]}, {"group_id": "h", "ranks": [0, 1]}]
    jobs = [
        {"job_id": "a", "nodes": [_comm("a0", 1, group="h"),
                                      _comm("a1", 0, ["a0"], group="g")]},
        {"job_id": "b", "nodes": [_comm("b0", 1, group="g"),
                                      _comm("b1", 0, ["b0"], group="h")]},
    ]
    with pytest.raises(ValueError, match="combined DAG/group ordering cycle"):
        parse_dag(_payload([], groups=groups, jobs=jobs, execution={}))


def test_jitter_sample_is_stable_by_seed_epoch_node_and_rank():
    args = (7, 3, "job", "compute", 0, 2.0, 0.5)
    first = sample_compute_duration(*args)
    assert sample_compute_duration(*args) == first
    assert 1.0 <= first <= 3.0
    assert sample_compute_duration(*args[:-2], 0.0, 0.5) == 0


def test_direct_graph_construction_uses_the_same_validation():
    collective = CollectiveSpec("all_reduce", 2, 8, "float32", (2,))
    group = GroupSpec(0, "g", (0, 1))
    graph = DagGraph((group,), (DagJob("job", (
        ComputeNode("root", (), 0.0),
        CommNode("comm", ("root",), "g", 0, 0.001, collective),
    )),))
    validate_graph(graph, world_size=2)
    invalid = DagGraph((group,), (DagJob("job", (
        ComputeNode("root", ("missing",), 0.0),
        CommNode("comm", (), "g", 0, 0.001, collective),
    )),))
    with pytest.raises(ValueError, match="missing deps"):
        validate_graph(invalid)


class _Handle:
    def __init__(self, task_id):
        self.task_id = task_id
        self.done = False
        self.state = type("State", (), {"value": "pending"})()

    def wait_host(self, timeout):
        return self.done


class _Runtime:
    def __init__(self):
        self.submitted = []
        self.declared = []
        self.handles = {}
        self.changed = threading.Event()
        self.failure = None
        self.aborts = []

    def submit(self, spec, binding, hint):
        self.submitted.append(spec.task_id)
        self.handles[spec.task_id] = _Handle(spec.task_id)
        self.changed.set()
        return self.handles[spec.task_id]

    def declare(self, spec, hint):
        self.declared.append((spec.task_id, hint.ready_after_s))
        self.changed.set()

    def complete(self, task_id):
        self.handles[task_id].done = True
        self.handles[task_id].state.value = "completed"
        self.changed.set()

    def abort(self, error, **kwargs):
        self.aborts.append((error, kwargs))
        if self.failure is None:
            self.failure = error


def _runner(raw, runtime, **kwargs):
    dag = parse_dag(raw)
    compute_fn = kwargs.pop("compute_fn", lambda _node, _stop: None)
    deadline = kwargs.pop("deadline", time.monotonic() + 2)
    return DagRunner(dag.graph, dag.graph.jobs[0], runtime, epoch=0, rank=0,
                     compute_fn=compute_fn, make_binding=lambda _comm: object(),
                     deadline=deadline, poll_interval=0.001, **kwargs)


def test_runner_reuses_precomputed_tail_map(monkeypatch):
    raw = _payload([_compute("root"), _comm("comm", 0, ["root"])])
    dag = parse_dag(raw)
    tails = compute_tails(dag.graph)
    monkeypatch.setattr(runner_module, "compute_tails",
                        lambda _graph: pytest.fail("precomputed tails should be reused"))
    runner = _runner(raw, _Runtime(), tails=tails)
    assert runner.tails is tails


def test_duplicate_completion_cannot_unlock_a_successor_twice():
    runner = _runner(_payload([_compute("root"), _comm("comm", 0, ["root"])]), _Runtime())
    runner.states["root"] = NodeState.RUNNING
    runner._complete("root")
    assert runner.remaining_deps["comm"] == 0
    assert runner.states["comm"] is NodeState.READY
    with pytest.raises(RuntimeError, match="completed from completed"):
        runner._complete("root")
    assert runner.remaining_deps["comm"] == 0


def test_submit_failure_aborts_once_and_never_runs_successor(monkeypatch):
    runtime = _Runtime()
    raw = _payload([_compute("root"), _comm("comm", 0, ["root"]),
                    _compute("successor", ["comm"])])
    runner = _runner(raw, runtime)

    def fail_submit(spec, _binding, _hint):
        runtime.submitted.append(spec.task_id)
        raise ValueError("submit failed")

    monkeypatch.setattr(runtime, "submit", fail_submit)
    with pytest.raises(ValueError, match="submit failed"):
        runner.run()
    assert len(runtime.aborts) == 1
    assert runtime.aborts[0][1]["node_id"] == "comm"
    assert runner.states["comm"] is NodeState.FAILED
    assert not any(event.get("node_id") == "successor" and event["kind"] == "compute_started"
                   for event in runner.event_log.events)


def test_deadline_timeout_reports_node_state_without_refreshing_budget():
    deadline = time.monotonic() - 1
    runtime = _Runtime()
    runner = _runner(_payload([_compute("root"), _comm("comm", 0, ["root"]),
                               _compute("successor", ["comm"])]),
                     runtime, deadline=deadline)
    with pytest.raises(TimeoutError, match="exceeded replay deadline"):
        runner.run()
    timeout = next(event for event in runner.event_log.events if event["kind"] == "job_timeout")
    assert timeout["states"] == {"root": "ready", "comm": "pending", "successor": "pending"}
    assert runner.deadline == deadline
    assert len(runtime.aborts) == 1


def _wait_for(predicate, runtime, timeout=1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        runtime.changed.wait(0.01)
        runtime.changed.clear()
    assert predicate(), "condition did not become true before timeout"


def test_chain_only_unblocks_after_handle_completion():
    runtime = _Runtime()
    raw = _payload([_compute("root"), _comm("first", 0, ["root"]),
                    _compute("next", ["first"]), _comm("last", 1, ["next"])])
    runner = _runner(raw, runtime)
    thread = threading.Thread(target=runner.run)
    thread.start()
    _wait_for(lambda: runtime.submitted == ["job/first"], runtime)
    assert "job/last" not in runtime.submitted
    runtime.complete("job/first")
    _wait_for(lambda: "job/last" in runtime.submitted, runtime)
    runtime.complete("job/last")
    thread.join(1)
    assert not thread.is_alive()
    assert runner.completed_node_ids == ("root", "first", "next", "last")


def test_fork_offers_every_ready_comm_while_serial_compute_runs_independently():
    runtime = _Runtime()
    compute_started = threading.Event()
    release_compute = threading.Event()
    second_compute_started = threading.Event()
    active_compute = 0
    peak_compute = 0
    compute_lock = threading.Lock()

    def compute(node, _stop):
        nonlocal active_compute, peak_compute
        with compute_lock:
            active_compute += 1
            peak_compute = max(peak_compute, active_compute)
        try:
            if node.node_id == "independent-a":
                compute_started.set()
                release_compute.wait(1)
            elif node.node_id == "independent-b":
                second_compute_started.set()
        finally:
            with compute_lock:
                active_compute -= 1

    groups = [{"group_id": "g", "ranks": [0, 1]}, {"group_id": "h", "ranks": [0, 1]}]
    nodes = [_compute("producer"), _comm("comm-a", 0, ["producer"], group="g"),
             _comm("comm-b", 0, ["producer"], group="h"),
             _compute("independent-a", ["producer"]), _compute("independent-b", ["producer"]),
             _compute("join", ["comm-a", "comm-b", "independent-a", "independent-b"]),
             _comm("comm-end", 1, ["join"], group="g")]
    raw = _payload(nodes, groups=groups, execution={f"job/{node['node_id']}": 0.0
                                                    for node in nodes if node["kind"] == "compute"})
    dag = parse_dag(raw)
    from runtime_comm_scheduler.runtime import EventLog
    events = EventLog("dag", 0)
    runner = DagRunner(dag.graph, dag.graph.jobs[0], runtime, epoch=0, rank=0,
                       make_binding=lambda _comm: object(), deadline=time.monotonic() + 2,
                       poll_interval=0.001, compute_fn=compute, event_log=events)
    thread = threading.Thread(target=runner.run)
    thread.start()
    _wait_for(lambda: set(runtime.submitted) == {"job/comm-a", "job/comm-b"}, runtime)
    assert compute_started.wait(1)
    assert not second_compute_started.is_set()
    assert peak_compute == 1
    runtime.complete("job/comm-a")
    _wait_for(lambda: any(event.get("task_id") == "job/comm-a"
                          and event["kind"] == "comm_completed_observed" for event in events.events), runtime)
    assert not any(event.get("node_id") == "join" and event["kind"] == "compute_started"
                   for event in events.events)
    runtime.complete("job/comm-b")
    release_compute.set()
    assert second_compute_started.wait(1)
    _wait_for(lambda: "job/comm-end" in runtime.submitted, runtime)
    runtime.complete("job/comm-end")
    thread.join(1)
    assert not thread.is_alive()
    assert peak_compute == 1


def test_lookahead_declares_only_safe_running_compute_frontier_once():
    runtime = _Runtime()
    started, release = threading.Event(), threading.Event()

    def compute(node, _stop):
        if node.node_id == "producer":
            started.set()
            release.wait(1)

    raw = _payload([
        _compute("root"),
        _comm("unrelated", 0, ["root"]),
        _compute("producer", ["root"], 0.2),
        _comm("comm", 1, ["producer"]),
        _compute("unrelated-compute", ["root"], 0.0),
    ])
    runner = _runner(raw, runtime, compute_fn=compute, enable_lookahead=True)
    thread = threading.Thread(target=runner.run)
    thread.start()
    assert started.wait(1)
    _wait_for(lambda: any(task == "job/comm" for task, _ in runtime.declared), runtime)
    task_id, prediction = next(item for item in runtime.declared if item[0] == "job/comm")
    assert task_id == "job/comm"
    assert prediction is not None and prediction >= 0
    declaration = next(event for event in runner.event_log.events
                       if event["kind"] == "comm_declared" and event.get("task_id") == task_id)
    assert declaration["prediction_base_us"] <= declaration["time_us"]
    assert declaration["predicted_ready_at_us"] == (
        declaration["prediction_base_us"] + round(declaration["ready_after_s"] * 1_000_000)
    )
    release.set()
    _wait_for(lambda: task_id in runtime.submitted, runtime)
    runtime.complete(task_id)
    runtime.complete("job/unrelated")
    thread.join(1)
    assert not thread.is_alive()
    assert sum(task == task_id for task, _ in runtime.declared) == 1


def test_compute_failure_aborts_and_does_not_submit_dependent_communication():
    runtime = _Runtime()

    def compute(*_args):
        raise RuntimeError("compute failed")

    raw = _payload([_compute("producer"), _comm("comm", 0, ["producer"])])
    with pytest.raises(RuntimeError, match="compute failed"):
        _runner(raw, runtime, compute_fn=compute).run()
    assert runtime.submitted == []
    assert len(runtime.aborts) == 1
