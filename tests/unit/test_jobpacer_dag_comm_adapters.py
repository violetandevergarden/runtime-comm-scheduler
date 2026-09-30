from __future__ import annotations

import sys
import threading
import time
from datetime import timedelta
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.jobpacer.runtime import dag_comm_adapters as adapters
from runtime_comm_scheduler.runtime import (
    CollectiveSpec, EventLog, GroupSpec, HandleState, LocalBinding, TaskSpec,
)
from runtime_comm_scheduler import TaskKey
from runtime_comm_scheduler.dag import CommNode, ComputeNode, DagGraph, DagJob


def test_epoch_failure_signal_preserves_each_ranks_first_cause(monkeypatch):
    class Store:
        def __init__(self):
            self.values = {}

        def compare_set(self, key, expected, desired):
            current = self.values.get(key, "")
            if current == expected:
                self.values[key] = desired
                current = desired
            return current.encode()

        def check(self, keys):
            return all(key in self.values for key in keys)

        def get(self, key):
            return self.values[key].encode()

        def set(self, key, value):
            self.values[key] = value.decode() if isinstance(value, bytes) else value

    store = Store()
    monkeypatch.setattr(adapters.dist.distributed_c10d, "_get_default_store", lambda: store)
    rank0 = adapters.EpochFailureSignal(4, 0, 2)
    rank1 = adapters.EpochFailureSignal(4, 1, 2)

    rank0.publish(RuntimeError("injected DAG binding failure"))
    rank0.publish(RuntimeError("follow-on cancellation"))

    assert "injected DAG binding failure" in str(rank0.failure())
    assert "injected DAG binding failure" in str(rank1.failure())

    rank1.publish(RuntimeError("injected completion probe failure"))
    assert "injected DAG binding failure" in str(rank0.failure())
    assert "injected DAG binding failure" in str(rank1.failure())


def test_epoch_failure_signal_waits_for_rank_teardown_acknowledgements(monkeypatch):
    class Store:
        def __init__(self):
            self.values = {}

        def compare_set(self, key, expected, desired):
            current = self.values.get(key, "")
            if current == expected:
                self.values[key] = desired
                current = desired
            return current.encode()

        def check(self, keys):
            return all(key in self.values for key in keys)

        def get(self, key):
            return self.values[key].encode()

        def set(self, key, value):
            self.values[key] = value.decode() if isinstance(value, bytes) else value

    store = Store()
    monkeypatch.setattr(adapters.dist.distributed_c10d, "_get_default_store", lambda: store)
    signals = [adapters.EpochFailureSignal(5, rank, 2) for rank in (0, 1)]
    signals[0].publish(RuntimeError("root failure"))
    signals[0].publish_teardown_ready()
    assert not signals[0].wait_for_teardown_ready(time.monotonic() - 1.0)
    signals[1].publish_teardown_ready()
    assert signals[0].wait_for_teardown_ready(time.monotonic() + 1.0)
    assert signals[1].wait_for_teardown_ready(time.monotonic() + 1.0)


def test_epoch_failure_signal_rejects_a_different_bare_order_on_one_rank(monkeypatch):
    class Store:
        def __init__(self):
            self.values = {}

        def compare_set(self, key, expected, desired):
            current = self.values.get(key, "")
            if current == expected:
                self.values[key] = desired
                current = desired
            return current.encode()

        def check(self, keys):
            return all(key in self.values for key in keys)

        def get(self, key):
            return self.values[key].encode()

        def set(self, key, value):
            self.values[key] = value.decode()

    store = Store()
    monkeypatch.setattr(adapters.dist.distributed_c10d, "_get_default_store", lambda: store)
    signals = [adapters.EpochFailureSignal(8, rank, 2) for rank in (0, 1)]
    errors = []

    def verify(rank):
        try:
            signals[rank].verify_common_contract(
                "bare-order", {"sequence": ["a", "b"] if rank == 0 else ["b", "a"]},
                time.monotonic() + 1.0,
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=verify, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)
    assert all(not thread.is_alive() for thread in threads)
    assert len(errors) == 2
    assert all(isinstance(error, ValueError) for error in errors)

class _Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


class _BoundWork:
    is_bound = True

    def __init__(self, clock: _Clock, timeouts: list[float]):
        self.clock, self.timeouts = clock, timeouts

    def wait(self, *, timeout):
        if isinstance(timeout, timedelta):
            timeout = timeout.total_seconds()
        self.timeouts.append(timeout)
        self.clock.now += min(0.2, timeout)


def _bare_graph():
    collective = CollectiveSpec("all_reduce", 1, 4, "float32", (1,))
    return DagGraph(
        (GroupSpec(0, "g0", (0, 1)), GroupSpec(0, "g1", (0, 1))),
        (
            DagJob("job0", (
                ComputeNode("root", (), 0.0),
                CommNode("g0-seq0", ("root",), "g0", 0, 0.01, collective),
                CommNode("g0-seq1", ("root",), "g0", 1, 0.01, collective),
            )),
            DagJob("job1", (
                ComputeNode("root", (), 0.0),
                CommNode("g1-seq0", ("root",), "g1", 0, 0.01, collective),
            )),
        ),
    )


def test_bare_order_is_stable_profile_independent_and_includes_submit_acceptance_gates():
    graph = _bare_graph()
    order = adapters.build_bare_order(graph, input_hash="frozen-input")
    changed_estimates = replace(
        graph,
        jobs=tuple(
            replace(job, nodes=tuple(
                replace(node, estimated_duration_s=9.0)
                if isinstance(node, ComputeNode) else replace(node, estimated_comm_s=8.0)
                for node in job.nodes
            ))
            for job in graph.jobs
        ),
    )
    profiled_order = adapters.build_bare_order(changed_estimates, input_hash="frozen-input")
    assert order.sequence == (
        "job0/g0-seq0", "job1/g1-seq0", "job0/g0-seq1",
    )
    assert order.job_order == ("job0", "job1")
    assert profiled_order.sequence == order.sequence
    assert profiled_order.digest == order.digest

    collective = next(node.collective for job in graph.jobs for node in job.nodes
                      if isinstance(node, CommNode))
    gate_graph = DagGraph(
        (GroupSpec(0, "first-group", (0, 1)), GroupSpec(0, "second-group", (0, 1))),
        (DagJob("gated", (
            ComputeNode("compute", (), 0.0),
            CommNode("after-compute", ("compute",), "first-group", 0, 0.01, collective),
            CommNode("gate-request", (), "second-group", 0, 0.01, collective),
        )),),
    )
    ungated = adapters.build_bare_order(gate_graph, input_hash="gated-input")
    gated = adapters.build_bare_order(
        gate_graph, submit_after={"gated/compute": ("gated/gate-request",)},
        input_hash="gated-input",
    )
    assert ungated.sequence == ("gated/gate-request", "gated/after-compute")
    assert gated.sequence == ("gated/gate-request", "gated/after-compute")

    cyclic_gate_graph = DagGraph(
        (GroupSpec(0, "cycle-group", (0, 1)),),
        (DagJob("cycle", (
            ComputeNode("compute", (), 0.0),
            CommNode("first", ("compute",), "cycle-group", 0, 0.01, collective),
            CommNode("second", (), "cycle-group", 1, 0.01, collective),
        )),),
    )
    with pytest.raises(ValueError, match="conflicts with DAG/submit_after"):
        adapters.build_bare_order(
            cyclic_gate_graph, submit_after={"cycle/compute": ("cycle/second",)},
        )


def test_bare_dispatcher_uses_same_order_for_skewed_ranks_and_allows_inflight_overlap(monkeypatch):
    graph = _bare_graph()
    # Rank 0 is not a member of job1's communicator. Its issue stream is the
    # rank-local projection of the common frozen task order.
    graph = replace(graph, groups=(graph.groups[0], GroupSpec(0, "g1", (1,))))
    order = adapters.build_bare_order(graph, input_hash="shared-input")
    receipts_by_rank = {0: [], 1: []}
    launch_events = {0: threading.Event(), 1: threading.Event()}

    class Receipt:
        def __init__(self):
            self.done = False

        def is_completed(self):
            return self.done

    class Executor:
        def __init__(self, device):
            self.rank = int(str(device).split(":")[-1])

        def launch(self, binding):
            receipt = Receipt()
            receipts_by_rank[self.rank].append((binding.tensor, receipt))
            expected = 2 if self.rank == 0 else 3
            if len(receipts_by_rank[self.rank]) == expected:
                launch_events[self.rank].set()
            return receipt

    class RecordingLog:
        def __init__(self, head_wait):
            self.rows = []
            self.head_wait = head_wait

        def record(self, kind, **fields):
            self.rows.append({"kind": kind, **fields})
            if kind == "bare_order_head_wait":
                self.head_wait.set()

    head_wait = threading.Event()
    monkeypatch.setattr(adapters, "CudaCollectiveExecutor", Executor)
    logs = {0: RecordingLog(head_wait), 1: RecordingLog(threading.Event())}
    runtimes = {
        rank: adapters.BareDagAdapter(
            graph, rank=rank, group_ranks={"g0": (0, 1), "g1": (1,)}, device=f"cuda:{rank}",
            order=order, submit_after={}, deadline_s=2.0, poll_interval_s=0.001,
            event_log=logs[rank],
        )
        for rank in (0, 1)
    }
    nodes = {
        f"{job.job_id}/{node.node_id}": node
        for job in graph.jobs for node in job.nodes if isinstance(node, CommNode)
    }

    def submit(rank, task_id):
        node = nodes[task_id]
        spec = TaskSpec(0, task_id.split("/", 1)[0], task_id, node.group_id,
                        node.group_seq, node.collective)
        binding = LocalBinding(task_id, object(), lambda: None, device=f"cuda:{rank}")
        return runtimes[rank].submit(spec, binding, SimpleNamespace())

    try:
        for runtime in runtimes.values():
            runtime.start()

        # Rank 0 has a later request ready first. Acceptance returns while the
        # dispatcher waits for the common sequence head.
        rank0_later = submit(0, "job0/g0-seq1")
        assert head_wait.wait(timeout=1.0)
        assert rank0_later.state is HandleState.PENDING
        assert receipts_by_rank[0] == []

        # Rank 1 reaches the head first, but may issue subsequent work before
        # those collectives physically complete.
        rank1_first = submit(1, "job0/g0-seq0")
        rank1_last = submit(1, "job1/g1-seq0")
        rank1_middle = submit(1, "job0/g0-seq1")
        rank0_first = submit(0, "job0/g0-seq0")

        assert launch_events[0].wait(timeout=1.0)
        assert launch_events[1].wait(timeout=1.0)
        expected_rank0 = ("job0/g0-seq0", "job0/g0-seq1")
        expected_rank1 = order.sequence
        assert runtimes[0].local_order == expected_rank0
        assert runtimes[1].local_order == expected_rank1
        assert runtimes[0].launch_order == list(expected_rank0)
        assert runtimes[1].launch_order == list(expected_rank1)
        assert runtimes[0].peak_inflight == 2
        assert runtimes[1].peak_inflight == 3
        assert all(handle.state is HandleState.BOUND for handle in (
            rank0_later, rank0_first, rank1_first, rank1_middle, rank1_last,
        ))

        for records in receipts_by_rank.values():
            for _task_id, receipt in records:
                receipt.done = True
        for rank, runtime in runtimes.items():
            runtime.finish_epoch(timeout=1.0)
            assert all(handle.wait_host(0) for handle in (
                (rank0_later, rank0_first) if rank == 0 else
                (rank1_first, rank1_middle, rank1_last)
            ))
    finally:
        for runtime in runtimes.values():
            runtime.close()


def test_bare_completion_probe_failure_aborts_pending_handles(monkeypatch):
    graph = _bare_graph()
    order = adapters.build_bare_order(graph)

    class Receipt:
        def is_completed(self):
            return False

    class Executor:
        def __init__(self, _device):
            pass

        def launch(self, _binding):
            return Receipt()

    class Probe:
        def is_completed(self, _receipt):
            raise RuntimeError("probe failed")

    monkeypatch.setattr(adapters, "CudaCollectiveExecutor", Executor)
    runtime = adapters.BareDagAdapter(
        graph, rank=0, group_ranks={"g0": (0, 1), "g1": (0, 1)}, device="cuda:0",
        order=order, submit_after={},
        deadline_s=2.0, poll_interval_s=0.002, event_log=EventLog("runtime", 0),
        completion_probe=Probe(),
    )
    runtime.start()
    node = next(node for node in graph.jobs[0].nodes if isinstance(node, CommNode))
    handle = runtime.submit(
        TaskSpec(0, "job0", "job0/g0-seq0", "g0", 0, node.collective),
        LocalBinding("tensor", object(), lambda: None, device="cuda:0"), SimpleNamespace(),
    )
    deadline = time.monotonic() + 1.0
    while runtime.failure is None and time.monotonic() < deadline:
        time.sleep(0.002)
    assert isinstance(runtime.failure, RuntimeError)
    with pytest.raises(RuntimeError, match="probe failed"):
        handle.wait_host(1.0)
    runtime.close()


def test_bare_missing_order_head_returns_later_handle_then_fails_by_deadline(monkeypatch):
    graph = _bare_graph()
    order = adapters.build_bare_order(graph)

    class Executor:
        def __init__(self, _device):
            pass

        def launch(self, _binding):
            raise AssertionError("dispatcher must not skip the absent order head")

    monkeypatch.setattr(adapters, "CudaCollectiveExecutor", Executor)
    runtime = adapters.BareDagAdapter(
        graph, rank=0, group_ranks={"g0": (0, 1), "g1": (0, 1)}, device="cuda:0",
        order=order, submit_after={}, deadline_s=1.0, poll_interval_s=0.002,
        event_log=EventLog("runtime", 0),
    )
    runtime.start()
    node = next(node for node in graph.jobs[0].nodes
                if isinstance(node, CommNode) and node.node_id == "g0-seq1")
    handle = runtime.submit(
        TaskSpec(0, "job0", "job0/g0-seq1", node.group_id, node.group_seq, node.collective),
        LocalBinding("tensor", object(), lambda: None, device="cuda:0"), SimpleNamespace(),
    )
    assert handle.state is HandleState.PENDING
    with pytest.raises(TimeoutError, match="did not drain"):
        runtime.finish_epoch(timeout=0.05)
    with pytest.raises(TimeoutError):
        handle.wait_host(0)
    runtime.close()


def test_bare_close_fails_accepted_handle_when_dispatcher_waits_for_order_head(monkeypatch):
    graph = _bare_graph()
    order = adapters.build_bare_order(graph)
    head_wait = threading.Event()

    class RecordingLog:
        def record(self, kind, **_fields):
            if kind == "bare_order_head_wait":
                head_wait.set()

    class Executor:
        def __init__(self, _device):
            pass

        def launch(self, _binding):
            raise AssertionError("dispatcher must remain at the missing order head")

    monkeypatch.setattr(adapters, "CudaCollectiveExecutor", Executor)
    runtime = adapters.BareDagAdapter(
        graph, rank=0, group_ranks={"g0": (0, 1), "g1": (0, 1)}, device="cuda:0",
        order=order, submit_after={}, deadline_s=10.0, poll_interval_s=0.001,
        event_log=RecordingLog(),
    )
    runtime.start()
    node = next(node for node in graph.jobs[0].nodes
                if isinstance(node, CommNode) and node.node_id == "g0-seq1")
    handle = runtime.submit(
        TaskSpec(0, "job0", "job0/g0-seq1", node.group_id, node.group_seq, node.collective),
        LocalBinding("tensor", object(), lambda: None, device="cuda:0"), SimpleNamespace(),
    )
    assert head_wait.wait(timeout=1.0)
    assert handle.state is HandleState.PENDING

    waiter_started = threading.Event()
    waiter_finished = threading.Event()
    wait_errors = []

    def wait_without_timeout():
        waiter_started.set()
        try:
            handle.wait_host()
        except BaseException as exc:
            wait_errors.append(exc)
        finally:
            waiter_finished.set()

    waiter = threading.Thread(target=wait_without_timeout)
    waiter.start()
    assert waiter_started.wait(timeout=1.0)
    runtime.close()
    assert waiter_finished.wait(timeout=1.0)
    waiter.join(timeout=1.0)

    assert not waiter.is_alive()
    assert handle.state is HandleState.FAILED
    assert len(wait_errors) == 1
    assert wait_errors[0] is runtime.failure
    assert isinstance(runtime.failure, RuntimeError)
    assert "closed before the epoch drained" in str(runtime.failure)
    assert not runtime._dispatcher.is_alive()
    assert not runtime._thread.is_alive()


def test_bare_launch_failure_wakes_an_already_accepted_handle(monkeypatch):
    graph = _bare_graph()
    order = adapters.build_bare_order(graph)

    class Executor:
        def __init__(self, _device):
            pass

        def launch(self, _binding):
            raise RuntimeError("injected bare launch failure")

    monkeypatch.setattr(adapters, "CudaCollectiveExecutor", Executor)
    runtime = adapters.BareDagAdapter(
        graph, rank=0, group_ranks={"g0": (0, 1), "g1": (0, 1)}, device="cuda:0",
        order=order, submit_after={}, deadline_s=2.0, poll_interval_s=0.002,
        event_log=EventLog("runtime", 0),
    )
    runtime.start()
    node = next(node for node in graph.jobs[0].nodes
                if isinstance(node, CommNode) and node.node_id == "g0-seq0")
    handle = runtime.submit(
        TaskSpec(0, "job0", "job0/g0-seq0", node.group_id, node.group_seq, node.collective),
        LocalBinding("tensor", object(), lambda: None, device="cuda:0"), SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="injected bare launch failure"):
        handle.wait_host(1.0)
    with pytest.raises(RuntimeError, match="injected bare launch failure"):
        runtime.finish_epoch(timeout=0.1)
    runtime.close()


def test_old_adapter_uses_one_deadline_for_each_work_and_scheduler_close(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(adapters.time, "monotonic", clock.monotonic)
    work_timeouts: list[float] = []
    close_timeouts: list[float] = []

    class Scheduler:
        def close(self, *, timeout):
            close_timeouts.append(timeout)

    adapter = object.__new__(adapters.LegacySchedulerDagAdapter)
    adapter._handles = {
        f"task-{index}": SimpleNamespace(work=_BoundWork(clock, work_timeouts))
        for index in range(2)
    }
    adapter._handles_lock = threading.Lock()
    adapter._closed = False
    adapter.failure_signal = None
    adapter.failure_drain_timeout_s = 10.0
    adapter._replay_deadline = 100.6
    adapter._cleanup_deadline = None
    adapter.event_log = EventLog("runtime", 0)
    adapter.scheduler = Scheduler()

    adapter.abort(RuntimeError("injected"), deadline=100.6)

    assert work_timeouts == pytest.approx([0.6, 0.4])
    assert close_timeouts == pytest.approx([0.2])
    assert clock.now == pytest.approx(100.4)


def test_old_adapter_abort_includes_submit_accepted_while_failure_starts():
    entered_submit = threading.Event()
    release_submit = threading.Event()
    abort_started = threading.Event()
    waited: list[float] = []
    close_timeouts: list[float] = []
    deadline = adapters.time.monotonic() + 2.0
    key = TaskKey(0, 0, "test", "g", 0, 0, 0)
    spec = TaskSpec(0, "job", "job/comm", "g", 0,
                    CollectiveSpec("all_reduce", 1, 4, "float32", (1,)))

    class Work:
        is_bound = True

        def wait(self, *, timeout):
            waited.append(timeout)

    class Scheduler:
        def submit(self, _intent):
            entered_submit.set()
            assert release_submit.wait(2.0)
            return Work()

        def close(self, *, timeout):
            close_timeouts.append(timeout)

    adapter = object.__new__(adapters.LegacySchedulerDagAdapter)
    adapter.task_keys = {spec.task_id: key}
    adapter.plan = SimpleNamespace(metadata=lambda _key: ("all_reduce", 4))
    adapter.scheduler = Scheduler()
    adapter.probe = object()
    adapter._handles = {}
    adapter._handles_lock = threading.Lock()
    adapter._closed = False
    adapter.failure_signal = None
    adapter.failure_drain_timeout_s = 2.0
    adapter._replay_deadline = 101.0
    adapter._cleanup_deadline = None
    adapter.event_log = EventLog("runtime", 0)
    binding = LocalBinding(object(), object(), lambda: None, device="cuda:0")
    submit_error: list[BaseException] = []

    def submit():
        try:
            adapter.submit(spec, binding, SimpleNamespace())
        except BaseException as exc:
            submit_error.append(exc)

    def abort():
        abort_started.set()
        adapter.abort(RuntimeError("injected"), deadline=deadline)

    submit_thread = threading.Thread(target=submit)
    submit_thread.start()
    assert entered_submit.wait(2.0)
    abort_thread = threading.Thread(target=abort)
    abort_thread.start()
    assert abort_started.wait(2.0)
    release_submit.set()
    submit_thread.join(2.0)
    abort_thread.join(2.0)

    assert not submit_thread.is_alive()
    assert not abort_thread.is_alive()
    assert submit_error == []
    assert len(adapter._handles) == 1
    assert len(waited) == 1 and 1.9 < waited[0] <= 2.0
    assert len(close_timeouts) == 1
    assert 0 <= close_timeouts[0] <= waited[0]


def test_raw_adapter_joins_inflight_binding_then_drains_all_works_to_same_deadline(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(adapters.time, "monotonic", clock.monotonic)
    work_timeouts: list[float] = []

    class Handle:
        def __init__(self, receipt=None):
            self.state = HandleState.BOUND
            self.receipt = receipt

        def fail(self, _error):
            self.state = HandleState.FAILED

    def receipt():
        return SimpleNamespace(backend_work=_BoundWork(clock, work_timeouts))

    late_binding = Handle()

    class Dispatcher:
        def __init__(self):
            self.join_timeouts = []

        def join(self, timeout):
            self.join_timeouts.append(timeout)
            clock.now += min(0.2, timeout)
            # Model launch() returning and binding its Work while abort joins.
            late_binding.receipt = receipt()

    dispatcher = Dispatcher()
    adapter = object.__new__(adapters.RawOrderedDagAdapter)
    adapter.failure_signal = None
    adapter.failure_drain_timeout_s = 10.0
    adapter._replay_deadline = 100.7
    adapter._cleanup_deadline = None
    adapter._condition = threading.Condition()
    adapter._failure = None
    adapter._stopping = False
    adapter._thread = dispatcher
    adapter._handles = {
        "task-0": Handle(receipt()),
        "task-1": Handle(receipt()),
        "task-2-in-launch": late_binding,
    }
    adapter.event_log = EventLog("runtime", 0)

    adapter.abort(RuntimeError("injected"), deadline=100.7)

    assert dispatcher.join_timeouts == pytest.approx([0.7])
    assert work_timeouts == pytest.approx([0.5, 0.3, 0.1])
    assert clock.now == pytest.approx(100.7)


@pytest.mark.parametrize("probe_fails", [False, True])
def test_raw_ordered_dispatch_records_physical_launch_peak_and_bounds_probe_failure(probe_fails):
    task_id = "job-0/comm-0"

    class Receipt:
        completion_source = "test-device-event"
        backend_work = None

        def is_completed(self):
            return True

    class Executor:
        def launch(self, _binding):
            return Receipt()

    class Probe:
        def is_completed(self, receipt):
            if probe_fails:
                raise RuntimeError("injected completion probe failure")
            return receipt.is_completed()

    handle = adapters._OrderedHandle(task_id, binding=object())
    adapter = object.__new__(adapters.RawOrderedDagAdapter)
    adapter.graph = object()
    adapter.order = (task_id,)
    adapter.rank = 0
    adapter.group_ranks = {"g0": (0, 1)}
    adapter.event_log = EventLog("runtime", 0)
    adapter.failure_signal = None
    adapter.completion_probe = Probe()
    adapter.failure_drain_timeout_s = 1.0
    adapter._replay_deadline = time.monotonic() + 1.0
    adapter._cleanup_deadline = None
    adapter.executor = Executor()
    adapter._condition = threading.Condition()
    adapter._handles = {task_id: handle}
    adapter._failure = None
    adapter._deadline = adapter._replay_deadline
    adapter._stopping = False
    adapter._started = True
    adapter._closed = False
    adapter._launch_order = []
    adapter._inflight = 0
    adapter.peak_inflight = 0
    adapter._comm_by_task = {task_id: SimpleNamespace(group_id="g0")}
    adapter._thread = threading.current_thread()

    adapter._run()

    if probe_fails:
        assert isinstance(adapter.failure, RuntimeError)
        assert handle.state is HandleState.FAILED
        assert adapter.launch_order == [task_id]
    else:
        assert adapter.failure is None
        assert handle.state is HandleState.COMPLETED
        assert adapter.launch_order == [task_id]
        assert adapter.peak_inflight == 1
        assert adapter._inflight == 0


def test_raw_ordered_submit_returns_while_static_head_waits_for_another_job():
    head, later = "job-0/comm-0", "job-1/comm-0"
    blocked = threading.Event()

    class RecordingLog:
        def __init__(self):
            self.rows = []

        def record(self, kind, **fields):
            self.rows.append({"kind": kind, **fields})
            if kind == "raw_ordered_static_head_blocked":
                blocked.set()

    class Executor:
        def launch(self, _binding):
            return SimpleNamespace(completion_source="test", is_completed=lambda: True)

    adapter = object.__new__(adapters.RawOrderedDagAdapter)
    adapter.order = (head, later)
    adapter.rank = 0
    adapter.group_ranks = {"g0": (0, 1)}
    adapter.event_log = RecordingLog()
    adapter.failure_signal = None
    adapter.completion_probe = None
    adapter.failure_drain_timeout_s = 1.0
    adapter._replay_deadline = time.monotonic() + 2.0
    adapter._cleanup_deadline = None
    adapter.deadline_s = 2.0
    adapter.poll_interval_s = 0.001
    adapter.executor = Executor()
    adapter._condition = threading.Condition()
    adapter._handles = {}
    adapter._failure = None
    adapter._deadline = adapter._replay_deadline
    adapter._stopping = False
    adapter._started = True
    adapter._closed = False
    adapter._launch_order = []
    adapter._inflight = 0
    adapter.peak_inflight = 0
    adapter.static_head_wait_s = 0.0
    adapter._comm_by_task = {
        head: SimpleNamespace(group_id="g0"), later: SimpleNamespace(group_id="g0")}
    adapter._thread = threading.Thread(target=adapter._run)
    adapter._thread.start()
    try:
        later_handle = adapter.submit(
            SimpleNamespace(task_id=later, group_id="g0"), binding=object(), hint=None)
        assert blocked.wait(timeout=1.0), "dispatcher did not report the static head wait"
        head_handle = adapter.submit(
            SimpleNamespace(task_id=head, group_id="g0"), binding=object(), hint=None)
        adapter._thread.join(timeout=1.0)
        assert not adapter._thread.is_alive()
        assert later_handle.state is HandleState.COMPLETED
        assert head_handle.state is HandleState.COMPLETED
        assert adapter.launch_order == [head, later]
        kinds = [row["kind"] for row in adapter.event_log.rows]
        assert kinds.index("raw_ordered_submit_return") < kinds.index("raw_ordered_static_head_blocked")
        assert any(row["kind"] == "raw_ordered_static_head_released" for row in adapter.event_log.rows)
    finally:
        if adapter._thread.is_alive():
            adapter.abort(RuntimeError("test cleanup"), deadline=time.monotonic() + 0.5)
            adapter._thread.join(timeout=1.0)
