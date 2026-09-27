from __future__ import annotations

import threading

from runtime_comm_scheduler.runtime.telemetry import EventLog


def test_thread_sharded_event_log_merges_all_rank_local_events_by_timestamp():
    log = EventLog("runtime", 0, thread_sharded=True)
    start = threading.Barrier(5)

    def write(worker: int) -> None:
        start.wait()
        for index in range(40):
            log.record("probe", task_id=f"task-{worker}", worker=worker, index=index)

    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(4)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    events = log.as_dict()["events"]
    assert len(events) == 160
    assert {event["worker"] for event in events} == {0, 1, 2, 3}
    assert [event["time_us"] for event in events] == sorted(
        event["time_us"] for event in events
    )
    for worker in range(4):
        indices = [event["index"] for event in events if event["worker"] == worker]
        assert indices == list(range(40))


def test_disabled_event_log_emits_no_shards_or_events():
    log = EventLog("runtime", 0, enabled=False, thread_sharded=True)
    log.record("discarded")
    log.record_at("discarded", 1)
    assert log.as_dict()["events"] == []
    assert log.events == []
