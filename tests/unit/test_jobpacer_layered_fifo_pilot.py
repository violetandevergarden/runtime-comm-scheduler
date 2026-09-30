from __future__ import annotations

from examples.jobpacer.diagnostics.layered_fifo_pilot import _head_wait


def test_pilot_head_wait_reports_uninstrumented_old_path_as_missing():
    raw = {"ranks": [{"rank": 0, "decision_records": []},
                     {"rank": 1, "decision_records": []}]}
    assert _head_wait(raw, "old-static-fifo") == {
        "source": "old plan adapter does not expose static-head wait",
        "seconds": None, "measured": False,
    }


def test_pilot_head_wait_sums_only_static_head_intervals_for_new_runtime():
    raw = {"ranks": [{"rank": 0, "decision_records": [
        {"kind": "idle_interval", "reason": "STATIC_HEAD_BLOCKED", "duration": 0.02},
        {"kind": "idle_interval", "reason": "CAPACITY_FULL", "duration": 0.03},
    ]}]}
    row = _head_wait(raw, "new-static-fifo")
    assert row["measured"] is True
    assert row["seconds"] == 0.02
    assert row["interval_count"] == 1


def test_pilot_head_wait_sums_bare_head_release_durations_per_rank():
    raw = {"ranks": [
        {"rank": 0, "runtime_events": [
            {"kind": "bare_order_head_released", "order_wait_s": 0.01},
            {"kind": "bare_order_head_released", "order_wait_s": 0.02},
        ]},
        {"rank": 1, "runtime_events": []},
    ]}
    row = _head_wait(raw, "bare-ordered")
    assert row["measured"] is True
    assert row["by_rank"] == [
        {"rank": 0, "seconds": 0.03, "released_heads": 2, "measured": True},
        {"rank": 1, "seconds": 0.0, "released_heads": 0, "measured": True},
    ]
