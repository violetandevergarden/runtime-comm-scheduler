"""Run the scheduler-free JobPacer Phase 1 baseline.

This entry point fixes execution to bare mode while retaining the exact
workload, process-group, trace, and result schema used by Phase 2.
"""

from __future__ import annotations

import sys

try:
    from .run_replay import main as replay_main
except ImportError:  # pragma: no cover - direct script execution
    from run_replay import main as replay_main


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--mode" in arguments:
        raise ValueError("run_phase1 fixes --mode=bare; do not pass --mode")
    return replay_main(["--mode", "bare", *arguments])


if __name__ == "__main__":
    sys.exit(main())
