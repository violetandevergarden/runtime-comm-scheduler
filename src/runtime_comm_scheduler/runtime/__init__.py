"""Stage 3.1 online collective runtime.

The package is deliberately small: control-plane objects are serializable and
rank-local bindings never cross the coordinator boundary.
"""

from .coordinator import CoordinatorState, CoordinatorError
from .executor import DirectExecutor, GlooExecutor, WorkIsCompletedProbe
from .handle import RuntimeHandle, HandleState
from .model import (
    CollectiveSpec,
    GroupSpec,
    LocalBinding,
    TaskHint,
    TaskSpec,
)
from .policy import (
    Action,
    Anticipated,
    Candidate,
    Dispatch,
    Idle,
    PolicySnapshot,
    Wait,
    make_policy,
)
from .runtime import RankRuntime
from .telemetry import EventLog, monotonic_us

__all__ = [
    "Action",
    "Anticipated",
    "Candidate",
    "CollectiveSpec",
    "CoordinatorError",
    "CoordinatorState",
    "DirectExecutor",
    "Dispatch",
    "EventLog",
    "GlooExecutor",
    "GroupSpec",
    "HandleState",
    "Idle",
    "LocalBinding",
    "PolicySnapshot",
    "RankRuntime",
    "RuntimeHandle",
    "TaskHint",
    "TaskSpec",
    "Wait",
    "WorkIsCompletedProbe",
    "make_policy",
    "monotonic_us",
]
