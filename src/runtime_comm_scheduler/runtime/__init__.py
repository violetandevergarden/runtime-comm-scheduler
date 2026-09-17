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
    BoundedLookaheadPolicy,
    Candidate,
    Dispatch,
    Done,
    FifoPolicy,
    Idle,
    LongestTailFirstPolicy,
    Policy,
    PolicySnapshot,
    StaticPolicy,
    Wait,
    make_policy,
)
from .runtime import RankRuntime
from .telemetry import EventLog, monotonic_us

__all__ = [
    "Action",
    "Anticipated",
    "BoundedLookaheadPolicy",
    "Candidate",
    "CollectiveSpec",
    "CoordinatorError",
    "CoordinatorState",
    "DirectExecutor",
    "Dispatch",
    "EventLog",
    "FifoPolicy",
    "GlooExecutor",
    "GroupSpec",
    "HandleState",
    "Idle",
    "LongestTailFirstPolicy",
    "LocalBinding",
    "Policy",
    "PolicySnapshot",
    "RankRuntime",
    "RuntimeHandle",
    "StaticPolicy",
    "TaskHint",
    "TaskSpec",
    "Wait",
    "WorkIsCompletedProbe",
    "make_policy",
    "monotonic_us",
]
