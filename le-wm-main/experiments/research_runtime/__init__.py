"""Small, dependency-free execution contracts shared by research workflows.

The runtime records reproducibility and execution state.  It deliberately does
not decide whether a scientific claim is supported.
"""

from .artifacts import artifact_is_valid, atomic_write_json, load_json
from .fingerprint import build_run_identity, fingerprint_file, fingerprint_inputs
from .stage_state import EvidenceStatus, ExecutionStatus, StageStateStore

__all__ = [
    "EvidenceStatus",
    "ExecutionStatus",
    "StageStateStore",
    "artifact_is_valid",
    "atomic_write_json",
    "build_run_identity",
    "fingerprint_file",
    "fingerprint_inputs",
    "load_json",
]
