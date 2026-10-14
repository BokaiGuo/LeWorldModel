"""Persistent execution/evidence states for resumable research stages."""

from __future__ import annotations

import signal
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json


class ExecutionStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    INTERRUPTED = "interrupted"


class EvidenceStatus(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    INSUFFICIENT = "insufficient"
    PARTIAL = "partial"
    NEGATIVE = "negative"
    SUPPORTED = "supported"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageStateStore:
    """Write stage state immediately so failures remain resumable and visible."""

    def __init__(self, path: Path, *, stage: str, input_fingerprint: str | None = None):
        self.path = Path(path)
        self.stage = stage
        self.input_fingerprint = input_fingerprint
        self.payload: dict[str, Any] = {
            "stage": stage,
            "execution_status": ExecutionStatus.PLANNED.value,
            "evidence_status": EvidenceStatus.NOT_EVALUATED.value,
            "input_fingerprint": input_fingerprint,
            "started_at": None,
            "finished_at": None,
            "error": None,
        }
        self._write()

    def _write(self) -> None:
        atomic_write_json(self.path, self.payload)

    def update(self, status: ExecutionStatus, *, error: str | None = None, **extra: Any) -> dict[str, Any]:
        self.payload["execution_status"] = status.value
        self.payload["error"] = error
        self.payload.update(extra)
        if status is ExecutionStatus.RUNNING:
            self.payload["started_at"] = _now()
        if status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.BLOCKED, ExecutionStatus.INTERRUPTED}:
            self.payload["finished_at"] = _now()
        self._write()
        return dict(self.payload)

    def mark_evidence(self, status: EvidenceStatus, *, reason: str | None = None) -> dict[str, Any]:
        self.payload["evidence_status"] = status.value
        if reason is not None:
            self.payload["evidence_reason"] = reason
        self._write()
        return dict(self.payload)

    def interrupted(self, signum: int, _frame: Any) -> None:
        self.update(ExecutionStatus.INTERRUPTED, error=f"signal:{signal.Signals(signum).name}")
        raise KeyboardInterrupt
