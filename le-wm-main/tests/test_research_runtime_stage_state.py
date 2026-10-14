import json

from experiments.research_runtime.stage_state import (
    EvidenceStatus,
    ExecutionStatus,
    StageStateStore,
)


def test_stage_state_persists_execution_and_evidence_separately(tmp_path):
    path = tmp_path / "stage_status.json"
    state = StageStateStore(path, stage="demo", input_fingerprint="abc")
    assert json.loads(path.read_text())["execution_status"] == "planned"
    state.update(ExecutionStatus.RUNNING)
    state.update(ExecutionStatus.COMPLETED, output_count=1)
    state.mark_evidence(EvidenceStatus.BLOCKED, reason="no heldout consequence")
    payload = json.loads(path.read_text())
    assert payload["execution_status"] == "completed"
    assert payload["evidence_status"] == "blocked"
    assert payload["output_count"] == 1
