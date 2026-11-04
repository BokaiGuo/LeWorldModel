import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import audit_reacher_pc_constrained_saved as audit
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup


def test_real_saved_row_rejects_action_and_selector_tampering():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "results/reacher_pc_constrained_cuda_20261102_v1/report.json").read_text())
    pf = json.loads((root / "results/reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json").read_text())
    mean, std = setup._load_action_stats(Path(pf["action_stats_path"]), "swm/ReacherDMControl-v0")
    original = report["rows"][0]
    assert audit.audit_row(original, mean, std)["selected"] is None
    row = copy.deepcopy(original)
    row["prefix_raw"][1][0][0] += 0.1
    with pytest.raises(ValueError, match="binding"):
        audit.audit_row(row, mean, std)
    row = copy.deepcopy(original)
    row["compiler_receipt"]["proposal_rows"][1]["eligible"] = True
    with pytest.raises(ValueError, match="selector"):
        audit.audit_row(row, mean, std)
