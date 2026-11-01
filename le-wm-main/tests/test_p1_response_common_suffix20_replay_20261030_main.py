import copy
import hashlib

import pytest

from experiments.spectral_reachability import p1_response_common_suffix20_replay_20261030_main as audit


def example():
    raw = b"\0" * 40
    candidates = [{"stage_Q": float(i), "raw_float32_bits": "00" * 80} for i in range(20)]
    report, selection, rows = [], [], []
    for i in range(8):
        arm = {"slot": i // 2, "arm": str(i % 2), "result": {"candidates": candidates}}
        selected = {"slot": i // 2, "arm": str(i % 2), "selected_indices": [0, 4, 5, 6]}
        sets = audit.source.candidate_sets(arm, selected)
        saved = {str(k): {"prefix_float32_hex": "00" * 40, "suffix_float32_hex": raw.hex(),
                          "suffix_sha256": hashlib.sha256(raw).hexdigest(),
                          "model_terminal_mse": float(k)} for k in range(7)}
        report.append(arm)
        selection.append(selected)
        rows.append({"slot": i // 2, "arm": str(i % 2), "candidate_sets": sets,
                     "candidates": saved, "winners": {"diverse": 0, "immediate": 0}})
    return {"status": "complete_model_only", "config": {"steps": 20}, "rows": rows}, \
        {"rows": report}, {"arms": selection}


def test_rejects_action_and_winner_tampering():
    args = example()
    audit.validate_result(*args)
    bad = copy.deepcopy(args)
    bad[0]["rows"][0]["candidates"]["0"]["suffix_float32_hex"] = "01" + "00" * 39
    with pytest.raises(ValueError, match="action bytes"):
        audit.validate_result(*bad)
    bad = copy.deepcopy(args)
    bad[0]["rows"][0]["winners"]["diverse"] = 4
    with pytest.raises(ValueError, match="winner"):
        audit.validate_result(*bad)
