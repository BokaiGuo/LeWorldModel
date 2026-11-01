import hashlib
import json

import numpy as np

from experiments.spectral_reachability import p1_response_diverse_backtrack_join_20261030_main as mod


def test_saved_selection_and_witnesses():
    path = mod.ROOT / "results/p1_response_diverse_backtrack_join_20261030_main_RUN01.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "ad5b2df40dc14d73e37c1b1b4913d76c20f224090e705892acd29f7852140d67"
    result = json.loads(path.read_bytes())
    source = mod.need_read(mod.ROOT / "results/p1_bounded_backtrack_host_20261030_MAIN01.json", mod.inventory.SHA)
    report = mod.need_read(mod.panel.source.REPORT, mod.panel.source.REPORT_SHA)
    assert len(result["rows"]) == 8 and result["physical_authorized"] is False
    for i, row in enumerate(result["rows"]):
        candidates = source["rows"][20*i:20*(i+1)]
        chosen = min(row["selected_indices"], key=lambda j: (candidates[j]["best_regularized"], j))
        assert chosen == row["chosen_index"]
        assert row["chosen_score"] == candidates[chosen]["best_regularized"]
        expected = np.concatenate((np.asarray(report["rows"][i]["result"]["candidates"][chosen]["prefix"],
                                               dtype="<f4"),
                                   np.asarray(candidates[chosen]["final"]["raw"], dtype="<f4").reshape(10, 2)))
        raw = bytes.fromhex(row["raw15_f32_hex"])
        assert raw == expected.tobytes()
        assert hashlib.sha256(raw).hexdigest() == row["raw15_sha256"]
