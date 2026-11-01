import numpy as np

from experiments.spectral_reachability import codex_p1_diag32_cp5_wiring_audit_20261101_exclusive as audit


def test_saved_inputs_and_budget():
    _, protocol, rows = audit.inputs()
    assert [row["role"] for row in rows] == protocol["roles"]
    assert audit.contract()["limits"] == {
        "encode": 3, "action_encoder": 12, "predict": 12, "reverse": 3, "wall_seconds": 600}
    for row in rows:
        assert row["task"]["root_context"]["latent_history"][-1] == row["latent"]
        assert np.asarray(row["first"]["actions"]).shape == (2, 5, 2)
        assert (row["first"]["candidate"], row["first"]["iteration"]) == (0, 0)
        assert (row["first_next"]["candidate"], row["first_next"]["iteration"]) == (0, 1)
