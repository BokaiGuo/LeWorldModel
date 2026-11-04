import numpy as np

from experiments import pc_v13_e0_prospective_mechanism_dev_20261104_codex as run


def test_control_bank_is_actual_first_layer_and_global_zero():
    action = np.full((5, 2), .2, np.float32)
    row = {"root": 0, "task": run.tasks()[0], "result": {
        "selected": np.full((15, 2), .1, np.float32).tolist(),
        "child": {"layers": [{"candidates": [{"chain": action.tolist(), "score": 1.2}]}]}}}
    adapted = run.adapt_row(row)
    assert adapted["result"]["first_bank"][0]["name"] == "zero"
    np.testing.assert_array_equal(adapted["result"]["first_bank"][1]["prefix"], action)
    assert adapted["result"]["prefix_scores"][1] == 1.2
    assert adapted["result"]["selected"]["chain"] == row["result"]["selected"]


def test_selected_index_adapter_restores_original_module(monkeypatch):
    original_local = run.intervention.local
    seen = {}

    def fake_calculate(row):
        out = run.intervention.local.build(row)
        seen["selected_index"] = out["controls"]["selected"]["candidate_index"]
        return {}, {}

    class FakeLocal:
        @staticmethod
        def build(_):
            return {"controls": {"selected": {"candidate_index": 42}}}

    monkeypatch.setattr(run.intervention, "local", FakeLocal)
    monkeypatch.setattr(run.intervention, "calculate", fake_calculate)
    monkeypatch.setattr(run, "adapt_row", lambda row: row)
    run.calculate({})
    assert seen["selected_index"] == -1
    assert run.intervention.local is FakeLocal
    monkeypatch.setattr(run.intervention, "local", original_local)
