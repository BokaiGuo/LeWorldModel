"""No neural execution: saved schema and metadata-only integration contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as probe


def saved_report():
    roots = [str(i) * 64 for i in range(4)]
    receipt = {"context_emb": np.zeros((1, 3, 192)).tolist(),
               "context_actions": np.zeros((1, 3, 10)).tolist(),
               "executed_suffix_params": [[0., 0.]],
               "predicted_terminal_vector": [0.] * 192}
    return {"protocol": {"roots": roots, "suffix_horizon": 1, "fixed_prefixes_per_root": 16},
            "rows": [{"root_sha256": root, "compiled": {"receipts": [dict(deepcopy(receipt), candidate_id=f"prefix_{i}") for i in range(16)]}}
                     for root in roots]}


class Model:
    def __init__(self):
        self.entries = 0

    def predict(self, embedding, action):
        self.entries += 1
        return np.zeros(192)


def fake_input():
    return SimpleNamespace(shape=(1, 3, 192), device=SimpleNamespace(type="cuda", index=0))


def test_complete_probe_control_flow_no_seventeenth_entry():
    row = saved_report()["rows"][0]
    model = Model()
    original = model.predict
    result = probe.probe_root(model, row, probe.Binding("p", "task", row["root_sha256"], "m"),
                              lambda saved: model.predict(fake_input(), fake_input()))
    assert model.entries == result["actual_predict_entries"] == 16
    assert result["root_sha256"] == row["root_sha256"]
    assert result["seventeenth_blocked_before_predict"]
    assert result["budget"]["attempted_predict_samples"] == 16
    assert result["budget"]["blocked_calls"] == 1
    assert len(result["predictions"]) == 16
    assert model.predict == original and "predict" not in vars(model)


def test_prediction_mismatch_restores_model():
    model = Model()
    original = model.predict
    row = saved_report()["rows"][0]
    def wrong(saved):
        model.predict(fake_input(), fake_input())
        return np.ones(192)
    with pytest.raises(ValueError, match="terminal mismatch"):
        probe.probe_root(model, row, probe.Binding("p", "t", "r", "m"), wrong)
    assert model.predict == original and "predict" not in vars(model)


def test_absolute_tolerance_no_relative_slack():
    assert probe.verify_prediction(np.ones(192) * probe.ATOL, np.zeros(192)) == probe.ATOL
    with pytest.raises(ValueError):
        probe.verify_prediction(np.ones(192) * (probe.ATOL + 1e-9), np.zeros(192))
    with pytest.raises(ValueError):
        probe.verify_prediction(np.full(192, np.nan), np.zeros(192))
    with pytest.raises(ValueError):
        probe.verify_prediction(np.zeros(191), np.zeros(192))


@pytest.mark.parametrize("defect", ["missing_root", "duplicate_root", "order", "missing_candidate", "duplicate_candidate", "emb_shape", "suffix_horizon", "nonfinite"])
def test_fail_closed_saved_receipts(defect):
    report = saved_report()
    if defect == "missing_root":
        report["rows"].pop()
    elif defect == "duplicate_root":
        report["protocol"]["roots"][1] = report["protocol"]["roots"][0]
    elif defect == "order":
        report["rows"].reverse()
    elif defect == "missing_candidate":
        report["rows"][0]["compiled"]["receipts"].pop()
    elif defect == "duplicate_candidate":
        report["rows"][0]["compiled"]["receipts"][1]["candidate_id"] = "prefix_0"
    elif defect == "emb_shape":
        report["rows"][0]["compiled"]["receipts"][0]["context_emb"] = []
    elif defect == "suffix_horizon":
        report["protocol"]["suffix_horizon"] = 2
    else:
        report["rows"][0]["compiled"]["receipts"][0]["predicted_terminal_vector"][0] = float("inf")
    with pytest.raises(ValueError):
        probe.validate_rows(report)


def test_saved_schema_and_sha_checked_without_cuda():
    root = Path(__file__).resolve().parents[1]
    path = root / "results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json"
    if not path.exists():
        pytest.skip("local sealed input unavailable")
    assert probe.digest(path) == probe.INPUT_SHA
    import json
    assert len(probe.validate_rows(json.loads(path.read_text()))) == 4


def test_input_sha_mismatch_precedes_model_import_or_output(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="compilation SHA"):
        probe.run(compilation=path, policy=tmp_path, output=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_exclusive_output_and_source_seal_order():
    # Static safety assertion; never executes neural/model setup in tests.
    import inspect
    source = inspect.getsource(probe.run)
    assert source.index("create_seal(") < source.index('require_cuda_allocation("cuda")')
    assert source.index('write_new(output / "protocol.json"') < source.index('load_compatible_pretrained(str(')
    assert "output.mkdir(exist_ok=False)" in source
    assert "p1._suffix_terminal(model, emb, act, suffix)" in source
    assert 'device="cuda"' in source


def test_write_new_never_overwrites(tmp_path):
    path = tmp_path / "report.json"
    probe.write_new(path, {"a": 1})
    with pytest.raises(FileExistsError):
        probe.write_new(path, {"a": 2})
