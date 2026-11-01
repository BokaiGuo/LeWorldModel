import hashlib
import json
import base64
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_two_stage_metered_plan_20261030_v1 as stage


def test_fixed_stage_contracts_distinguish_history_and_keep_axes():
    path = Path(__file__).resolve().parents[1] / "results/codex_p1_three_prefix_blind_inputs_20261030_v1.json"
    rows = json.loads(path.read_bytes())["rows"]
    sources = stage.base.bindings()
    binding = {k: hashlib.sha256(k.encode()).hexdigest() for k in
               ("rgb_sha256", "previous_latent_sha256", "executed_packet_sha256")}
    for row in rows:
        _, first = stage.contract_for(row, sources, 1, binding)
        _, second = stage.contract_for(row, sources, 2, binding)
        assert first["identity"]["history"] != second["identity"]["history"]
        assert first["levels"][0]["caps"] == second["levels"][0]["caps"] == stage.CAPS
        assert stage.CAPS["predict_calls"] == 42
        assert stage.CAPS["backward_calls"] == 20
        assert stage.CAPS["replay_calls"] == 1
        assert first["identity"]["optimizer_config"] == stage.base.spec.digest(stage.bridge.CONFIG)


def test_failed_encoding_retains_terminal_ledger_and_failure_report(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "results/codex_p1_three_prefix_blind_inputs_20261030_v1.json"
    row = json.loads(path.read_bytes())["rows"][0]
    png = base64.b64decode(row["cp5_png_base64"])
    sources = stage.base.bindings()

    def failed_encode(pixels, model, budget, transform, device):
        def fail():
            raise RuntimeError("synthetic encoder failure")
        budget.call("encoder", 1, fail)

    monkeypatch.setattr(stage.bridge, "encode_fresh", failed_encode)
    folder = tmp_path / "stage"
    with pytest.raises(RuntimeError, match="report retained"):
        stage.run_stage(folder, row, 1, png, None, None, None, None, "cpu", sources)
    report = json.loads((folder/"report.json").read_bytes())
    assert report["status"] == "failed" and report["result"] is None
    assert report["error"]["message"] == "synthetic encoder failure"
    account = stage.base.consumer.consume_encoder(folder/"encoder.jsonl", report["encoder_sha256"],
                                                  stage.ENC_CAPS, stage.WALL)
    assert account["status"] == "failed" and account["attempted"]["encoder"] == 1
    assert not (folder/"planning").exists()
