import pytest
import torch

from experiments.pc_input_permissions import STATE, freeze_audit
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context


def test_context_is_bound_and_wrong_config_fails(tmp_path):
    weights = tmp_path / "weights"
    weights.write_text("weights")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model_kwargs:\n  wrapper_kwargs:\n    ctxt_window: 2\n")
    audit = freeze_audit(STATE, "pointmaze_jepa", [weights], proprio_fields=("x", "y", "vx", "vy"),
                         preprocessing="test", clock="old context 1")
    model = torch.nn.Linear(1, 1)
    model.ctxt_window = 1
    new_audit = configure_context(model, audit, cfg)
    assert model.ctxt_window == 2
    assert "context window 2" in new_audit.clock
    assert new_audit.fingerprint != audit.fingerprint
    assert audit.clock == "old context 1"
    cfg.write_text("model_kwargs:\n  wrapper_kwargs:\n    ctxt_window: 1\n")
    with pytest.raises(ValueError):
        new_audit.verify()
    with pytest.raises(ValueError, match="context window"):
        configure_context(model, audit, cfg)
