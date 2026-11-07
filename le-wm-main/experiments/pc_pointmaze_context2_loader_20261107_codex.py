"""Explicit local state-assisted model loader bound to the evaluation context."""
from pathlib import Path

import yaml

from experiments.pc_input_permissions import InputAudit, file_hash
from experiments.pc_pointmaze_multimodal_objective_20261107_codex import CONFIG
from experiments.pc_state_track_smoke_20261106_codex import load_model


def configure_context(model, audit, config_path=CONFIG):
    cfg = yaml.safe_load(Path(config_path).read_text())
    window = cfg["model_kwargs"]["wrapper_kwargs"]["ctxt_window"]
    if type(window) is not int or window != 2:
        raise ValueError("expected explicitly configured context window 2")
    if not hasattr(model, "ctxt_window"):
        raise ValueError("model wrapper lacks context attribute")
    model.ctxt_window = window
    dependencies = list(audit.dependencies)
    for path in (Path(config_path).resolve(), Path(__file__).resolve()):
        item = (str(path), file_hash(path))
        prior = [old for old in dependencies if old[0] == item[0]]
        if prior and prior[0] != item:
            raise ValueError("dependency conflict")
        if not prior:
            dependencies.append(item)
    updated = InputAudit(audit.track_id, audit.family, tuple(dependencies), audit.proprio_fields,
                         audit.preprocessing,
                         "one observed frame; context window 2; five environment steps per action packet")
    return updated.verify()


def load_context2():
    model, dataset, audit = load_model()
    return model, dataset, configure_context(model, audit)
