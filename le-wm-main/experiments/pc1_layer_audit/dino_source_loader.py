"""Explicit state-assisted loader; historical loaders and RGB permissions unchanged."""
from dataclasses import replace
from pathlib import Path

from experiments.pc_input_permissions import STATE, file_hash
from experiments.pc1_layer_audit import dino_source_compatible as positional
from experiments.pc1_layer_audit import dino_source_compatible_v2 as compatible
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint


def load_model(*, track):
    if track != STATE:
        raise ValueError("this PointMaze checkpoint requires full proprio; RGB-only is forbidden")
    from experiments.pc1_layer_audit.camera import load_model as original_load
    model, dataset, audit = original_load()
    fingerprint = model_fingerprint(model)
    compatible.install_source_interpolation(model.model.encoder.base_model.model)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("source compatibility must not change weights")
    paths = dict(audit.dependencies)
    paths.update({str(Path(p).resolve()): file_hash(p) for p in
                  (__file__, positional.__file__, compatible.__file__)})
    audit = replace(audit, dependencies=tuple(paths.items()),
                    preprocessing=audit.preprocessing + "; source offset 0.1 and fused QKV v2").verify()
    return model, dataset, audit
