from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.decision_benchmark import build_pointmaze_v22_c3_action_supported_pool as pool
from experiments.decision_benchmark import run_pointmaze_c3_action_supported_v22_cuda as v22
from experiments.gdd_wm import preflight_pointmaze_c3_action_supported_v22 as preflight


def test_v22_uses_new_roots_but_declares_v20_action_support() -> None:
    assert pool._high()["new_root_id_range"] == [28872, 29319]
    assert pool._direct()["new_root_id_range"] == [29320, 29703]
    assert preflight.HIGH_RANGE == (28872, 29320)
    assert preflight.DIRECT_RANGE == (29320, 29704)
    assert preflight.SCHEMA == panel.C3_ACTION_SUPPORTED_V22_PREFLIGHT_SCHEMA
    assert "202611404" in inspect.getsource(v22.main)  # V20 support-bank construction seed


def test_v22_requires_exact_hash_reuse_not_seed_based_approximation() -> None:
    source = inspect.getsource(preflight.preflight)
    assert "v20_action_bank_exactly_reused" in source
    assert "v20_held_action_hashes" in source
    builder = inspect.getsource(panel.build_panel)
    assert "fixed held action bank does not exactly match" in builder
    assert "permitted_prior_hashes" in builder
    runner = inspect.getsource(v22.main)
    assert "FIXED_HELD_ACTIONS" in runner
    assert "ALLOWED_PRIOR_ACTION_HASHES" in runner
    provenance = inspect.getsource(panel.frozen_preflight_provenance)
    assert "V22 preflight lacks exact frozen-model action-support evidence" in provenance
    assert "C3_ACTION_SUPPORTED_V22_PREFLIGHT_SCHEMA" in provenance
    assert "set(held_hashes) != set(declared_hashes)" in builder
    assert "len(set(held_hashes)) != HELD_CANDIDATES" in builder


def test_v22_runtime_provenance_retains_the_frozen_action_hashes() -> None:
    """V22 must receive hashes from the runtime provenance receipt, not only disk."""
    contexts = v22.runner._contexts()
    _roots, _hashes, _templates, context_sha256 = panel._all_prior_identities(list(contexts))
    receipt = panel.frozen_preflight_provenance(
        v22.runner.ROOT / "results" / "pointmaze_c3_action_supported_v22_static_preflight_20261023.json",
        dataset_root=v22.runner.ROOT / "data" / "point_maze_v69_c3_action_supported_20261023",
        prior_context_sha256=context_sha256,
        split_counts={"train": 256, "calibration": 256, "outcome": 256},
        candidate_design="velocity_matched_braking_pairs",
        goal_qpos=np.asarray([0.8, 0.8], dtype=np.float64),
        construction={
            "root_seed": 202611603,
            "held_action_seed": 202611404,
            "train_action_seed": 202611405,
            "root_id_min": 28872,
            "root_id_max_exclusive": 29704,
        },
    )
    declared = receipt["candidate_bank"]["v20_held_action_hashes"]
    assert isinstance(declared, list)
    assert len(declared) == 16
    assert len(set(declared)) == 16


def test_v22_wrapper_passes_the_exact_frozen_bank_to_v21_builder(tmp_path: Path, monkeypatch) -> None:
    """Exercise v22.main -> v21._build, not merely its source text.

    This regression guards the failure that reached the host GPU job: V22 set
    the module global correctly, but ``build_panel`` later regenerated a
    velocity-matched candidate bank and silently discarded it.
    """
    candidate_bank = np.arange(16 * 20 * 2, dtype=np.float64).reshape(16, 20, 2) / 100.0
    panel = tmp_path / "v20_panel"
    panel.mkdir()
    records = [
        {
            "instance_id": "v20_outcome_root",
            "split": "outcome",
            "candidate_id": index,
            "action_sequence": candidate_bank[index].tolist(),
            "candidate_action_hash": f"hash-{index}",
        }
        for index in range(16)
    ]
    (panel / "temporal_ood_v6_context.json").write_text(json.dumps({"records": records}), encoding="utf-8")
    for relative in (
        "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar",
        "results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "external/local_dinov2").mkdir(parents=True)

    captured: dict[str, object] = {}
    monkeypatch.setattr(v22.runner, "ROOT", tmp_path)
    monkeypatch.setattr(v22.runner, "V20_PANEL", panel)
    # ``v22.main`` configures these V21 globals.  Register their original
    # values with pytest before invoking it so this integration test cannot
    # leak V22 configuration into the V21 protocol tests that share a process.
    for name in (
        "DATASET", "OUTPUT", "PREFLIGHT", "ROOT_RANGE", "SPLIT_COUNTS",
        "ROOT_SEED", "HELD_ACTION_SEED", "TRAIN_ACTION_SEED", "PREFLIGHT_MODULE", "POOL_MODULE",
        "FIXED_HELD_ACTIONS", "ALLOWED_PRIOR_ACTION_HASHES",
    ):
        monkeypatch.setattr(v22.runner, name, getattr(v22.runner, name))
    monkeypatch.setattr(v22.runner.builder, "build_panel", lambda *args, **kwargs: captured.update(kwargs) or {"built": True})
    monkeypatch.setattr(v22.runner, "main", lambda: v22.runner._build(tuple()))

    assert v22.main() == {"built": True}
    assert np.array_equal(captured["fixed_held_actions"], candidate_bank)
    assert captured["allowed_prior_action_hashes"] == {f"hash-{index}" for index in range(16)}
