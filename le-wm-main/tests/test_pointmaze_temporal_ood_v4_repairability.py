from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v4_repairability as v4


def _record(root: str, split: str, candidate: int, state: float = 0.0) -> dict[str, object]:
    action = np.repeat(np.asarray([[0.15 + candidate * 0.1, 0.05]], dtype=np.float64), 20, axis=0)
    return {
        "instance_id": root, "split": split, "candidate_id": candidate,
        "source_latent_model": [0.1, 0.2],
        "source_history_model": [0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.1, 0.2],
        "history_actions_raw": np.repeat(np.asarray([[0.1, 0.0]], dtype=np.float64), 15, axis=0).tolist(),
        "source_state_raw": [state, state, state, state],
        "action_sequence": action.tolist(),
    }


def test_moderators_do_not_read_privileged_source_state() -> None:
    first = [_record("root", "calibration", 0, state=0.0), _record("root", "calibration", 1, state=0.0)]
    second = [_record("root", "calibration", 0, state=99.0), _record("root", "calibration", 1, state=-99.0)]
    x1, roots1, names1 = v4.moderator_features(first)
    x2, roots2, names2 = v4.moderator_features(second)
    assert roots1 == roots2 == ["root"]
    assert names1 == names2
    assert np.allclose(x1, x2)


def _write_development_panel(tmp_path: Path, panel: str) -> tuple[Path, Path]:
    records, action_roots, history_roots = [], {"calibration": {}, "outcome": {}}, {"calibration": {}, "outcome": {}}
    for index in range(8):
        split = "calibration" if index % 2 == 0 else "outcome"
        root = f"{panel}-root-{index}"
        records.extend([_record(root, split, 0), _record(root, split, 1)])
        action = 0.03 if index % 3 == 0 else 0.0
        history = 0.0 if index % 3 == 0 else 0.02
        action_roots[split][root] = {"regret": action}
        history_roots[split][root] = {"regret": history}
    context = tmp_path / f"{panel}_context.json"
    report = tmp_path / f"{panel}_report.json"
    context.write_text(json.dumps({"metadata": {}, "records": records}))
    report.write_text(json.dumps({"selection": {"action_only": {key: {"roots": value} for key, value in action_roots.items()}, "history_latent_action": {key: {"roots": value} for key, value in history_roots.items()}}}))
    return context, report


def test_development_selector_is_panel_aware_and_frozen(tmp_path: Path) -> None:
    pairs = [_write_development_panel(tmp_path, name) for name in ("v1", "v2", "v3")]
    target = tmp_path / "selector.json"
    package = v4.fit_selector([item[0] for item in pairs], [item[1] for item in pairs], target)
    assert target.exists()
    assert package["development_panels"] == ["v1", "v2", "v3"]
    assert package["panel_root_aware_cv"].startswith("leave-one-panel-out")
    assert len(package["development_cv"]) == 3 * len(v4.THRESHOLD_GRID)
    assert package["threshold_grid"] == list(v4.THRESHOLD_GRID)


def test_v4_seeded_action_banks_are_disjoint() -> None:
    held = v4.v3_action_sequences(seed=v4.ACTION_SEED, train=False)
    train = v4.v3_action_sequences(seed=v4.TRAIN_ACTION_SEED, train=True)
    assert not {v4.sequence_hash(value) for value in held} & {v4.sequence_hash(value) for value in train}
    assert not {v4.temporal_template_fingerprint(value) for value in held} & {v4.temporal_template_fingerprint(value) for value in train}
