import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p1_source_replay_blind_20261101_exclusive as mod


QUEUE = mod.ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV01.json"
QUEUE_SHA = "6756ad9f3ccf058baa422520d3058dbbdc10e7ac86ff1091b53b719f5a1fe847"
REPORT_SHA = "05ce658129367dcad6d8bdf9a05e963758d1eec144ad760cbd05786dd7e7a354"
SLOT = "dev_reused"


def test_exact_development_replay_and_blind_export(tmp_path):
    replay_dir = tmp_path / "replay"
    result = mod.replay(QUEUE, QUEUE_SHA, SLOT, 0, REPORT_SHA, replay_dir)
    assert result["status"] == "exact_development_replay"
    assert result["steps_returned"] == 10 and result["frames_verified"] == 3
    assert not result["independent_confirmation"] and not result["P1_admission"]

    replay_sha = mod.producer.sha(replay_dir / "report.json")
    payload = tmp_path / "root_blind_inputs.npz"
    manifest = tmp_path / "root_blind_manifest.json"
    exported = mod.export_blind(QUEUE, SLOT, 0, replay_dir, replay_sha, payload, manifest)
    assert exported["members"] == ["goal_rgb", "raw_action_history", "rgb_history"]
    assert not exported["contains_seed"] and not exported["contains_physical_state"]
    decoded = mod.blind.load_blind(payload, exported["blind_sha256"])
    assert set(decoded) == {"goal_rgb", "raw_action_history", "rgb_history"}
    assert decoded["rgb_history"].shape == (3, 224, 224, 3)
    assert np.array_equal(decoded["raw_action_history"], np.zeros((10, 2), np.float32))


def test_confirmation_slot_and_nonexact_replay_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="slot"):
        mod.replay(QUEUE, QUEUE_SHA, "not_a_confirmation", 0, REPORT_SHA, tmp_path / "bad")
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    fake = {
        "status": "replay_failed",
        "exact": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    mod.producer.q.write_new(fake_dir / "report.json", fake)
    with pytest.raises(ValueError, match="exact development replay"):
        mod.export_blind(QUEUE, SLOT, 0, fake_dir,
                         mod.producer.sha(fake_dir / "report.json"),
                         tmp_path / "bad.npz", tmp_path / "bad.json")


def test_qualification_preserves_unknown_source_boundary(tmp_path):
    payload = tmp_path / "qualification_blind_inputs.npz"
    np.savez_compressed(
        payload,
        rgb_history=np.zeros((3, 224, 224, 3), np.uint8),
        raw_action_history=np.zeros((10, 2), np.float32),
        goal_rgb=np.zeros((224, 224, 3), np.uint8),
    )
    blind_manifest = tmp_path / "blind.json"
    mod.producer.q.write_new(blind_manifest, {
        "status": "blind_inputs_verified_development_only",
        "implementation_sha256": mod.producer.sha(Path(mod.__file__)),
        "blind_path": str(payload),
        "blind_sha256": mod.producer.sha(payload),
        "members": ["goal_rgb", "raw_action_history", "rgb_history"],
        "confirmation_authorized": False,
    })
    history_path = mod.ROOT / "results/p1_future_root_history_inventory_20261101_codex_LOCAL01.json"
    authority_path = mod.ROOT / "results/codex_p1_source_authority_gap_20261030_LOCAL02.json"
    result = mod.qualify(
        blind_manifest, mod.producer.sha(blind_manifest),
        history_path, mod.producer.sha(history_path),
        authority_path, mod.producer.sha(authority_path),
    )
    assert result["engineering_ready"]
    assert result["status"].endswith("confirmation_blocked")
    assert result["unknown_training_source_is_not_independence"]
    assert not result["confirmation_authorized"]
    assert not result["independent_confirmation"]
    assert not result["P1_admission"]
    assert len(result["training_missing"]) == 7
    assert json.loads(blind_manifest.read_bytes())["confirmation_authorized"] is False
