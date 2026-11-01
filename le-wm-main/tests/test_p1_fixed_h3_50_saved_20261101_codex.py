"""CPU-only saved checker fixtures; no neural or physical execution."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import p1_fixed_h3_50_saved_20261101_codex as check


def test_inherited_censor_never_replayed_or_padded():
    old = dict(eligible=False, old_status="canvas_censored", old_steps=31,
               old_stop_reason=dict(step=31), old_final_checkpoint=dict(step=31))
    row = dict(status="inherited_censored", old_status="canvas_censored", steps=31,
               stop_reason=old["old_stop_reason"], old_final_checkpoint=old["old_final_checkpoint"],
               replayed=False, executed_raw=[], checkpoints=[], nodes=[], node_ledgers=[], encodings=[],
               physical_replay_steps=0, physical_continuation_steps=0, warmup_steps=0,
               observed_cp40=None, observed_cp50=None)
    check.verify_inherited(row, old)
    altered = deepcopy(row)
    altered["physical_replay_steps"] = 1
    with pytest.raises(ValueError, match="inherited terminal"):
        check.verify_inherited(altered, old)
    altered = deepcopy(row)
    altered["observed_cp50"] = {}
    with pytest.raises(ValueError, match="inherited terminal"):
        check.verify_inherited(altered, old)


def test_checkpoint_hash_and_metrics_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "frame.png"
    pixels = np.zeros((224, 224, 3), np.uint8)
    Image.fromarray(pixels).save(path)
    monkeypatch.setattr(check.ctrl.fb.common.metrics, "metrics", lambda state, goal: {"object_xy": 2})
    monkeypatch.setattr(check.ctrl.fb.common.metrics.qualification, "native",
                        lambda goal, state: dict(joint_xy_error=3, initial_success=False))
    frame = dict(step=40, state=[0.] * 7, png=str(path), file_sha256=check.sha(path),
                 pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                 metrics=dict(object_xy=2, joint_xy=3, native_success=False))
    row = dict(checkpoints=[frame], steps=40, observed_cp40=frame, observed_cp50=None)
    assert check.check_points(row, tmp_path, [0.] * 7)[40] == frame
    broken = deepcopy(row)
    broken["checkpoints"][0]["pixel_sha256"] = "0"*64
    with pytest.raises(ValueError, match="actual RGB"):
        check.check_points(broken, tmp_path, [0.] * 7)
