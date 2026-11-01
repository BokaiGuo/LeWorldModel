import hashlib
import struct
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import codex_p1_fixed_suffix_censored_episode_20261030 as m


def inputs(tmp_path):
    state = [100., 520., 200., 300., 0., 0., 0.]
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(pixels).save(path)
    command = np.array([0.1, -0.2], dtype="<f4")
    rec = dict(phase="plan", step=1, overlay_index=0, condition="off", png=path.name,
               file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               pixel_sha256=m.native.moving.image_sha(pixels), state=state,
               snapshot=dict(state=state, state_float64_bits=struct.pack("<7d", *state).hex()))
    row = dict(last_observation=rec, initial_exact=True, secondary_errors=[])
    plan = dict(raw15_float32_hex=(command.tobytes() + bytes(14 * 8)).hex())
    transition = dict(command=command, actual=command.copy(), obs={"state": np.array(state)})
    try:
        m.native.moving.guard(pixels, np.array(state))
    except m.native.moving.Rejected as exc:
        return exc, plan, transition, row, SimpleNamespace(path=tmp_path)
    raise AssertionError("guard unexpectedly accepted")


def test_out_of_bounds_frame_is_censored(tmp_path):
    exc, plan, transition, row, store = inputs(tmp_path)
    proof = m.verify_censor(exc, plan, 1, transition, row, store)
    assert proof["out_of_bounds"] == [{"coordinate": 1, "value": 520.0}]
    assert proof["step"] == 1


@pytest.mark.parametrize("tamper", ["action", "frame", "state", "exception"])
def test_rejects_nonmatching_evidence(tmp_path, tamper):
    exc, plan, transition, row, store = inputs(tmp_path)
    if tamper == "action":
        transition["actual"][0] = 0.5
    elif tamper == "frame":
        row["last_observation"]["file_sha256"] = "bad"
    elif tamper == "state":
        transition["obs"]["state"][1] = 300
    else:
        exc = m.native.moving.Rejected("fresh RGB/canvas guard")
    with pytest.raises(Exception):
        m.verify_censor(exc, plan, 1, transition, row, store)
