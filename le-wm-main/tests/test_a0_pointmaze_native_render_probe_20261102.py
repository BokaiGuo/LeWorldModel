import json
from pathlib import Path

import mujoco

from experiments import a0_pointmaze_native_render_probe_20261102 as probe


def test_xml_builder_keeps_official_visual_sites_and_action_contract():
    xml = probe.official_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "particle_site") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_site") >= 0
    assert (model.nq, model.nv, model.nu) == (2, 2, 2)
    assert model.opt.timestep == 0.01


def test_saved_probe_rows_and_boundaries():
    path = probe.OUTPUT
    report = json.loads(path.read_text())
    assert report["script_sha256"] == probe.digest(Path(probe.__file__))
    assert {(row["root"], row["frame"]) for row in report["rows"]} == {
        (root, frame) for root in probe.ROOTS for frame in probe.FRAMES
    }
    assert report["backend"]["official_legacy_binding_identical"] is False
    assert all(0 <= row["mean_absolute_pixel_error"] <= 255 for row in report["rows"])
    assert all(0 <= row["fraction_nonidentical_pixels"] <= 1 for row in report["rows"])
