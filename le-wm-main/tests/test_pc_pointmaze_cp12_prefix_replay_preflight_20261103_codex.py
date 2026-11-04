import json

import numpy as np
import pytest

from experiments import pc_pointmaze_cp12_prefix_replay_preflight_20261103_codex as preflight


def test_rgb_exact_gate_and_bad_shapes():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    assert preflight.compare_frame(image, image)["equal"]
    changed = image.copy()
    changed[1, 1, 2] = 7
    result = preflight.compare_frame(image, changed)
    assert not result["equal"] and result["different_channels"] == 1 and result["max_abs"] == 7
    with pytest.raises(ValueError):
        preflight.compare_frame(image, image[:-1])
    with pytest.raises(ValueError):
        preflight.compare_frame(image.astype(float), image)


def test_refuses_overwrite(monkeypatch, tmp_path):
    output = tmp_path / "already_exists.json"
    output.touch()
    monkeypatch.setattr(preflight, "OUT", output)
    with pytest.raises(FileExistsError):
        preflight.main()


def test_saved_gate_and_bindings():
    if not preflight.OUT.exists():
        pytest.skip("preflight receipt absent")
    result = json.loads(preflight.OUT.read_text())
    assert result["source_sha256"] == preflight.h.sha256_file(preflight.SOURCE)
    assert result["source_npz_sha256"] == preflight.h.sha256_file(preflight.SOURCE.with_suffix(".npz"))
    assert result["protocol_sha256"] == preflight.h.sha256_file(preflight.PROTOCOL)
    assert [r["root"] for r in result["rows"]] == [8343, 8345, 8349, 8351]
    assert all(len(r["frames"]) == 13 for r in result["rows"])
    assert result["pass_exact_all"] == all(f["equal"] for r in result["rows"] for f in r["frames"])
    assert not result["pass_exact_all"]
    assert result["counts"]["suffix_commands"] == result["counts"]["model_calls"] == 0
