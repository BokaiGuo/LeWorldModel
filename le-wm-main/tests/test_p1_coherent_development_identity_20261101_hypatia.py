from copy import deepcopy
import math

import pytest

from experiments.spectral_reachability import p1_coherent_development_identity_20261101_hypatia as m


@pytest.fixture(scope="module")
def result():
    return m.generate()


def test_real_coherent_root_is_exclusion_not_fresh(result):
    assert result["record"]["role"] == "known_development_root_exclude_not_candidate"
    assert result["record"]["reset_seed"] == 202798201
    assert result["new_candidates_selected"] == 0
    assert result["freshness_certified"] is False
    assert result["development_disjoint_certified"] is False
    assert result["training_disjoint_certified"] is False
    assert result["P1_admission"] is False


def test_pose_tolerance_wrap_and_independent_velocity():
    assert m.same_pose([1, 2, 3, 4, 0], [1 + 5e-10, 2, 3, 4, 2 * math.pi])
    assert not m.same_pose([1, 2, 3, 4, 0], [1 + 2e-9, 2, 3, 4, 0])
    assert not m.same_pose([1, 2, 3, 4, 0], [1, 2, 3, 4, 2e-12])


def test_binding_and_root_mismatch_rejected():
    bindings = {k: {"path": str(m.ROOT / p), "sha256": h} for k, (p, h) in m.INPUTS.items()}
    values = {k: m.base.decode(m.base.read_bound(v["path"], v["sha256"])) for k, v in bindings.items()}
    bad = deepcopy(values)
    bad["target"]["rows"][0]["root_state"][0] += 1
    with pytest.raises(ValueError, match="root identity"):
        m.build(bad, bindings, "s", values["history"]["implementation_sha256"])
    bad = deepcopy(values)
    bad["target"]["sources"][bindings["witness"]["path"]] = "0" * 64
    with pytest.raises(ValueError, match="does not bind"):
        m.build(bad, bindings, "s", values["history"]["implementation_sha256"])


def test_output_exclusive_and_reverified(tmp_path, monkeypatch, result):
    monkeypatch.setattr(m, "generate", lambda: result)
    path = tmp_path / (m.NAME + "_test.json")
    digest = m.create(path)
    assert m.verify(path, digest)["P1_admission"] is False
    with pytest.raises(FileExistsError):
        m.create(path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        m.verify(path, digest)
