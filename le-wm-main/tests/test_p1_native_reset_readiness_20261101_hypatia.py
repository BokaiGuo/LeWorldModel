import hashlib

import pytest

from experiments.spectral_reachability import p1_native_reset_readiness_20261101_hypatia as m


@pytest.fixture(scope="module")
def readiness():
    return m.generate()


def test_real_source_is_read_only_and_not_authority(readiness):
    checks = readiness["native_reset_static_inspection"]
    assert checks["direct_set_state_calls"] == 2
    assert checks["inferred_space_steps_per_reset"] == 2
    assert checks["direct_render_calls"] == 1
    assert readiness["reset_calls"] == readiness["physical_calls"] == readiness["neural_calls"] == 0
    assert readiness["candidate_roster"] is None
    assert readiness["freshness_certified"] is False
    assert readiness["source_run_authorized"] is False
    assert readiness["P1_admission"] is False


def test_changed_reset_structure_is_rejected():
    text = b"""class PushT:
 def reset(self):
  self._set_state(0)
  self.render()
 def _set_state(self, state):
  self.space.step(0)
"""
    with pytest.raises(ValueError, match="unsupported"):
        m.inspect_reset(text)


def test_exclusive_create_reconstruct_and_tamper(tmp_path, monkeypatch, readiness):
    monkeypatch.setattr(m, "generate", lambda: readiness)
    output = tmp_path / (m.NAME + "_test.json")
    digest = m.create(output)
    assert m.verify(output, digest)["source_run_authorized"] is False
    with pytest.raises(FileExistsError):
        m.create(output)
    output.write_text("{}")
    with pytest.raises(ValueError, match="SHA"):
        m.verify(output, digest)
    with pytest.raises(ValueError, match="drift"):
        m.verify(output, hashlib.sha256(b"{}").hexdigest())
