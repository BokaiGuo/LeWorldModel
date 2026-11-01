import json

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_source_supported_p1 import _disjoint_reference_binding


def test_confirmation_binding_excludes_all_reference_episodes(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"status": "completed", "rows": [
        {"seed": 1, "source_episode": 12}, {"seed": 2, "source_episode": 15}]}))
    episodes, bindings = _disjoint_reference_binding([path], {3, 4})
    assert episodes == {12, 15}
    assert bindings[0]["path"] == str(path.resolve())
    assert len(bindings[0]["sha256"]) == 64
    with pytest.raises(ValueError, match="overlap"):
        _disjoint_reference_binding([path], {2, 4})


@pytest.mark.parametrize("payload", [
    {"status": "fail", "rows": [{"seed": 1, "source_episode": 12}]},
    {"status": "completed", "rows": []},
    {"status": "completed", "rows": [{"seed": 1}]},
])
def test_confirmation_requires_completed_source_episode_evidence(tmp_path, payload):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source-episode reference"):
        _disjoint_reference_binding([path], {3})
