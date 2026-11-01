import hashlib
import json

import pytest
import torch

from experiments.spectral_reachability import codex_p1_full_cem_source_20261101_exclusive as m
from experiments.spectral_reachability.codex_p1_full_cem_controller_20261101_exclusive import tensor_sha


@pytest.mark.parametrize("root", [0, 1])
def test_saved_source_uses_rebound_cp10_target(root):
    result = m.inspect(root)
    target = json.loads((m.ROOT / "results/e3_cp10_target_cuda_20261031_RUN01.json").read_bytes())
    goal = torch.tensor(target["rows"][root]["latent"], dtype=torch.float32).unsqueeze(0)
    assert result["identity"]["goal_sha256"] == tensor_sha(goal)
    assert result["identity"]["root_sha256"] == result["rebound_goal_binding"]["root_sha256"]
    assert result["identity"]["root_sha256"] != result["original_binding"]["root_sha256"]
    assert not result["transitive_sources_verified"]
    assert not result["runtime_identity_verified"]


def test_tampered_source_rejected(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"x": 2}')
    with pytest.raises(ValueError, match="digest"):
        m.pinned_json(path, hashlib.sha256(b'{"x": 1}').hexdigest())


def test_invalid_root_rejected():
    with pytest.raises(ValueError, match="root"):
        m.inspect(True)
