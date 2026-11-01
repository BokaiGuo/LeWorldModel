from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p1_pusher_latent_saved_analysis_20261030 as m


def sample():
    encoded, physical = [], []
    for i in range(16):
        plan = dict(slot=i, arm="a", role="r")
        points = [dict(step=5*j, state=[float(j), 0., 1., 2., 3., 0., 0.],
                       pixel_sha256=str(j)) for j in range(4)]
        physical.append(dict(index=i, plan=plan, checkpoints=points, goal=[0.]*7))
        encoded.append(dict(index=i, **plan, actual_latents=[[float(j)]*192 for j in range(4)],
                            goal_latent=[0.]*192))
    return dict(rows=encoded), dict(rows=physical)


def test_same_pose_pairs_and_dedup():
    encoded, physical = sample()
    result = m.analyze(encoded, physical)
    assert len(result["pairs"]) == 16 * 6
    assert result["pairs"][0]["pusher_distance"] == [0., 1.]
    assert result["pairs"][0]["goal_latent_distance"][1] > 0


def test_wrong_role_fails():
    encoded, physical = sample()
    altered = deepcopy(physical)
    altered["rows"][0]["plan"]["role"] = "wrong"
    with pytest.raises(Exception):
        m.analyze(encoded, altered)
