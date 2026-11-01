from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import codex_p2_symmetric_h3_saved_checks_20261030_v1 as saved


def fixture():
    goal = np.ones(192)
    case = dict(goal=goal.tolist(), projection=np.eye(192).tolist())
    tags = ["nominal"]+[f"i{i}_a{a}" for i in range(2) for a in saved.run.ALPHAS]+["replay0", "replay1"]
    rows = []
    for tag in tags:
        z = np.zeros(192)
        rows.append(dict(tag=tag, logits=[0.]*30,
                         contexts=[dict(latent_history=[z.tolist()]) for _ in range(3)],
                         projected_terminal=z.tolist(), cost=1.))
    return case, dict(root=0, arm="suffix", winner=rows[0]), rows


def test_tie_and_fixed_order():
    case, row, candidates = fixture()
    assert saved.check_candidates(case, row, candidates)["winner_tag"] == "nominal"
    candidates[1]["tag"] = "wrong"
    with pytest.raises(ValueError): saved.check_candidates(case, row, candidates)


@pytest.mark.parametrize("mutation", ["cost", "terminal", "prefix", "replay", "winner"])
def test_tampering_rejected(mutation):
    case, row, candidates = deepcopy(fixture())
    if mutation == "cost": candidates[1]["cost"] = 0.
    if mutation == "terminal": candidates[1]["projected_terminal"][0] = 1.
    if mutation == "prefix": candidates[1]["logits"][0] = .1
    if mutation == "replay": candidates[-1]["contexts"][0]["latent_history"][0][0] = .1
    if mutation == "winner": row["winner"] = candidates[1]
    with pytest.raises(ValueError): saved.check_candidates(case, row, candidates)
