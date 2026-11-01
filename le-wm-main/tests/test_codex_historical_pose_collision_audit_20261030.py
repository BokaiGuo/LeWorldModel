import copy
import json
import math

import pytest

from experiments.spectral_reachability import codex_historical_pose_collision_audit_20261030 as m


def test_real_index():
    raw = m.INDEX.read_bytes()
    assert m.hashlib.sha256(raw).hexdigest() == m.INDEX_SHA
    result = m.audit(json.loads(raw))
    assert result["records"] == 80
    assert result["freshness_certified"] is False
    assert sum(result["component_sizes"]) == 80


def test_pose_alias_and_boundaries():
    assert m.same_pose([0, 0, 0, 0, 0], [1e-9, 0, 0, 0, 2 * math.pi])
    assert not m.same_pose([0, 0, 0, 0, 0], [1.01e-9, 0, 0, 0, 0])
    assert m.identity([0, 0, 0, 0, 0, 1, 2])[2] == m.identity(
        [-0.0, 0, 0, 0, 0, 3, 4])[2]


def test_nontransitive_pose_relation():
    a, b, c = ([x, 0, 0, 0, 0] for x in (0, 0.75e-9, 1.5e-9))
    assert m.same_pose(a, b) and m.same_pose(b, c) and not m.same_pose(a, c)


def test_non_clique_reported():
    index = json.loads(m.INDEX.read_bytes())
    for i, x in enumerate((0, 0.75e-9, 1.5e-9)):
        row = index["observed_prefix_records"][i]
        row["state7"] = [x, 0, 0, 0, 0, 0, 0]
        state_hash, pose, pose_hash = m.identity(row["state7"])
        row.update(root_state_sha256=state_hash, physical_pose=pose, physical_pose_key=pose_hash)
    assert [0, 1, 2] in m.audit(index)["non_clique_components"]


@pytest.mark.parametrize("mutation", ["state", "duplicate", "authority", "count", "nan"])
def test_fail_closed(mutation):
    index = json.loads(m.INDEX.read_bytes())
    index = copy.deepcopy(index)
    if mutation == "state":
        index["observed_prefix_records"][0]["state7"][0] += 1
    elif mutation == "duplicate":
        index["observed_prefix_records"][1]["evidence"] = index["observed_prefix_records"][0]["evidence"]
    elif mutation == "authority":
        index["history_complete"] = True
    elif mutation == "count":
        index["counts"]["observed_prefix_records"] = 79
    else:
        index["observed_prefix_records"][0]["state7"][0] = float("nan")
    with pytest.raises(ValueError):
        m.audit(index)
