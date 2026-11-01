import json

import pytest

from experiments.spectral_reachability import p1_exploration_seed_preseal_20261101_hypatia as mod


def test_fixed_budget_and_all_eight_seeds_retained():
    raw = mod.TEMPLATE.read_bytes()
    assert mod.digest(raw) == mod.TEMPLATE_SHA
    template = json.loads(raw)
    config, candidates = mod.construct(template, [mod.LOW + 1, mod.LOW + 2, mod.LOW + 100])
    assert config["root_seeds"] == [mod.LOW + 1, mod.LOW + 2]
    assert [x["reset_seed"] for x in candidates["candidates"][2:]] == list(range(mod.LOW + 100, mod.LOW + 108))
    assert all(x["state7"] is None for x in candidates["candidates"])


def test_equal_draws_are_retained_not_replaced():
    template = json.loads(mod.TEMPLATE.read_bytes())
    config, candidates = mod.construct(template, [mod.LOW, mod.LOW, mod.LOW])
    assert config["root_seeds"] == [mod.LOW, mod.LOW]
    assert len(candidates["candidates"]) == 10


def test_exclusive_write(tmp_path):
    path = tmp_path / "proposal.json"
    mod.write_exclusive(path, {"a": 1})
    with pytest.raises(FileExistsError):
        mod.write_exclusive(path, {"a": 2})


def test_invalid_draw_rejected():
    with pytest.raises(ValueError):
        mod.construct(json.loads(mod.TEMPLATE.read_bytes()), [True, mod.LOW, mod.LOW])
