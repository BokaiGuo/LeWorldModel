import copy

import pytest

from experiments.spectral_reachability import p1_adam_checkpoint_physical_pair_20261031 as module


def actual_inputs():
    sources = {path: module.pairing.read(path, sha) for path, sha in module.SOURCES.items()}
    root = module.ROOT / "results"
    prepared = {s: sources[root / f"p1_adam_checkpoint_physical_{s}_20261031_prepare_RUN01.json"]
                for s in (5, 10)}
    consumed = {s: sources[root / f"p1_adam_checkpoint_physical_{s}_20261031_consume_RUN01.json"]
                for s in (5, 10)}
    return (sources[module.pairing.ANCHORS], sources[module.physical.PLAN], prepared, consumed,
            sources[root / "p1_uphill_multistart_pairing_20261030_main_RUN01.json"],
            sources[module.physical.prior.PLAN])


def test_join_budgets_and_preserve_censoring():
    result = module.join(*actual_inputs())
    assert len(result["rows"]) == 24
    for row in result["rows"]:
        assert set(row["budgets"]) == {"5", "10", "20"}
        for budget in row["budgets"].values():
            assert (budget["endpoint"] is not None) == (budget["status"] == "probe_completed")


def test_rejects_twenty_action_mismatch_and_prefix_mismatch():
    args = list(actual_inputs())
    args[5] = copy.deepcopy(args[5])
    args[5]["rows"][0]["selected_action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="twenty score and action identity"):
        module.join(*args)
    args = list(actual_inputs())
    args[3] = copy.deepcopy(args[3])
    args[3][5]["rows"][0]["condition"] = "tampered"
    with pytest.raises(ValueError, match="identity binding"):
        module.join(*args)
