from experiments.spectral_reachability import codex_four_arm_common_rgb_root1_20261030_main as module
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec


def test_second_root_roster_and_protocol():
    rows = module.roster()
    assert module.protocol()["root_slot"] == 1
    assert [r["arm"] for r in rows] == list(spec.ARMS)
    assert all(r["source_slot"] == 1 and r["episode_id"] == "s1_" + r["arm"] for r in rows)
    assert module.protocol()["fairness_proven"] is False
    assert module.protocol()["P1_admission"] is False


def test_second_root_uses_distinct_bridge():
    assert module.bridge.__name__.endswith("codex_four_arm_common_suffix_bridge_root1_20261030_main")
