import json

from experiments.spectral_reachability import p1_uphill_fixedtail_physical_20261030 as fixed


def test_saved_roster_matches_all_three_roles_to_each_sealed_tail():
    path = fixed.physical.ROOT / "results/p1_uphill_fixedtail_physical_20261030_prepare_RUN01.json"
    prepared = json.loads(path.read_bytes())
    h10 = json.loads(fixed.physical.bundle.PREP.read_bytes())
    assert len(prepared["roster"]) == 24
    for i, row in enumerate(prepared["roster"]):
        source = next(x for x in h10["probe_banks"][i // 3] if x["name"] == "original_selected_tail")
        assert row["index"] == i
        assert row["raw10_float32_hex"] == source["raw10_float32_hex"]
        assert row["raw10_sha256"] == source["raw10_sha256"]
        assert "plan_row_sha256" not in row
