from experiments.spectral_reachability import codex_p2_h3_zero_pair_analysis_20261030_v1 as m

def test_actual_fixed_four_pairs():
    r=m.compare()
    assert len(r["roots"])==4 and not r["total_resource_matched"]
    assert all(set(x["arms"])=={"joint","suffix"} for x in r["roots"])
    assert all(not x["zero_success"] for x in r["roots"])
