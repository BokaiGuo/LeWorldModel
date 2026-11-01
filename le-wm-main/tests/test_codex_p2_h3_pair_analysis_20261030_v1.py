import numpy as np
from experiments.spectral_reachability import codex_p2_h3_pair_analysis_20261030_v1 as m

def test_actual_pairs_and_periodic_angle():
    a=m.analyze()
    assert len(a["pairs"]) == 4 and len(a["rows"]) == 8
    assert all(p["successes"] == [False,False] and p["same_steps"] == 15 for p in a["pairs"])
    assert m.pose([0,0,0,0,2*np.pi,0,0],[0]*7)["angle"] < 1e-12
