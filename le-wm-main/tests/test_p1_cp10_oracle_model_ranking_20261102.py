from experiments.spectral_reachability import p1_cp10_oracle_model_ranking_20261102 as subject


def test_model_inputs_exclude_oracle_state_and_outcome():
    spec = subject.protocol()
    assert len(spec["rows"]) == 2
    assert all(len(row["actions"]) == 64 for row in spec["rows"])
    assert all(not ({"cp10_snapshot", "goal_state", "terminal", "reward"} & set(row))
               for row in spec["rows"])
    assert not spec["oracle_state_used_in_model"] and not spec["P1_admission"]
