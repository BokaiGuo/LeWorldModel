from experiments.spectral_reachability.p1_reverse_witness_20261101_main import PHASES, eligibility


def test_task_eligibility_is_not_method_passage():
    states = [[100., 100., 0., 0., 0., 0., 0.] for _ in range(21)]
    states[5] = [0., 100., 0., 0., 0., 0., 0.]
    states[20] = [100., 100., 30., 0., 0., 0., 0.]
    assert sum(x[2] for x in PHASES) == 20
    assert all(eligibility(states).values())
    states[5][2] = 1.
    assert not eligibility(states)["cp5_object_unchanged"]
    assert eligibility(states[:10]) == {"complete20": False}
