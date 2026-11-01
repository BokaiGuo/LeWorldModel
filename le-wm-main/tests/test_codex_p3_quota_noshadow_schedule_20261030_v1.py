from experiments.spectral_reachability import codex_p3_quota_noshadow_schedule_20261030_v1 as sched


def test_all_arms_use_only_actual_eligible_candidates():
    for policy, expected in (("uniform", [10, 10]), ("signal", [15, 5]), ("reverse", [5, 15])):
        q = sched.queue(policy, [2., 1.])
        assert q["quotas"] == expected
        assert len(q["jobs"]) == 20
        assert [sum(j["root"] == i for j in q["jobs"]) for i in (0, 1)] == expected
        assert q["per_root_predict_caps"] == [2 * n for n in expected]
        assert q["predict_cap"] == 40
