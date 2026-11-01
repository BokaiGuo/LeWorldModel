from experiments.spectral_reachability import codex_p1_rgb_latent_task_compare_20261030_v1 as compare


def test_comparisons_never_promote_early_stop_and_report_disagreements():
    rows = []
    for slot in (0, 1):
        for condition in ("off", "on"):
            for n in range(6):
                rows.append({"index": len(rows), "identity": {"root_slot": slot, "condition": condition},
                             "full_horizon": n != 0, "latent_goal_distance": float(n),
                             "posthoc_pusher_error": float(-n)})
    values = compare.compare_order(rows)
    assert len(values) == 60
    assert sum(v["status"] == "early_stop" for v in values) == 20
    assert sum(v["status"] == "disagree" for v in values) == 40
