import itertools

from experiments.pc_pusht_task1_pose_angle_factor_20261107_codex import effects


def test_factor_effects_and_invalid_pair():
    rows = []
    for p, x, a in itertools.product(("failure", "success"), repeat=3):
        rows.append({"pusher": p, "position": x, "angle": a,
                     "goal_cost": 2 * (p == "success") + 3 * (x == "success") + 5 * (a == "success"),
                     "valid": (p, x, a) != ("success", "success", "success")})
    result = effects(rows)
    assert len(result) == 12
    for row in result:
        expected = {"pusher": 2, "position": 3, "angle": 5}[row["factor"]]
        assert row["delta_cost"] == (None if row["success_key"] == ("success",) * 3 else expected)
