from experiments.calibrate_pusht_orientation_gate import authorize, group_key, split_by_stratum


def row(root_id, source_index, edge="0:0", fraction_bin=0, sign=1.0, mode=1.0, separation=0.05, rmse=0.02):
    return {
        "root_id": root_id,
        "source_index": source_index,
        "edge_id": edge,
        "fraction_bin": fraction_bin,
        "sign_consistent": sign,
        "successor_mode_agreement_pm": mode,
        "paired_separation": separation,
        "paired_antisymmetry_rmse": rmse,
    }


def test_split_is_root_disjoint_within_each_stratum():
    rows = [row(index, index) for index in range(4)]
    calibration, evaluation = split_by_stratum(rows, 0.5)
    assert {item["root_id"] for item in calibration}.isdisjoint({item["root_id"] for item in evaluation})
    assert len(calibration) == 2
    assert len(evaluation) == 2


def test_gate_requires_all_predeclared_conditions():
    args = type("Args", (), {
        "min_calibration_roots": 2,
        "min_sign_consistency": 0.75,
        "min_mode_agreement": 0.75,
        "min_separation": 0.02,
        "max_antisymmetry_rmse": 0.06,
    })()
    authorized, stats = authorize([row(0, 0), row(1, 1)], args)
    assert authorized
    assert stats["authorized"]
    denied, _ = authorize([row(0, 0, sign=0.0), row(1, 1)], args)
    assert not denied


def test_group_key_keeps_edge_and_fraction_stratum():
    assert group_key(row(0, 0, edge="1:2", fraction_bin=3)) == "1:2__bin3"
