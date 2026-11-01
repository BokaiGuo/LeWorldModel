from experiments.spectral_reachability import codex_p1_rgb_action_mapping_audit_20261030_v1 as audit


def test_axis_sign_and_small_motion():
    a = {"pixel_xy": [10, 10], "posthoc_native_xy": [30, 30]}
    b = {"pixel_xy": [15, 9], "posthoc_native_xy": [41, 27]}
    result = audit.audit_pixel_pair(a, b)
    assert result[0]["classification"] == "same_sign"
    assert result[1]["classification"] == "below_pixel_floor"
    b["pixel_xy"] = [5, 15]
    result = audit.audit_pixel_pair(a, b)
    assert [r["classification"] for r in result] == ["opposite_or_zero", "opposite_or_zero"]
