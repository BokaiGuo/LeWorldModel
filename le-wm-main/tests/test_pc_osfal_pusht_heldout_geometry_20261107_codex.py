from experiments.pc_osfal_pusht_heldout_geometry_20261107_codex import tasks, audit_preconditions


def test_new_geometry_panel_is_disjoint_from_old_and_has_16_unique_roots():
    audit_preconditions()
    rows = tasks()
    assert len({(tuple(row["position"]), row["angle"]) for row in rows}) == 16
