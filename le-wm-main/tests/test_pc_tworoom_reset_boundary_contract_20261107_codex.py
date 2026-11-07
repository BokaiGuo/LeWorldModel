import pytest

from experiments.pc_tworoom_reset_boundary_contract_20261107_codex import border_contract, require_legal_border


def test_border_bounds_match_local_collision_geometry():
    assert border_contract([21, 203], 14, 7, 224)["valid"]
    assert not border_contract([92.4, 207.2], 14, 7, 224)["valid"]
    with pytest.raises(ValueError, match="outside"):
        require_legal_border([92.4, 207.2], 14, 7, 224)


def test_invalid_geometry_fails_closed():
    with pytest.raises(ValueError):
        border_contract([1, 2], 100, 20, 224)
