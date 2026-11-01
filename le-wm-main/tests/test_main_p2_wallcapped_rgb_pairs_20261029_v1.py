import pytest
from experiments.spectral_reachability import main_p2_wallcapped_rgb_pairs_20261029_v1 as m


def values():
    return {(r, a, s): dict(model_Q=1. if a == 'joint' else 0., real_Q=0. if a == 'joint' else 1.,
                           joint_xy=0., object_xy=0., wrapped_angle=0.)
            for r in (0, 1) for a in ('joint', 'composition') for s in (5, 10, 15)}


def test_key_join_not_root1_positional_order():
    v = values()
    ordered = list(v.items())[:6] + list(reversed(list(v.items())[6:]))
    assert m.pair(dict(ordered)) == m.pair(v)
    assert all(p['composition_minus_joint']['model_Q'] == -1 for p in m.pair(v))
    assert all(p['model_to_real'] == 'reversed' for p in m.pair(v))
    assert all(p['real_to_joint'] == 'tie' for p in m.pair(v))


def test_missing_stays_sixth_pair_null():
    v = values()
    for k in ('real_Q', 'joint_xy', 'object_xy', 'wrapped_angle'):
        v[1, 'joint', 15][k] = None
    p = m.pair(v)
    assert len(p) == 6 and p[-1]['model_to_real'] == 'missing'
    assert p[-1]['composition_minus_joint']['model_Q'] == -1
    assert p[-1]['composition_minus_joint']['joint_xy'] is None


def test_missing_key_rejected():
    v = values(); del v[1, 'joint', 15]
    with pytest.raises(ValueError, match='twelve'):
        m.pair(v)


def test_duplicate_rejected():
    row = dict(root_slot=1, arm='joint', checkpoint=15)
    with pytest.raises(ValueError, match='duplicate'):
        m.keyed([row, row])


def test_mse_float64_and_nonfinite():
    assert m.mse([1]*192, [0]*192) == 1.
    with pytest.raises(ValueError, match='finite'):
        m.mse([float('nan')]*192, [0]*192)
