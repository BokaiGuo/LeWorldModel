"""Saved arithmetic and exact actual-authority regression; no model/physics."""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_spatial_patch_attribution_20261028_v1 as m


@pytest.fixture
def arrays():
    g = np.zeros((256, 384), np.float32)
    p = {w: np.full_like(g, i+1) for i, w in enumerate(m.WORDS)}
    r = {w: np.full_like(g, 2*(i+1)) for i, w in enumerate(m.WORDS)}
    rgb = {w: np.full((224, 224, 3), i, np.uint8) for i, w in enumerate(m.WORDS)}
    return g, p, r, rgb, np.zeros((224, 224, 3), np.uint8)


def test_identity_all_patches_tiles_and_units(arrays):
    data = m.decompose(*arrays)
    assert len(data['rows']) == 4 and len(data['pairs']) == 2
    for i, row in enumerate(data['rows']):
        x = (i+1)**2
        assert row['full_mean']['predicted_goal'] == x
        assert row['full_mean']['real_goal'] == 4*x
        assert row['full_mean']['prediction_real'] == x
        assert row['full_mean']['cross'] == 2*x
        assert row['full_mean']['raw_rgb_goal'] == i*i
        assert len(row['tiles']) == 16
        for k, total in row['full_mean'].items():
            assert sum(t['full_metric_contribution'][k] for t in row['tiles']) == total
            assert np.sum(row['patch_full_metric_contribution'][k]) == total
    assert data['pairs'][0]['full_mean']['real_feature_pair'] == 4


def test_negative_cross_not_clipped(arrays):
    g, p, r, rgb, goal_rgb = arrays
    r = {w: g.copy() for w in m.WORDS}
    row = m.decompose(g, p, r, rgb, goal_rgb)['rows'][0]
    assert row['full_mean']['cross'] == -2 and row['full_mean']['real_goal'] == 0


def test_single_patch_order_and_fixed_tile(arrays):
    g, p, r, rgb, gr = arrays
    p = {w: g.copy() for w in m.WORDS}
    r = {w: g.copy() for w in m.WORDS}
    r['xy'][3*16+7] = 2
    rgb['xy'][:] = 0
    rgb['xy'][42:56, 98:112] = 10
    row = m.decompose(g, p, r, rgb, gr)['rows'][0]
    assert row['patch_channel_mean']['real_goal'][3][7] == 4
    assert row['patch_channel_mean']['raw_rgb_goal'][3][7] == 100
    assert row['tiles'][1]['full_metric_contribution']['real_goal'] == 4/256
    assert sum(t['patches'] for t in row['tiles']) == 256


@pytest.mark.parametrize('fault', ['missing', 'order', 'nan', 'dtype', 'shape'])
def test_abi_failclosed(arrays, fault):
    g, p, r, rgb, gr = arrays
    if fault == 'missing': r.pop('xy')
    elif fault == 'order': r = dict(reversed(list(r.items())))
    elif fault == 'nan': r['xy'][0, 0] = np.nan
    elif fault == 'dtype': r['xy'] = r['xy'].astype(np.float64)
    else: r['xy'] = r['xy'][:-1]
    with pytest.raises(ValueError): m.decompose(g, p, r, rgb, gr)


def test_allzero_figure_no_scale_divide(arrays):
    g, _, _, rgb, gr = arrays
    p = {w: g.copy() for w in m.WORDS}
    picture = m.figure(m.decompose(g, p, p, rgb, gr), rgb, gr)
    assert picture.count('<title>') == 4*4*256
    assert picture.count('data:image/png;base64,') == 5
    assert 'nan' not in picture


@pytest.fixture(scope='module')
def actual():
    root = m.ROOT/'results'
    args = (root/'agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json',
            root/'agent_p5_terminal_goal_attribution_20261028_v5_cuda_MAIN01/report.json',
            root/'agent_p5_terminal_goal_attribution_20261028_v5_consume_MAIN01.json')
    value, picture = m.build(*args)
    return args, value, picture


def test_actual_full_saved_consumers_and_score_reconstruction(actual):
    _, value, picture = actual
    assert [r['word'] for r in value['rows']] == list(m.WORDS)
    assert value['new_neural_API'] == value['new_physics'] == 0
    assert value['scientific_admission'] is False and value['task_success'] is None
    assert value['semantic_regions'] is value['causal_attribution'] is None
    assert picture.startswith('<svg')
    assert all(m.sha(p) == h for p, h in value['bindings'].items())
    for row in value['rows']:
        assert row['identity_max_abs_residual'] < 1e-10
        for k, total in row['full_mean'].items():
            assert np.isclose(sum(t['full_metric_contribution'][k] for t in row['tiles']), total, rtol=1e-13)


def test_actual_cli_analyze_verify(actual, tmp_path):
    args, _, _ = actual
    out, svg = tmp_path/'analysis.json', tmp_path/'patches.svg'
    cli = ['analyze', '--manifest', str(args[0]), '--producer-report', str(args[1]), '--receipt', str(args[2]),
           '--report', str(out), '--svg', str(svg)]
    assert m.main(cli) == 0
    assert m.main(['verify', '--report', str(out), '--svg', str(svg)]) == 0
    with pytest.raises(ValueError, match='exclusive'): m.main(cli)


@pytest.mark.parametrize('fault', ['patch', 'tile', 'missingword', 'science', 'svg', 'authority'])
def test_actual_saved_mutations(actual, tmp_path, fault):
    _, value, picture = actual
    v = deepcopy(value)
    if fault == 'patch': v['rows'][0]['patch_channel_mean']['real_goal'][0][0] += 1
    elif fault == 'tile': v['rows'][0]['tiles'].pop()
    elif fault == 'missingword': v['rows'].pop()
    elif fault == 'science': v['scientific_admission'] = True
    elif fault == 'svg': picture += '<!--tamper-->'
    else: v['authority']['receipt'] = v['authority']['report']
    out, svg = tmp_path/'a.json', tmp_path/'a.svg'
    out.write_text(json.dumps(v)); svg.write_text(picture)
    with pytest.raises(ValueError): m.verify(out, svg)
