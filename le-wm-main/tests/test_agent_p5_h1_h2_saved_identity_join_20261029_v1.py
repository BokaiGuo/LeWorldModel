import copy

import pytest

from experiments.spectral_reachability import agent_p5_h1_h2_saved_identity_join_20261029_v1 as module


def fixture():
    rows = [dict(index=i, root_slot=i // 82, condition=('off', 'on')[(i // 41) % 2],
                 task_id=str(i), selection_digest='digest' + str(i)) for i in range(164)]
    pairs = [dict(slot=p, coord=j % 10, h=(0.01, 0.001)[j // 10],
                  minus_index=41*p+1+2*j, plus_index=41*p+2+2*j)
             for p in range(4) for j in range(20)]
    reports = {}
    for name, (stem, _, sha, status, report_status) in module.SOURCES.items():
        receipt = dict(schema=stem+'.consume', status=status, requests=164, pairs=80, report_sha256=sha)
        rpairs = [dict(x, root_slot=rows[x['minus_index']]['root_slot'],
                       condition=rows[x['minus_index']]['condition']) for x in pairs]
        report = dict(schema=stem+'.report', status=report_status, errors=[], rows=copy.deepcopy(rows), pairs=rpairs)
        reports[name] = receipt, report
    return dict(rows=rows, pairs=pairs), reports


def test_full_join_and_boundaries():
    physical, reports = fixture()
    result = module.join(physical, reports)
    assert (result['rows'], result['pairs'], result['fresh_roots']) == (164, 80, 0)
    assert not result['P5_admission'] and not result['full_strict_rerun']


@pytest.mark.parametrize('corruption', ['row', 'pair', 'condition', 'receipt', 'missing', 'errors'])
def test_rejects_identity_and_binding_corruption(corruption):
    physical, reports = fixture()
    receipt, report = reports['h2']
    if corruption == 'row': report['rows'][5]['selection_digest'] = 'wrong'
    if corruption == 'pair': report['pairs'][0]['plus_index'] = 4
    if corruption == 'condition': report['pairs'][0]['condition'] = 'on'
    if corruption == 'receipt': receipt['report_sha256'] = 'wrong'
    if corruption == 'missing': del reports['h1']
    if corruption == 'errors': report['errors'] = ['incomplete']
    with pytest.raises(ValueError):
        module.join(physical, reports)
