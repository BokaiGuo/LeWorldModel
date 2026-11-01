"""RAM fixtures only; no saved research outcome loading or test temp files."""
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.spectral_reachability import p1_moving_baseline_budget_matrix_20261027_v1 as m


def fixture():
    r = dict(schema=m.INPUT_SCHEMA, fixed_baselines=list(m.ARMS), cohort=list(m.ARMS),
        equal_total_compute=False, equal_search_domain=False, official_reproduction=False,
        independent_confirmation=False, P1_admission=False, P2_admission=False,
        manifest=dict(path='/ram/manifest.json', sha256='1'*64), rows=[], runs=[], physical_runs=[], RGB_runs=[],
        inherited_encoding_shared_once=m.actual({k: dict(attempted=8, completed=8)
            for k in ('encoder', 'projector')}, 'RAM'))
    counts = [136, 134, 72, 84, 1802, 88]
    for i, (a, n) in enumerate(zip(m.ARMS, counts)):
        sid = str(i)*64
        c = dict(horizon=2, packet_steps=5, history='H3 nonzero history',
            mapping='packet map', bank='fixed bank' if i in (3, 5) else 'not applicable',
            per_root_budget=dict(predict=n), numerical=dict(SDP='MATH'))
        r['runs'].append(dict(source_id=sid, arms=[a], contract=deepcopy(c),
            latency_seconds=m.unknown('not metered'), provenance=dict(report=dict(path='/ram/report'+str(i), sha256=sid))))
        for s in (0, 1):
            res = {k: m.actual(dict(attempted=0, completed=0), 'RAM explicit absence') for k in m.COUNTERS}
            for k in ('predict', 'action_encoder', 'forward_samples'):
                res[k] = m.actual(dict(attempted=n, completed=n), 'RAM receipts')
            rev = 32 if i == 0 else 900 if i == 4 else 192 if i == 5 else 0
            ad = 64 if i == 0 else 1800 if i == 4 else 2 if i == 5 else 0
            res.update(AD_forward_in_predict=m.actual(ad, 'included'), mathematical_reverse_directions=m.actual(rev, 'unit depends on method'),
                raw_predict_API_resources={}, raw_resources={})
            if i in (0, 4):
                res['backward'] = m.actual(dict(attempted=rev, completed=rev), 'scalar reverse')
            if i == 5:
                res['raw_resources']['vjp_directions'] = dict(attempted=192, completed=192)
                for k, v in [('jacobian', 1), ('svd', 2)]:
                    res[k] = m.actual(dict(attempted=v, completed=v), 'RAM')
                res['backward'] = m.unknown('no separate hook')
            res['pred_projector'] = (m.actual(dict(attempted=n, completed=n), 'RAM')
                if i == 3 else m.unknown('not metered'))
            r['rows'].append(dict(arm=a, source_slot=s, task_sha256=m.TASK_SHAS[s],
                source_id=sid, contract=deepcopy(c), resources=m.actual(res, 'RAM'),
                model_Q='POISON_OUTCOME_NOT_TO_BE_READ', native_success='POISON'))
    for key in ('physical_runs', 'RGB_runs'):
        for j, arms in enumerate((list(m.ARMS[:3]), [m.ARMS[3]], list(m.ARMS[4:]))):
            n = 2 * len(arms)
            run = dict(source_id=str(j+6)*64, arms=arms, provenance=dict(report=dict(path='/ram/'+key+str(j), sha256='a'*64)),
                latency_seconds=m.unknown('not measured'), native_successes='POISON')
            if key == 'physical_runs':
                run['counts'] = dict(physics_step=dict(attempted=204*n, completed=204*n),
                                     public_step=dict(attempted=20*n, completed=20*n))
            else:
                run['resources'] = {k: dict(attempted=2*n, completed=2*n) for k in ('encoder', 'projector')}
            r[key].append(run)
    return r


def render(r=None):
    return m.assemble(fixture() if r is None else r, dict(path='/ram/comparison.json', sha256=m.INPUT_SHA))


def test_six_methods_fixed_two_roots_counter_totals_no_module_sum():
    x = render()
    assert len(x['methods']) == 6 and len(x['pairwise_budget_contracts']) == 15
    assert [d['two_root_totals']['predict']['value']['completed'] for d in x['methods']] == [272,268,144,168,3604,176]
    for d in x['methods']:
        assert [z['source_slot'] for z in d['root_receipts']] == [0, 1]
        assert d['two_root_totals']['predict']['value'] == d['two_root_totals']['action_encoder']['value']
        assert 'total_forward' not in d


def test_derivative_units_and_overlap_are_not_added():
    x = render()['methods'];g,s = x[4]['two_root_totals'],x[5]['two_root_totals']
    assert g['predict']['value']['completed'] == 3604
    assert g['AD_forward_in_predict']['value'] == 3600
    assert g['scalar_loss_reverse_sweeps']['value'] == 1800
    assert g['scalar_output_VJP_directions']['value'] is None
    assert s['scalar_output_VJP_directions']['value'] == dict(attempted=384,completed=384)
    assert s['scalar_loss_reverse_sweeps']['value'] is None
    assert s['backward']['value'] is None and s['jacobian']['value']['completed'] == 2


def test_cohorts_once_single_arm_exact_multiarm_unallocated_encoding_once():
    x = render()
    assert len(x['physical_cohorts']) == len(x['RGB_cohorts']) == 3
    assert [r['counts']['physics_step']['completed'] for r in x['physical_cohorts']] == [1224,408,816]
    assert x['methods'][0]['per_method_physics']['value'] is None
    assert x['methods'][3]['per_method_physics']['value']['physics_step']['completed'] == 408
    assert x['inherited_encoding_shared_once']['value']['encoder']['completed'] == 8
    assert all(z['standalone_inherited_encoding']['value']['encoder']['completed']==8 for z in x['methods'])


def test_no_outcomes_or_scientific_winners_no_input_mutation():
    r=fixture();old=deepcopy(r);x=render(r);data=m.canonical(x)
    assert r==old and b'POISON' not in data and b'native_success' not in data and b'model_Q' not in data
    assert x['equal_total_compute'] is x['P1_admission'] is False
    assert len(data)+len(m.markdown(x)) < m.CAP
    assert 'NA不是0' in m.markdown(x).decode()


@pytest.mark.parametrize('status',['missing','partial','unmeasured'])
def test_unknown_one_root_never_sums_as_zero(status):
    r=fixture();r['rows'][0]['resources']['value']['predict']=m.unknown('absent',status)
    assert render(r)['methods'][0]['two_root_totals']['predict']['value'] is None


def test_missing_resource_envelope_and_horizon_remain_null():
    r=fixture()
    for row in r['rows'][:2]:
        row['resources']=m.unknown('partial','partial');row['contract'].pop('horizon')
    r['runs'][0]['contract'].pop('horizon')
    x=render(r)['methods'][0]
    assert x['future_packets']['value'] is None
    assert all(c['value'] is None for c in x['two_root_totals'].values())


def test_gradient_horizon_only_from_explicit_shapes_not_H3_label():
    r=fixture()
    for c in [r['runs'][0]['contract']]+[x['contract'] for x in r['rows'][:2]]:
        c.pop('horizon');c.update(logits_shape=[2,10],raw_commands_shape=[10,2])
    assert render(r)['methods'][0]['future_packets']['value']==2
    assert render()['methods'][4]['future_packets']['value']==2


def test_equal_predict_alone_never_certifies_equal_compute():
    r=fixture()
    for row in r['rows'][2:4]:
        row['resources']['value']['predict']=deepcopy(r['rows'][0]['resources']['value']['predict'])
    pair=render(r)['pairwise_budget_contracts'][0]
    assert pair['same_predict_counter'] is True and pair['equal_total_compute_established'] is False
    assert pair['equal_search_domain_established'] is False and len(pair['remaining_contracts'])==3


@pytest.mark.parametrize('edit',['schema','cohort','root_missing','root_duplicate','bool_slot','task','equal_compute',
    'negative','bool_counter','overcount','unknown_zero','bad_contract','extra_run','duplicate_physics','overlapping_RGB'])
def test_rejects_invented_or_mixed_resources(edit):
    r=fixture();v=r['rows'][0]['resources']['value']
    if edit=='schema':r['schema']='old'
    elif edit=='cohort':r['cohort'].reverse()
    elif edit=='root_missing':r['rows'].pop()
    elif edit=='root_duplicate':r['rows'][-1]=deepcopy(r['rows'][-2])
    elif edit=='bool_slot':r['rows'][0]['source_slot']=False
    elif edit=='task':r['rows'][0]['task_sha256']='f'*64
    elif edit=='equal_compute':r['equal_total_compute']=True
    elif edit=='negative':v['predict']['value']['attempted']=-1
    elif edit=='bool_counter':v['predict']['value']['attempted']=True
    elif edit=='overcount':v['predict']['value']['completed']=999
    elif edit=='unknown_zero':v['pred_projector']=dict(status='unmeasured',value=0,reason='absent')
    elif edit=='bad_contract':r['rows'][0]['contract']['horizon']=3
    elif edit=='extra_run':r['runs'].append(dict(source_id='z'*64))
    elif edit=='duplicate_physics':r['physical_runs'].append(deepcopy(r['physical_runs'][0]))
    else:r['RGB_runs'][1]['arms'].append(m.ARMS[0])
    with pytest.raises(ValueError):render(r)


def test_missing_downstream_has_no_cost_invented():
    r=fixture();r['physical_runs']=[];r['RGB_runs']=[]
    x=render(r)
    assert not x['physical_cohorts'] and not x['RGB_cohorts']
    assert all(d['per_method_physics']['value'] is None and not d['physics_cohort_ids'] for d in x['methods'])


def test_strict_external_SHA_and_duplicate_JSON(monkeypatch):
    raw=b'{"x":1,"x":2}'
    monkeypatch.setattr(Path,'stat',lambda p:SimpleNamespace(st_size=len(raw)))
    monkeypatch.setattr(Path,'read_bytes',lambda p:raw)
    with pytest.raises(ValueError,match='only approved'):m.read_pinned('/ram/x','b'*64)
    with pytest.raises(ValueError,match='SHA mismatch'):m.read_pinned('/ram/x',m.INPUT_SHA)
    with pytest.raises(ValueError,match='duplicate'):json.loads(raw,object_pairs_hook=m.unique_pairs)


def test_reserve_rejects_before_any_write(monkeypatch):
    monkeypatch.setattr(m,'read_pinned',lambda *a:(fixture(),dict(path='/ram/x',sha256=m.INPUT_SHA)))
    monkeypatch.setattr(Path,'exists',lambda p:False)
    monkeypatch.setattr(Path,'is_dir',lambda p:True)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=m.FLOOR))
    monkeypatch.setattr(Path,'mkdir',lambda p:pytest.fail('must not write'))
    with pytest.raises(ValueError,match='reserve'):m.run('/ram/x',m.INPUT_SHA,'/ram/new')


def test_cap_rejects_before_any_write(monkeypatch):
    monkeypatch.setattr(m,'CAP',1)
    monkeypatch.setattr(m,'read_pinned',lambda *a:(fixture(),dict(path='/ram/x',sha256=m.INPUT_SHA)))
    monkeypatch.setattr(Path,'mkdir',lambda p:pytest.fail('must not write'))
    with pytest.raises(ValueError,match='cap'):m.run('/ram/x',m.INPUT_SHA,'/ram/new')


def test_cli_exists_help_only(monkeypatch,capsys):
    monkeypatch.setattr('sys.argv',[m.NAME,'--help'])
    with pytest.raises(SystemExit) as e:m.main()
    assert e.value.code==0 and '--comparison-sha' in capsys.readouterr().out
