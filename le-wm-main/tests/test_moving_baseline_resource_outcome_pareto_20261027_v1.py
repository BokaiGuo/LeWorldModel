"""RAM metadata only, including IO mocks. No model/physics/scientific experiment."""
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import moving_baseline_resource_outcome_pareto_20261027_v1 as p


def ref(label):
    return dict(path='/ram/' + label + '.json', sha256=p.digest(label))


def fixture():
    b = p.budget
    tasks = []
    for s in (0, 1):
        contract = dict(observed_steps=[0, 5, 10], source_slot=s)
        task = dict(task_contract_sha256=p.own_plan_sha(contract), root_context=dict(latent_history=[[s, 0], [s, 1], [s, 2]]),
                    goal=[1, 1], horizon=2, packet_steps=5)
        tasks.append(dict(source_slot=s, with_target=False, task_id=p.own_plan_sha(task),
            blind_sha256=p.digest(['blind', s]), native_root_sha256=None, task_contract=contract,
            task=task))
    c = dict(schema=b.INPUT_SCHEMA, cohort=list(p.ARMS), fixed_baselines=list(p.ARMS),
        equal_total_compute=False, equal_search_domain=False, official_reproduction=False,
        independent_confirmation=False, P1_admission=False, P2_admission=False,
        manifest=ref('manifest'), rows=[], runs=[], physical_runs=[], RGB_runs=[],
        inherited_encoding_shared_once=b.actual({k: dict(attempted=8, completed=8)
            for k in ('encoder', 'projector')}, 'RAM shared actual receipt'))
    consumes, physical, rgb, arm_source = {}, {}, {}, {}
    for i, arms in enumerate(([p.ARMS[0]], list(p.ARMS[1:3]), [p.ARMS[3]], [p.ARMS[4]], [p.ARMS[5]])):
        sid = p.digest(['source', i])
        contract = dict(horizon=2, packet_steps=5, per_root_budget=dict(scope='RAM sealed'))
        c['runs'].append(dict(source_id=sid, arms=arms, contract=contract,
            provenance=dict(report=ref('model' + str(i)), consume=ref('consume' + str(i))),
            latency_seconds=b.unknown('not metered')))
        consumes[sid] = dict(rows=[])
        for arm in arms: arm_source[arm] = sid
    for arm_index, arm in enumerate(p.ARMS):
        sid = arm_source[arm]; run = next(r for r in c['runs'] if r['source_id'] == sid)
        for s in (0, 1):
            t = tasks[s]; n = [136, 134, 72, 84, 1802, 88][arm_index]
            commands = [[arm_index / 10, s / 10] for _ in range(10)]
            identity = dict(source_slot=s, arm=arm, task_id=t['task_id'], task_sha256=b.TASK_SHAS[s], blind_sha256=t['blind_sha256'])
            plan = dict(**identity, commands=commands, task_contract_sha256=t['task']['task_contract_sha256'],
                        model_terminal_cost=(arm_index + s + 1) / 10)
            if arm == p.ARMS[3]:
                plan.pop('task_contract_sha256'); plan['terminal_cost'] = plan.pop('model_terminal_cost')
            consumes[sid]['rows'].append(plan)
            resources = {k: b.actual(dict(attempted=0, completed=0), 'RAM explicit zero') for k in b.COUNTERS}
            for k in ('predict', 'forward_samples', 'action_encoder'):
                resources[k] = b.actual(dict(attempted=n, completed=n), 'RAM meter')
            resources.update(AD_forward_in_predict=b.actual(0, 'RAM schedule'),
                mathematical_reverse_directions=b.actual(0, 'RAM schedule'), raw_resources={}, raw_predict_API_resources={})
            resources['pred_projector'] = b.unknown('not measured')
            c['rows'].append(dict(**identity, source_id=sid, contract=deepcopy(run['contract']),
                model_status='actual', physical_status='actual', rgb_status='actual',
                resources=b.actual(resources, 'RAM model ledger'), latency_seconds=b.unknown('no per-root meter'),
                commands_sha256=b.actual(p.digest(commands), 'same command'),
                physical_selection_digest=b.actual(p.digest(['selection', s, arm]), 'RAM saved seal'),
                model_Q=b.actual((arm_index + s + 1) / 10, 'RAM selected prediction'),
                model_root_Q=b.actual(1 + s, 'RAM original root'), realQ_root=b.actual(1 + s, 'RAM root'),
                realQ_prefix5=b.actual(.5 + s, 'RAM actual prefix'), realQ_terminal10=b.actual(.2 + s + arm_index / 100, 'RAM actual terminal'),
                native_success=b.actual(False, 'RAM native receipt'), full_horizon=b.actual(True, 'RAM horizon'),
                physical_returned_steps=b.actual(10, 'RAM public returns'), RGB_latents={}))
    for ci, arms in enumerate((list(p.ARMS[:3]), [p.ARMS[3]], list(p.ARMS[4:]))):
        phid = p.digest(['physical', ci]); ph = dict(rows=[])
        rg_prov = dict(report=ref('RGB' + str(ci))); rgid = b.digest(rg_prov); rg = dict(rows=[])
        n = 2 * len(arms)
        c['physical_runs'].append(dict(source_id=phid, arms=arms, provenance=dict(report=ref('physical' + str(ci))),
            counts=dict(physics_step=dict(attempted=204*n, completed=204*n), public_step=dict(attempted=20*n, completed=20*n)),
            latency_seconds=b.unknown('no isolated latency')))
        c['RGB_runs'].append(dict(arms=arms, provenance=rg_prov,
            resources={k: dict(attempted=2*n, completed=2*n) for k in ('encoder', 'projector')},
            latency_seconds=b.unknown('no RGB latency')))
        ph['counts'] = deepcopy(c['physical_runs'][-1]['counts'])
        rg['resources'] = deepcopy(c['RGB_runs'][-1]['resources'])
        for s in (0, 1):
            for arm in arms:
                row = next(r for r in c['rows'] if r['source_slot'] == s and r['arm'] == arm)
                sid = arm_source[arm]; plan = next(r for r in consumes[sid]['rows'] if r['source_slot'] == s and r['arm'] == arm)
                run = next(r for r in c['runs'] if r['source_id'] == sid)
                row['physical_source_id'] = phid
                ph['rows'].append(dict(root_slot=s, arm=arm, root_state=[s, 3, 4], native_success=False,
                    full_horizon=True, own_returned=10, selection_digest=row['physical_selection_digest']['value'],
                    initial_metrics=dict(native_success=False, native_distance=30),
                    final_metrics=dict(native_success=False, native_distance=20 + s, native_angle_term=-.1,
                                       object_position_error=5, wrapped_object_angle_error=.1, object_success=None)))
                for role in ('prefix5', 'terminal10'):
                    latent = dict(file=arm + str(s) + role + '.npy', sha256=p.digest(['latent', s, arm, role]), dtype='float32', shape=[192])
                    identity = {k: row[k] for k in ('task_id', 'task_sha256', 'blind_sha256')}
                    item = dict(**identity, source_slot=s, root_slot=s, arm=arm, role=role, Q='identity192',
                        root=tasks[s]['task']['root_context']['latent_history'][-1], goal=tasks[s]['task']['goal'],
                        selection_digest=row['physical_selection_digest']['value'], own_plan_sha256=p.own_plan_sha(plan),
                        compiler_report_sha256=run['provenance']['report']['sha256'])
                    rg['rows'].append(dict(**identity, source_slot=s, arm=arm, role=role, status='encoded', provenance=item,
                        metrics=dict(root_Q=1+s, actual_Q=row['realQ_' + role]['value']), latent=latent))
                    row['RGB_latents'][role] = latent
        physical[phid], rgb[rgid] = ph, rg
    matrix = b.assemble(c, dict(path='/ram/comparison.json', sha256=p.COMPARISON_SHA))
    return matrix, c, tasks, consumes, physical, rgb


def test_complete_saved_join_no_mutation_and_no_fabricated_root_hash():
    bundle = fixture(); old = deepcopy(bundle); x = p.assemble(*bundle)
    assert bundle == old and len(x['rows']) == 12 and len(x['pairwise']) == 30
    assert all(r['binding']['upstream_native_root_sha256']['value'] is None for r in x['rows'])
    assert all(r['binding']['physical_root_state_sha256'] for r in x['rows'])
    assert all(r['binding_sha256'] == p.digest(r['binding']) for r in x['rows'])
    assert [r['costs']['predict']['value']['completed'] for r in x['rows']] == [136,136,134,134,72,72,84,84,1802,1802,88,88]
    assert all(z['status'] == 'unknown' for z in x['pairwise'])
    assert all(z['status'] == 'unknown' for z in x['frontier'])
    assert x['equal_total_compute'] is x['global_equal_budget_winner'] is x['author_reported_values_used'] is False
    assert b'NA' in p.markdown(x) or b'unknown' in p.markdown(x)


def test_unique_cohorts_shared_model_and_overlap_no_refunds():
    x = p.assemble(*fixture()); entries = x['ledger']['entries']
    assert len(entries) == 17  # inherited1 + model5 + consumerCPU5 + physical3 + RGB3
    assert len([e for e in entries if e['kind'] == 'inherited_encoding']) == 1
    assert len([e for e in entries if e['kind'] == 'physics']) == 3
    sampling = next(e for e in entries if e['kind'] == 'model' and e['arms'] == list(p.ARMS[1:3]))
    assert sampling['counters']['predict']['value']['completed'] == 412
    assert entries[0]['receipt']['value']['encoder']['completed'] == 8
    assert x['ledger']['cross_unit_total'] is None
    assert any(e['relation'] == 'subset_not_extra_forward' for e in x['ledger']['overlap'])
    assert all(r['charge_semantics'] == 'view_only_no_new_charge_no_refund' for r in x['rows'])


def complete_pair():
    a = deepcopy(p.assemble(*fixture())['rows'][0]); b = deepcopy(a)
    for axis in p.AXES:
        value = (dict(Q_scores=dict(attempted=2,completed=2), SVD=dict(attempted=2,completed=2))
                 if axis in p.PROFILES else dict(attempted=2, completed=2) if axis in p.COUNTER_COSTS else 2)
        for r in (a, b):
            r['costs'][axis] = dict(**p.actual(value, 'RAM complete MOCK'), unit=axis, scope='RAM same pipeline')
    return a, b


@pytest.mark.parametrize('mode,expected', [('equal','tie'), ('cheaper','a_dominates_b'),
    ('better','a_dominates_b'), ('reverse','b_dominates_a'), ('tradeoff','nondominated_pair')])
def test_dominance_math_on_complete_ram_vectors(mode, expected):
    a, b = complete_pair()
    if mode in ('cheaper','tradeoff'):
        a['costs']['svd']['value'] = dict(attempted=1, completed=1)
    if mode == 'better': a['outcomes']['native_success']['value'] = True
    if mode == 'reverse': b['outcomes']['realQ_terminal10']['value'] = .1
    if mode == 'tradeoff': a['outcomes']['realQ_terminal10']['value'] = 3
    assert p.dominance(a, b)['status'] == expected


@pytest.mark.parametrize('axis', p.AXES)
def test_any_unknown_cost_axis_prevents_all_verdicts_even_known_loss(axis):
    a, b = complete_pair(); a['costs'][axis] = p.unknown('not metered')
    b['outcomes']['realQ_terminal10']['value'] = 100  # no partial-axis dominance
    assert p.dominance(a, b)['status'] == 'unknown'


@pytest.mark.parametrize('key', ['unit','scope'])
def test_unit_scope_mismatch_unknown(key):
    a, b = complete_pair(); a['costs']['predict'][key] = 'different'
    assert p.dominance(a,b)['status'] == 'unknown'


@pytest.mark.parametrize('edit', ['context','derivatives','partial','outcome','horizon'])
def test_incompatible_or_incomplete_evidence_unknown(edit):
    a,b = complete_pair()
    if edit == 'context': a['comparison_context']['goal_sha256'] = 'f'*64
    elif edit == 'derivatives': a['derivative_semantics'] = 'output_VJP'
    elif edit == 'partial': a['costs']['predict']['value']['attempted'] = 3
    elif edit == 'outcome': a['outcomes']['native_success'] = p.unknown('missing')
    else: a['outcomes']['full_horizon']['value'] = False
    assert p.dominance(a,b)['status'] == 'unknown'


@pytest.mark.parametrize('bad', [True, -1, float('nan'), float('inf')])
def test_bad_cost_values_fail(bad):
    a,b = complete_pair(); a['costs']['pipeline_walltime_seconds']['value'] = bad
    with pytest.raises(ValueError): p.dominance(a,b)


def test_missing_zero_boolean_and_counter_rejected_signed_angle_allowed():
    with pytest.raises(ValueError): p.typed(dict(status='missing',value=0,reason='none'))
    with pytest.raises(ValueError): p.typed(p.actual(1,'mock'),'bool')
    with pytest.raises(ValueError): p.typed(p.actual(dict(attempted=True,completed=0),'mock'),'counter')
    assert p.typed(p.actual(-.3,'literal signed native angle'),'signed')['value'] == -.3


def test_cpu_operation_profile_never_scalarized_or_partially_accepted():
    a,b=complete_pair()
    a['costs']['CPU_complete_profile']['value']=4
    with pytest.raises(ValueError,match='profile'): p.dominance(a,b)
    a,b=complete_pair()
    a['costs']['CPU_complete_profile']['value'].pop('SVD')
    assert p.dominance(a,b)['status']=='unknown'
    a,b=complete_pair()
    a['costs']['CPU_complete_profile']['value']['SVD']['attempted']=3
    assert p.dominance(a,b)['status']=='unknown'


@pytest.mark.parametrize('edit',['extra_physical','physical_count','extra_RGB','RGBcount','native_int','bool_slot'])
def test_exact_cohort_and_literal_receipt_types_required(edit):
    bundle=fixture(); ph=next(iter(bundle[4].values())); rg=next(iter(bundle[5].values()))
    if edit=='extra_physical': ph['rows'].append(dict(ph['rows'][0],arm='extraneous'))
    elif edit=='physical_count': ph['counts']['public_step']['completed']-=1
    elif edit=='extra_RGB': rg['rows'].append(dict(rg['rows'][0],arm='extraneous'))
    elif edit=='RGBcount': rg['resources']['encoder']['completed']-=1
    elif edit=='native_int': ph['rows'][0]['native_success']=0
    else: next(iter(bundle[3].values()))['rows'][0]['source_slot']=False
    with pytest.raises(ValueError): p.assemble(*bundle)


@pytest.mark.parametrize('edit', ['matrix_cost','matrix_source','comparison_arm','task','task_contract','blind',
    'commands','selection','modelQ','native','realQ','RGBgoal','RGBplan','RGBlatent','native_root','duplicate_plan','duplicate_physical','duplicate_RGB'])
def test_sha_and_identity_join_fail_closed(edit):
    m,c,t,cons,phys,rgb = fixture(); row = c['rows'][0]; sid = row['source_id']
    plan = cons[sid]['rows'][0]; ph = next(iter(phys.values())); rg = next(iter(rgb.values()))
    if edit == 'matrix_cost': m['methods'][0]['root_receipts'][0]['counters']['predict']['value']['completed'] = 1
    elif edit == 'matrix_source': m['methods'][0]['source_id'] = 'f'*64
    elif edit == 'comparison_arm': row['arm'] = 'reported_author_arm'
    elif edit == 'task': t[0]['task_id'] = 'f'*64
    elif edit == 'task_contract': t[0]['task']['task_contract_sha256'] = 'f'*64
    elif edit == 'blind': plan['blind_sha256'] = 'f'*64
    elif edit == 'commands': plan['commands'][0][0] = .77
    elif edit == 'selection': ph['rows'][0]['selection_digest'] = 'f'*64
    elif edit == 'modelQ': plan['model_terminal_cost'] = 99
    elif edit == 'native': ph['rows'][0]['native_success'] = True
    elif edit == 'realQ': rg['rows'][0]['metrics']['actual_Q'] = 999
    elif edit == 'RGBgoal': rg['rows'][0]['provenance']['goal'] = [9,9]
    elif edit == 'RGBplan': rg['rows'][0]['provenance']['own_plan_sha256'] = 'f'*64
    elif edit == 'RGBlatent': rg['rows'][0]['latent'] = dict(rg['rows'][0]['latent'],sha256='f'*64)
    elif edit == 'native_root': ph['rows'][1]['root_state'] = [99,3,4]
    elif edit == 'duplicate_plan': cons[sid]['rows'].append(deepcopy(plan))
    elif edit == 'duplicate_physical': ph['rows'].append(deepcopy(ph['rows'][0]))
    else: rg['rows'].append(deepcopy(rg['rows'][0]))
    with pytest.raises(ValueError): p.assemble(m,c,t,cons,phys,rgb)


def test_reported_poison_fields_not_used_and_native_null_not_filled():
    bundle = fixture(); bundle[1]['author_reported_values'] = dict(native_success=1,realQ=0)
    next(iter(bundle[4].values()))['rows'][0]['final_metrics']['object_position_error'] = None
    x = p.assemble(*bundle)
    assert x['rows'][0]['outcomes']['native_success']['value'] is False
    assert x['rows'][0]['outcomes']['object_position_error']['value'] is None
    assert 'author_reported_values' not in x


def test_duplicate_ledger_receipt_rejected():
    m,*_ = fixture()
    m['physical_cohorts'][1]['provenance'] = deepcopy(m['physical_cohorts'][0]['provenance'])
    with pytest.raises(ValueError,match='receipt charged twice'): p.ledger(m)


def test_saved_reader_ram_sha_duplicate_nonfinite_gzip_caps_and_scope(monkeypatch):
    data = b'{"x":1}\n'; virtual = Path('/ram/results/file.json')
    monkeypatch.setattr(Path,'resolve',lambda x:x)
    monkeypatch.setattr(Path,'is_file',lambda _:True)
    monkeypatch.setattr(Path,'is_symlink',lambda _:False)
    monkeypatch.setattr(Path,'stat',lambda _:SimpleNamespace(st_size=len(data)))
    monkeypatch.setattr(Path,'read_bytes',lambda _:data)
    reader=p.SavedReader('/ram/results')
    def load(): return reader.load(dict(path=str(virtual),sha256=p.sha(data)))
    assert load()==dict(x=1); reader.recheck()
    with pytest.raises(ValueError,match='SHA mismatch'): reader.raw(dict(path=str(virtual),sha256='f'*64))
    with pytest.raises(ValueError): reader.raw(dict(path='/outside/file',sha256=p.sha(data)))
    for raw in (b'{"x":1,"x":2}',b'{"x":NaN}'):
        data=raw; reader=p.SavedReader('/ram/results')
        with pytest.raises(ValueError): load()
    virtual=Path('/ram/results/task.json.gz'); data=gzip.compress(b'{"task":1}')
    reader=p.SavedReader('/ram/results'); assert load()==dict(task=1)
    data=gzip.compress(b'x'*1000); monkeypatch.setattr(p,'CAP',100)
    reader=p.SavedReader('/ram/results')
    with pytest.raises(ValueError,match='decompressed'): load()


def test_output_exclusive_scope_and_no_overwrite():
    for path in ('/tmp/' + p.STEM + '_new', str(p.ROOT/'results/old_result'),
                 str(p.ROOT/'results/p1_moving_baseline_budget_matrix_20261027_v1_main01')):
        with pytest.raises(ValueError,match='exclusive'): p.run('/unused','/unused',path)


def test_run_io_mock_generates_exact_two_files_and_rechecks(monkeypatch):
    report = p.assemble(*fixture()); writes={}; rechecks=[]
    reader=SimpleNamespace(refs={'/ram/input':'a'*64}, recheck=lambda:rechecks.append(True))
    monkeypatch.setattr(p,'load_bundle',lambda *_:(reader,fixture()))
    monkeypatch.setattr(p,'assemble',lambda *_:deepcopy(report))
    monkeypatch.setattr(p.Path,'exists',lambda _:False)
    monkeypatch.setattr(p.Path,'is_symlink',lambda _:False)
    monkeypatch.setattr(p.Path,'mkdir',lambda _:None)
    monkeypatch.setattr(p.Path,'read_bytes',lambda _:b'RAM implementation')
    monkeypatch.setattr(p.shutil,'disk_usage',lambda _:SimpleNamespace(free=10**10))
    monkeypatch.setattr(p.os,'fsync',lambda _:None)
    monkeypatch.setattr(p.os,'open',lambda *_:123)
    monkeypatch.setattr(p.os,'close',lambda _:None)
    class File(io.BytesIO):
        def __init__(self,path): super().__init__(); self.path=path
        def __exit__(self,*_): writes[self.path.name]=self.getvalue(); self.close()
        def fileno(self): return 123
    def op(path,mode):
        assert mode=='xb'
        return File(path)
    monkeypatch.setattr(p.Path,'open',op)
    result=p.run('matrix','comparison',p.ROOT/'results'/ (p.STEM+'_RAM'))
    assert set(writes)=={'pareto.json','pareto.md'} and len(rechecks)==2
    assert result['rows']==12 and result['unknown_pairs']==30
    assert json.loads(writes['pareto.json'])['inputs_sha256']==reader.refs
    monkeypatch.setattr(p.shutil,'disk_usage',lambda _:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='reserve'): p.run('m','c',p.ROOT/'results'/(p.STEM+'_RAM'))
