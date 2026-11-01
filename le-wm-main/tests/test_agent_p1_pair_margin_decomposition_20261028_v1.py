"""Small saved-only fixtures; no production build, model or environment execution."""
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p1_pair_margin_decomposition_20261028_v1 as m


def vec(value=0):
    return np.full(192, value, dtype='<f4')


def inputs(i=0):
    anchor, word = divmod(i, 5)
    task, role = divmod(anchor, 2)
    prefix = np.zeros((5, 2), '<f4')
    suffix = np.full((5, 2), word / 8, '<f4')
    goal = vec()
    pred = np.stack((vec(9), vec(8), vec(1 + word / 8)))
    actual = None if i == 12 else vec(1 + (4-word) / 8)
    context = {'latent_history': [vec().tolist()]*3, 'action_history': [[0.]*10]*2}
    blind = dict(context=context, goal=goal.tolist(), q=np.eye(192).tolist())
    plan = dict(index=i, root_slot=task//2, condition=('off', 'on')[task%2], role=('Z', 'S')[role],
                library_index=word, library_label=m.LIBRARY[word], anchor_id=f'a{anchor}',
                task_id=f't{task}', blind_sha256=f'{anchor+1:064x}', teacher_context_sha256=m.digest(context),
                prefix_commands=prefix.tolist(), suffix_commands=suffix.tolist(),
                commands=np.concatenate((prefix, suffix)).tolist(), model_Q=float(np.mean(pred[-1].astype(float)**2)))
    plan['plan_digest'] = m.digest(plan)
    pw = dict(goal_sha256=m.array_sha(goal), shifted_context={'file': f'p{i}'})
    c = dict(original_plan=plan, prediction_witness=pw, prediction_digest=m.digest(pw),
             suffix_raw_bits=suffix.tobytes().hex(), full10_raw_bits=np.concatenate((prefix,suffix)).tobytes().hex())
    native_goal = [0.]*7
    state = [float(word)]*7
    metrics = {k: None if 'coverage' in k else False if k == 'task_success' else float(word) for k in m.METRICS}
    row = {k: deepcopy(plan[k]) for k in m.IDENTITY}
    row.update(job_digest=f'{i+100:064x}', goal192=goal.tolist(), physical_goal=native_goal,
               outline_pose=[256.,256.,.5], goal_binding=dict(state=native_goal,pose=[256.,256.,.5],pixel_sha256='e'*64),
               array_receipts=dict(goal={'file': f'g{anchor}'},prediction=pw['shifted_context'],
                                   actual=None if i == 12 else {'file': f'a{i}'}),
               suffix_raw_bits=suffix.tobytes().hex(), predicted_Q=plan['model_Q'],
               actual_Q=None if actual is None else float(np.mean(actual.astype(float)**2)),
               actual_return_absolute_step=20, actual_return_event_sha256=f'{i+200:064x}',
               returned_state=state, physical_metrics=metrics, native_success=False,
               missing_reason='guard_rejected' if i==12 else None,
               boundary_trace={str(t): dict(state=state,absolute_step=t) for t in (10,15,20)})
    join = dict(index=i,plan_digest=plan['plan_digest'],prediction_digest=c['prediction_digest'],
                job_digest=row['job_digest'], teacher_context_sha256=plan['teacher_context_sha256'],
                prefix_sha256=m.array_sha(prefix),suffix_sha256=m.array_sha(suffix),
                full10_sha256=m.array_sha(np.concatenate((prefix,suffix))),
                latent_goal_sha256=m.array_sha(goal), native_goal=native_goal,
                actual_start15=state, actual_root10=state,actual_return_event_sha256=row['actual_return_event_sha256'],
                actual20_verified=True, exact_existing_execution_reuse=True,newly_executed=False)
    # JSON files are independent objects: mutating a join must not mutate its authority too.
    return tuple(deepcopy(x) for x in (row, join, c, blind, goal, pred, actual))


def slots():
    return [m.validate_slot(*inputs(i)) for i in range(40)]


def test_exact_mse_normalization_not_sum_or_double_division():
    p = vec(); p[0] = 12
    x = m.decompose(p, vec(), vec())
    assert m.unpack(x['exact']['predicted_Q']) == Fraction(144,192)
    assert x['float64']['predicted_Q'] == .75
    assert m.unpack(x['exact']['identity_residual']) == 0


def test_pair_exact_identity_and_common_response_factorization():
    a,b = slots()[:2]
    p = m.pair(a,b)
    e = {k:m.unpack(v) for k,v in p['exact'].items()}
    assert e['observed_minus_predicted_margin'] == e['cross_i']-e['cross_j']+e['quadratic_i']-e['quadratic_j']
    assert e['observed_minus_predicted_margin'] == e['common_bias_projection']+e['relative_response_projection']+e['relative_response_quadratic']
    assert e['identity_residual'] == e['factorization_residual'] == 0
    assert p['model_to_observed']['classification'] == 'discordant'


def test_unknown12_preserves_all80_and_physical_comparisons():
    report = m.assemble(slots())
    assert len(report['slots']) == 40 and len(report['pairs']) == 80
    assert report['summary']['model_to_observed']['available'] == 76
    assert report['summary']['model_to_observed']['unknown'] == 4
    unknown = [p for p in report['pairs'] if p['missing_indices']]
    assert [(p['i'],p['j']) for p in unknown] == [(10,12),(11,12),(12,13),(12,14)]
    assert all(p['exact'] is None and p['float64'] is None and p['missing_indices'] == [12] for p in unknown)
    assert all(p['model_to_physical']['all_xy_error']['available'] for p in unknown)
    assert all(not p['observed_to_physical']['all_xy_error']['available'] for p in unknown)
    assert all('_math' not in row for row in report['slots'])
    assert len(m.canonical(report)) < 1024**2


def test_metric_axes_are_separate_and_success_ties_not_wins():
    p = m.pair(*slots()[:2])
    assert p['observed_to_physical']['task_success']['classification'] == 'right_tie_only'
    assert p['observed_to_physical']['native_coverage']['classification'] == 'unknown'
    assert set(p['model_to_physical']) == set(m.METRICS)


@pytest.mark.parametrize('change', ['time','context','raw_zero_sign','goal_zero_sign','state_zero_sign',
                                  'physical_goal','identity','prediction_receipt','predicted_cost','actual_cost',
                                  'metric','missing_actual','q','goal_shape','pred_shape','actual_dtype'])
def test_slot_rejects_coordinate_identity_and_bit_mutations(change):
    r,j,c,b,g,p,a = deepcopy(inputs())
    if change == 'time': r['actual_return_absolute_step'] = 15
    elif change == 'context': b['context']['action_history'][0][0] = -0.
    elif change == 'raw_zero_sign': c['original_plan']['suffix_commands'][0][0] = -0.
    elif change == 'goal_zero_sign': g[0] = -0.
    elif change == 'state_zero_sign': j['actual_start15'][0] = -0.
    elif change == 'physical_goal': r['physical_goal'][0] = 1.
    elif change == 'identity': r['task_id'] = 'other'
    elif change == 'prediction_receipt': r['array_receipts']['prediction'] = {'file':'other'}
    elif change == 'predicted_cost': r['predicted_Q'] = np.nextafter(r['predicted_Q'],2.)
    elif change == 'actual_cost': r['actual_Q'] = np.nextafter(r['actual_Q'],3.)
    elif change == 'metric': r['physical_metrics']['all_xy_error'] = float('nan')
    elif change == 'missing_actual': a = None
    elif change == 'q': b['q'][0][0] = 2.
    elif change == 'goal_shape': g = g[None]
    elif change == 'pred_shape': p = p[-1]
    else: a = a.astype(np.float64)
    with pytest.raises(ValueError): m.validate_slot(r,j,c,b,g,p,a)


@pytest.mark.parametrize('change', ['drop','reorder','fill12','cross_goal','cross_context','physical_goal_sign'])
def test_full_roster_and_pair_stratum_fail_closed(change):
    ss = slots()
    if change == 'drop': ss.pop()
    elif change == 'reorder': ss[0],ss[1] = ss[1],ss[0]
    elif change == 'fill12': ss[12]['actual_Q'] = 0.
    elif change == 'cross_goal': ss[1]['goal_sha256'] = '0'*64
    elif change == 'cross_context': ss[1]['context_sha256'] = '0'*64
    else: ss[1]['physical_goal_sha256'] = '0'*64
    with pytest.raises(ValueError): m.assemble(ss)


def test_rounding_is_descriptive_not_tolerance_or_selection():
    p = vec(); p[0] = np.float32(.1)
    a = vec(); a[0] = np.float32(.3)
    d = m.decompose(p,a,vec())
    assert m.unpack(d['exact']['identity_residual']) == 0
    assert set(d['float64']) >= {'identity_residual','rounding_vs_exact'}
    assert m.classify(0.,-0.)['classification'] == 'both_tie'
    assert m.classify(np.nextafter(0.,1.),-np.nextafter(0.,1.))['classification'] == 'discordant'


def receipt(tmp_path, a):
    buf = BytesIO(); np.save(buf,a,allow_pickle=False); raw=buf.getvalue()
    p=tmp_path/'array.npy.gz'; p.write_bytes(gzip.compress(raw))
    return dict(file=p.name,file_sha256=m.sha(p),npy_sha256=hashlib.sha256(raw).hexdigest(),
                array_sha256=m.array_sha(a),dtype=str(a.dtype),shape=list(a.shape))


@pytest.mark.parametrize('change',[None,'file','npy','array','dtype','shape','escape','trailing'])
def test_saved_array_loader_binds_all_layers(tmp_path,change):
    meta=receipt(tmp_path,vec())
    if change in ('file','npy','array'): meta[change+'_sha256' if change!='file' else 'file_sha256']='0'*64
    elif change=='dtype':meta['dtype']='float64'
    elif change=='shape':meta['shape']=[1,192]
    elif change=='escape':meta['file']='../array.npy.gz'
    elif change=='trailing':
        p=tmp_path/meta['file'];raw=gzip.decompress(p.read_bytes())+b'x';p.write_bytes(gzip.compress(raw))
        meta['file_sha256']=m.sha(p);meta['npy_sha256']=hashlib.sha256(raw).hexdigest()
    if change:
        with pytest.raises(ValueError):m.load_array(tmp_path,meta)
    else:assert m.load_array(tmp_path,meta).tobytes()==vec().tobytes()


def test_verify_rebuild_and_compact_exclusive_output(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    report=m.assemble(slots());p=tmp_path/'results'/(m.NAME+'_fixture.json')
    m.write(p,report)
    assert m.verify(p,m.sha(p),report)['verified']
    with pytest.raises(FileExistsError):m.write(p,report)
    changed=deepcopy(report);changed['pairs'][0]['exact']['identity_residual']['numerator']='1'
    p.write_bytes(m.canonical(changed))
    with pytest.raises(ValueError):m.verify(p,m.sha(p),report)
    with pytest.raises(ValueError):m.write(tmp_path/'results/old.json',report)


def test_load_authority_must_rebuild_not_hash_only(monkeypatch):
    from types import SimpleNamespace
    original=dict(rows=[1],nested={'signed':0.})
    fake=SimpleNamespace(teacher=SimpleNamespace(read=lambda *args:original),build=lambda:deepcopy(original))
    monkeypatch.setattr(m,'upstream',lambda:fake)
    assert m.load_authority()[0] == original
    fake.build=lambda:dict(rows=[1],nested={'signed':-0.})
    with pytest.raises(ValueError):m.load_authority()


def test_core_has_no_neural_or_physics_imports():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    modules=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    modules += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert not any(s.startswith(('torch','gym','pymunk','mujoco','stable_worldmodel')) for s in modules)


@pytest.mark.parametrize('change',[None,'authority_flags','row_order','blind_Q','array_signed_goal','candidate_goal','native_time'])
def test_full_build_with_original_saved_layout_and_real_compressed_arrays(tmp_path,monkeypatch,change):
    from types import SimpleNamespace
    cf,af = tmp_path/'compiler',tmp_path/'actual'
    cf.mkdir();af.mkdir()
    def save(folder,name,a):
        buf=BytesIO();np.save(buf,a,allow_pickle=False);raw=buf.getvalue();p=folder/name
        p.write_bytes(gzip.compress(raw))
        return dict(file=name,file_sha256=m.sha(p),npy_sha256=hashlib.sha256(raw).hexdigest(),
                    array_sha256=m.array_sha(a),shape=list(a.shape),dtype=str(a.dtype))
    rows,joins,candidates,blind_inputs=[],[],[],[]
    for i in range(40):
        r,j,c,b,g,p,a=inputs(i)
        r['array_receipts']=dict(goal=save(cf,f'g{i}.gz',g),prediction=save(cf,f'p{i}.gz',p),
                                 actual=None if a is None else save(af,f'a{i}.gz',a))
        c['prediction_witness']['shifted_context']=deepcopy(r['array_receipts']['prediction'])
        c['prediction_digest']=m.digest(c['prediction_witness']);j['prediction_digest']=c['prediction_digest']
        rows.append(r);joins.append(j);candidates.append(c)
        if i%5==0:blind_inputs.append(dict(blind_input=b))
    authority=dict(schema='agent_p1_model_library_selector_evaluation_20261028_v1.evaluation',
        status='exact_seen_execution_reuse_exploratory_not_admitted',fixed_slots=40,actual_latent_available=39,
        missing_actual_latent=[12],known_roots=2,fresh_roots=0,fresh_confirmation=False,P1_admission=False,
        bindings={'one_authority': 'f'*64},rows=rows,exact_reuse_witnesses=joins)
    selector_path=tmp_path/'selector.json';model_path=cf/'report.json'
    def read(path,*args):
        if path==selector_path:return dict(candidates=candidates)
        assert path==model_path
        return dict(rows=blind_inputs)
    old=SimpleNamespace(teacher=SimpleNamespace(read=read),SELECTOR=selector_path,SELECTOR_SHA='s'*64,PANEL_SHA='p'*64,
        selector=SimpleNamespace(REPORT=model_path,REPORT_SHA='m'*64,RECEIPT_SHA='c'*64),
        panel=SimpleNamespace(base=SimpleNamespace(rank=SimpleNamespace(REPORT=af/'report.json',REPORT_SHA='a'*64,RECEIPT_SHA='r'*64))))
    if change=='authority_flags':authority['fresh_confirmation']=True
    elif change=='row_order':rows[0],rows[1]=rows[1],rows[0]
    elif change=='blind_Q':blind_inputs[0]['blind_input']['q'][0][0]=2.
    elif change=='array_signed_goal':
        g=vec();g[0]=-0.;rows[0]['array_receipts']['goal']=save(cf,'mutated_goal.gz',g)
    elif change=='candidate_goal':candidates[0]['prediction_witness']['goal_sha256']='0'*64
    elif change=='native_time':rows[0]['actual_return_absolute_step']=15
    monkeypatch.setattr(m,'load_authority',lambda:(authority,old))
    if change:
        with pytest.raises(ValueError):m.build()
    else:
        report=m.build()
        assert report['authority']['full_saved_native_rebuilt']
        assert report['summary']['model_to_observed']['available']==76
        assert len(report['pairs'])==80 and report['new_neural_calls']==report['new_physics_calls']==0
        assert report['protocol']['MSE_divisor']==192
        assert 'bindings' not in report and len(m.canonical(report))<1024**2


def test_pure_math_perturbed_case_matches_direct_fraction_margin():
    a,b=slots()[:2]
    g=vec(.03);p=vec(.1);q=vec(.7);x=vec(.55);y=vec(.21)
    for row,pred,obs in ((a,p,x),(b,q,y)):
        terms=m.decompose(pred,obs,g)
        row.update(terms);row['predicted_Q']=terms['float64']['predicted_Q'];row['actual_Q']=terms['float64']['observed_Q']
    pair=m.pair(a,b);e={k:m.unpack(v) for k,v in pair['exact'].items()}
    pg, qg = Fraction(float(p[0]))-Fraction(float(g[0])),Fraction(float(q[0]))-Fraction(float(g[0]))
    xg, yg = Fraction(float(x[0]))-Fraction(float(g[0])),Fraction(float(y[0]))-Fraction(float(g[0]))
    assert e['predicted_margin']==pg**2-qg**2
    assert e['observed_margin']==xg**2-yg**2
    assert e['identity_residual']==e['factorization_residual']==0


def test_consume_rebuild_required_and_no_admission(tmp_path,monkeypatch):
    report=m.assemble(slots());p=tmp_path/'own.json';p.write_bytes(m.canonical(report));calls=[]
    def build():calls.append(1);return deepcopy(report)
    monkeypatch.setattr(m,'build',build)
    out=m.consume(p,m.sha(p))
    assert calls==[1] and out['comparable_pairs']==76 and out['unknown_pairs']==4
    assert out['P1_admission'] is out['C2_claim'] is out['fresh_confirmation'] is False
