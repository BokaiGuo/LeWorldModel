"""RAM JSON metadata only. No simulator, CUDA or neural calls."""
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_subgoal_gap_assets_20261028_v1 as m


def fixture(tmp_path):
    def put(name,body):
        p=tmp_path/(name+'.json');data=m.c.encode(body)
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
        return m.ref(p)
    model=put('checkpoint_metadata',dict(RAM_ONLY=True));normal=dict(mean=[0.,0.],std=[1.,1.])
    evaluator=put('native_source_metadata',dict(RAM_ONLY=True))
    metric=dict(sha256='a'*64,projection=[[1.,0.]])
    cases=[];rows=[];compiled={}
    for i in range(4):
        root=dict(root_id=str(i),context=dict(latent_history=[[float(i),0.]],action_history=[]))
        domain=dict(normalization_sha256=m.c.digest(normal))
        h=m.c.digest(root);cases.append(dict(root_index=i,root_sha256=h,root=root|dict(goal_rgb_sha256='b'*64),
            metric_sha256='a'*64,action_domain=domain))
        root=cases[-1]['root'];w=dict(receipt=dict(root=root,provenance=dict(checkpoint_sha256=model['sha256'],
            encoder_sha256=model['sha256'],metric_sha256='a'*64)),edges=[dict(output_context=root['context'])])
        attempts=[dict(index=j,completed=True,witness=deepcopy(w),actions=[[0.,0.]]) for j in range(40)]
        rows.append(dict(root_slot=i,root_sha256=h,complete=True,faults=dict(primary=None,secondary=[]),
            results=[dict(attempts=[{}]*80,complete=True,error=None),dict(attempts=attempts,complete=True,error=None)]))
        compiled[h]=dict(scoring=dict(metric=metric),goal_vectors=[[1.,2.]])
    protocol=dict(context=dict(cases=cases,normalization=normal))
    report=dict(status='known_root_completed',error=None,model_unchanged=True,source_unchanged=True)
    return [protocol,report,rows,compiled,dict(source=evaluator)],dict(bindings={model['path']:model['sha256']},put=put)


def test_actual_runner_prepare_seam_stays_blocked(tmp_path):
    args,kw=fixture(tmp_path);spec,inventory=m.assemble(*args,**kw);p=m.r.prepare(spec)
    assert p['status']=='blocked' and p['trial_denominator']==4 and p['arm_denominator']==12
    assert spec['assets']['learned_checkpoint'] is spec['assets']['learned_pairing'] is None
    assert spec['assets']['controller'] is spec['assets']['adapter_source'] is None
    assert all(t['proposal_inputs'][m.c.ARMS[1]] is None for t in spec['trials'])
    assert all(x['compiled_candidate_index']==0 and x['native_success'] is None for x in inventory['roots'])
    assert all(x['flops'] is None and x['predict_calls']==0 for x in spec['phase_caps'].values())
    m.r.verify(p,m.c.digest(p))
    with pytest.raises(ValueError,match='missing assets/controller'):
        m.r.run_fixture(p,m.c.digest(p),None,lambda e:pytest.fail('never dispatch'))


@pytest.mark.parametrize('damage',['root_drop','row_reorder','failed','missing_candidate','duplicate','history','metric','model','index'])
def test_mutations_fail_closed(tmp_path,damage):
    args,kw=fixture(tmp_path);p,report,rows,compiled,_=args
    if damage=='root_drop':rows.pop()
    elif damage=='row_reorder':rows.reverse()
    elif damage=='failed':report['error']='preserved failure'
    elif damage=='missing_candidate':rows[0]['results'][1]['attempts'].pop()
    elif damage=='duplicate':p['context']['cases'][1]['root_sha256']=p['context']['cases'][0]['root_sha256']
    elif damage=='history':rows[0]['results'][1]['attempts'][0]['witness']['receipt']['root']['context']['latent_history']=[[9.,9.]]
    elif damage=='metric':rows[0]['results'][1]['attempts'][0]['witness']['receipt']['provenance']['metric_sha256']='c'*64
    elif damage=='model':kw['bindings']={}
    else:rows[0]['results'][1]['attempts'][0]['index']=1
    with pytest.raises(ValueError):m.assemble(*args,**kw)


def test_no_winner_or_cost_selection(tmp_path):
    args,kw=fixture(tmp_path)
    args[2][0]['results'][1]['winner']={'index':39,'cost':-9999}
    args[2][0]['results'][1]['attempts'][0]['cost']=9999
    spec,inventory=m.assemble(*args,**kw)
    query=spec['trials'][0]['proposal_inputs'][m.c.ARMS[2]]
    assert m.c.bound(query['path'],query['sha256'])['candidate_index']==0
    assert inventory['source_candidate_denominator']==480


def test_asset_drift_rejected_by_runner(tmp_path):
    args,kw=fixture(tmp_path);spec,_=m.assemble(*args,**kw);p=m.r.prepare(spec)
    Path(spec['assets']['model']['path']).write_bytes(b'changed RAM metadata')
    with pytest.raises(ValueError):m.r.verify(p,m.c.digest(p))


def test_unknown_identities_are_real_descriptors_not_future_hashes(tmp_path):
    args,kw=fixture(tmp_path);spec,_=m.assemble(*args,**kw)
    for key in ('controller','observation','budget'):
        p=tmp_path/('unknown_'+key+'.json');d=m.c.bound(p,m.r.file_sha(p))
        assert d['supported'] is False and spec['assets'][key] is None
        assert all(t['shared'][key]==m.r.file_sha(p) for t in spec['trials'])


def test_canonical_saved_roundtrip_preserves_runner_blocker_order(tmp_path):
    args,kw=fixture(tmp_path);spec,_=m.assemble(*args,**kw)
    spec=m.c.parse(m.c.encode(spec));p=m.r.prepare(spec)
    assert m.r.prepare(m.c.parse(m.c.encode(spec)))==m.c.parse(m.c.encode(p))
    assert spec['assets']['q'] is None
