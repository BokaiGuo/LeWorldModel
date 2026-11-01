"""Mathematical and saved-contract fixtures only; no neural forward."""
from copy import deepcopy
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p2_composition_paired_20261027_v2 as m
from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS


def arrays():
    jac = np.zeros((192,20))
    jac[:20] = np.diag(np.arange(20,0,-1))
    goal = np.zeros(192)
    goal[:20] = np.linspace(-2,2,20)
    return jac,np.zeros(192),goal,np.eye(192)


def inputs():
    context = dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
        action_dim=10,latent_history=[[0.]*192 for _ in range(3)],action_history=[[0.]*10,[0.]*10])
    def replay(c,actions):
        for start in range(0,len(actions),5):
            packet = np.asarray(actions[start:start+5]).reshape(10)
            terminal = np.array(c['latent_history'][-1])
            terminal[:10] += packet
            c['latent_history'] = c['latent_history'][1:]+[terminal.tolist()]
            c['action_history'] = c['action_history'][1:]+[packet.tolist()]
        return c
    def suffix(c,p,i):
        jac = np.zeros((192,10))
        jac[:10] = np.eye(10)
        result = m.suffix_solution(p,jac,c['latent_history'][-1],arrays()[2],np.eye(192))
        result['input_context_sha256'] = m.digest(c)
        return result
    return dict(bank=m.sra.spectral_bank(*arrays()),
        root=dict(root_id='fixture-root',root_rgb_sha256='a'*64,goal_rgb_sha256='b'*64,context=context),
        provenance={k:'c'*64 for k in PROVENANCE_KEYS},
        action_domain=dict(name='raw',units='official PushT live command',low=[-1.,-1.],high=[1.,1.],
            action_block=5,normalization_sha256='d'*64),
        target_metric=lambda c:float(np.mean((np.asarray(c['latent_history'][-1])-arrays()[2])**2)),
        replay=replay,suffix=suffix,fixture=True)


def test_real_pair_all_slots_same_prefix_joint_cap_and_witness():
    args = inputs()
    original = deepcopy(args['root'])
    results = [m.run_arm(arm=arm,**args) for arm in m.ARMS]
    assert args['root']==original
    for arm,result in zip(m.ARMS,results):
        assert result['complete'] and len(result['attempts'])==40
        assert len(result['ledger'])==84
        assert sum(e['phase']=='planning' for e in result['ledger'])==80
        assert sum(e['phase']=='replay' for e in result['ledger'])==4
        best = min(result['attempts'],key=lambda a:a['cost'])
        assert result['winner']['candidate_id']==best['candidate_id']
        for i,a in enumerate(result['attempts']):
            assert np.linalg.norm(a['logits'])<=m.sra.CAP*(1+1e-12)
            assert a['prefix_logits']==args['bank']['logits'][i][:10]
            assert a['actions'][:5]==args['bank']['commands'][i][0]
            assert len(a['witness']['edges'])==(1 if arm==m.ARMS[0] else 2)
            assert len(a['witness']['receipt']['actions'])==10
    assert any(a['actions']!=b['actions'] for a,b in zip(results[0]['attempts'],results[1]['attempts']))


def test_suffix_true_bounded_ridge_uses_remaining_joint_not_packet_cap():
    p = np.ones(10)*.4
    j = np.zeros((192,10)); j[:10] = np.eye(10)
    goal = np.zeros(192); goal[:10] = 10
    result = m.suffix_solution(p,j,np.zeros(192),goal,np.eye(192))
    step = np.array(result['suffix_logits'])
    assert result['svd_calls']==result['bounded_solve_calls']==1
    assert np.isclose(np.linalg.norm(step),np.sqrt(m.sra.CAP**2-p@p))
    assert np.isclose(np.linalg.norm(np.r_[p,step]),m.sra.CAP)
    kkt = (j.T@j+(.002+result['solve']['ball_multiplier'])*np.eye(10))@step-j.T@goal
    assert np.max(np.abs(kkt))<1e-10


def test_zero_cap_no_hidden_svd_or_solve(monkeypatch):
    monkeypatch.setattr(np.linalg,'svd',lambda *a,**k:pytest.fail('zero-cap SVD'))
    prefix = np.zeros(10); prefix[0] = m.sra.CAP
    r = m.suffix_solution(prefix,np.zeros((192,10)),np.zeros(192),np.ones(192),np.eye(192))
    assert r['remaining_radius']==0 and r['suffix_logits']==[0.]*10
    assert r['svd_calls']==r['bounded_solve_calls']==0


@pytest.mark.parametrize('defect',['prefix','jac','nan'])
def test_invalid_suffix_rejected(defect):
    p,j,n,g,q = np.zeros(10),np.zeros((192,10)),np.zeros(192),np.zeros(192),np.eye(192)
    if defect=='prefix': p[0]=m.sra.CAP+1
    if defect=='jac': j=np.zeros((192,20))
    if defect=='nan': n[0]=np.nan
    with pytest.raises(ValueError): m.suffix_solution(p,j,n,g,q)


def test_first_tie_is_slot_zero_both_arms():
    args = inputs(); args['target_metric']=lambda c:1.
    for arm in m.ARMS:
        r=m.run_arm(arm=arm,**args)
        assert r['winner']['candidate_id']==arm+'_00'


@pytest.mark.parametrize('defect',['suffix','metric','replay','wrong_context'])
def test_failure_retained_no_deployable_winner(defect):
    args = inputs()
    if defect in ('suffix','wrong_context'):
        original = args['suffix']
        def suffix(c,p,i):
            if defect=='suffix' and i==0: raise RuntimeError('fixture failure')
            value = original(c,p,i)
            if defect=='wrong_context': value['input_context_sha256']='e'*64
            return value
        args['suffix']=suffix
    if defect=='metric': args['target_metric']=lambda c:float('nan')
    if defect=='replay':
        original = args['replay']; count=0
        def replay(c,a):
            nonlocal count
            count+=1
            if count==81: raise RuntimeError('fixture replay failure')
            return original(c,a)
        args['replay']=replay
    r=m.run_arm(arm=m.ARMS[1],**args)
    assert not r['complete'] and r['winner'] is None and len(r['attempts'])==40


def test_bad_metadata_rejected_before_dispatch():
    args=inputs(); args['provenance']['source_sha256']='not-sha'
    args['replay']=lambda *a:pytest.fail('dispatch before metadata validation')
    with pytest.raises(ValueError): m.run_arm(arm=m.ARMS[0],**args)


def test_host_only_and_no_prefix_neural_gradient():
    args=inputs(); args['fixture']=False
    with pytest.raises(ValueError,match='CUDA'): m.run_arm(arm=m.ARMS[0],**args)
    with pytest.raises(ValueError,match='CUDA'):
        m.suffix_cuda(None,SimpleNamespace(type='cpu'),None,None,None,None,None,None)
    source=inspect.getsource(m.suffix_cuda)
    assert 'vectorize=True' in source and source.count('torch.autograd.functional.jacobian(')==1
    assert 'torch.tanh(u).reshape(5,2)' in source
    cuda=inspect.getsource(m.cuda)
    assert cuda.index('load_prepared(output)')<cuda.index('require_cuda_allocation')
    assert 'sample_limit=252' in cuda


def accounting_fixture():
    entries=[]
    def append(scope):
        for kind in ('action_encoder','predict'):
            entries.append(dict(kind=kind,scope=deepcopy(scope),completed=True,samples=1,context_tokens=3))
    for stage in ('nominal','joint_jacobian'):
        for _ in range(2): append(dict(arm=m.sra.ARM,phase='compilation',path=[stage]))
    results=[]
    for arm in m.ARMS:
        ledger=[]
        for i in range(40):
            for packet in range(2):
                if arm==m.ARMS[1] and packet==1:
                    for stage in ('nominal','jacobian'): append(dict(arm=arm,phase='suffix_compilation',path=[i,stage]))
                path=[i,packet]
                append(dict(arm=arm,phase='planning',path=path))
                ledger.append(dict(phase='planning',path=path,predict_calls=1,predict_samples=1,completed=True))
        for i in (40,41,42,42):
            append(dict(arm=arm,phase='replay',path=[i]))
            ledger.append(dict(phase='replay',path=[i],predict_calls=1,predict_samples=1,completed=True))
        results.append(dict(complete=True,ledger=ledger))
    budget=dict(phases=[dict(phase=phase,predict_calls=n,predict_samples=n,completed=True)
        for phase,n in [('planning',4),('planning',80),('evaluation',4),('planning',160),('evaluation',4)]])
    return entries,budget,results


def test_unique_shared_attributed_and_full_context_accounting():
    s=m.SCHEDULE
    assert s['unique_predict_per_root']==4+84+164==252
    assert s['standalone_attributed_predict']==[88,168]
    assert sum(s['standalone_attributed_predict'])!=252
    assert 4*252==1008 and 4*252*3==3024
    assert 4*41==164 and 4*41*192==31488
    assert m.accounting_ok(*accounting_fixture())
    e,b,r=accounting_fixture(); e[0]['context_tokens']=2
    assert not m.accounting_ok(e,b,r)
    e,b,r=accounting_fixture(); b['phases'][3]['predict_calls']=80
    assert not m.accounting_ok(e,b,r)


def test_operations_not_predict_only_matching():
    ops=[dict(kind=k,completed=True) for k in ('jacobian','bounded_ridge_svd','mode_svd')]
    ops += [dict(kind='suffix_jacobian',completed=True) for _ in range(40)]
    ops += [dict(kind='suffix_solve',completed=True,svd_calls=int(i>3),bounded_solve_calls=int(i>3)) for i in range(40)]
    assert m.operation_counts(ops)==dict(jacobian_calls=41,svd_calls=38,bounded_solve_calls=37,scalar_output_vjp_directions=7872)
    ops.append(dict(kind='suffix_jacobian',completed=False))
    assert m.operation_counts(ops)['jacobian_calls']==41


def test_actual_resource_receipt_shared_cost_is_not_double_counted():
    e,b,r=accounting_fixture()
    for arm,result in zip(m.ARMS,r):
        result.update(arm=arm,selection_evaluations=40)
    receipt=m.resource_receipt(e,r,[])
    assert receipt['unique']['predict_calls']==252
    assert receipt['shared']['predict_calls']==4
    assert [receipt['exclusive'][a]['predict_calls'] for a in m.ARMS]==[84,164]
    assert [receipt['standalone_attributed'][a]['predict_calls'] for a in m.ARMS]==[88,168]
    assert receipt['unique']['predict_context_tokens']==receipt['unique']['encoder_context_tokens']==756


def test_full_context_chain_not_just_terminal_vector():
    args=inputs()
    r=m.run_arm(arm=m.ARMS[1],**args)
    a=r['attempts'][0]
    w=deepcopy(a['witness'])
    contexts=[edge['input_context'] for edge in w['edges']]+[w['edges'][-1]['output_context']]
    contexts[0]['action_history'][0][0]=.123
    with pytest.raises(ValueError,match='disconnected'):
        m.saved_witness(args['root'],args['provenance'],args['action_domain'],contexts,
            [edge['actions'] for edge in w['edges']],a['candidate_id'])


@pytest.fixture
def prepared(tmp_path):
    path=tmp_path/m.TAG
    m.prepare(m.historical.DEFAULT_REFERENCE,path)
    return path


def test_prepare_model_free_and_bound(prepared,monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    monkeypatch.setattr(base,'require_cuda_allocation',lambda *a:pytest.fail('CUDA'))
    monkeypatch.setattr(base,'load_compatible_pretrained',lambda *a,**k:pytest.fail('model'))
    _,p,s=m.load_prepared(prepared)
    assert p['schedule']==m.SCHEDULE and p['expected_four_root_predict']==1008
    assert len(p['context']['cases'])==4 and not s['P2_admission']
    assert str(m.Path(m.__file__)) in p['source_paths']
    assert s['new_neural_calls']==0 and s['new_physical_steps']==0
    with pytest.raises(FileExistsError): m.prepare(m.historical.DEFAULT_REFERENCE,prepared)
    with pytest.raises(ValueError,match='regular nonsymlink evidence'): m.selected_actions(prepared)


def test_protocol_tamper_rejected(prepared):
    path=prepared/'protocol.json'
    p=json.loads(path.read_text()); p['schedule']['damping']=.003
    # Intentional corruption confined to pytest temporary fixture, not sealed assets.
    with path.open('w') as stream: json.dump(p,stream)
    with pytest.raises(ValueError): m.load_prepared(prepared)


def test_archived_source_tamper_rejected(prepared):
    s=json.loads((prepared/'prepared.json').read_text())
    archive=m.Path(s['source_seals'][0]['output_dir'])/'sources'/'000.source'
    with archive.open('a') as stream: stream.write('\n# deliberate pytest corruption\n')
    with pytest.raises(ValueError): m.load_prepared(prepared)


def test_exporter_all_slots_and_tamper_on_explicit_math_fixture(prepared):
    """Fabricated report in pytest tmp only; NOT CUDA or scientific evidence.

    Exercise the exporter after the real source/input verifier. The mathematical
    transitions and synthetic instrumentation are deliberately marked fixtures
    here, even where production metadata is imitated to cover its parser branch.
    """
    _,p,s=m.load_prepared(prepared)
    source=m.historical.audit.read_json(p['context']['compilation'])
    compiled={r['root_sha256']:r['compiled'] for r in source['rows']}
    rows=[]
    fixture=inputs()
    for case in p['context']['cases']:
        c=compiled[case['root_sha256']]
        goal=c['goal_vectors'][0]; q=c['scoring']['metric']['projection']
        bank=m.sra.spectral_bank(arrays()[0],case['root']['context']['latent_history'][-1],goal,q)
        m.full20.write(prepared/f'root_{case["root_index"]}_bank.json',bank)
        def local(context,prefix,i):
            jac=arrays()[0][:,:10]
            result=m.suffix_solution(prefix,jac,context['latent_history'][-1],goal,q)
            result['input_context_sha256']=m.digest(context)
            return result
        results=[m.run_arm(arm=arm,bank=bank,root=case['root'],
            provenance=m.provenance_for(p,s,case,bank,arm),action_domain=case['action_domain'],
            target_metric=m.full20.metric_for(c),replay=fixture['replay'],suffix=local,fixture=True)
            for arm in m.ARMS]
        assert all(r['complete'] for r in results)
        for result in results:
            # Local test-only fabrication for production export branch coverage.
            result['fixture_only']=False
        entries,budget,_=accounting_fixture()
        ops=[dict(kind=k,completed=True) for k in ('jacobian','bounded_ridge_svd','mode_svd')]
        for i,a in enumerate(results[1]['attempts']):
            ops.extend([dict(kind='suffix_jacobian',index=i,completed=True),
                dict(kind='suffix_solve',index=i,completed=True,svd_calls=a['suffix_compilation']['svd_calls'],
                    bounded_solve_calls=a['suffix_compilation']['bounded_solve_calls'])])
        rows.append(dict(root_sha256=case['root_sha256'],root_index=case['root_index'],results=results,
            actual_model_entries=entries,actual_prediction_budget=budget,operations=ops,
            completed_operations=m.operation_counts(ops),predict_calls=252,action_encoder_calls=252,
            predict_context_tokens=756,encoder_context_tokens=756,complete=True,error=None,
            resource_receipt=m.resource_receipt(entries,results,ops)))
    report=dict(schema=m.TAG+'.report',status='calibration_completed',protocol_sha256=m.digest(p),
        source_seals=s['source_seals'],rows=rows,gates=m.calibration_gates(rows,True,True),
        test_fixture_not_cuda=True,**deepcopy(m.BOUNDARY))
    m.full20.write(prepared/'report.json',report)
    exported=m.selected_actions(prepared)
    assert len(exported['actions'])==8
    assert all(v.shape==(10,2) for v in exported['actions'].values())
    assert exported['verification_work']['neural_predict_calls']==0
    # Corrupt a nonwinning slot: exporter must validate ALL slots, not only winner.
    report['rows'][0]['results'][1]['attempts'][39]['actions'][0][0]+=.01
    with (prepared/'report.json').open('w') as stream: json.dump(report,stream)
    with pytest.raises(ValueError,match='action/witness'): m.selected_actions(prepared)
