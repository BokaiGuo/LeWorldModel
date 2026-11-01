"""Pure array/linear mathematical callbacks and metadata; no neural execution."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p2_full20_gradient_mpc_20261027_v1 as m
from experiments.spectral_reachability.d2p_composition_witness_v2 import PROVENANCE_KEYS


def fixture_inputs():
    context = dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
        action_dim=10,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
    compiled = dict(goal_vectors=[([.5]+[0.]*191)],scoring=dict(metric=dict(projection=np.eye(192).tolist(),sha256='f'*64)))
    def replay(context,actions):
        for start in range(0,len(actions),5):
            packet = np.asarray(actions[start:start+5]).reshape(10)
            value = list(context['latent_history'][-1])
            value[0] += float(packet.sum()/10)
            context['latent_history'] = context['latent_history'][1:]+[value]
            context['action_history'] = context['action_history'][1:]+[packet.tolist()]
        return context
    metric = m.full20.metric_for(compiled)
    root = dict(root_id='a'*64,root_rgb_sha256='b'*64,goal_rgb_sha256='c'*64,context=context)
    class ArrayObjective:
        def evaluate(self,logits,index,need_gradient,event,ledger,journal):
            counter = m.full20._Ledger(replay,m.ARM)
            try:
                terminal = counter.call(context,event['actions'],'planning',[index])
            finally:
                ledger.extend(counter.events)
            cost = metric(terminal)
            event.update(forward_completed=True,cost=cost,differentiated_cost=cost,terminal_context=deepcopy(terminal))
            gradient = None
            if need_gradient:
                operation = dict(kind='autograd_grad',candidate_index=index,completed=True,
                    scalar_objective_vjp_directions_requested=1,extra_forward_requested=0)
                event['derivative'] = operation
                gradient = np.full(20,2*(terminal['latent_history'][-1][0]-.5)/192/10)
            return terminal,gradient
    return dict(root=root,provenance={k:'d'*64 for k in PROVENANCE_KEYS},
        action_domain=dict(name='raw',units='official PushT live command',low=[-1.,-1.],high=[1.,1.],
            action_block=5,normalization_sha256='e'*64),objective=ArrayObjective(),replay=replay,
        target_metric=metric,fixture=True),compiled


def metadata_for(result,provenance,root_id):
    entries = []
    for dispatch in result['ledger']:
        for kind in ('action_encoder','predict'):
            entries.append(dict(kind=kind,samples=1,context_tokens=3,completed=dispatch['completed'],
                scope=dict(arm=m.ARM,phase=dispatch['phase'],path=dispatch['path'])))
    budget = dict(binding=dict(protocol_id=provenance['source_sha256'],task_id=m.TAG,
        root_id=root_id,frozen_model_id=provenance['checkpoint_sha256']),method=m.ARM,predict_sample_limit=84,
        attempted_predict_samples=84,blocked_calls=0,phases=[dict(phase=p,predict_calls=n,predict_samples=n,
            completed=True,failed_predict_calls=0) for p,n in (('planning',80),('evaluation',4))])
    return entries,budget


def test_fixed_algorithm_and_exact_fullH2_budget():
    s = m.SCHEDULE
    assert s['candidates']==40 and s['optimizer_steps']==39 and s['learning_rate']==.1
    assert s['total_predict_samples']==40*2+4==84 and 4*84==336
    assert s['autograd_grad_calls']==s['scalar_objective_vjp_directions']==39
    assert s['random_draws']==s['restarts']==0 and not s['line_search'] and not s['early_stop']
    assert s['model_optimizer_steps']==s['physical_env_steps']==0
    assert m.CAP==float(np.sqrt(10)*.75)
    assert not m.BOUNDARY['official_paper_implementation'] and not m.BOUNDARY['fair_total_budget_comparison']


def test_gradient_step_chain_and_joint_projection():
    u = np.linspace(-.2,.2,20)
    g = np.linspace(-1.,1.,20)
    updated,chain = m.gradient_step(u,g)
    expected = g*(1-np.tanh(u)**2)
    assert np.array_equal(chain,expected)
    assert np.array_equal(updated,u-.1*expected)
    cap,_ = m.gradient_step(np.zeros(20),np.full(20,1e4))
    assert np.isclose(np.linalg.norm(cap),m.CAP)
    axis,_ = m.gradient_step(np.zeros(20),np.r_[1e4,np.zeros(19)])
    assert np.linalg.norm(axis[:10])>m.compiler.CAP
    assert np.array_equal(m.commands_for(updated),np.tanh(updated).astype(np.float32).reshape(2,5,2))


@pytest.mark.parametrize('defect',['shape','nan','complex','bool','outside','grad_nan','grad_shape'])
def test_invalid_kernel_inputs_refuse(defect):
    u,g = np.zeros(20),np.ones(20)
    if defect=='shape': u=np.zeros(19)
    if defect=='nan': u[0]=np.nan
    if defect=='complex': u=np.zeros(20,dtype=complex)
    if defect=='bool': u=np.zeros(20,dtype=bool)
    if defect=='outside': u[0]=m.CAP+1
    if defect=='grad_nan': g[0]=np.nan
    if defect=='grad_shape': g=np.zeros(19)
    with pytest.raises(ValueError): m.gradient_step(u,g)


def test_all40_costs_39updates_no_extra_nominal_or_derivative_forward():
    args,_ = fixture_inputs()
    root = deepcopy(args['root'])
    result = m.run_trajectory(**args)
    assert args['root']==root
    assert result['candidate_and_replay_complete'] and result['fixture_only']
    assert len(result['attempts'])==40 and len(result['ledger'])==84
    assert result['planning_predict_samples']==80 and result['replay_predict_samples']==4
    assert result['optimizer_steps']==result['completed_autograd_grad_calls']==39
    assert all('cost' in e and 'terminal_context' in e for e in result['attempts'])
    assert 'derivative' not in result['attempts'][-1]
    assert not result['attempts'][-1]['optimizer_step_completed']
    assert result['attempts'][0]['logits']==[0.]*20
    best = min(result['attempts'],key=lambda e:e['cost'])
    assert result['selected_candidate_id']==best['candidate_id']
    assert result['witness']['receipt']['actions']==best['actions']
    assert len(result['witness']['edges'])==1 and len(result['witness']['receipt']['actions'])==10
    assert not result['P2_admission'] and not result['composition_admission']


def test_stable_first_tie_still_charges_all40_duplicates():
    args,_ = fixture_inputs()
    original = args['objective'].evaluate
    def zero_gradient(*call):
        terminal,gradient = original(*call)
        return terminal,np.zeros(20) if gradient is not None else None
    args['objective'].evaluate = zero_gradient
    result = m.run_trajectory(**args)
    assert result['candidate_and_replay_complete']
    assert result['selected_candidate_id']==m.ARM+'_00'
    assert len(result['ledger'])==84 and result['optimizer_steps']==39
    assert all(e['cost']==result['attempts'][0]['cost'] for e in result['attempts'])


@pytest.mark.parametrize('kind',['forward','backward','gradient_nan','replay','replay_mismatch'])
def test_failure_retains_work_and_never_refills_or_exposes_winner(kind):
    args,_ = fixture_inputs()
    original = args['objective'].evaluate
    def evaluate(u,i,needed,event,ledger,journal):
        terminal,gradient = original(u,i,needed,event,ledger,journal)
        if i==3:
            if kind=='forward': raise RuntimeError('fixture forward downstream failure')
            if kind=='backward':
                event['derivative']['completed']=False
                raise RuntimeError('fixture backward failure')
            if kind=='gradient_nan': gradient[0]=np.nan
        return terminal,gradient
    args['objective'].evaluate=evaluate
    if kind in ('replay','replay_mismatch'):
        replay = args['replay']
        def broken(c,a):
            if kind=='replay': raise RuntimeError('fixture replay failure')
            out = replay(c,a)
            out['latent_history'][-1][0] += 1
            return out
        args['replay']=broken
    result = m.run_trajectory(**args)
    assert not result['candidate_and_replay_complete']
    assert result['witness'] is result['selected_candidate_id'] is None
    assert len(result['attempts'])==40
    if kind in ('forward','backward','gradient_nan'):
        assert result['planning_predict_samples']==8 and result['replay_predict_samples']==0
        assert all(e['blocked'] for e in result['attempts'][4:])
        assert 'cost' in result['attempts'][3]
    else:
        assert result['planning_predict_samples']==80
        assert result['replay_predict_samples'] in (1,4)


def test_production_refuses_callbacks_and_cpu_objective():
    args,_ = fixture_inputs()
    args['fixture']=False
    with pytest.raises(ValueError,match='CUDA'): m.run_trajectory(**args)
    with pytest.raises(ValueError,match='CUDA'):
        m.CudaObjective(None,SimpleNamespace(type='cpu'),None,None,None)


def synthetic_row():
    args,compiled = fixture_inputs()
    case = dict(root_index=0,root_sha256=args['root']['root_id'],root=args['root'],action_domain=args['action_domain'],
        metric_sha256=compiled['scoring']['metric']['sha256'])
    protocol = dict(context=dict(cases=[case]),schedule=m.SCHEDULE,boundary=m.BOUNDARY)
    prepared = dict(source_seals=[dict(manifest_sha256='d'*64)])
    args['provenance']=m.provenance_for(protocol,prepared,case)
    result = m.run_trajectory(**args)
    # Synthetic metadata only, NOT a production model callback or real evidence.
    result['fixture_only']=False
    entries,budget = metadata_for(result,args['provenance'],case['root_sha256'])
    row = dict(root_index=0,root_sha256=case['root_sha256'],root=case['root'],action_domain=case['action_domain'],
        complete=True,result=result,actual_model_entries=entries,actual_prediction_budget=budget)
    return row,protocol,prepared,compiled


def test_array_only_saved_trajectory_verifier():
    row,protocol,prepared,compiled = synthetic_row()
    actions = m.verify_result(row,protocol,prepared,compiled)
    assert actions.shape==(10,2)
    assert m.accounting_ok(row['actual_model_entries'],row['actual_prediction_budget'],row['result'],
        m.provenance_for(protocol,prepared,protocol['context']['cases'][0]),row['root_sha256'])


@pytest.mark.parametrize('defect',['logits','commands','cost','gradient','update','winner','fixture','root',
    'entry','budget_root','budget_limit','backward','extra_last_backward','missing_candidate','receipt','goal'])
def test_saved_trajectory_mutations_reject(defect):
    row,protocol,prepared,compiled = synthetic_row()
    result = row['result']
    if defect=='logits': result['attempts'][1]['logits'][0] += .1
    if defect=='commands': result['attempts'][1]['commands'][0][0][0] += .1
    if defect=='cost': result['attempts'][1]['cost'] += .1
    if defect=='gradient': result['attempts'][1]['raw_gradient'][0] += .1
    if defect=='update': result['attempts'][1]['next_logits'][0] += .1
    if defect=='winner': result['selected_candidate_id']='cem_00'
    if defect=='fixture': result['fixture_only']=True
    if defect=='root': row['root']=dict(row['root'],root_id='0'*64)
    if defect=='entry': row['actual_model_entries'].pop()
    if defect=='budget_root': row['actual_prediction_budget']['binding']['root_id']='0'*64
    if defect=='budget_limit': row['actual_prediction_budget']['predict_sample_limit']=88
    if defect=='backward': result['completed_autograd_grad_calls']=38
    if defect=='extra_last_backward': result['attempts'][-1]['derivative']=dict(completed=True)
    if defect=='missing_candidate': result['attempts'].pop()
    if defect=='receipt': result['witness']['receipt']['actions'][0][0] += .1
    if defect=='goal': compiled['goal_vectors'][0][0] += 1
    with pytest.raises(ValueError): m.verify_result(row,protocol,prepared,compiled)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    def forbidden(*a,**kw): pytest.fail('execution during model-free prepare')
    for name in ('require_cuda_allocation','load_compatible_pretrained','collect_nonterminal_boundary_contact_roots'):
        monkeypatch.setattr(base,name,forbidden)
    monkeypatch.setattr(base.gym,'make',forbidden)
    monkeypatch.setattr(m,'cuda',forbidden)
    output = tmp_path/m.TAG
    m.prepare(m.historical.DEFAULT_REFERENCE,output)
    return output


def test_prepare_binds_exact_contexts_full20_Q_and_own_exporter(prepared):
    output,protocol,sealed = m.load_prepared(prepared)
    assert sealed['new_neural_calls']==sealed['new_physical_steps']==0
    assert [c['root_index'] for c in protocol['context']['cases']]==[0,1,2,3]
    assert protocol['schedule']==m.SCHEDULE and protocol['frozen_before_gradient_cuda']
    assert protocol['expected_four_root_totals']['predict_samples']==336
    assert protocol['expected_four_root_totals']['autograd_grad_calls']==156
    assert not protocol['physical_outcomes_used']
    assert not (output/'report.json').exists() and not (output/'started.json').exists()
    own = {str(Path(m.__file__)),str(m.PROJECT/'tests'/f'test_{m.TAG}.py'),str(m.PROJECT/'innerdocs'/f'{m.TAG}.md')}
    assert own<=set(protocol['source_paths'])
    assert m.main(['verify','--output',str(output)]) is None
    with pytest.raises(FileExistsError): m.prepare(m.historical.DEFAULT_REFERENCE,output)


@pytest.mark.parametrize('defect',['schedule','context','archive','args'])
def test_prepare_tampering_rejected(prepared,defect):
    protocol = m.historical.audit.read_json(prepared/'protocol.json')
    sealed = m.historical.audit.read_json(prepared/'prepared.json')
    if defect in ('archive','args'):
        seal = Path(sealed['source_seals'][0]['output_dir'])
        target = seal/('sources/000.source' if defect=='archive' else 'args.json')
        target.write_text('test-only artifact corruption')
    else:
        if defect=='schedule': protocol['schedule']['optimizer_steps']=40
        if defect=='context': protocol['context']['cases'][0]['root']['context']['latent_history'][0][0] += 1
        (prepared/'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError): m.load_prepared(prepared)


def test_reference_contract_never_parses_outcomes_or_sampler_report(monkeypatch):
    reader = m.historical.audit.EvidenceReader()
    original = reader.load
    opened = []
    def restricted(path,*args):
        text = str(path)
        assert 'physical' not in text and 'admission' not in text
        assert Path(path)!=m.historical.DEFAULT_REFERENCE/'report.json'
        opened.append(text)
        return original(path,*args)
    monkeypatch.setattr(reader,'load',restricted)
    context = m.reference_contract(m.historical.DEFAULT_REFERENCE,reader)
    assert len(context['cases'])==4
    assert all('p2_sra_matched_comparison' not in p for p in opened)
    reader.unchanged()


def test_static_cuda_has_one_autograd_no_detached_H2_and_math_backend():
    source = inspect.getsource(m.CudaObjective.evaluate)
    assert source.count('torch.autograd.grad(')==1
    assert source.count('self.transition(')==1 and 'for packet in range(2)' in source
    before_terminal = source[:source.index('terminal = dict(')]
    assert '.detach(' not in before_terminal
    assert 'with torch.set_grad_enabled(need_gradient)' in source
    runtime = inspect.getsource(m.cuda)
    assert runtime.index('load_prepared(output)')<runtime.index('require_cuda_allocation')
    assert 'sample_limit=84' in runtime and 'sdpa_kernel(SDPBackend.MATH)' in runtime
    assert 'torch.use_deterministic_algorithms(True)' in runtime
    assert 'load_compatible_pretrained' not in inspect.getsource(m.prepare)
    assert 'full==end==best[\'terminal\']' in inspect.getsource(m.run_trajectory)
    assert '.eval().requires_grad_(False)' in runtime
    assert 'before==checked.tensor_state_hash(model)' in runtime


def test_actual_instrumentation_guard_on_metadata_only_84_calls():
    from experiments.spectral_reachability.d2p_adaptive_budget_v2 import Binding,PredictionBudget
    class Encoder:
        def forward(self,value): return value
    class Model:
        action_encoder = Encoder()
        def predict(self,emb,act): return 'shape metadata only, no neural tensor work'
    model = Model()
    args,_ = fixture_inputs()
    result = m.run_trajectory(**args)
    replay = SimpleNamespace(_p2_scope=None)
    token = SimpleNamespace(shape=(1,3,192),device=SimpleNamespace(type='cuda',index=0))
    provenance = dict(source_sha256='a'*64,checkpoint_sha256='d'*64)
    root = 'b'*64
    binding = Binding(provenance['source_sha256'],m.TAG,root,provenance['checkpoint_sha256'])
    with m.full20.actual_entries(model,replay) as entries, PredictionBudget(
            model,binding=binding,method=m.ARM,sample_limit=84) as guard:
        for name,start,stop in (('planning',0,80),('evaluation',80,84)):
            with guard.phase(name,method=m.ARM,horizon_tokens=2):
                for dispatch in result['ledger'][start:stop]:
                    replay._p2_scope = dict(arm=m.ARM,phase=dispatch['phase'],path=dispatch['path'])
                    model.predict(token,model.action_encoder.forward(token))
        actual = guard.receipt()
    assert m.accounting_ok(entries,actual,result,provenance,root)
    assert sum(e['context_tokens'] for e in entries if e['kind']=='predict')==252
    assert not m.accounting_ok(entries[:-2],actual,result,provenance,root)

