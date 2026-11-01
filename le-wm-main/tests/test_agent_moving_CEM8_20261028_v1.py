"""RAM-only backend/interface mocks, no neural model, tmp, GPU or simulator."""
from copy import deepcopy
from io import BytesIO
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_moving_CEM8_20261028_v1 as m

# Reuse immutable NumPy fixture builders, not previous model/experiment runners.
spec=importlib.util.spec_from_file_location('next_suffix_fixture',Path(__file__).with_name('test_moving_history_observed_suffix_cuda_20261027_v1.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class RAM:
    def __init__(self):self.files={};self.events=[]
    def _write(self,name,data,**kw):
        assert name not in self.files
        self.files[name]=data
    def event(self,e):self.events.append(deepcopy(e))


def execute(slot=0,arm=None,fail=None,condition="off"):
    b=h.branch(slot,arm);b.update(condition=condition,anchor=m.ORDER.index((slot,condition,b['arm'])));b['branch_id']=m.core.digest(b);store=RAM();meter=NS(stage=None,expected_input=None,last_native=None,require_native=True)
    modules=m.Modules(NS(),store,'branch_00',meter)
    backend=h.Backend(b,modules,fail);row=dict(index=0,branch_id=b['branch_id'])
    try:m.evaluate(b,backend,meter,store,row)
    finally:row.update(module_ledger=deepcopy(modules.entries),resources=modules.counts())
    return b,store,row,backend


def load(store,a):
    data=store.files[a['file']];assert m.prior.observed.enc.digest(data)==a['sha256']
    return np.load(BytesIO(data),allow_pickle=False)


@pytest.mark.parametrize('slot',[0,1])
@pytest.mark.parametrize('arm',m.ARMS)
@pytest.mark.parametrize('condition',['off','on'])
def test_full_fixed64_plus_one_independent_replay_native_path(slot,arm,condition):
    b,s,r,backend=execute(slot,arm,condition=condition);m.validate_row(b,r);m.native_arrays(b,r,lambda x:load(s,x))
    assert backend.calls==list(range(65))
    assert len(r['evaluations'])==65 and len(r['module_ledger'])==195
    assert r['resources']=={k:dict(attempted=65,completed=65) for k in m.Modules.ORDER}
    p=m.model_plan(b,r)
    assert p['commands'][:5]==b['prefix_commands'] and p['commands'][5:]==r['suffix_commands']
    assert p['blind_sha256']==b['blind_sha256'] and p['prefix_arm']==arm
    assert p['selection_sha256']==m.core.digest({k:v for k,v in p.items() if k!='selection_sha256'})
    assert sum(map(len,s.files.values()))<2*1024**2
    events=s.events;kinds=[e['kind'] for e in events]
    assert kinds.index('proposal_bank_frozen')<kinds.index('candidate_attempt')
    freeze=kinds.index('selection_frozen_before_replay')
    assert len([e for e in events[:freeze] if e['kind']=='candidate_return'])==64
    assert events[freeze+1]['receipt']['phase']=='independent_winner_replay'


def test_original_nonzero_history_and_no_second_scale():
    b,s,r,_=execute();raw=np.asarray(b['observed_raw_history'],np.float32)
    assert np.array_equal(raw[:5],m.prior.planner.fixed_raw()[5:])
    assert np.array_equal(raw[5:],b['prefix_commands'])
    expected=m.prior.core.normalized_tokens(b,raw)
    assert np.array_equal(b['observed_context']['action_history'],expected)
    ae=load(s,r['module_ledger'][0]['inputs'][0])[0]
    assert np.array_equal(ae[:2],expected)


@pytest.mark.parametrize('bad',['cost','raw','proposal','mean','std','elite','selected','replay','history','denominator'])
def test_full_candidate_update_release_tampering_rejected(bad):
    b,_,r,_=execute()
    if bad=='cost':r['evaluations'][0]['cost']+=1
    elif bad=='raw':r['evaluations'][0]['commands'][0][0]+=.01
    elif bad=='proposal':r['iterations'][0]['params'][0][0]+=.01
    elif bad=='mean':r['iterations'][1]['mean_before'][0]+=.01
    elif bad=='std':r['iterations'][0]['std_after'][0]+=.01
    elif bad=='elite':r['iterations'][0]['elite'].reverse()
    elif bad=='selected':r['selected_index']=(r['selected_index']+1)%64
    elif bad=='replay':r['evaluations'][-1]['context']['latent_history'][-1][0]+=.01
    elif bad=='history':r['evaluations'][0]['context']['action_history'][0][0]+=.01
    else:r['evaluations'].pop()
    with pytest.raises(ValueError):m.validate_row(b,r)


@pytest.mark.parametrize('stage',[0,1,15,16,63,64])
def test_failed_predict_preserves_every_attempt_and_partial_array(stage):
    b=h.branch();b['condition']='off';s=RAM();meter=NS(stage=None,expected_input=None,last_native=None,require_native=True)
    modules=m.Modules(NS(),s,'branch_00',meter);backend=h.Backend(b,modules,stage)
    row=dict(index=0,branch_id=b['branch_id'])
    with pytest.raises(RuntimeError,match='original forward failed'):m.evaluate(b,backend,meter,s,row)
    assert backend.calls==list(range(stage+1))
    assert len(row['evaluations'])==stage+1 and row['evaluations'][-1]['status']=='attempted'
    assert modules.entries[-1]['resource']=='predict' and not modules.entries[-1]['completed']
    assert len(modules.entries[-1]['inputs'])==2 and modules.failed
    assert modules.counts()['predict']==dict(attempted=stage+1,completed=stage)
    assert modules.counts()['action_encoder']==dict(attempted=stage+1,completed=stage+1)
    assert s.files and len([e for e in s.events if e['kind']=='candidate_return'])==stage


@pytest.mark.parametrize('target',['context','action','bridge','output','stage'])
def test_saved_native_array_bridge_rejects_mutation(target):
    b,s,r,_=execute()
    if target=='stage':r['module_ledger'][1]['stage']=1
    else:
        entry=r['module_ledger'][0]['inputs'][0] if target=='action' else r['module_ledger'][1]['inputs'][0 if target=='context' else 1]
        if target=='output':entry=r['module_ledger'][2]['output']
        a=load(s,entry).copy()
        if target=='output':a[-1,0]+=.01  # Returned successor is last projected row, not earlier history outputs.
        else:a.flat[0]+=.01
        new=m.prior.save_array(s,'tampered.npy',a);entry.clear();entry.update(new)
    with pytest.raises(ValueError):m.native_arrays(b,r,lambda x:load(s,x))


def test_projection_is_declared_proposal_change_not_dispatch_clip():
    cap=m.contract()['packet_logit_cap'];a=np.full((16,10),100.)
    out=m.project(a,cap)
    assert np.allclose(np.linalg.norm(out,axis=1),cap)
    assert not m.contract()['projection_density_correction']
    assert len(set(m.contract()['seeds']))==8


def test_stable_ties_are_oldest_candidate_not_last_or_best_replay():
    b=h.branch();b['condition']='off';s=RAM();meter=NS(stage=None,expected_input=None,last_native=None,require_native=False)
    class Constant:
        def step(self,context,raw):
            incoming=m.prior.planner.incoming(b,context,raw)
            return dict(context,latent_history=deepcopy(context['latent_history'][1:])+[context['latent_history'][-1]],action_history=incoming['action_history'][-2:])
    row=dict(branch_id=b['branch_id']);m.evaluate(b,Constant(),meter,s,row);m.validate_row(b,row)
    assert row['selected_index']==0


def test_invalid_prepare_no_output_before_gate(monkeypatch):
    def bad(*a):raise ValueError('external prepare SHA mismatch')
    monkeypatch.setattr(m,'verify',bad)
    monkeypatch.setattr(m,'Store',lambda *a:pytest.fail('output before verify'))
    with pytest.raises(ValueError,match='SHA mismatch'):m.run('/RAM/p','a'*64,'/RAM/o',m.AUTH)


def test_MAIN_authorization_precedes_any_preparation(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('verification before authorization'))
    with pytest.raises(ValueError,match='MAIN'):m.run('/RAM/p','a'*64,'/RAM/o','wrong')


def test_cli_help_no_producer_or_model(capsys):
    with pytest.raises(SystemExit) as e:m.main(['--help'])
    assert e.value.code==0 and 'consume' in capsys.readouterr().out


def test_known_task_not_same_compute_or_official_paper():
    c=m.contract()
    assert c['total']['predict']==520 and c['per_branch']['predict']==65
    assert c['old_suffix_evaluations']==c['nominal_evaluations']==c['padding']==c['AD']==c['SVD']==c['new_physics']==0
    assert c['P1_admission'] is c['equal_total_compute'] is c['official_reproduction'] is False


def material_fixture():
    cases=[];tasks={}
    for ordinal,(slot,condition,arm) in enumerate(m.ORDER):
        i,p,item,z,entry,_=h.fixture(slot,arm)
        b=h.branch(slot,arm)
        p.update(root_slot=slot,condition=condition,selection_digest=p['selection_sha256'])
        # Separate on/off goal/latent values force same-condition identity checks.
        task=deepcopy(i['gn_task']);task['goal'][0]=float(ordinal)
        task['root_context']['latent_history'][1][0]=float(ordinal)
        context=deepcopy(b['observed_context'])
        context['latent_history'][:2]=deepcopy(task['root_context']['latent_history'][1:])
        tasks[slot,condition,arm]=task
        for variant in m.factorial.VARIANTS:
            cases.append(dict(index=len(cases),anchor=ordinal,root_slot=slot,condition=condition,arm=arm,
                task_id=p['task_id'],selection_digest=p['selection_digest'],original_plan=deepcopy(p),
                prefix=deepcopy(p['commands'][:5]),context=deepcopy(context),goal=deepcopy(task['goal']),
                variant=variant,scale=.1 if variant=='teacher_small' else 1.))
    return cases,tasks


def test_fixed8_assembler_all_conditions_and_no_outcome_inputs():
    cases,tasks=material_fixture();before=deepcopy(cases);bs=m.assemble(cases,tasks)
    assert [(b['source_slot'],b['condition'],b['arm']) for b in bs]==m.ORDER
    assert len(set(b['branch_id'] for b in bs))==8 and cases==before
    assert [b['goal'][0] for b in bs]==list(range(8))
    assert all(b['q']==np.eye(192).tolist() for b in bs)
    # No physical outcome/cost can affect which anchors are included.
    for c in cases:c['native_success']=bool(c['anchor']%2);c['actual_cost']=-999
    assert m.assemble(cases,tasks)==bs


@pytest.mark.parametrize('bad',['missing','duplicate','condition','prefix','history','goal','scale','warmup','float64','selected'])
def test_assembler_rejects_cross_condition_or_legacy_inputs(bad):
    cs,ts=material_fixture();c=cs[1]
    if bad=='missing':cs.pop()
    elif bad=='duplicate':cs[4]=deepcopy(cs[1])
    elif bad=='condition':c['condition']='on'
    elif bad=='prefix':c['prefix'][0][0]+=.01
    elif bad=='history':c['context']['action_history'][0][0]+=.01
    elif bad=='goal':c['goal'][0]+=.01
    elif bad=='scale':c['scale']=.1
    elif bad=='warmup':ts[m.ORDER[0]]['root_raw_action_history'][0][0]=0
    elif bad=='float64':c['original_plan']['commands'][0][0]+=1e-14
    else:c['selection_digest']='f'*64
    with pytest.raises(ValueError):m.assemble(cs,ts)


def test_new_seed_rule_is_blind_unique_independent_from_observed4():
    values=[m.seed(*k) for k in m.ORDER]
    assert values==[20261028+800000+104729*i for i in range(8)]
    assert len(set(values))==8
    assert not set(values)&set(m.frozen.contract()['seeds'])
    with pytest.raises(ValueError):m.seed(True,'off',m.ARMS[0])


def test_literal_raw_float32_authority_is_not_second_tanh():
    b,s,r,_=execute()
    for e in r['evaluations']:
        raw=np.asarray(e['commands'],np.float32)
        assert np.array_equal(raw,e['commands'])
        if e['index']==0:assert not np.array_equal(raw,np.tanh(raw))
        ae=load(s,r['module_ledger'][3*e['index']]['inputs'][0])[0,-1]
        assert np.array_equal(ae,m.core.normalized_tokens(b,raw)[0])
    p=m.model_plan(b,r)
    assert p['raw_authority']=='literal_float32_same_model_and_dispatch'
    assert p['condition']==b['condition'] and p['commands'][5:]==r['evaluations'][r['selected_index']]['commands']


def test_shared_axes_not_equal_total_compute_or_double_counted():
    c=m.contract();s=c['shared_resource_axes']
    assert c['total']['predictor_samples']==520 and c['jacobian']==0
    assert s['factorial_teacher8_subset']['predict']==104
    assert s['factorial_teacher8_subset']['VJP_directions']==1536
    assert s['factorial_all24']['predict']==312
    assert c['inherited_RGB'] is None and not c['equal_FLOPs']


def test_RAM_eight_branch_full_saved_consume_and_actual_size_bound(monkeypatch):
    """Entire own CLI consumer contract without tmp, torch, simulator or real results."""
    import gzip
    import hashlib
    import json
    data={};branches=[];rows=[];events=[]
    for index,(slot,condition,arm) in enumerate(m.ORDER):
        b=h.branch(slot,arm);b.update(condition=condition,anchor=index)
        b['branch_id']=m.core.digest(b);branches.append(b)
        s=RAM();meter=NS(stage=None,expected_input=None,last_native=None,require_native=True)
        modules=m.Modules(NS(),s,f'branch_{index:02d}',meter)
        backend=h.Backend(b,modules)
        row=dict(index=index,source_slot=slot,prefix_arm=arm,condition=condition,anchor=index,branch_id=b['branch_id'])
        events.append(dict(kind='branch_started',branch=deepcopy(b)))
        m.evaluate(b,backend,meter,s,row)
        row.update(module_ledger=deepcopy(modules.entries),resources=modules.counts())
        row['plan']=m.model_plan(b,row);rows.append(row)
        events.extend(s.events);events.append(dict(kind='branch_sealed',receipt=deepcopy(row)))
        data.update(s.files)
        data[f'branch_{index:02d}.json.gz']=gzip.compress(m.canonical(row),mtime=0)
    prepared=dict(branches=branches,contract=m.contract(),expected_model_hash='a'*64)
    data['binding.json']=m.canonical(dict(prepared='/RAM/prepare',prepare_sha256='b'*64))
    data['events.jsonl.gz']=gzip.compress(b''.join(m.canonical(e) for e in events),mtime=0)
    hashes={name:hashlib.sha256(value).hexdigest() for name,value in data.items()}
    report=dict(schema=m.NAME+'.report',status='teacher8_CEM_commands_sealed_not_admitted',errors=[],first_error=None,
        commands_released=True,model_unchanged=True,source_verified=True,P1_admission=False,P3_admission=False,
        prepared='/RAM/prepare',prepare_sha256='b'*64,contract=m.contract(),rows=rows,artifact_sha256=hashes,
        model_hash_before='a'*64,model_hash_after='a'*64,
        resources={k:dict(attempted=520,completed=520) for k in m.Modules.ORDER})
    data['report.json']=m.canonical(report)
    actual_size=sum(map(len,data.values()))
    # Fixture AE embeddings are 3x10, real AE embeddings are 3x192 (twice per forward).
    native_shape_margin=520*2*3*(192-10)*4
    assert actual_size+native_shape_margin<m.CAP
    assert len(data['report.json'])<m.FINAL
    class VirtualPath:
        def __init__(self,path):self.path=str(path)
        def __str__(self):return self.path
        @property
        def name(self):return self.path.rsplit('/',1)[-1]
        def absolute(self):return self
        def __truediv__(self,name):return VirtualPath(self.path+'/'+str(name))
        def iterdir(self):return [self/n for n in data]
        def is_symlink(self):return False
        def stat(self):return NS(st_size=len(data[self.name]))
        def read_bytes(self):return data[self.name]
    def read(path,*args):
        content=path.read_bytes()
        return json.loads(gzip.decompress(content) if path.name.endswith('.gz') else content)
    monkeypatch.setattr(m,'Path',VirtualPath)
    monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(m.prior.blind,'read',read)
    monkeypatch.setattr(m,'verify',lambda path,expected:prepared)
    expected=hashlib.sha256(data['report.json']).hexdigest()
    out=m.consume('/RAM/output',expected)
    assert out['status']=='teacher8_CEM_commands_verified_not_admitted'
    assert len(out['plans'])==8
    assert [(p['root_slot'],p['condition'],p['prefix_arm']) for p in out['plans']]==m.ORDER
    assert all(p['whole_run_report_sha256']==expected for p in out['plans'])
    # Re-sealed wrong resource totals must not bypass the independent census.
    report['resources']['predict']['completed']=519;data['report.json']=m.canonical(report)
    with pytest.raises(ValueError,match='520'):
        m.consume('/RAM/output',hashlib.sha256(data['report.json']).hexdigest())


def test_RAM_CLI_prepare_verify_no_model_or_disk(monkeypatch,capsys):
    import json
    cs,ts=material_fixture()
    p=dict(schema=m.NAME+'.prepare',branches=m.assemble(cs,ts),contract=m.contract())
    frozen=m.canonical(p);digest=m.prior.observed.enc.digest(frozen)
    monkeypatch.setattr(m,'build',lambda:deepcopy(p))
    def write(path,raw):
        assert path=='/RAM/prepare' and raw==frozen and len(raw)<=m.PREP_CAP
        return dict(file=path,sha256=digest,bytes=len(raw))
    monkeypatch.setattr(m,'write_new',write)
    assert m.main(['prepare','--output','/RAM/prepare'])==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt['prepare_sha256']==digest and receipt['contract']['branches']==8
    monkeypatch.setattr(m,'sha',lambda path:digest)
    monkeypatch.setattr(m.prior.blind,'read',lambda *a:deepcopy(p))
    assert m.main(['verify','--prepared','/RAM/prepare','--prepare-sha',digest])==0
    assert json.loads(capsys.readouterr().out)==dict(status='verified_no_execution',branches=8)
    with pytest.raises(ValueError,match='SHA'):
        m.verify('/RAM/prepare','0'*64)
