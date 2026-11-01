"""RAM actions/state/images and tiny metadata; no real model, physics or consume."""
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_fixed_execution_physical_20261028_v1 as m
from agent_p3_fixed_execution_contract_20261028_v1_test import complete as model_ram
from test_agent_allmode_physical_20261028_v1 import fixture as physical_ram, MemoryStore, Env, fake_place, h, reseal


@lru_cache(maxsize=1)
def _fixture():
    panel,result,_=model_ram();_,data,*_=physical_ram();tasks=deepcopy(panel['base']['tasks'])
    for s,t in enumerate(tasks):t['binding']['root_sha256']=data['roots'][s]['root_sha256']
    requests=[]
    for prefix in result['prefixes']:
        req=deepcopy(prefix['request']);req['binding']=deepcopy(tasks[req['slot']]['binding']);requests.append(req)
    winners=deepcopy(result['winners'])
    for w in winners:w['task_binding_sha256']=m.fixed.digest(tasks[w['slot']]['binding'])
    verified=dict(schema=m.PRODUCER+'.consume',status='saved_prefix_verified_not_admitted',report_sha256=h('CUDA'),
        prefixes=[dict(slot=q['slot'],mode=q['mode'],status='model_verified_physical_blocked',request=q,error=None) for q in requests],
        winners=winners,counts=dict(task_methods=4,candidates=32,candidate_returns=32,selected_prefixes=4,prefix_returns=4),
        equal_requested_execution_horizon=True,equal_actual_physical_execution_horizon=None,
        actual_physics_status='unknown_not_run',per_branch=[],**m.FLAGS)
    plans=[dict(index=i,slot=req['slot'],root_slot=req['slot'],mode=req['mode'],condition='off',arm=req['mode'],
        task_id=req['binding']['task_id'],selection_digest=m.fixed.digest(req),commands=deepcopy(req['commands']),request=deepcopy(req)) for i,req in enumerate(requests)]
    p=dict(schema=m.NAME+'.prepare',contract=m.contract(),arguments={},producer_consume=verified,
        producer_report_sha256=h('CUDA'),producer_prepare_sha256=h('CUDA prepare'),source_report_sha256=m.SOURCE_SHA,
        tasks=tasks,plans=plans,source_slots=[deepcopy(r['slot']) for r in data['roots']],source_roots=m.allmode.root_metadata(data),
        warm_traces=[deepcopy(r['warm_trace']) for r in data['roots']],bindings={})
    return p,data,verified


def fixture():return deepcopy(_fixture())


def test_fixed5_requests_no_H3_tail_execution():
    p,data,v=fixture();assert m.validate_requests(v,p['tasks'],p['producer_report_sha256'])==p['plans']
    assert [len(x['commands']) for x in p['plans']]==[5]*4
    assert [x['request']['nonexecuted_suffix_steps'] for x in p['plans']]==[0,10,0,10]


@pytest.mark.parametrize('fault',['denominator','mode','binding','root_context','raw','prefix_hash','full_hash','H','release','extra','status','winner','float64'])
def test_request_identity_and_prefix_seal_failclosed(fault):
    p,data,v=fixture();q=v['prefixes'][0]['request']
    if fault=='denominator':v['prefixes'].pop()
    elif fault=='mode':q['mode']='adaptive'
    elif fault=='binding':q['binding']['task_id']=h('bad')
    elif fault=='root_context':q['root_context_sha256']=h('bad')
    elif fault=='raw':q['commands'].append([0.,0.])
    elif fault=='prefix_hash':q['commands_sha256']=h('bad')
    elif fault=='full_hash':q['full_plan_actions_sha256']=h('bad')
    elif fault=='H':q['planning_horizon_packets']=True
    elif fault=='release':q['physical_release']=True
    elif fault=='extra':q['outcome']=1
    elif fault=='status':v['status']='partial_or_failed'
    elif fault=='winner':v['winners'][0]['slot']=1
    else:q['commands'][0][0]=.123456789012345
    with pytest.raises((ValueError,KeyError)):m.validate_requests(v,p['tasks'],p['producer_report_sha256'])


def test_no_MAIN_authority_before_reads_output_env(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('unauthorized metadata read'))
    with pytest.raises(ValueError):m.run('/missing',h('x'),'/missing',lambda:pytest.fail('env'),authorization=None)


def test_native_success_not_native7_reward_threshold():
    calc=m.metrics([0.]*7,[1.,1.,1.,1.,.01,10000.,10000.])
    assert calc['native_success'] is True and calc['native7_reward']<-10000 and calc['joint_xy_error']==2.


def execute(monkeypatch,*,factory=Env,p=None,data=None):
    fp,fd,_=fixture();p=fp if p is None else p;data=fd if data is None else data
    store=MemoryStore();envs=[]
    def make():e=factory();envs.append(e);return e
    monkeypatch.setattr(m,'place',fake_place);monkeypatch.setattr(m,'validate_env',lambda _:None)
    monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(store.files[Path(path).name]).hexdigest())
    r=m.execute(p,data,store,make,authorization=m.AUTH,prepared='/RAM/fixed5_prepare.json',prepare_sha256=h('prepare'))
    return r,store,envs,p


@lru_cache(maxsize=1)
def completed():
    with pytest.MonkeyPatch.context() as mp:return execute(mp)


def test_full4_five_only_51_images_and_nested_resources():
    r,s,envs,p=completed()
    assert r['status']==m.RECORDED,r['errors']
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    assert len(m.validate_saved(r,s.events,p))==4
    for row,env,plan in zip(r['rows'],envs,p['plans'],strict=True):
        assert env.closed and len(env.commands)==15 and row['own_returned']==5
        assert np.asarray(env.commands[10:],np.float32).tobytes()==m.commands(plan['commands']).tobytes()
        assert row['prefix51']['complete'] and len(row['prefix51']['frames'])==51
        assert [cp['step'] for cp in row['checkpoints']]==[0,5]
        assert row['physical_receipt']['returned_steps']==5


@pytest.mark.parametrize('fault',['terminal','truncate'])
@pytest.mark.parametrize('at',[5,11,13,15])
def test_native_earlystop_no_padding(monkeypatch,fault,at):
    r,s,envs,p=execute(monkeypatch,factory=lambda:Env(fault=fault,at=at))
    assert all(e.closed and len(e.commands)==at for e in envs)
    assert all(row['own_returned']==max(at-10,0) for row in r['rows'])
    if not (fault=='terminal' and at==5):assert len(m.validate_saved(r,s.events,p))==4


@pytest.mark.parametrize('mutation',['count','request','raw','metric','gate','denominator','checkpoint','receipt','actual','source','equal','scope'])
def test_saved_resealed_tamper_failclosed(mutation):
    r,s,_,p=completed();r=deepcopy(r);events=deepcopy(s.events);row=r['rows'][0]
    if mutation=='count':r['counts']['public_step']['completed']-=1
    elif mutation=='request':row['request']['binding']['task_id']=h('wrong')
    elif mutation=='raw':row['commands'][0][0]=.5
    elif mutation=='metric':row['responses'][0]['reward']=0
    elif mutation=='gate':r['P3_admission']=True
    elif mutation=='denominator':r['rows'].pop()
    elif mutation=='checkpoint':row['checkpoints'][0]['pixel_sha256']=h('wrong')
    elif mutation=='receipt':row['physical_receipt']['returned_steps']=0
    elif mutation=='actual':row['actual_returns'][0]['actual_bits']='00'
    elif mutation=='source':p=deepcopy(p);p['source_roots'][0]['goal'][0]+=1
    elif mutation=='equal':r['equal_actual_physical_execution_horizon']=False
    else:r['total_internal_physics_scope']='unknown'
    for event in events:
        if event['kind']=='branch_closed' and event['receipt']['index']==0:event['receipt']=deepcopy(row)
    reseal(events,r)
    with pytest.raises((ValueError,KeyError)):m.validate_saved(r,events,p)


def test_saved_consume_full_bytes_and_unchanged_upstream_audit(monkeypatch):
    r,s,_,p=completed();files=deepcopy(s.files)
    monkeypatch.setattr(m,'verify',lambda *a:(deepcopy(p),{}))
    monkeypatch.setattr(m,'saved_artifacts',lambda _:deepcopy(files))
    out=m.consume('/RAM_ONLY',hashlib.sha256(files['report.json']).hexdigest())
    assert out['rows']==r['rows'] and len(out['outcomes'])==4
    assert out['producer_consume']==p['producer_consume'] and out['artifact_authenticity_verified']
    assert all(o['contract_audit']['artifact_authenticity_verified'] is False for o in out['outcomes'])
    assert all(out[k] is False for k in m.FLAGS)
    assert out['new_physics_calls']==out['new_neural_calls']==0


@pytest.mark.parametrize('fault',['png','extra','binding','journal','external','drift','verify'])
def test_saved_byte_boundaries(monkeypatch,fault):
    _,s,_,p=completed();files=deepcopy(s.files);expected=hashlib.sha256(files['report.json']).hexdigest();calls=[]
    def verify(*a):
        if fault=='verify':raise ValueError('complete source verification failure')
        return deepcopy(p),{}
    monkeypatch.setattr(m,'verify',verify)
    if fault=='png':files[next(n for n in files if n.endswith('.png'))]=b'bad'
    elif fault=='extra':files['orphan.bin']=b'extra'
    elif fault=='binding':files['execution_binding.json']=b'{}'
    elif fault=='journal':files['journal.jsonl']=files['journal.jsonl'][:-1]
    elif fault=='external':expected=h('bad')
    def saved(_):
        calls.append(1);out=deepcopy(files)
        if fault=='drift' and len(calls)>1:out['late']=b'bad'
        return out
    monkeypatch.setattr(m,'saved_artifacts',saved)
    with pytest.raises(ValueError):m.consume('/RAM_ONLY',expected)


@pytest.mark.parametrize('fault',['substitute','dtype','step_raise','render_after_return','constructor'])
def test_failure_preserves_denominator_return_vs_validation(monkeypatch,fault):
    class Mock(Env):
        def step(self,c):
            value=super().step(c)
            if len(self.commands)==11:
                if fault=='dtype':self.latest_action=self.latest_action.astype(np.float64)
                if fault=='step_raise':raise RuntimeError('RAM native callback failure')
            return value
        def render(self):
            if fault=='render_after_return' and len(self.commands)==11:raise RuntimeError('RAM render failure after native return')
            return super().render()
    def factory():
        if fault=='constructor':raise RuntimeError('RAM constructor failure')
        return Mock(fault=fault if fault in ('substitute','dtype','step_raise') else None,at=11)
    r,s,envs,p=execute(monkeypatch,factory=factory)
    assert r['status']=='partial_or_failed' and len(r['rows'])==4
    assert len(m.validate_saved(r,s.events,p))==4
    for row in r['rows']:
        assert row['native_success'] is None and row['responses']==[]
        if fault in ('substitute','dtype','constructor'):assert row['physical_receipt'] is None
        elif fault=='step_raise':
            assert row['own_attempted']==1 and row['own_returned']==0
            assert row['physical_receipt']['stop_reason']=='engineering_failure' and row['physical_receipt']['returned_steps']==0
        else:
            assert row['own_returned']==1 and len(row['actual_returns'])==1
            assert row['physical_receipt']['returned_steps']==1 and row['physical_receipt']['native_success'] is None


def bridge_fixture():
    p,data,_=fixture();tasks=p['tasks'];slots=[];exports=[];packets=[]
    for s,t in enumerate(tasks):
        c=dict(source_slot=s,**{k:t['binding'][k] for k in ('root_sha256','task_id','task_sha256','task_contract_sha256','blind_sha256')})
        task_contract={'RAM':'contract'+str(s)};c['task_contract_sha256']=t['binding']['task_contract_sha256']=m.fixed.digest(task_contract)
        e=dict(source_slot=s,task_id=t['binding']['task_id'],blind_sha256=t['binding']['blind_sha256'],with_target=False,
            task_contract=task_contract,task=deepcopy(t))
        slots.append(c);exports.append(e);packets.append(deepcopy(data['roots'][s]['packets']['off']))
    return tasks,slots,exports,packets,data


@pytest.mark.parametrize('fault',[None,'slot','contract','blind','target','context','goal','projection','normalizer','root','rawzero','raworder','RGB','goalRGB'])
def test_exact_encoded_to_native_source_bridge(fault):
    tasks,slots,exports,packets,data=bridge_fixture()
    if fault=='slot':slots[0]['source_slot']=1
    elif fault=='contract':slots[0]['task_contract_sha256']=h('bad')
    elif fault=='blind':exports[0]['blind_sha256']=h('bad')
    elif fault=='target':exports[0]['with_target']=True
    elif fault=='context':exports[0]['task']['root_context']['latent_history'][0][0]+=1
    elif fault=='goal':exports[0]['task']['goal'][0]+=1
    elif fault=='projection':exports[0]['task']['projection'][0][0]+=1
    elif fault=='normalizer':exports[0]['task']['action_mean'][0]+=1
    elif fault=='root':data['roots'][0]['root_sha256']=h('bad')
    elif fault=='rawzero':packets[0]['raw_action_history'][:]=0
    elif fault=='raworder':packets[0]['raw_action_history']=packets[0]['raw_action_history'][::-1].copy()
    elif fault=='RGB':packets[0]['rgb_history'][0,0,0,0]^=1
    elif fault=='goalRGB':packets[0]['goal_rgb'][0,0,0]^=1
    if fault is None:assert m.bridge(tasks,slots,exports,packets,data)==m.allmode.root_metadata(data)
    else:
        with pytest.raises((ValueError,KeyError,TypeError)):m.bridge(tasks,slots,exports,packets,data)


def material_boundary(monkeypatch,tmp_path):
    p,data,v=fixture();v['report_sha256']=m.CUDA_REPORT_SHA;calls=[]
    args=dict(producer_output=str(tmp_path),producer_report_sha256=m.CUDA_REPORT_SHA,producer_source_sha256=m.SUPPORTED_SOURCE_SHA,
        producer_prepared=str(m.ROOT/m.CUDA_PREPARE),producer_prepare_sha256=m.CUDA_PREPARE_SHA)
    code=m.ROOT/'experiments/spectral_reachability'/(m.PRODUCER+'.py');coldpath=m.ROOT/'results/agent_p3_resume_20261027_next_prepare01.json'
    sources={str(coldpath):h('cold')};slots=[];exports={};packets={}
    for i,t in enumerate(p['tasks']):
        path='/RAM/export'+str(i);blind='/RAM/blind'+str(i)
        sources[path]=t['binding']['task_sha256'];sources[blind]=t['binding']['blind_sha256']
        slots.append(dict(task_export=path));exports[path]={'RAM_export':i};packets[blind]=data['roots'][i]['packets']['off']
    prepared=dict(panel=dict(base=dict(tasks=p['tasks'])),sources=sources)
    report=dict(schema=m.PRODUCER+'.report',status='saved_model_prefix_not_admitted',errors=[],prepared=args['producer_prepared'],
        prepare_sha256=m.CUDA_PREPARE_SHA)
    docs={str(tmp_path/'report.json'):report,args['producer_prepared']:prepared,str(coldpath):dict(slots=slots)}
    hashes={str(f):h(str(f)) for f in [*m.own_files(),*m.allmode.own_files(),Path(m.codec.__file__).resolve()]}
    hashes.update(sources);hashes.update({str(code):m.SUPPORTED_SOURCE_SHA,str(Path(m.fixed.__file__)):m.FIXED_SHA,
        str(Path(m.allmode.__file__)):m.HELPER_SHA,str(tmp_path/'report.json'):m.CUDA_REPORT_SHA,args['producer_prepared']:m.CUDA_PREPARE_SHA})
    def consumed(*a):calls.append('real_consume_boundary');return deepcopy(v)
    module=SimpleNamespace(__file__=str(code),consume=consumed)
    monkeypatch.setattr(m,'load_producer',lambda _:module)
    monkeypatch.setattr(m,'sha',lambda path:hashes[str(path)])
    monkeypatch.setattr(m,'read',lambda path,cap:deepcopy(docs[str(path)]))
    monkeypatch.setattr(m,'source_material',lambda:deepcopy(data))
    monkeypatch.setattr(m,'read_export',lambda path:deepcopy(exports[str(path)]))
    monkeypatch.setattr(m,'validate_export',lambda value:calls.append('export_validation_boundary'))
    monkeypatch.setattr(m,'load_blind',lambda path,expected:deepcopy(packets[str(path)]))
    monkeypatch.setattr(m,'bridge',lambda *a:m.allmode.root_metadata(data))
    return args,docs,hashes,module,calls


def test_material_mandatory_full_producer_consume_and_entire_result(monkeypatch,tmp_path):
    args,docs,hashes,module,calls=material_boundary(monkeypatch,tmp_path)
    out,data=m.material(args)
    assert calls==['real_consume_boundary','export_validation_boundary','export_validation_boundary']
    assert len(out['plans'])==4 and out['producer_consume']['winners']
    assert out['producer_report_sha256']==m.CUDA_REPORT_SHA and out['producer_prepare_sha256']==m.CUDA_PREPARE_SHA
    assert all(out['bindings'][k]==v for k,v in hashes.items())


@pytest.mark.parametrize('fault',['source','helper','contract','report','prepare','future','otherbatch','status','errors','binding','consume','cold','drift'])
def test_material_prechecks_and_real_consume_failure_propagate(monkeypatch,tmp_path,fault):
    args,docs,hashes,module,calls=material_boundary(monkeypatch,tmp_path)
    report=docs[str(tmp_path/'report.json')]
    if fault=='source':hashes[module.__file__]=h('bad')
    elif fault=='helper':hashes[str(Path(m.allmode.__file__))]=h('bad')
    elif fault=='contract':hashes[str(Path(m.fixed.__file__))]=h('bad')
    elif fault=='report':hashes[str(tmp_path/'report.json')]=h('bad')
    elif fault=='prepare':hashes[args['producer_prepared']]=h('bad')
    elif fault=='future':args['producer_report_sha256']='0'*64
    elif fault=='otherbatch':args['producer_report_sha256']=h('another terminal batch')
    elif fault=='status':report['status']='partial_or_failed'
    elif fault=='errors':report['errors']=[{'failure':'retained'}]
    elif fault=='binding':report['prepare_sha256']=h('bad')
    elif fault=='consume':module.consume=lambda *a:(_ for _ in ()).throw(ValueError('real consumer rejected'))
    elif fault=='cold':hashes[str(m.ROOT/'results/agent_p3_resume_20261027_next_prepare01.json')]=h('bad')
    else:
        original=module.consume
        def drift(*a):
            value=original(*a);hashes[str(tmp_path/'report.json')]=h('postconsume drift');return value
        module.consume=drift
    with pytest.raises((ValueError,KeyError)):m.material(args)
    if fault in ('source','helper','contract','report','prepare','future','otherbatch','status','errors','binding','consume'):assert calls==[]


def test_prepare_exclusive_before_consume_and_verify_rebuild(monkeypatch,tmp_path):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR+m.PREP_CAP+m.CAP))
    calls=[];p={'RAM':'metadata','arguments':{}}
    monkeypatch.setattr(m,'material',lambda a:(calls.append(a) or (deepcopy(p),{})))
    path=tmp_path/(m.NAME+'_prepare_RAM.json');receipt=m.prepare(str(path),{})
    assert m.verify(str(path),receipt['prepare_sha256'])[0]==p
    before=len(calls)
    with pytest.raises(ValueError):m.prepare(str(path),{})
    assert len(calls)==before
    p['RAM']='changed input'
    with pytest.raises(ValueError):m.verify(str(path),receipt['prepare_sha256'])


def test_store_reentry_and_output_floor(tmp_path,monkeypatch):
    path=tmp_path/(m.NAME+'_RAM')
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR+m.CAP))
    m.Store(path)
    with pytest.raises(ValueError):m.Store(path)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda _:SimpleNamespace(free=m.FLOOR))
    with pytest.raises(ValueError):m.Store(tmp_path/(m.NAME+'_low'))


def test_cli_no_authority_before_environment_import(monkeypatch):
    monkeypatch.setattr(m,'run',lambda *a,**kw:pytest.fail('unauthorized run'))
    with pytest.raises(ValueError):m.main(['run','--prepared','/missing'])


def test_cli_authorized_run_stdout_is_compact_no_row_dump(monkeypatch,capsys):
    report,_,_,_=completed()
    monkeypatch.setattr(m,'run',lambda *a,**kw:deepcopy(report))
    assert m.main(['run','--prepared','/RAM/p','--prepare-sha',h('p'),'--output','/RAM_ONLY',
        '--authorization',m.AUTH])==0
    out=capsys.readouterr().out
    assert len(out)<4096 and 'run_summary' in out and '"rows"' not in out and 'execute only' in out
