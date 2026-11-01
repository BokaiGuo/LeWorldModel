"""Explicit RAM-only typed-contract fixtures, not ready/execution evidence."""
from copy import deepcopy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_p3_physical_commit_preconditions_20261029_v1 as m

factory=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_parent_native_adapter_20261029_v1.py'))['task']


def frame(tmp,step):
    p=tmp/('frame'+str(step)+'.png');pixels=np.full((224,224,3),step,np.uint8)
    Image.fromarray(pixels).save(p);state=[0.]*7
    return dict(absolute_step=step,plan_step=step-10,png=str(p),file_sha256=m.sha(p),
        pixel_sha256=m.io.hashlib.sha256(pixels.tobytes()).hexdigest(),state=state,snapshot={'state':state,'fixture':True})


def data(tmp):
    t=factory();mean=np.asarray(t['action_mean'],np.float32);std=np.asarray(t['action_std'],np.float32)
    raw=np.zeros((10,2),np.float32);incoming=m.tokens(raw,mean,std).reshape(2,10)
    t['root_context']['action_history']=incoming.tolist();t['binding']['context_sha256']=m.core.digest(t['root_context'])
    frames=[frame(tmp,k) for k in (5,10,15)];cp=frames[-1]
    trace=[]
    for phase,step in [('warmup',k) for k in range(11)]+[('plan',k) for k in range(6)]:
        actual=frames[0] if phase=='warmup' and step==5 else frames[1] if (phase=='warmup' and step==10) or (phase=='plan' and step==0) else cp
        for k in range(3):
            receipt={name:deepcopy(actual[name]) for name in ('state','snapshot','file_sha256','pixel_sha256')}
            receipt.update(phase=phase,step=step,overlay_index=k,condition='on' if k==1 else 'off')
            trace.append(dict(pre={'snapshot':deepcopy(actual['snapshot'])},post={'snapshot':deepcopy(actual['snapshot'])},receipt=receipt))
    anchor=dict(root_sha256=t['binding']['root_sha256'],task_id=t['binding']['task_id'],condition='off',
        goal=[0.]*7,goal_pose=[0.]*3,renderer_sha256='b'*64,checkpoint=cp,
        prefix51=trace,
        warm_raw10=raw.tolist(),prefix_raw5=raw[:5].tolist())
    history=dict(frames=frames,raw10=raw.tolist(),latent3=t['root_context']['latent_history'],history_tokens=incoming.tolist(),
        goal192=t['goal'],normalizer_sha256=m.core.digest([t['action_mean'],t['action_std']]),context_sha256=m.core.digest(t['root_context']))
    commands=np.full((5,2),.25,np.float32);token=m.tokens(commands,mean,std).reshape(1,10)
    after=deepcopy(t['root_context']);after['action_history']=[incoming[-1].tolist(),token[0].tolist()]
    after['latent_history']=after['latent_history'][1:]+[after['latent_history'][-1]]
    plan=dict(actions=commands.tolist(),path=[after])
    release=dict(plan_available=True,status='released',released_plan=plan,core_plan=deepcopy(plan),whole_replay=[after],segmented_replay=[after],
        guards_passed=True,errors=[],context_sha256=m.core.digest(t['root_context']),task_sha256=m.core.digest(t))
    ledger=m.core.Ledger(dict.fromkeys(m.core.AXES,10000));ledger.scope={'fixture':True}
    ledger.call('replay',lambda:None);ledger.call('replay',lambda:None)
    source=tmp/'source.py';source.write_text('# fixture only\n')
    anchor['renderer_sha256']=m.sha(source)
    model=dict(expected_model_hash=t['binding']['model_sha256'],before={'original_state_dict':t['binding']['model_sha256']},
        after={'original_state_dict':t['binding']['model_sha256']},loaded_bytes=128,
        sources_before={str(source):m.sha(source)},sources_after={str(source):m.sha(source)})
    mi=dict(latent3=deepcopy(history['latent3']),history_tokens=incoming.tolist(),outgoing_tokens=np.concatenate([incoming,token]).tolist(),
        raw_commit5=commands.tolist(),goal192=t['goal'],input_fields=['latent3','history_tokens','outgoing_tokens','goal192'])
    return dict(task=t,anchor=anchor,restore_reference=deepcopy(anchor),actual_history=history,
        model_input=mi,release=release,resources=ledger.receipt(),model=model)


def write_json(p,obj):p.write_bytes(m.canonical(obj));return dict(path=str(p),sha256=m.sha(p))


def material(tmp,mutation=None):
    d=data(tmp)
    if mutation:mutation(d)
    second=deepcopy(d)
    second['task']['binding']['root_sha256']=m.core.digest('second known RAM root')
    second['task']['binding']['task_id']=m.core.digest('second known RAM task')
    for key in ('anchor','restore_reference'):
        second[key]['root_sha256']=second['task']['binding']['root_sha256']
        second[key]['task_id']=second['task']['binding']['task_id']
    second['release']['task_sha256']=m.core.digest(second['task'])
    c=m.template();roles={section:'physical' if section in ('anchor','restore_reference') else 'history' if section=='actual_history' else 'planner' for section in m.SECTIONS}
    for role in ('planner','physical','history'):
        report=dict(schema='RAM_'+role+'.report',data=[{s:v for s,v in datum.items() if roles[s]==role} for datum in (d,second)])
        rr=write_json(tmp/(role+'_report.json'),report)
        receipt=dict(schema='RAM_'+role+'.consume',status='RAM_validated',report_sha256=rr['sha256'],source_sha=m.sha(tmp/'source.py'))
        sr=write_json(tmp/(role+'_receipt.json'),receipt)
        c['documents'][role+'_report']=rr;c['documents'][role+'_receipt']=sr
        c['authorities'][role]=dict(report=role+'_report',receipt=role+'_receipt',report_schema=report['schema'],
            receipt_schema=receipt['schema'],receipt_status=receipt['status'],report_sha_pointer='/report_sha256',
            source_bindings=[dict(file=dict(path=str(tmp/'source.py'),sha256=m.sha(tmp/'source.py')),
                sha_ref={'$ref':role+'_receipt','pointer':'/source_sha'})])
    for row in c['slots']:
        row['projection']={s:{'$ref':roles[s]+'_report','pointer':'/data/'+str(row['root'])+'/'+s} for s in m.SECTIONS}
    return c,d


def test_missing_real_inputs_block_all4():
    out=m.build(m.template())
    assert out['status']=='BLOCKED' and len(out['rows'])==4 and len(out['missing'])==6
    assert all(not out[k] for k in m.FLAGS)
    assert all(r['status']=='BLOCKED' for r in out['rows'])


def test_full_typed_fixture_never_claims_ready_or_runtime_restore(tmp_path):
    c,d=material(tmp_path);out=m.build(c)
    assert out['status']=='saved_inputs_validated_pending_runtime_guard'
    assert all(not out[k] for k in m.FLAGS)
    assert len(out['rows'])==4 and all(not r['runtime_restore_proven'] for r in out['rows'])
    assert out['rows'][0]['commit_raw5']==d['release']['released_plan']['actions']
    assert out['rows'][0]['resource_completed']['replay_calls']==2


@pytest.mark.parametrize('kind',('release','authority','projection','section'))
def test_missing_gate_is_blocked_not_ready(tmp_path,kind):
    c,_=material(tmp_path,lambda d:d['release'].update(plan_available=False) if kind=='release' else None)
    if kind=='authority':c['authorities']['history']=None
    if kind=='projection':c['slots'][0]['projection']=None
    if kind=='section':c['slots'][0]['projection'].pop('model')
    out=m.build(c);assert out['status']=='BLOCKED' and out['ready'] is False


@pytest.mark.parametrize('kind',('prefix51','traceorder','overlay','renderstate','cp5join','renderer','restore','statezero','time','history','double_normalization','outgoing','raw100','rawbox',
    'length','core_plan','whole','guard','source','model','bytes','resources','unknown','goal','context','privileged'))
def test_typed_saved_mutations_reject(tmp_path,kind):
    def change(d):
        if kind=='prefix51':d['anchor']['prefix51'].pop()
        elif kind=='traceorder':d['anchor']['prefix51'][0]['receipt']['step']=1
        elif kind=='overlay':d['anchor']['prefix51'][0]['receipt']['condition']='on'
        elif kind=='renderstate':d['anchor']['prefix51'][0]['post']={'bad':True}
        elif kind=='cp5join':d['anchor']['prefix51'][-3]['receipt']['pixel_sha256']='0'*64
        elif kind=='renderer':d['anchor']['renderer_sha256']='0'*64
        elif kind=='restore':d['restore_reference']['goal'][0]=1
        elif kind=='statezero':d['restore_reference']['checkpoint']['state'][0]=-0.
        elif kind=='time':d['actual_history']['frames'][0]['absolute_step']=6
        elif kind=='history':d['actual_history']['raw10'][0][0]=.25
        elif kind=='double_normalization':d['actual_history']['history_tokens']=m.tokens(np.asarray(d['actual_history']['history_tokens'],np.float32).reshape(10,2),np.asarray(d['task']['action_mean'],np.float32),np.asarray(d['task']['action_std'],np.float32)).reshape(2,10).tolist()
        elif kind=='outgoing':d['model_input']['outgoing_tokens'][0][0]=7.
        elif kind=='raw100':d['model_input']['raw_commit5']=(np.asarray(d['model_input']['raw_commit5'],np.float32)/100).tolist()
        elif kind=='rawbox':d['release']['released_plan']['actions'][0][0]=2.
        elif kind=='length':d['release']['released_plan']['actions'].pop()
        elif kind=='core_plan':d['release']['core_plan']['actions'][0][0]=.5
        elif kind=='whole':d['release']['whole_replay']=[]
        elif kind=='guard':d['release']['guards_passed']=False
        elif kind=='source':d['model']['sources_after']={}
        elif kind=='model':d['model']['after']['original_state_dict']='0'*64
        elif kind=='bytes':d['model']['loaded_bytes']=0
        elif kind=='resources':d['resources']['completed']['replay_calls']=3
        elif kind=='unknown':d['resources']['FLOPs']=0
        elif kind=='goal':d['model_input']['goal192'][0]=.25
        elif kind=='context':d['release']['context_sha256']='0'*64
        else:d['model_input']['input_fields'].append('native_state')
    c,_=material(tmp_path,change)
    with pytest.raises(ValueError):m.build(c)


def test_section_cannot_import_unattested_document(tmp_path):
    c,_=material(tmp_path);c['documents']['unattested']=c['documents']['planner_report']
    c['slots'][0]['projection']['release']['$ref']='unattested'
    with pytest.raises(ValueError,match='authenticated planner'):m.build(c)


def test_failed_attempt_no_release_stays_paid_in_blocked_manifest(tmp_path):
    def failure(d):
        d['release']['plan_available']=False
        ledger=m.core.Ledger(dict.fromkeys(m.core.AXES,10));ledger.scope={'fixture':'failure'}
        def fail():raise RuntimeError('RAM only')
        with pytest.raises(RuntimeError):ledger.call('metric',fail)
        d['resources']=ledger.receipt()
    c,_=material(tmp_path,failure);out=m.build(c)
    assert out['status']=='BLOCKED'
    assert out['rows'][0]['resource_attempted']['metric_calls']==1
    assert out['rows'][0]['resource_completed']['metric_calls']==0


def test_same_root_time_control_cannot_cross_wire(tmp_path):
    c,_=material(tmp_path)
    for section,projection in c['slots'][1]['projection'].items():
        projection['pointer']=projection['pointer'].replace('/data/0/','/data/1/')
    with pytest.raises(ValueError,match='same-root/time'):m.build(c)


def test_strict_sha_join_and_no_literal_evidence(tmp_path):
    c,_=material(tmp_path);c['authorities']['planner']['report_sha_pointer']='/schema'
    with pytest.raises(ValueError,match='SHA join'):m.build(c)
    c,_=material(tmp_path);c['slots'][0]['projection']['release']=True
    with pytest.raises(ValueError,match='no literal'):m.build(c)


def test_source_bytes_mutation_rejected(tmp_path):
    c,_=material(tmp_path);(tmp_path/'source.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='source bytes'):m.build(c)


def test_source_binding_cannot_be_unjoined_claim(tmp_path):
    c,_=material(tmp_path);c['authorities']['physical']['source_bindings'][0]['sha_ref']['pointer']='/report_sha256'
    with pytest.raises(ValueError,match='source SHA explicitly'):m.build(c)


def test_pointer_escape_and_array_index():
    assert m.pointer({'a/b':[{'~':5}]},'/a~1b/0/~0')==5
    for p in ('/a~2b','/a~1b/01'):
        with pytest.raises(ValueError):m.pointer({'a/b':[1]},p)


def test_real_cli_blocked_writer_and_no_overwrite(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    p=tmp_path/'input.json';ref=write_json(p,m.template());out=tmp_path/'results'/(m.NAME+'_fixture.json')
    args=['--contract',str(p),'--contract-sha',ref['sha256'],'--output',str(out)]
    m.main(args);receipt=json.loads(capsys.readouterr().out)
    assert receipt['status']=='BLOCKED' and receipt['sha256']==m.sha(out)
    assert json.loads(out.read_bytes())['ready'] is False
    with pytest.raises(ValueError,match='exclusive'):m.main(args)


def test_import_template_no_torch_physics():
    code=f'from experiments.spectral_reachability import {m.NAME} as m; import sys; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules; m.main(["--template"])'
    p=subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert p.returncode==0 and json.loads(p.stdout)['slots'][0]['projection'] is None
