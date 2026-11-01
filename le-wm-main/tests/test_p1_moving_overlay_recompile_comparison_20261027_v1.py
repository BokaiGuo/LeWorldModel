"""RAM-only protocol fixtures. No producer/model/env import or actual artifacts."""
from copy import deepcopy
from io import BytesIO
import json
import math

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p1_moving_overlay_recompile_comparison_20261027_v1 as c


def h(v):return c.sha(c.canonical(v))


def seal(events):
    previous=None;out=[]
    for i,e in enumerate(events):
        e=dict(e,index=i,previous_sha256=previous);e['event_sha256']=h(e)
        previous=e['event_sha256'];out.append(e)
    return out,b''.join(c.canonical(e)+b'\n' for e in out)


@pytest.fixture(scope='module')
def saved():
    """Synthetic all8 complete physical ledger + true producer-shaped saved graph."""
    mem={};prefix='/RAM';refs={}
    def put(name,value):
        data=value if isinstance(value,bytes) else c.canonical(value)
        path=prefix+'/'+name;mem[path]=data;return dict(file=path,sha256=c.sha(data))
    pixels=np.zeros((224,224,3),dtype=np.uint8);buf=BytesIO();Image.fromarray(pixels).save(buf,format='PNG')
    image=buf.getvalue();image_sha=c.sha(image);pixel_sha=c.sha(pixels.tobytes())
    initial=[0.,0.,0.,0.,0.,0.,0.];goals={s:[30.+s,0.,10.,0.,.1,0.,0.] for s in (0,1)}
    original_prepare={'mock':'RAM original prepare'};se=[dict(kind='run_started',prepare=original_prepare)]
    pack_rows=[];exports=[]
    for s in (0,1):
        root=c.sha(np.asarray(initial+goals[s],dtype=np.float64).tobytes())
        pack_rows.append(dict(slot=dict(index=s),root_sha256=root,blind_payload_sha256=h(['old',s])))
        se.extend([dict(kind='native_return',slot=s,phase='construction',step=1,state=goals[s]),
                   dict(kind='fresh_observation',slot=s,phase='replay0',step=10,state=initial,snapshot=dict(state=initial))])
        artifacts={}
        for mode in ('off','on'):
            name=f's{s}_{mode}_blind_inputs.npz';ref=put(name,('RAM only '+name).encode())
            artifacts[mode]=dict(path=name,sha256=ref['sha256'],goal_pixel_sha256=pixel_sha)
        exports.append(dict(source_slot=s,old_root_sha256=root,artifacts=artifacts))
    se.append(dict(kind='run_finished',status='fixed_slots_processed',counts={'mock':2}))
    se,sb=seal(se);refs['original_source_journal']=put('sourcejournal.jsonl',sb)
    pack=dict(schema=c.SOURCEPACK_SCHEMA,sourcepack_complete=True,status='fixed_slots_processed',errors=[],
              rows=pack_rows,counts={'mock':2},prepare_id=h(original_prepare),journal_chain_tail=se[-1]['event_sha256'])
    refs['original_sourcepack']=put('sourcepack.json',pack)
    refs['overlay_source_report']=put('overlayreport.json',dict(schema=c.OVERLAY+'.report',
        status='paired_moving_overlay_recorded',full_protocol=True,errors=[],blind_exports=exports))
    refs['physical_source']=put('physical.py',b'RAM source, never imported')
    refs['planner_source']=put('planner.py',b'RAM compiler source, never imported')
    full=[];planner_rows=[];artifacts={}
    for i,(s,mode,arm) in enumerate(c.CELLS):
        e=exports[s]['artifacts'][mode]
        p=dict(schema=c.PLANNER+'.plan',root_slot=s,condition=mode,arm=arm,commands=np.zeros((10,2),np.float32).tolist(),
            task_id=h(['task',s,mode]),blind_path='/RAM/'+e['path'],blind_sha256=e['sha256'],
            source_report_sha256=refs['overlay_source_report']['sha256'],with_target=mode=='on',selection_uses_physics=False,
            P1_admission=False,prediction={'mock_only':True})
        p['selection_digest']=h(p);full.append(p);name=f'plan_{i:02}.plan.json';pr=put(name,p);artifacts[name]=pr['sha256']
        planner_rows.append({**{f:p[f] for f in ('root_slot','condition','arm','selection_digest')},'plan_file':name,
                             'resources':{'predict':78 if arm==c.ARMS[0] else 66},'AD_API_resources':{'mock':True}})
    planner=dict(schema=c.PLANNER+'.report',status='overlay_recompiled_plans_sealed_not_admitted',errors=[],
        model_unchanged=True,source_verified=True,commands_released=True,
        source_report_sha256=refs['overlay_source_report']['sha256'],rows=planner_rows,
        artifact_sha256=artifacts,contract={'mock_only':True,'predict':576})
    refs['planner_report']=put('plannerreport.json',planner)
    handoff=[dict(p,planner_report_sha256=refs['planner_report']['sha256']) for p in full]
    keys='root_slot condition arm commands selection_digest task_id blind_path blind_sha256 source_report_sha256 planner_report_sha256'.split()
    plans=[{f:p[f] for f in keys} for p in handoff]
    prepared=dict(schema=c.PHYSICAL+'.prepare',plans=plans,contract={'caps':c.CAPS},compiler_handoff_sha256=h(handoff),
        bindings={r['file']:r['sha256'] for r in refs.values()},
        planner_report_sha256=refs['planner_report']['sha256'],source_report_sha256=refs['overlay_source_report']['sha256'])
    prepared['bindings'].update({p['blind_path']:p['blind_sha256'] for p in plans})
    refs['physical_prepare']=put('prepare.json',prepared)
    binding=dict(prepared=refs['physical_prepare']['file'],prepare_sha256=refs['physical_prepare']['sha256'],
        planner_report_sha256=refs['planner_report']['sha256'],source_report_sha256=refs['overlay_source_report']['sha256'])
    refs['physical_execution_binding']=put('execution_binding.json',binding)
    events=[dict(kind='run_started',prepare_sha256=refs['physical_prepare']['sha256'],plans=plans,contract=prepared['contract'])]
    rows=[];warm=np.array([[.1,0]]*5+[[0,.1]]*5,dtype=np.float32)
    for i,p in enumerate(plans):
        s,mode,arm=c.key(p);where={f:p[f] for f in ('root_slot','condition','arm','selection_digest')};where['branch']=i
        row=dict(index=i,**{f:p[f] for f in ('root_slot','condition','arm','selection_digest','task_id')},
            status='horizon_completed',checkpoints=[],missing_checkpoints=[],native_success=False,initial_exact=True,
            root_state=initial,goal=goals[s],goal_pixel_sha256=pixel_sha,warmup_returned=10,own_attempted=10,own_returned=10,PNG=66,
            first_error=None,secondary_errors=[],trajectory=[])
        for phase in ('warmup','plan'):
            for t in range(11):
                state=initial.copy()
                if phase=='plan':state[0]=t*(2 if mode=='on' else 1)
                if t:
                    command=warm[t-1] if phase=='warmup' else np.asarray(p['commands'],np.float32)[t-1]
                    bits=command.tobytes().hex()
                    for kind in ('operation_attempted','operation_completed'):
                        events.append(dict(kind=kind,**where,resource='public_step',phase=phase,step=t,raw_bits=bits))
                    events.append(dict(kind='returned_transition',**where,phase=phase,step=t,requested_bits=bits,
                        actual_bits=bits,actual_dtype='<f4',state=state,snapshot=dict(state=state),goal_state=goals[s]))
                for j in range(3):
                    name=f'b{i}_{phase}_{t}_{j}.png';mem['/RAM/'+name]=image
                    receipt=dict(phase=phase,step=t,condition='on' if j==1 else 'off',overlay_index=j,png=name,
                        file_sha256=image_sha,pixel_sha256=pixel_sha,state=state,snapshot=dict(state=state))
                    invariant=dict(snapshot=dict(state=state),renderer='mock')
                    events.append(dict(kind='fresh_observation',**where,receipt=receipt,pre=invariant,post=invariant))
                    if j==(0 if mode=='off' else 1):
                        row['trajectory'].append(receipt)
                        if phase=='plan' and t in (0,5,10):
                            row['checkpoints'].append({f:receipt[f] for f in ('step','png','file_sha256','pixel_sha256','state','snapshot')})
        for resource,cap in c.CAPS.items():
            if resource=='public_step':continue
            # All caps divisible by8; mock metering is not a simulator.
            for _ in range(cap//8):
                for kind in ('operation_attempted','operation_completed'):events.append(dict(kind=kind,**where,resource=resource))
        events.append(dict(kind='branch_closed',**where,status=row['status'],own_returned=10,first_error=None));rows.append(row)
    counts={k:dict(attempted=v,completed=v) for k,v in c.CAPS.items()}
    events.append(dict(kind='run_finished',status='moving_overlay_recompile_physical_recorded',counts=counts,errors=[]))
    events,jb=seal(events);refs['physical_journal']=put('journal.jsonl',jb)
    report=dict(schema=c.PHYSICAL+'.report',status='moving_overlay_recompile_physical_recorded',rows=rows,errors=[],
        fixed_denominator=8,inputs_final_verified=True,counts=counts,journal_chain_tail=events[-1]['event_sha256'],journal_sha256=refs['physical_journal']['sha256'],
        execution_binding=binding,execution_binding_sha256=refs['physical_execution_binding']['sha256'],
        prepare_sha256=refs['physical_prepare']['sha256'],planner_report_sha256=refs['planner_report']['sha256'],
        source_report_sha256=refs['overlay_source_report']['sha256'])
    refs['physical_report']=put('report.json',report)
    own=dict(schema=c.NAME+'.binding',files=refs);ref=put('binding.json',own)
    targets=c.source_targets(pack,se)
    return dict(mem=mem,binding=ref,pack=pack,source_events=se,report=report,prepared=prepared,plans=plans,
                events=events,targets=targets,frame=lambda name:mem['/RAM/'+name])


def run(s,**updates):
    kw={k:s[k] for k in ('report','prepared','events','targets','plans')};kw.update(updates)
    return c.assemble(**kw,read_frame=s['frame'])


def test_actual_contract_end_to_end_RAM(saved,monkeypatch):
    monkeypatch.setattr(c,'SOURCEPACK_SHA',c.sha(saved['mem']['/RAM/sourcepack.json']))
    b=saved['binding'];out=c.material(b['file'],b['sha256'],read=saved['mem'].__getitem__)
    assert len(out['rows'])==8 and out['physical_counts']['public_step']['completed']==160
    assert out['physical_counts']['physics_step']['completed']==1632
    assert out['actual_neural_calls']==out['actual_physics_calls']==0
    assert len(out['binding']['frame_file_sha256'])==528
    assert all(r['actual_RGB']['actual_latent_Q'] is None for r in out['rows'])
    assert len(c.canonical(out))<c.CAP


def test_stable_identity_not_array_order(saved):
    report=deepcopy(saved['report']);report['rows'].reverse()
    result=run(saved,report=report,plans=list(reversed(saved['plans'])))
    assert [(r['root_slot'],r['condition'],r['arm']) for r in result['rows']]==list(c.CELLS)
    assert all(r['on_minus_off_terminal']['native_distance']<0 for r in result['descriptive']['per_root'])
    assert result['descriptive']['by_arm'][c.ARMS[0]]['root_denominator']==2
    assert all(r['strict_non_greedy_vs_raw_zero'] is None for r in result['rows'])


def test_missing10_remains_null_full_denominator(saved):
    report=deepcopy(saved['report']);row=report['rows'][0]
    events=deepcopy(saved['events']);report['status']='partial_or_failed'
    report['errors']=[dict(stage='branch',type='MockCaptureError',message='RAM fixture')]
    events[-1]['status']=report['status'];events[-1]['errors']=report['errors']
    row['status']='engineering_failed';row['first_error']=report['errors'][0]
    close=next(e for e in events if e['kind']=='branch_closed' and e['branch']==0)
    close['status']=row['status'];close['first_error']=row['first_error']
    row['checkpoints']=[cp for cp in row['checkpoints'] if cp['step']!=10];row['missing_checkpoints']=[10]
    row['trajectory']=[cp for cp in row['trajectory'] if not(cp['phase']=='plan' and cp['step']==10)]
    out=run(saved,report=report,events=events)
    assert out['rows'][0]['terminal10'] is None and len(out['rows'])==8
    assert out['descriptive']['by_arm'][c.ARMS[0]]['on_minus_off']['native_distance'] is None
    assert out['rows'][0]['first_error']['type']=='MockCaptureError'


@pytest.mark.parametrize('field,value',[('selection_digest','0'*64),('task_id','0'*64),('root_state',[1.]*7),('goal',[1.]*7)])
def test_branch_binding_rejected(saved,field,value):
    report=deepcopy(saved['report']);report['rows'][0][field]=value
    with pytest.raises(ValueError):run(saved,report=report)


def test_checkpoint_cannot_disagree_with_journal(saved):
    report=deepcopy(saved['report']);cp=deepcopy(report['rows'][0]['checkpoints'][1]);cp['state'][0]+=1
    cp['snapshot']['state']=cp['state'];report['rows'][0]['checkpoints'][1]=cp
    with pytest.raises(ValueError):run(saved,report=report)


@pytest.mark.parametrize('kind,field,value',[
    ('operation_attempted','raw_bits','00'*8),('returned_transition','actual_bits','00'*8),
    ('returned_transition','actual_dtype','<f8')])
def test_no_raw_action_clipping_or_rescaling(saved,kind,field,value):
    events=deepcopy(saved['events']);e=next(e for e in events if e['kind']==kind and e.get('phase')=='warmup')
    e[field]=value
    with pytest.raises(ValueError):run(saved,events=events)


def test_full_counts_not_selected_only(saved):
    report=deepcopy(saved['report']);events=deepcopy(saved['events'])
    report['counts']['public_step']['completed']=80;events[-1]['counts']=report['counts']
    with pytest.raises(ValueError,match='recount'):run(saved,report=report,events=events)


def test_actual_return_state_cannot_be_replaced_by_render_label(saved):
    events=deepcopy(saved['events']);e=next(e for e in events if e['kind']=='returned_transition')
    e['state']=[123.]*7
    with pytest.raises(ValueError,match='returned transition'):run(saved,events=events)


def test_accepted_triplet_invariant_rejected(saved):
    events=deepcopy(saved['events']);e=next(e for e in events if e['kind']=='fresh_observation')
    e['post']={'corrupt':True}
    with pytest.raises(ValueError,match='snapshot'):run(saved,events=events)


@pytest.mark.parametrize('field',['file_sha256','pixel_sha256'])
def test_png_three_layer_validation(saved,field):
    cp=deepcopy(saved['report']['rows'][0]['checkpoints'][0]);cp[field]='0'*64
    with pytest.raises(ValueError):c.png(saved['frame'](cp['png']),cp)


def test_official_RGB_not_mock32():
    buf=BytesIO();Image.fromarray(np.zeros((32,32,3),np.uint8)).save(buf,format='PNG');data=buf.getvalue()
    with pytest.raises(ValueError,match='RGB224'):c.png(data,dict(file_sha256=c.sha(data),pixel_sha256='0'*64))


def test_hashchain_detects_edits(saved):
    raw=b''.join(c.canonical(e)+b'\n' for e in saved['source_events'])
    assert c.chain(raw,saved['pack']['journal_chain_tail'])==saved['source_events']
    with pytest.raises(ValueError):c.chain(raw.replace(b'construction',b'constructioX'),saved['pack']['journal_chain_tail'])


def test_original_goal_not_endpoint_labels(saved):
    events=deepcopy(saved['source_events']);next(e for e in events if e['kind']=='native_return')['state'][0]+=1
    with pytest.raises(ValueError,match='identity'):c.source_targets(saved['pack'],events)


@pytest.mark.parametrize('offset',[-4*math.pi,-2*math.pi,0,2*math.pi,4*math.pi])
def test_wrapped_2pi_invariance(offset):
    goal=np.zeros(7);state=np.zeros(7);state[4]=.8+offset
    m=c.metrics(goal,state)
    assert m['wrapped_angle_error']==pytest.approx(.8)
    assert m['wrapped_allxy_success'] is False


def test_native_multiturn_bug_visible_not_silently_fixed():
    state=np.zeros(7);state[4]=2*math.pi+.8;m=c.metrics(np.zeros(7),state)
    assert m['negative_native_angle'] and m['native_success'] and not m['wrapped_allxy_success']
    assert m['native_wrapped_success_disagree']


def test_velocity_reward_not_success_condition():
    state=np.zeros(7);state[5:]=100;m=c.metrics(np.zeros(7),state)
    assert m['native_success'] and m['native_reward']==pytest.approx(-math.sqrt(20000))
    assert m['success_has_velocity_condition'] is False


@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'{"a":NaN}'])
def test_strict_json(raw):
    with pytest.raises(ValueError):c.parse(raw)


def test_external_binding_sha_first(saved):
    with pytest.raises(ValueError,match='binding SHA'):
        c.material(saved['binding']['file'],'0'*64,read=saved['mem'].__getitem__)


def test_dependency_hash_before_assembly(saved,monkeypatch):
    monkeypatch.setattr(c,'SOURCEPACK_SHA',c.sha(saved['mem']['/RAM/sourcepack.json']))
    mem=dict(saved['mem']);mem['/RAM/physical.py']+=b'changed'
    with pytest.raises(ValueError,match='physical_source'):
        c.material(saved['binding']['file'],saved['binding']['sha256'],read=mem.__getitem__)


@pytest.mark.parametrize('name',['plan_00.plan.json','b0_plan_5_0.png'])
def test_saved_compiler_or_frame_corruption(saved,monkeypatch,name):
    monkeypatch.setattr(c,'SOURCEPACK_SHA',c.sha(saved['mem']['/RAM/sourcepack.json']))
    mem=dict(saved['mem']);mem['/RAM/'+name]+=b'corrupt'
    with pytest.raises(ValueError):
        c.material(saved['binding']['file'],saved['binding']['sha256'],read=mem.__getitem__)


def test_fixed8_never_subset(saved):
    with pytest.raises(ValueError):c.index_rows(saved['report']['rows'][:-1])
    with pytest.raises(ValueError):c.index_rows([saved['report']['rows'][0]]*8)


def test_help_and_mock_cli(saved,monkeypatch,capsys):
    with pytest.raises(SystemExit) as error:c.main(['--help'])
    assert error.value.code==0
    monkeypatch.setattr(c,'material',lambda *args:dict(status='RAM_only'))
    monkeypatch.setattr(c,'write_new',lambda *args:'a'*64)
    assert c.main(['--binding','/RAM/binding.json','--binding-sha','a'*64,'--output','/RAM/no-write.json'])==0
    assert 'RAM_only' in capsys.readouterr().out


def test_disk_floor_before_open(monkeypatch):
    monkeypatch.setattr(c.shutil,'disk_usage',lambda _:type('Disk',(),{'free':c.FLOOR-1})())
    with pytest.raises(ValueError,match='floor'):c.write_new('/RAM/not-created.json',{})
