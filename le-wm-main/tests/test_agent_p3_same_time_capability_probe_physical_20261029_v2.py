"""Inert RAM and temporary disk only; no environments, models or upstream consume."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import agent_p3_same_time_capability_probe_physical_20261029_v2 as m


def context(actions):
    return dict(action_history=np.asarray(actions,np.float32).reshape(2,10).tolist(),latent_history=[[0.0]*192 for _ in range(3)])


def state(root,step):return [100.0+root,100.0+step,200.0,200.0,0.25,1.0,2.0]


def snapshot(s):
    return dict(state=s,state_float64_bits=struct.pack('<7d',*s).hex(),
        bodies=dict(agent=dict(original_unknown=[0.0,-0.0]),block=dict(original_unknown='retained')))


PIXELS=np.full((224,224,3),180,dtype=np.uint8)
BUF=io.BytesIO();Image.fromarray(PIXELS).save(BUF,format='PNG');PNG=BUF.getvalue()
FILE_SHA=hashlib.sha256(PNG).hexdigest();PIXEL_SHA=hashlib.sha256(PIXELS.tobytes()).hexdigest()


def observation(root,phase,step,overlay):
    s=state(root,step);snap=snapshot(s)
    pre=dict(snapshot=snap,goal_state=[0.0]*7,goal_pose=[256.0,256.0,0.7853981633974483],
             goal_rgb_sha256=PIXEL_SHA,renderer=dict(fixed_original_unknown='kept'))
    return dict(kind='fresh_observation',index=step*3+overlay,previous_sha256='a'*64,event_sha256='b'*64,
        branch=root,root_slot=root,arm='source_role',condition='off',selection_digest='c'*64,
        pre=pre,post=deepcopy(pre),receipt=dict(phase=phase,step=step,overlay_index=overlay,
            condition='on' if overlay==1 else 'off',png=f'original_{phase}_{step}_{overlay}.png',
            file_sha256=FILE_SHA,pixel_sha256=PIXEL_SHA,state=s,snapshot=deepcopy(snap)))


def result(root,step,raw,*,truncated=False):
    s=state(root,step)
    return dict(state=s,snapshot=snapshot(s),reward=-float(np.linalg.norm(np.asarray(s,np.float64))),
        terminated=False,truncated=truncated,goal_state=[0.0]*7,
        actual_bits=''.join(m.bits([raw],1)),actual_dtype='<f4')


@pytest.fixture
def inputs():
    refs=dict(full_witness=dict(path='/inert/fw.json',sha256='1'*64),
        stageB={k:dict(path='/inert/'+k+'.json',sha256=str(i+2)*64) for i,k in enumerate(m.B_PINS)},
        U=[dict(path=f'/inert/u{i}.json',sha256='8'*64) for i in range(5)])
    warm=[[0.125,0.0] for _ in range(10)];commands=[[0.25,-0.0] for _ in range(10)]
    w=dict(schema=m.fw.NAME+'.witness',actual_MAIN_authority=False,stageB_ready=False,rows=[],
           authorities={k:dict(path='/A/'+k,sha256='9'*64) for k in m.fw.PINS})
    plans=[];exports=[]
    for i,(root,role,kind,slot) in enumerate(m.fw.ORDER):
        obs=[observation(root,ph,st,ov) for ph in ('warmup','plan') for st in range(11) for ov in range(3)]
        task=dict(root_context=context(commands),action_mean=[0.0,0.0],action_std=[1.0,1.0])
        oldtask=dict(task,root_context=context(warm[-5:]+commands[:5]))
        ident=dict(root_slot=root,role=role,index=i)
        source=dict(state=state(root,10),snapshot=snapshot(state(root,10)),path='/inert/source.png')
        ref=dict(index=i,root_slot=root,role=role,observations=obs,warm_raw10=warm,control_raw10=commands,
            source_row=dict(goal=[0.0]*7,root_state=state(root,0),plan=dict(model_task=oldtask)),
            stageA_task=dict(task=task,identity=ident),stageA_row=dict(source=source),
            source_root=dict(goal=[0.0]*7,goal_pixels={'off':PIXEL_SHA}),
            source_slot_metadata=dict(reset_seed=51+root,root_seed=31+root,index=root),
            placement_events=[dict(native_goal=[0.0]*7,requested_state=state(root,0))],
            unknown_semantics=dict(rng='unknown',public_state='unknown',contact_cache='unknown'))
        ref['actual_response10']=m.response([0.0]*7,state(root,0),10,result(root,10,commands[-1]))
        w['rows'].append(ref)
        raw5=[[0.5,0.0] for _ in range(5)];bb=''.join(m.bits(raw5,5));a=refs['stageB']
        plan=dict(index=i,root_slot=root,role=role,checkpoint=10,probe_checkpoint=15,
            task=task,endpoint_identity=ident,source=source,causal_raw10=commands,
            whole=[{'fixture_context':True}],segmented=[{'fixture_context':True}],raw5=raw5,
            native_seal_sha256='d'*64,native_call_ids=dict(signal=i+2,whole=7+2*i,segmented=8+2*i),
            raw_float32_bits=bb,raw_float32_sha256=hashlib.sha256(bytes.fromhex(bb)).hexdigest(),
            authority=dict(prepared=a['prepare']['path'],prepare_sha256=a['prepare']['sha256'],
                report=a['report']['path'],report_sha256=a['report']['sha256'],
                completion=a['completion']['path'],completion_sha256=a['completion']['sha256']))
        plan['witness_sha256']=m.compact_digest(plan);plans.append(plan)
        exports.append(dict(schema=m.B+'.witness',receipt=a['receipt']['path'],receipt_sha256=a['receipt']['sha256'],
                            plan=deepcopy(plan),evidence_kind='MAIN_SHA_bound_saved_receipt'))
    counts=dict(jacobian_API=7,VJP_directions=1344,predict_API=24,solve_calls=7,replay_calls=10,metric_calls=22)
    resource=dict(attempted=counts,completed=counts,caps=counts)
    a=refs['stageB'];results=dict(status='complete_NOT_admitted',signals=[{}]*7,probes=[{}]*5,metrics=[{}]*22,
                                resource_receipt=resource,native_seal_sha256='d'*64)
    bp=dict(schema=m.B+'.prepare',bridge_authority=w['authorities'])
    br=dict(schema=m.B+'.report',prepared=a['prepare']['path'],prepare_sha256=a['prepare']['sha256'],
        status='complete_NOT_admitted',errors=[],result=results,sources_verified=True,model_before={'weights':'fixture'},model_after={'weights':'fixture'})
    bc=dict(schema=m.B+'.completion',prepared=a['prepare']['path'],prepare_sha256=a['prepare']['sha256'],report_sha256=a['report']['sha256'])
    bs=dict(schema=m.B+'.consume',status='saved_complete_NOT_admitted',fixed_signals=7,fixed_probes=5,
        verification=dict(full_native_closure=True,native_calls=17),prepared=a['prepare']['path'],prepare_sha256=a['prepare']['sha256'],
        report_sha256=a['report']['sha256'],completion_sha256=a['completion']['sha256'],
        bridge_authority=w['authorities'],plans=plans,**{k:results[k] for k in ('signals','probes','metrics','resource_receipt')})
    return w,bp,br,bc,bs,exports,refs


def prepared(inputs):return m.assemble(*inputs,fixture=True)


def drive_fixture(output,p,w,index=0,stop=None,break_ref=False):
    job=p['jobs'][index];ref=deepcopy(w['rows'][job['endpoint']])
    if break_ref:ref['observations'][50]['pre']['renderer']['fixed_original_unknown']='tamper'
    binding=dict(schema=m.NAME+'.job_binding',job=job,fixture_only=True,
        prepared='/inert/prepare.json',prepare_sha256='1'*64,authority=p['authority'],guard_transport_fields=list(m.TRANSPORT)+['receipt/png'])
    store=m.Store(str(output),binding,p['census']);store.emit('run_started',binding_sha256=m.digest(binding))
    meter=m.Meter(store,job,ref);closed=dict(status='horizon_completed',error=None,cleanup_errors=[])
    def reset_call():
        for _ in range(2):meter.call('set_state',lambda:meter.call('physics_step',lambda:None))
        meter.call('goal_setter',lambda:None);meter.call('goal_render',lambda:None)
        return dict(goal_state=[0.0]*7,state=state(job['root_slot'],0),snapshot=snapshot(state(job['root_slot'],0)))
    try:
        meter.call('constructor',lambda:None)
        meter.call('reset',reset_call,role='seed',seed=51+job['root_slot'])
        meter.call('reset',reset_call,role='placement',seed=0,requested_state=state(job['root_slot'],0))
        for phase,commands in (('warmup',job['warm10']),('plan',job['raw15'])):
            if phase=='plan':meter.call('goal_bind',lambda:meter.call('goal_setter',lambda:None))
            for step in range(len(commands)+1):
                stopping=stop==(phase,step)
                if step:
                    raw=commands[step-1]
                    def public_call():
                        for _ in range(10):meter.call('physics_step',lambda:None)
                        return result(job['root_slot'],step,raw,truncated=stopping)
                    meter.call('public_step',public_call,phase=phase,step=step,raw_bits=''.join(m.bits([raw],1)))
                for overlay in range(3):
                    meter.call('explicit_render',lambda:None)
                    e=observation(job['root_slot'],phase,step,overlay)
                    rc=e['receipt'];rc['png']=f'b{index}_{phase}_{step:02}_{overlay}.png'
                    rc['file_sha256']=store.frame(rc['png'],PIXELS)
                    store.emit('fresh_observation',**meter.where,receipt=rc,pre=e['pre'],post=e['post'])
                if stopping:closed['status']='native_stopped';raise m.NativeStop()
    except m.NativeStop:
        closed['error']=None
    except BaseException as e:closed.update(status='engineering_failed',error=m.error(e))
    finally:
        meter.call('close',lambda:None,cleanup=True)
        store.emit('branch_closed',result=closed);store.emit('run_finished',result=closed)
    seal=m.publish(store,ref)
    return store,ref,seal


def test_assemble_fixed10_and_two_authority_domains(inputs):
    p=prepared(inputs)
    assert [(j['root_slot'],j['source_role'],j['probe']) for j in p['jobs']]==[
        (r,role,probe) for r,role,_,_ in m.fw.ORDER for probe in ('U','Z2')]
    assert not p['full_witness_stageB_flag_unchanged'] and p['stageB_connected']
    assert p['fixture_only']
    assert all('rng' not in e['receipt']['snapshot'] for r in inputs[0]['rows'] for e in r['observations'])
    for j in p['jobs'][1::2]:assert ''.join(m.bits(j['raw15'][-5:],5))=='00'*40


@pytest.mark.parametrize('mutation',('partial','plan_export','whole','task','source','rawbits','paid','bridge','count','model','U_time'))
def test_authority_joins_failclosed(inputs,mutation):
    w,bp,br,bc,bs,exports,refs=inputs
    if mutation=='partial':bs['status']='saved_partial_NOT_admitted'
    elif mutation=='plan_export':exports[0]['plan']['role']='Z'
    elif mutation=='whole':bs['plans'][0]['whole'][0]['change']=True
    elif mutation=='task':w['rows'][0]['stageA_task']['task']['action_mean']=[1.0,1.0]
    elif mutation=='source':w['rows'][0]['stageA_row']['source']['state'][0]=0.0
    elif mutation=='rawbits':bs['plans'][0]['raw_float32_bits']='00'*40
    elif mutation=='paid':bs['resource_receipt']['completed']=dict(bs['resource_receipt']['completed'],predict_API=23)
    elif mutation=='bridge':bp['bridge_authority']={}
    elif mutation=='count':bs['fixed_probes']=4
    elif mutation=='model':br['model_after']={'changed':True}
    elif mutation=='U_time':bs['plans'][0]['checkpoint']=5
    with pytest.raises(ValueError):prepared(inputs)


def test_actual_fields_budget(inputs):
    c=m.census(inputs[0])
    assert sum(m.COUNTS.values())==374 and c['total_events']==834
    assert c['conservative_total_bytes']<m.CAP
    assert c['journal_upper_bytes']<=m.QUOTAS['journal.jsonl']
    assert c['report_upper_bytes']<=m.QUOTAS['report.json']
    assert m.COUNTS['physics_step']==2*m.COUNTS['reset']+10*m.COUNTS['public_step']
    assert m.CAP==67108864 and m.RESERVE==8388608 and m.FLOOR==209715200 and m.WALL==900


def test_large_real_field_bound_rejects_no_truncation(inputs):
    inputs[0]['rows'][0]['observations'][0]['pre']['original_unknown']='x'*(400*1024)
    with pytest.raises(ValueError,match='actual field budget'):m.census(inputs[0])


@pytest.fixture(scope='module')
def unused_module_marker():return 'no native imports'


@pytest.mark.parametrize('index',range(10))
def test_all10_whole_disk_81png834events(inputs,tmp_path,index):
    p=prepared(inputs);store,ref,seal=drive_fixture(tmp_path/f'job{index}',p,inputs[0],index)
    rec=m.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref,permit_fixture=True)
    assert rec['event_count']==834 and len(rec['row']['observations'])==81
    assert len(rec['row']['operations'])==374 and rec['row']['guards']==[51,66]
    assert rec['row']['counts']=={k:dict(attempted=n,completed=n) for k,n in m.COUNTS.items()}
    assert rec['row']['missing_checkpoints']==[] and rec['row']['endpoint15']['step']==15
    with pytest.raises(ValueError,match='RAM never production'):m.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref)


def test_guard_mismatch_no_control6_and_earlystop_missing(inputs,tmp_path):
    p=prepared(inputs)
    store,ref,seal=drive_fixture(tmp_path/'bad_ref',p,inputs[0],break_ref=True)
    r=m.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref,permit_fixture=True)['row']
    assert r['status']=='engineering_failed' and r['endpoint15'] is None
    assert r['counts']['public_step']['attempted']==15 and r['guards']==[]
    store,ref,seal=drive_fixture(tmp_path/'early',p,inputs[0],stop=('plan',12))
    r=m.consume_job(str(store.path),seal['report_sha256'],seal['completion_sha256'],store.binding,ref,permit_fixture=True)['row']
    assert r['status']=='native_stopped' and r['missing_checkpoints']==[15]
    assert r['counts']['public_step']['attempted']==22 and r['guards']==[51,66]


def test_guard_durable_failure_never_dispatches_next(inputs,tmp_path):
    p=prepared(inputs);job=p['jobs'][0];ref=inputs[0]['rows'][0]
    binding=dict(job=job,fixture_only=True);store=m.Store(str(tmp_path/'failedguard'),binding,p['census'])
    meter=m.Meter(store,job,ref)
    # No observations yet: guard failure itself is durable, no intent/callback.
    called=[]
    with pytest.raises(ValueError):meter.call('public_step',lambda:called.append(True),phase='plan',step=6,raw_bits='00'*8)
    assert called==[] and not meter.receipts and not meter.guarded


@pytest.mark.parametrize('tamper',('png','extra_file','report','chain','missing_png'))
def test_disk_tamper(inputs,tmp_path,tamper):
    store,ref,s=drive_fixture(tmp_path/'out',prepared(inputs),inputs[0]);p=store.path
    if tamper=='png':next(p.glob('*.png')).write_bytes(b'corrupt')
    elif tamper=='extra_file':(p/'unknown.bin').write_bytes(b'x')
    elif tamper=='report':
        r=m.read(p/'report.json');r['row']['unknowns']['rng']='fabricated';(p/'report.json').write_bytes(m.raw(r)+b'\n')
    elif tamper=='chain':
        f=p/'journal.jsonl';f.write_bytes(f.read_bytes().replace(b'run_started',b'run_changed',1))
    else:next(p.glob('*.png')).unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        m.consume_job(str(p),s['report_sha256'],s['completion_sha256'],store.binding,ref,permit_fixture=True)


def test_failed_final_reserve_and_unknown_not_refunded(inputs,tmp_path):
    p=prepared(inputs);job=p['jobs'][0];binding=dict(job=job,fixture_only=True)
    store=m.Store(str(tmp_path/'partial'),binding,p['census']);store.emit('run_started',binding_sha256=m.digest(binding))
    meter=m.Meter(store,job,inputs[0]['rows'][0]);meter.call('constructor',lambda:None)
    closed=dict(status='engineering_failed',error={'type':'WriteFailure','message':'fixture'},cleanup_errors=[])
    seal=m.failed_publication(store,meter,closed,ValueError('incomplete publication'))
    r=m.consume_failure(str(store.path),seal['failure_sha256'],binding)
    assert r['counts'] is None and r['endpoint15'] is None and r['unknown_final_callbacks']
    assert r['declarations']['observed_meter_counts']['constructor']['attempted']==1


def test_no_production_on_bad_auth_or_incomplete_authority(inputs,tmp_path):
    with pytest.raises(ValueError,match='MAIN explicit'):m.run('/not/a/prepare','0'*64,str(tmp_path/'out'),'wrong')
    bad=m.references();bad['stageB']['receipt']['sha256']='0'*64
    with pytest.raises(ValueError,match='actual MAIN fixed'):m.material(bad)
    assert not (tmp_path/'out').exists()


def test_no_stageB_import_consume_no_globals_patch():
    import ast
    t=ast.parse(Path(m.__file__).read_text())
    assert not any(isinstance(n,ast.Global) for n in ast.walk(t))
    imports=[n.module for n in ast.walk(t) if isinstance(n,ast.ImportFrom)]
    assert not any(x and (m.B in x or 'torch' in x or 'pickle' in x) for x in imports)


def test_constructor_source_guard_without_importing_environment():
    m.inert_constructor('class PushT:\n def __init__(self): self.seed = 0\n')
    for method in ('reset','step','render','_set_state','_set_goal_state','_setup'):
        with pytest.raises(ValueError,match='constructor inert'):
            m.inert_constructor(f'class PushT:\n def __init__(self): self.{method}()\n')


def test_accepted_guard_write_failure_prevents_dispatch(inputs,tmp_path,monkeypatch):
    p=prepared(inputs);job=p['jobs'][0];ref=inputs[0]['rows'][0]
    store=m.Store(str(tmp_path/'durable'),dict(job=job,fixture_only=True),p['census'])
    meter=m.Meter(store,job,ref);called=[]
    # A proof may be valid, but release still requires durable journal delivery.
    monkeypatch.setattr(m,'guard_proof',lambda *a:{'fixture_valid_proof':True})
    def refuse(*args,**kwargs):raise OSError('fixture fsync/write refusal')
    monkeypatch.setattr(store,'write',refuse)
    with pytest.raises(OSError,match='refusal'):
        meter.call('public_step',lambda:called.append(True),phase='plan',step=11,raw_bits='00'*8)
    assert not called and not meter.receipts and not meter.guarded


def test_wall_deadline_does_not_disable_final_reserve(inputs,tmp_path):
    p=prepared(inputs);now=[0.0]
    store=m.Store(str(tmp_path/'deadline'),{'fixture_only':True},p['census'],clock=lambda:now[0])
    now[0]=901.0
    with pytest.raises(ValueError,match='900s'):store.check()
    store.put('failure.json',{'status':'fixture_deadline_only'},final=True)
    assert m.read(store.path/'failure.json')['status']=='fixture_deadline_only'


def test_81_incompressible_real_png_disk_budget(inputs,tmp_path):
    p=prepared(inputs);store=m.Store(str(tmp_path/'noise81'),{'fixture_only':True},p['census'])
    rng=np.random.default_rng(20261029);sizes=[]
    b=m.png_budget()
    assert b==dict(scanline_bytes=150752,deflate_upper_bytes=171963,idat_block_bytes=65536,
        idat_chunks_upper=4,png_fixed_bytes=45,png_upper_bytes=172056)
    tight=150752+(150752>>12)+(150752>>14)+(150752>>25)+13
    for i in range(81):
        pixels=rng.integers(0,256,(224,224,3),dtype=np.uint8)
        name=f'noise{i:02}.png';h=store.frame(name,pixels);path=store.path/name
        assert m.sha(str(path))==h
        body=path.read_bytes();sizes.append(len(body));m.check_png_stream(body)
        with Image.open(path) as im:assert np.array_equal(np.asarray(im),pixels)
    assert min(sizes)>tight and max(sizes)<=b['png_upper_bytes']
    assert len(list(store.path.glob('*.png')))==81
    assert sum(sizes)<=81*b['png_upper_bytes']
    assert 81*b['png_upper_bytes']+sum(m.QUOTAS.values())+m.RESERVE<=m.CAP==64*1024**2
    with pytest.raises(ValueError,match='81 bounded PNG'):store.frame('noise82.png',pixels)


def test_png_budget_rejects_ancillary_chunk_and_excess_idat():
    # Runtime proof is enforced on actual writer output; not an assumption
    # that every possible Pillow plugin/configuration follows this grammar.
    extra=struct.pack('>I',0)+b'tEXt'+b'\x00'*4
    with pytest.raises(ValueError,match='minimal Pillow'):
        m.check_png_stream(PNG[:33]+extra+PNG[33:])
    empty=struct.pack('>I',0)+b'IDAT'+b'\x00'*4
    with pytest.raises(ValueError,match='minimal Pillow'):
        m.check_png_stream(PNG[:-12]+empty*5+PNG[-12:])
