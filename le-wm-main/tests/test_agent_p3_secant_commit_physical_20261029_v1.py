"""RAM arrays and real temporary Store only; no native physics or fullstrict."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_p3_secant_commit_physical_20261029_v1 as m
import test_agent_p3_actual_cp5_commit_physical_20261029_v2 as reference

old=reference.old
_temps=[]


def fixture(allfour=False,none=False):
    _,data,args,c,u,b=reference.fixture()
    args['delivery_completion_sha']=old.h('RAM completion')
    c['schema']=m.PRODUCER+'.prepare';c['rules']['protocol_sha256']=m.DELIVERY_PROTOCOL_SHA
    c['groups']=[]
    u.update(schema=m.PRODUCER+'.consume',status='all4_secant_model_deliveries_verified_NOT_physical',
        groups=4,candidate_denominator=16,completion_sha256=args['delivery_completion_sha'],
        protocol_sha256=m.DELIVERY_PROTOCOL_SHA,accounting={'RAM_inherited':True})
    for i,arm in enumerate(u['arms']):
        d=arm['delivery'];d['schema']=m.PRODUCER+'.delivery'
        d['source_row_sha256']=old.h(('row',i));d['source_row_pointer']=f'/rows/{i}/result/rows/0'
        arm['selected_index']=0
        g=dict(selected_index=0,eligible_order=[0],items=[dict(row_sha256=d['source_row_sha256'],
            origin_pointer=d['source_row_pointer'],actions=deepcopy(d['raw_edge']))])
        if none or not allfour and i:
            arm.update(status='no_witness',model_delivery_verified=False,no_witness=True,delivery=None,
                selected_index=None,eligible_order=[])
            g.update(selected_index=None,eligible_order=[])
        c['groups'].append(g)
    p=m.assemble(args,c,u,b,data)
    return p,data,args,c,u,b


def execute(p=None,data=None,*,fault=None,at=18,allfour=False):
    if p is None:p,data,*_=fixture(allfour=allfour)
    tmp=tempfile.TemporaryDirectory(prefix='secant_commit_RAM_');_temps.append(tmp)
    root=Path(tmp.name);prepared=root/'RAM_prepared.json';prepared.write_bytes(m.raw(p))
    store=m.Store(root/(m.NAME+'_RAM'));envs=[]
    def factory():
        e=old.Env(fault=fault,at=at);envs.append(e);return e
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m.native,'place',old.fake_place);mp.setattr(m.native,'validate_env',lambda _:None)
        r=m.execute(p,data,store,factory,authorization=m.AUTH,prepared=str(prepared),prepare_sha256=m.sha(prepared))
    events=[m.native.decode(x) for x in (store.path/'journal.jsonl').read_bytes().splitlines(keepends=True)]
    return r,store,p,data,envs,events


@lru_cache(maxsize=2)
def complete(allfour=False):return execute(allfour=allfour)


def test_current_one_delivery_plus_three_null_full_denominator():
    r,s,p,data,envs,events=complete()
    assert r['status']==m.RECORDED,r['errors']
    assert [x['status'] for x in r['rows']]==['horizon_completed']+['no_witness']*3
    assert len(m.validate_saved(r,events,p))==4 and len(envs)==1 and envs[0].closed
    assert len(envs[0].commands)==20
    assert r['counts']['public_step']==dict(attempted=20,completed=20)
    assert r['counts']['physics_step']==dict(attempted=204,completed=204)
    assert len(list(s.path.glob('*.png')))==66
    for row in r['rows'][1:]:
        assert row['plan']['commands'] is row['plan']['model_delivery'] is row['final_state'] is row['native_success'] is None


def test_full4_shape_census_real_store_and_copies():
    r,s,p,_,envs,events=complete(True);c=m.shape_census()
    assert r['status']==m.RECORDED,r['errors']
    assert len(m.validate_saved(r,events,p))==4 and len(envs)==4
    assert r['counts']=={k:dict(attempted=v,completed=v) for k,v in m.contract()['caps'].items()}
    files=list(s.path.iterdir());pngs=[f for f in files if f.suffix=='.png']
    assert len(pngs)==264 and c['PNG_count']==264 and c['PNG_per_group']==66
    assert max(f.stat().st_size for f in pngs)<=c['PNG_file_upper_bytes']
    assert sum(f.stat().st_size for f in files if f.suffix!='.png')<=c['metadata_upper_bytes']
    assert c['total_upper_bytes']<=m.CAP and c['working_upper_bytes']<=m.CAP-m.FINAL
    assert sum(e['kind']=='branch_closed' for e in events)==4
    assert sum(e['kind']=='fresh_observation' for e in events)==264
    ops=[e for e in events if e['kind'] in ('operation_dispatch_intent','operation_returned','operation_failed')]
    assert len(ops)==2*len(r['actual_operation_receipts'])
    # Actual receipts persist twice in journal (intent+terminal), once in report.
    assert c['operation_receipt_persisted_copies']==3
    assert c['prefix_persisted_copies']==c['row_persisted_copies']==2
    print('RAM_CENSUS',json.dumps(dict(PNG_count=len(pngs),PNG_bytes=sum(f.stat().st_size for f in pngs),
        metadata_bytes=sum(f.stat().st_size for f in files if f.suffix!='.png'),
        operation_receipts=len(r['actual_operation_receipts']),operation_events=len(ops),
        observation_events=sum(e['kind']=='fresh_observation' for e in events),
        branch_rows=sum(e['kind']=='branch_closed' for e in events),report_rows=len(r['rows'])),sort_keys=True))
    for i,e in enumerate(envs):
        np.testing.assert_array_equal(np.asarray(e.commands[15:],np.float32),p['plans'][i]['model_delivery']['commit_raw5'])
        assert len(e.commands)==20 # H3 lookahead never dispatched


@pytest.mark.parametrize('allfour',[False,True])
def test_real_store_full_png_journal_consume(tmp_path,monkeypatch,allfour):
    r,s,p,data,_,events=complete(allfour)
    monkeypatch.setattr(m,'verify',lambda path,h:(deepcopy(p),deepcopy(data)))
    out=m.consume(s.path,m.sha(s.path/'report.json'),m.sha(s.path/'completion.json'))
    assert out['saved_PNG_verified']==(264 if allfour else 66)
    assert len(out['rows'])==4 and all(out[k] is False for k in m.FLAGS)
    assert out['completion']['inclusive_wall_seconds']>=r['wall_seconds']
    assert all(line==m.raw(e) for line,e in zip((s.path/'journal.jsonl').read_bytes().splitlines(keepends=True),events,strict=True))


def test_real_store_two_events_reader_hashchain(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_two'))
    s.emit('RAM_one',value=1);s.emit('RAM_two',value=2)
    lines=(s.path/'journal.jsonl').read_bytes().splitlines(keepends=True)
    es=[m.native.decode(line) for line in lines]
    assert len(es)==2 and all(line==m.raw(e) for line,e in zip(lines,es,strict=True))
    assert es[1]['previous_sha256']==es[0]['event_sha256']
    assert es[1]['event_sha256']==m.source.digest({k:v for k,v in es[1].items() if k!='event_sha256'})==s.previous
    with pytest.raises(ValueError):s.write('journal.jsonl',m.raw(es[1])+b'\n',append=True)


@pytest.mark.parametrize('kind',['schema','status','completion','missing_slot','selected','rowsha','pointer','raw','segments','goal','tokens','flags','alias'])
def test_assembly_rejects_wrong_authority_or_source(kind):
    _,data,args,c,u,b=fixture();d=u['arms'][0]['delivery']
    if kind=='schema':u['schema']='oldR.consume'
    elif kind=='status':u['status']='partial'
    elif kind=='completion':u['completion_sha256']=old.h('wrong')
    elif kind=='missing_slot':u['arms'].pop()
    elif kind=='selected':d['candidate_index']=1
    elif kind=='rowsha':d['source_row_sha256']=old.h('wrong')
    elif kind=='pointer':d['source_row_pointer']='/rows/3'
    elif kind=='raw':d['raw_edge'][0][0]=.25
    elif kind=='segments':d['verification'][1]['segments']=[15]
    elif kind=='goal':b['roots'][0]['anchor']['goal'][0]+=.25
    elif kind=='tokens':b['roots'][0]['history']['history_tokens'][0][0]+=.25
    elif kind=='flags':u['physical_release']=True
    else:d['C3']=d.pop('full_C3_demonstrated')
    with pytest.raises((ValueError,KeyError,IndexError)):m.assemble(args,c,u,b,data)


@pytest.mark.parametrize('key',m.UPSTREAM_FLAGS)
@pytest.mark.parametrize('bad',[True,0,None,'false','ABSENT'])
def test_false_flags_are_not_defaulted(key,bad):
    flags=dict.fromkeys(m.UPSTREAM_FLAGS,False)
    if bad=='ABSENT':flags.pop(key)
    else:flags[key]=bad
    with pytest.raises(ValueError):m.upstream_flags(flags,'RAM')


@pytest.mark.parametrize('kind',['state','snapshot','pixels','png','warmraw','history_frame','tokens'])
def test_exact_guard_rejects_before_control6(kind):
    p,data,*_=fixture();ref=p['prefix_reference'][0]
    if kind=='state':ref['checkpoint']['state'][0]+=.25
    elif kind=='snapshot':ref['checkpoint']['snapshot']['state_float64_bits']='wrong'
    elif kind=='pixels':ref['checkpoint']['pixel_sha256']=old.h('wrong')
    elif kind=='png':ref['frames'][-3]['receipt']['file_sha256']=old.h('wrong')
    elif kind=='warmraw':ref['warm_raw10'][0][0]=.25
    elif kind=='history_frame':ref['history_frames'][0]['pixel_sha256']=old.h('wrong')
    else:ref['history_tokens'][0][0]+=.25
    reference.reseal_plan(p,0)
    r,s,_,_,envs,events=execute(p,data)
    assert r['status']=='partial_or_failed' and len(envs)==1 and len(envs[0].commands)==15
    assert len(m.validate_saved(r,events,p))==4
    assert not any(e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step'
        and e['receipt'].get('phase')=='plan' and e['receipt'].get('step',0)>=6 for e in events)


def test_guard_is_durable_before_first_commit():
    r,s,p,_,_,events=complete()
    guard=next(e for e in events if e['kind']=='cp5_guard')
    before=[e for e in events if e['kind']=='fresh_observation' and e['index']<guard['index']]
    commit=next(e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['resource']=='public_step'
        and e['receipt'].get('phase')=='plan' and e['receipt'].get('step')==6)
    assert len(before)==51 and guard['accepted'] and guard['index']<commit['index']
    row=r['rows'][0];meter=m.Meter(type('Broken',(),{'emit':lambda *a,**kw:(_ for _ in ()).throw(OSError('fsync'))})())
    meter.where=dict(branch=0);meter.guard_context(row,p['prefix_reference'][0],p['plans'][0])
    meter.receipts=[x for x in r['actual_operation_receipts'] if x['branch']==0 and x['resource']=='public_step'
        and (x['phase']=='warmup' or x['step']<=5)]
    called=[]
    with pytest.raises(OSError):meter.call('public_step',lambda:called.append(True),phase='plan',step=6)
    assert not called and not meter.guarded


@pytest.mark.parametrize('at',[5,14,18])
def test_earlystop_preserves_full_denominator_no_padding(at):
    r,s,p,_,envs,events=execute(fault='truncate',at=at)
    assert r['status']==m.RECORDED,r['errors']
    assert len(envs)==1 and len(envs[0].commands)==at and r['rows'][0]['status']=='native_stopped'
    assert r['rows'][0]['missing_checkpoints'] and len(m.validate_saved(r,events,p))==4


@pytest.mark.parametrize('kind',['unknown','guard_missing','guard_late','guard_proof','action','label','counter','missing_frame'])
def test_saved_event_mutations_rejected(kind):
    r,_,p,_,_,events=deepcopy(complete())
    if kind=='unknown':events[1]['kind']='claim'
    elif kind=='guard_missing':events[:]=[e for e in events if e['kind']!='cp5_guard']
    elif kind=='guard_late':
        i=next(i for i,e in enumerate(events) if e['kind']=='cp5_guard');e=events.pop(i);events.insert(i+2,e)
    elif kind=='guard_proof':next(e for e in events if e['kind']=='cp5_guard')['proof']['actual_raw10_float32_sha256']=old.h('wrong')
    elif kind=='action':next(e for e in events if e['kind']=='returned_transition')['actual_bits']='00'
    elif kind=='label':next(e for e in events if e['kind']=='returned_transition')['reward']+=1
    elif kind=='counter':r['counts']['physics_step']['completed']-=1
    else:events.pop(next(i for i,e in enumerate(events) if e['kind']=='fresh_observation'))
    old.reseal(events,r)
    with pytest.raises((ValueError,KeyError)):m.validate_saved(r,events,p)


def test_png_uncompressible_and_metadata_hard_caps(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_caps'));c=m.shape_census()
    pixels=np.random.default_rng(7).integers(0,256,(224,224,3),dtype=np.uint8)
    buf=io.BytesIO();Image.fromarray(pixels).save(buf,format='PNG')
    assert len(buf.getvalue())<=c['PNG_file_upper_bytes']
    s.write('noise.png',buf.getvalue())
    with pytest.raises(ValueError):s.write('large.png',b'x'*(c['PNG_file_upper_bytes']+1))
    with pytest.raises(ValueError):s.put('execution_binding.json',dict(x='x'*m.METADATA_CAP))


@pytest.mark.parametrize('kind',['external_sha','wall','files','prepare','flags','failure'])
def test_completion_mutations_refused(tmp_path,monkeypatch,kind):
    r,s,p,data,_,_=complete();out=tmp_path/(m.NAME+'_mutated');out.mkdir()
    for f in s.path.iterdir():(out/f.name).write_bytes(f.read_bytes())
    cp=json.loads((out/'completion.json').read_bytes());expected=m.sha(out/'report.json')
    if kind=='wall':cp['inclusive_wall_seconds']=901
    elif kind=='files':cp['artifact_sha256'].pop(next(k for k in cp['artifact_sha256'] if k.endswith('.png')))
    elif kind=='prepare':cp['prepare_sha256']=old.h('wrong')
    elif kind=='flags':cp['P1_admission']=True
    elif kind=='failure':(out/'completion_failure.json').write_bytes(m.raw(dict(failed=True)))
    (out/'completion.json').write_bytes(m.raw(cp));h=m.sha(out/'completion.json')
    monkeypatch.setattr(m,'verify',lambda *args:(p,data))
    with pytest.raises((ValueError,KeyError)):m.consume(out,expected,old.h('wrong') if kind=='external_sha' else h)


def test_saved_actual_assembly_only_no_fullstrict(monkeypatch):
    args=m.DEFAULT_ARGUMENTS;c=m.read(args['delivery_prepared'],m.META);u=m.read(args['delivery_receipt'],m.META)
    assert m.sha(args['delivery_receipt'])==args['delivery_receipt_sha']
    b=m.read(args['bridge'],m.META)
    prior=m.read(m.ROOT/'results/agent_p3_h3_witness_physical_20261029_v1_prepare_MAIN01.json',m.META)
    data=dict(roots=[dict(slot=s,warm_trace=t) for s,t in zip(prior['source_slots'],prior['warm_traces'],strict=True)])
    monkeypatch.setattr(m.native.allmode,'root_metadata',lambda data:deepcopy(prior['source_roots']))
    p=m.assemble(args,c,u,b,data)
    assert [x['witness_status'] for x in p['plans']]==['model_delivery_verified']+['no_witness']*3
    assert p['plans'][0]['model_delivery']['candidate_index']==u['arms'][0]['selected_index']
    assert p['planner_authority']['inherited_accounting']==u['accounting']


def test_cold_import_and_authorization():
    code=f'from experiments.spectral_reachability import {m.NAME}; import sys; assert "torch" not in sys.modules and "pymunk" not in sys.modules'
    r=subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    p,data,*_=fixture();called=[]
    with pytest.raises(ValueError):m.execute(p,data,None,lambda:called.append(True),authorization='wrong',prepared='/RAM/p',prepare_sha256=old.h('p'))
    assert not called


def test_entire_upstream_rebuild_call_and_canonical_receipt_not_status(tmp_path,monkeypatch):
    args=deepcopy(m.DEFAULT_ARGUMENTS);rp=tmp_path/'receipt.json'
    persisted=dict(status='fixture',nested=dict(sha='original',items=[1,2,3]))
    rp.write_bytes(m.raw(persisted));args['delivery_receipt']=str(rp);seen=[]
    class Producer:
        @staticmethod
        def consume(*values):seen.append(values);return deepcopy(persisted)
    assert m.rebuild_delivery(args,Producer)==persisted
    assert seen==[(Path(args['delivery_output']),args['delivery_report_sha'],args['delivery_prepared'],
        args['delivery_prepare_sha'],args['delivery_completion_sha'])]
    changed=deepcopy(persisted);changed['nested']['items'][2]=4;rp.write_bytes(m.raw(changed))
    with pytest.raises(ValueError,match='entire actual saved secant'):m.rebuild_delivery(args,Producer)


def test_completion_post_report_hash_and_final_failure(tmp_path):
    s=m.Store(tmp_path/(m.NAME+'_publication'));s.put('execution_binding.json',dict(RAM=True));s.emit('RAM',value=1)
    report=dict(execution_binding=dict(prepared='/RAM/prepare',prepare_sha256=old.h('prepare')),wall_seconds=1)
    def checkpoint():
        assert (s.path/'report.json').exists() and m.sha(s.path/'report.json')
        return 2
    result=m.publish(s,report,0,clock=checkpoint)
    cp=m.read(s.path/'completion.json');assert cp['inclusive_wall_seconds']==2 and cp['report_sha256']==result['report_sha256']
    failed=m.Store(tmp_path/(m.NAME+'_failed_publication'));failed.put('execution_binding.json',dict(RAM=True));failed.emit('RAM',value=1)
    times=iter([2,901])
    with pytest.raises(ValueError,match='final publication wall'):m.publish(failed,report,0,clock=lambda:next(times))
    assert (failed.path/'completion_failure.json').exists()
