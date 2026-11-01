"""Four-slot saved integration fixture: fabricated arrays and transitions, never evidence."""
from copy import deepcopy
import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import agent_p3_fullcommit_live_20261028_v1_tests as fixture
from experiments.spectral_reachability import p3_matched_order_feedback_host_v2_codex_20261029 as host
from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_v2_codex_20261029 as saved
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_v2_codex_20261029 as encoder


class FastFixtureStore(host.storage.Store):
    """Same event ABI; batch fixture IO avoids per-event fsync, not a durability test."""
    def write(self,name,data,category,append=False):
        self.check(category,len(data));p=self.output/name
        assert not p.is_symlink() and (append and name=='events.jsonl' or not p.exists())
        with p.open('ab' if append else 'xb') as f:assert f.write(data)==len(data)
        self.used[category]+=len(data)


class BufferedRAM:
    kind='RAM_FIXTURE'
    def __init__(self,task,sink):self.task=task;self.sink=sink
    def invoke(self,name,args,ledger):
        buffer=fixture.Events();backend=fixture.RamBackend(self.task,buffer)
        result=getattr(backend,name)(*args,ledger)
        for event in buffer.events:self.sink.event(event)
        return result
    def signal(self,request,ledger):return self.invoke('signal',(request,),ledger)
    def replay(self,context,actions,segmented,ledger):return self.invoke('replay',(context,actions,segmented),ledger)


def make_fixture(tmp_path):
    warm=fixture.m.old.kernel.moving.warmup_commands();roots=[]
    for i in range(2):
        root,_=fixture.request(slot=i)
        root['root_context']['action_history']=fixture.m.moving.normalized(root,warm)
        root['binding']['context_sha256']=host.life.seam.core.digest(root['root_context']);roots.append(root)
    sources=[dict(warm_trace=[],packets={'off':{'raw_action_history':warm.tolist()}}) for _ in roots]
    store=FastFixtureStore(tmp_path/'joint');host.life.seam.need(len(roots)==2,'fixture')
    def create(slot,root,policy):
        episode=f'matched_slot_{slot:02}'
        def record(origin,resource,detail):return store.event(origin,'record',resource,store.index,detail)
        def meter(name,region):
            detail=dict(episode_id=episode,region=region,units=1);token=store.index
            store.event('physical','intent','native_meter/'+name,token,detail)
            store.event('physical','return','native_meter/'+name,token,detail)
        def observe(phase,step):
            image=np.full((224,224,3),root*35+(0 if phase=='warmup' else step),np.uint8)
            buf=io.BytesIO();Image.fromarray(image).save(buf,format='PNG');raw=buf.getvalue()
            name=f's{slot}_{phase}_{step}.png';store.write(name,raw,'physical_png')
            frame=dict(file=name,sha256=host.storage.base.fsha(raw),pixel_sha256=host.storage.base.fsha(image.tobytes()),shape=[224,224,3],dtype='uint8')
            record('physical','native_frame',dict(frame=frame))
            receipt=dict(phase=phase,step=step,path=str(store.output/name),file_sha256=frame['sha256'],
                         pixel_sha256=frame['pixel_sha256'],snapshot={'fixture':True},state=[float(step)])
            record('physical','native_fresh_observation',dict(episode_id=episode,receipt=receipt))
            if phase=='warmup' and slot in (0,2):
                sources[root]['warm_trace'].append(dict(receipt=dict(receipt,overlay_index=0)))
            return image,frame,raw
        meter('constructor','setup');meter('reset','setup');meter('reset','setup')
        observe('warmup',0)
        for i in range(1,11):
            meter('public_step','warmup');observe('warmup',i)
            record('physical','native_warmup_return',dict(terminated=False,truncated=False))
        meter('goal_bind','goal_bind');observe('plan',0)
        record('physical','native_initial_native_evaluation',dict(success=False))
        record('physical','native_matched_root_accepted',dict(root_sha256=roots[root]['binding']['root_sha256']))
        class Session:
            episode_id=episode
            def packet(self,start,raw):
                for i,command in enumerate(raw):
                    step=start+i+1;bits=command.tobytes().hex()
                    record('physical','native_authorized_dispatch',dict(episode_id=episode,step=step,raw_bits=bits))
                    meter('public_step','prefix' if start==0 else 'suffix')
                    self.image,self.frame,self.png=observe('plan',step)
                    record('physical','native_control_return',dict(episode_id=episode,step=step,raw_bits=bits,terminated=False,truncated=False))
                return dict(raw=raw.copy(),rgb_bytes=self.png,end=start+5,stopped=False,episode_id=episode)
            def encode(self,boundary,rgb):
                prep=encoder.original.pixels.preprocess_pixels(self.image)
                hidden=np.zeros((1,257,192),np.float32);out=np.full((1,192),boundary/100,np.float32)
                values=dict(pixels=self.image,preprocessed=prep,encoder_input=prep.reshape(1,3,224,224),
                    encoder_hidden=hidden,projector_input=hidden[:,0,:],projector_output=out,latent=out.reshape(192))
                refs={role:store.array('encoding',role,value) for role,value in values.items()}
                for kind in ('encoder','projector'):
                    token=store.index;store.event('encoding','intent',kind,token,dict(boundary=boundary))
                    store.event('encoding','return',kind,token,dict(boundary=boundary))
                record('encoding','encoding_complete',dict(boundary=boundary,frame=self.frame,arrays=refs))
                return dict(latent=out.reshape(192),input_sha256=host.storage.base.fsha(rgb),end=boundary,episode_id=episode)
        return Session(),warm[-5:].copy(),lambda:meter('close','cleanup')
    def backend(task,sink):return host.prior.RecordingBackend(BufferedRAM(task,sink),task,sink,lambda:None)
    rows=host.life.execute(roots,store,create,backend,permit_ram=True)['rows']
    assert all(r['status']=='callback_feedback_complete_not_certified' for r in rows),rows
    prepared=dict(precision={'fixture':True},cuda_allocated_bytes=100,cuda_reserved_bytes=100)
    original=dict(tasks=roots,expected_model_hash='RAM_FIXTURE')
    report=dict(schema=host.NAME+'.report',status='four_complete_candidate',prepared='fixture',prepare_sha256='fixture',
        coordination='FAKE_FIXTURE_ONLY',rows=rows,errors=[],model_before={'original_state_dict':'RAM_FIXTURE'},
        model_after={'original_state_dict':'RAM_FIXTURE'},classes_before={},classes_after={},precision=[prepared['precision']]*2,
        gpu=dict(device='cuda:0',allocation_sum=4.0,visible_nvidia_nodes=['FAKE_FIXTURE']),peak_bytes=[0,0],
        sources_verified=True,hooks_restored=True,elapsed_seconds=1,journal_sha256=host.source.host.sha(store.output/'events.jsonl'),
        journal_tail=store.previous,event_count=store.index,category_bytes=store.used,pending=[],known_roots=2,fresh_roots=0,
        training=False,equal_FLOPs=False,scientific_pass=False)
    report_sha=host.write_new(store.output/'report.json',report)
    completion_sha=host.write_new(store.output/'completion.json',dict(schema=host.NAME+'.completion',report_sha256=report_sha,
        complete=True,boundary='after_report_fsync',scientific_pass=False))
    return store.output,report_sha,completion_sha,prepared,original,dict(roots=sources)


def test_complete_four_slot_joint_saved_and_corrupted_frame(tmp_path,monkeypatch):
    output,report_sha,completion_sha,prepared,original,data=make_fixture(tmp_path)
    monkeypatch.setattr(host,'verify',lambda *args:prepared)
    monkeypatch.setattr(host.source.host,'verify_prepare',lambda *args:original)
    monkeypatch.setattr(host.source.physical,'source_material',lambda:data)
    receipt=saved.consume(output,report_sha,completion_sha)
    assert len(receipt['slots'])==4 and all(x['physical_steps']==20 and x['encoding_boundaries']==4 for x in receipt['slots'])
    assert not receipt['scientific_pass']
    journal=host.storage.inspect_journal(output/'events.jsonl',host.source.host.sha(output/'events.jsonl'))
    events=journal['events'];wrapped,positions=saved._events_with_positions(events,0)
    row=host.source.host.design.read_pinned(output/'report.json',report_sha)['rows'][0]
    def slot(ev=events,wr=wrapped,pos=positions):
        return saved._slot(original['tasks'][0],data['roots'][0],row,wr,pos,ev,output)
    # Keep the original test ledger untouched while attacking each independent semantic check.
    changed=list(events);i=next(i for i,e in enumerate(events) if e['resource']=='native_authorized_dispatch')
    changed[i]=deepcopy(changed[i]);changed[i]['detail']['raw_bits']='f'*16
    with pytest.raises(ValueError,match='public dispatch'):slot(ev=changed)
    changed=list(events);i=next(i for i,e in enumerate(events) if e['origin']=='encoding' and e['resource']=='encoder' and e['kind']=='return')
    changed.pop(i)
    with pytest.raises(ValueError,match='encoder hook lifecycle'):slot(ev=changed)
    altered=list(wrapped);i=next(i for i,e in enumerate(wrapped) if e['payload']['kind']=='actual_context')
    altered[i]=deepcopy(altered[i]);altered[i]['payload']['context_sha256']='0'*64
    with pytest.raises(ValueError):slot(wr=altered)
    altered=list(wrapped);i=next(i for i,e in enumerate(wrapped) if e['call'] is not None and e['payload']['kind']=='io_return')
    altered.pop(i)
    with pytest.raises((ValueError,AssertionError,KeyError,IndexError)):saved._native_calls(altered,next(
        e['payload']['result'] for e in wrapped if e['payload']['kind']=='slot_final'))
    journal_path=output/'events.jsonl';journal_bytes=journal_path.read_bytes();broken=bytearray(journal_bytes);broken[100]^=1
    journal_path.write_bytes(broken)
    with pytest.raises(ValueError):saved.consume(output,report_sha,completion_sha)
    journal_path.write_bytes(journal_bytes)
    frame=output/'s0_plan_5.png';blob=bytearray(frame.read_bytes());blob[-1]^=1;frame.write_bytes(blob)
    with pytest.raises(ValueError):saved.consume(output,report_sha,completion_sha)
