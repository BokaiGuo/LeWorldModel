"""Synthetic transport and injected cleanup faults only; no neural CPU/GPU."""
import io
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import p3_matched_order_feedback_host_v2_codex_20261029 as host
from experiments.spectral_reachability import p3_matched_order_feedback_host_saved_v2_codex_20261029 as saved
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_v2_codex_20261029 as encoder
from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as native_signal


class Handle:
    def __init__(self,collection,item):self.collection=collection;self.item=item
    def remove(self):self.collection.remove(self.item)


class Module:
    def __init__(self,fail=False):
        self._forward_hooks={};self._forward_pre_hooks={};self._backward_hooks={};self.fail=fail
    def forward(self,*args):return args
    def register_forward_pre_hook(self,fn):
        key=len(self._forward_pre_hooks)+1;self._forward_pre_hooks[key]=fn
        class Hook:
            def remove(s):del self._forward_pre_hooks[key]
        return Hook()
    def register_forward_hook(self,fn):
        if self.fail:raise RuntimeError('injected second registration failure')
        key=len(self._forward_hooks)+1;self._forward_hooks[key]=fn
        class Hook:
            def remove(s):del self._forward_hooks[key]
        return Hook()


class Model:
    def __init__(self):
        for name in ('encoder','projector','action_encoder','predictor','pred_proj'):setattr(self,name,Module())
    def modules(self):return [getattr(self,n) for n in ('encoder','projector','action_encoder','predictor','pred_proj')]


def test_encoder_registration_failure_cleanup_and_signal_wrapper_exception():
    model=Model();snapshot=host.hook_snapshot(model);handles=[];model.encoder.fail=True
    with pytest.raises(RuntimeError):
        try:encoder.register_hooks(model,lambda n:None,lambda n:None,None,handles)
        finally:encoder.remove_hooks(handles)
    assert host.hooks_restored(snapshot)
    observer=object.__new__(native_signal.NativeSignal);observer.model=model
    with pytest.raises(RuntimeError):
        with observer.observe():raise RuntimeError('injected planning failure')
    assert host.hooks_restored(snapshot)


def test_partial_saved_consumer_keeps_unreferenced_array(tmp_path,monkeypatch):
    store=host.storage.Store(tmp_path/'partial')
    store.event('physical','record','matched_run_start',store.index,{'fixture':True})
    orphan=store.array('encoding','latent',np.zeros((192,),dtype=np.float32))
    rows=[dict(slot=s,root=r,policy=p,status='not_started',result_sha256=None,cleanup='not_required',error=None)
          for s,r,p in host.life.seam.ROSTER]
    report=dict(schema=host.NAME+'.report',status='partial_or_failed',prepared='fixture',prepare_sha256='fixture',
        rows=rows,errors=[{'stage':'injected'}],scientific_pass=False,journal_sha256=host.source.host.sha(store.output/'events.jsonl'),
        journal_tail=store.previous,event_count=store.index,category_bytes=store.used,pending=[])
    report_sha=host.write_new(store.output/'report.json',report)
    completion_sha=host.write_new(store.output/'completion.json',dict(schema=host.NAME+'.completion',report_sha256=report_sha,
        complete=False,boundary='after_report_fsync',scientific_pass=False))
    monkeypatch.setattr(host,'verify',lambda *args:{})
    monkeypatch.setattr(host.source.host,'verify_prepare',lambda *args:{})
    monkeypatch.setattr(host.source.physical,'source_material',lambda:{})
    result=saved.consume(store.output,report_sha,completion_sha)
    assert result['status']=='partial_preserved_not_certified' and result['inventory']['orphans'][0]['file']==orphan['file']
    assert not result['full_saved_strict'] and not result['scientific_pass']
    (store.output/orphan['file']).write_bytes(b'tampered')
    with pytest.raises(ValueError,match='orphan byte accounting'):saved.consume(store.output,report_sha,completion_sha)


def test_fixed_four_slot_sixteen_real_shape_saved_encoder_payloads(tmp_path):
    store=host.storage.Store(tmp_path/'payloads'); frames=[]
    for slot in range(4):
        for boundary in (5,10,15,20):
            image=np.full((224,224,3),slot*32+boundary,dtype=np.uint8)
            buf=io.BytesIO();Image.fromarray(image).save(buf,format='PNG');png=buf.getvalue()
            name=f'frame_{slot}_{boundary}.png';store.write(name,png,'physical_png')
            frame=dict(file=name,sha256=host.storage.base.fsha(png),pixel_sha256=host.storage.base.fsha(image.tobytes()),
                       shape=[224,224,3],dtype='uint8')
            store.event('physical','record','native_frame',store.index,{'frame':frame})
            prep=encoder.original.pixels.preprocess_pixels(image)
            hidden=np.full((1,257,192),slot/10,np.float32);output=np.full((1,192),boundary/100,np.float32)
            arrays=dict(pixels=image,preprocessed=prep,encoder_input=prep.reshape(1,3,224,224),
                        encoder_hidden=hidden,projector_input=hidden[:,0,:],projector_output=output,latent=output.reshape(192))
            refs={role:store.array('encoding',role,value) for role,value in arrays.items()}
            store.event('encoding','record','encoding_complete',store.index,dict(boundary=boundary,frame=frame,arrays=refs))
            frames.append((frame,refs,boundary))
    host.write_new(store.output/'report.json',{'fixture':True})
    host.write_new(store.output/'completion.json',{'fixture':True})
    events=host.storage.inspect_journal(store.output/'events.jsonl',host.source.host.sha(store.output/'events.jsonl'))['events']
    inventory=saved._artifacts(store.output,events,{'category_bytes':store.used})
    assert inventory['referenced']==16*8 and not inventory['orphans']
    for frame,refs,boundary in frames:
        assert saved.encoder.inspect(store.output,{k:frame[k] for k in ('file','sha256','pixel_sha256')},refs,boundary)['saved_io_connected']
    wrong=dict(frames[0][0],pixel_sha256='0'*64)
    with pytest.raises(ValueError):saved.encoder.inspect(store.output,{k:wrong[k] for k in ('file','sha256','pixel_sha256')},frames[0][1],5)
