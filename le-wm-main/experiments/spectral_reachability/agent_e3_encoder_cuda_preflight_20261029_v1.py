"""Two fixed synthetic RGB frames, HOST CUDA encoder only; never a task result."""
import argparse
import base64
import hashlib
import io
import os
from pathlib import Path
import time
import traceback
import numpy as np

from experiments.spectral_reachability import agent_e3_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_native_capture_transport_20261029_v1 as capture

NAME='agent_e3_encoder_cuda_preflight_20261029_v1'
AUTH='MAIN_E3_ENCODER_IDENTITY_FIXTURE_CUDA_ONLY'
need=host.need


def sources(binding):
    result=dict(binding['prepared']['bindings_sha256']);host.launch.host.source.merge(result,binding['additional_sources'])
    required=[str(Path(m.__file__).resolve()) for m in (host,host.capacity,host.identity,capture,host.identity.encoder)]
    required.append(str(Path(__file__).resolve()))
    host.launch.episode.old.guard.validate_sources(result,required)
    return result,required


def record(store,name,detail):store.event('physical','record',name,store.index,detail)


def snapshot(store,stage,value):
    raw=host.launch.host.raw(value);need(0<len(raw)<=16*1024**2,'bounded guard snapshot')
    sha=hashlib.sha256(raw).hexdigest();chunks=[raw[i:i+6000] for i in range(0,len(raw),6000)]
    for i,chunk in enumerate(chunks):
        record(store,'preflight_guard_chunk',dict(stage=stage,index=i,total=len(chunks),sha256=sha,
               data=base64.b64encode(chunk).decode('ascii')))
    record(store,'preflight_guard_'+stage,dict(sha256=sha,total=len(chunks),bytes=len(raw)))


def fixed_frame(boundary):
    need(type(boundary) is int and boundary in (10,15),'fixed boundary')
    return np.full((224,224,3),0 if boundary==10 else 127,dtype=np.uint8)


def run(authority_path,authority_sha,output,authorization):
    need(authorization==AUTH,'explicit preflight-only HOST CUDA authorization')
    started=time.monotonic();store=host.launch.pub.Store(output,started=started)
    record(store,'preflight_started',dict(schema=NAME,scientific_admission=False,physics_calls=0))
    error=None
    try:
        binding=host.authority(authority_path,authority_sha);bindings,required=sources(binding)
        record(store,'preflight_authority',dict(path=authority_path,sha256=authority_sha,cell=binding['cell']))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        gpu=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        record(store,'preflight_cuda',dict(preflight=gpu))
        loaded,proof=host.capacity.load_bound(binding['prepared'],torch,device,store,binding['prepare_sha'])
        record(store,'preflight_capacity',proof)
        guard=host.launch.episode.old.guard
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
            before=guard.snapshot(loaded.model,torch,device,bindings,required)
            guard.validate_pair(before,before,binding['prepared']['expected_model_hash']);snapshot(store,'before',before)
            primary=None
            try:
                bridge=capture.CaptureStore(store)
                for boundary in (10,15):
                    name=f'preflight_{boundary}.png';bridge.frame(name,fixed_frame(boundary));frame=bridge.frames[name]
                    record(store,'preflight_encode_intent',dict(boundary=boundary,frame=frame))
                    latent,_=loaded.encode(torch,frame,boundary,lambda:store.check('encoding_arrays',0))
                    need(np.asarray(latent).shape==(192,) and np.isfinite(latent).all(),'finite latent')
                    record(store,'preflight_encode_return',dict(boundary=boundary))
            except BaseException as exc:
                primary=exc;raise
            finally:
                try:
                    after=guard.snapshot(loaded.model,torch,device,bindings,required)
                    guard.validate_pair(before,after,binding['prepared']['expected_model_hash']);snapshot(store,'after',after)
                except BaseException as exc:
                    if primary is None:raise
                    primary.add_note('post-encoder guard failure: '+repr(exc))
        sources(binding)
        need(host.launch.host.sha(authority_path)==authority_sha,'unchanged authority')
        record(store,'preflight_return',dict(boundaries=[10,15],scientific_admission=False,physics_calls=0))
    except BaseException as exc:
        error=exc
        try:record(store,'preflight_failure',dict(type=type(exc).__name__,message=str(exc)[:2048],
                                                  traceback=traceback.format_exc()[-8192:]))
        except BaseException as secondary:exc.add_note('preflight failure journal: '+repr(secondary))
    seal=store.publish('partial' if error else 'completed')
    return dict(publication=seal,complete=seal['complete'] and error is None,
                error_type=None if error is None else type(error).__name__,scientific_admission=False)


def read_snapshot(events,stage):
    end=[e for e in events if e['resource']=='preflight_guard_'+stage and e['origin']=='physical']
    chunks=[e for e in events if e['resource']=='preflight_guard_chunk' and e['origin']=='physical'
            and e['detail'].get('stage')==stage]
    need(len(end)==1 and bool(chunks),'one complete guard snapshot')
    d=end[0]['detail'];need(set(d)=={'sha256','total','bytes'} and type(d['total']) is int
         and d['total']==len(chunks) and type(d['bytes']) is int and 0<d['bytes']<=16*1024**2,'guard bound')
    parts=[];last=-1
    for i,e in enumerate(chunks):
        x=e['detail'];need(set(x)=={'stage','index','total','sha256','data'} and x['index']==i
            and x['total']==d['total'] and x['sha256']==d['sha256'] and last<e['index']<end[0]['index']
            and type(x['data']) is str and len(x['data'])<=8000,'guard chunk chronology')
        part=base64.b64decode(x['data'],validate=True)
        need(len(part)==6000 if i<len(chunks)-1 else 0<len(part)<=6000,'guard chunk size')
        parts.append(part);last=e['index']
    raw=b''.join(parts);need(len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'],'guard hash')
    return host.launch.pub.base.decode(raw),end[0]['index']


def consume(output,report_sha,completion_sha,authority_path,authority_sha):
    transport=host.launch.pub.consume(output,report_sha,completion_sha)
    binding=host.authority(authority_path,authority_sha);bindings,required=sources(binding)
    p=Path(output);report=host.launch.pub.base.read_bytes(p/'report.json',host.launch.pub.FINAL_CAP)
    journal=host.launch.pub.base.read_bytes(p/'events.jsonl',host.launch.pub.base.CAP)
    need(host.launch.pub.base.fsha(report)==report_sha and
         host.launch.pub.base.fsha(journal)==host.launch.pub.base.decode(report)['journal_sha256'],'stable journal')
    events=[host.launch.pub.base.decode(x) for x in journal.splitlines()]
    def select(name):return [e for e in events if e['origin']=='physical' and e['resource']==name]
    starts=select('preflight_started');auth=select('preflight_authority')
    need(len(starts)==1 and starts[0]['index']==0 and starts[0]['detail']==dict(schema=NAME,scientific_admission=False,physics_calls=0),
         'preflight schema')
    if not transport['complete']:
        failure=select('preflight_failure');need(len(failure)==1,'partial failure retained')
        return dict(status='partial',error=failure[0]['detail'],scientific_admission=False,physics_calls=0,
                    encoder_calls_certified=False,full_saved_strict=False)
    need(len(auth)==1 and auth[0]['detail']==dict(path=authority_path,sha256=authority_sha,cell=binding['cell']),
         'actual authority join')
    identity=host.identity.consume(output,report_sha,completion_sha,binding['prepared'],binding['prepare_sha'],bindings)
    need([r['boundary'] for r in identity['encodes']]==[10,15] and
         all(r['outcome']=='return' for r in identity['encodes']) and identity['pending'] is None,'two bound calls')
    encoding=[e for e in events if e['origin']=='encoding']
    expected=[(boundary,role,kind) for boundary in (10,15) for role,kind in
              (('encoder','intent'),('encoder','return'),('projector','intent'),('projector','return'),('encoding_complete','record'))]
    need([(e['detail'].get('boundary'),e['resource'],e['kind']) for e in encoding]==expected,
         'all-and-only two encoder/projector calls')
    before,bi=read_snapshot(events,'before');after,ai=read_snapshot(events,'after')
    guard=host.launch.episode.old.guard;guard.validate_pair(before,after,binding['prepared']['expected_model_hash'])
    for cls in before['classes']:
        need(bindings.get(cls['path'])==cls['sha256'],'guard class source pinned')
    bind=[e for e in events if e['resource']=='encoder_identity_bind' and e['origin']=='physical']
    proofs=select('preflight_capacity');cuda=select('preflight_cuda')
    need(len(bind)==len(proofs)==len(cuda)==1 and auth[0]['index']<cuda[0]['index']<bind[0]['index']<proofs[0]['index']<bi,
         'cuda, loader, capacity and guard chronology')
    host.verify_capacity(bind[0],proofs[0],journal)
    results=[];last=bi
    for boundary in (10,15):
        intents=[e for e in select('preflight_encode_intent') if e['detail'].get('boundary')==boundary]
        returns=[e for e in select('preflight_encode_return') if e['detail'].get('boundary')==boundary]
        done=[e for e in events if e['origin']=='encoding' and e['resource']=='encoding_complete'
              and e['detail'].get('boundary')==boundary]
        need(len(intents)==len(returns)==len(done)==1 and last<intents[0]['index']<done[0]['index']<returns[0]['index']<ai,
             'preflight encoder chronology')
        frame=intents[0]['detail']['frame'];need(done[0]['detail']['frame']==frame,'same frame')
        from PIL import Image
        png=host.launch.pub.base.read_bytes(p/frame['file'],host.launch.pub.PNG_CAP)
        with Image.open(io.BytesIO(png)) as img: pixels=np.asarray(img)
        need(pixels.tobytes()==fixed_frame(boundary).tobytes(),'fixed synthetic RGB')
        arrays={}
        for role,desc in done[0]['detail']['arrays'].items():
            b=host.launch.pub.base.read_bytes(p/desc['file'],2*1024**2);need(hashlib.sha256(b).hexdigest()==desc['sha256'],'array SHA')
            s=io.BytesIO(b);arrays[role]=np.load(s,allow_pickle=False);need(s.tell()==len(b),'array EOF')
        host.identity.encoder.verify_arrays(png,arrays);results.append(boundary);last=returns[0]['index']
    end=select('preflight_return');need(len(end)==1 and ai<end[0]['index']==len(events)-1 and
        end[0]['detail']==dict(boundaries=[10,15],scientific_admission=False,physics_calls=0),'terminal preflight')
    return dict(status='two_encoder_io_verified',boundaries=results,scientific_admission=False,physics_calls=0,
                neural_parity_recomputed=False,full_saved_strict=False,identity=identity)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('run','consume'):
        a=sub.add_parser(name)
        for opt in ('authority','authority-sha256','output'):a.add_argument('--'+opt,required=True)
        if name=='run':a.add_argument('--authorization',required=True)
        else:
            a.add_argument('--report-sha256',required=True);a.add_argument('--completion-sha256',required=True)
    x=p.parse_args()
    r=run(x.authority,x.authority_sha256,x.output,x.authorization) if x.command=='run' else consume(
        x.output,x.report_sha256,x.completion_sha256,x.authority,x.authority_sha256)
    print(host.launch.host.raw(r).decode())


if __name__=='__main__':main()
