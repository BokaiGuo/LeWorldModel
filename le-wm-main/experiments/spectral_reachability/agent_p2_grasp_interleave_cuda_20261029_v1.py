"""Fixed-input original/HOST GRASP CUDA diagnostic; never run without MAIN authorization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.spectral_reachability import agent_p2_intersection_host_cuda_20261029_v1 as host
from experiments.spectral_reachability import agent_p2_grasp_interleave_audit_20261029_v1 as audit

NAME = 'agent_p2_grasp_interleave_cuda_20261029_v1'
AUTH = 'MAIN_P2_GRASP_INTERLEAVE_CUDA_ONLY'
ROOT = host.ROOT
CAP = 64 * 1024**2
FLOOR = 200 * 1024**2
need = host.need


def own():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def build(protocol, digest):
    audit.inspect_sources()
    p = host.common.build(protocol, digest)
    host.common.join_tasks(p['slots'])
    p['bindings_sha256'].update({str(f):host.sha(f) for f in own()})
    for path, sha in audit.SOURCES.values():
        p['bindings_sha256'][str(ROOT/path)] = sha
    host.old.parent.check_files(p)
    return dict(schema=NAME+'.prepare', protocol=str(Path(protocol).resolve()), protocol_sha256=digest,
                input=p, root=0, fixed_x='float64 zeros(30)', fixed_v='original linear interpolation',
                order=['original','host'], comparisons='exact dtype shape stride raw bytes',
                prediction_calls=6, backward_calls=2, scientific_admission=False)


def verify(path, digest):
    path=Path(path);need(host.sha(path)==digest and path.stat().st_size<=8*1024**2,'prepare SHA/size')
    raw=path.read_bytes();p=json.loads(raw);need(host.box.raw(p)==raw,'canonical prepare')
    host.equal(p,build(p['protocol'],p['protocol_sha256']),'prepare rebuilt');return p


def write(path, value):
    path=Path(path).absolute();data=host.box.raw(value)
    need(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(data)<=CAP,'exclusive bounded artifact')
    with path.open('xb') as f:
        need(f.write(data)==len(data),'complete write');f.flush();os.fsync(f.fileno())
    host.old.fsync_dir(path.parent)
    return dict(path=str(path),sha256=host.sha(path))


class Store:
    def __init__(self,path):
        import shutil
        self.path=Path(path).absolute();need(self.path.name.startswith(NAME+'_') and not self.path.exists() and
            not self.path.is_symlink() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'exclusive output and floor')
        self.path.mkdir();self.events=[];self.start=time.monotonic();self.event_bytes=0
        with (self.path/'events.jsonl').open('xb') as f:f.flush();os.fsync(f.fileno())
    def check(self):need(time.monotonic()-self.start<900,'wall900')
    def event(self,e):
        import shutil
        self.check();line=host.box.raw(e)+b'\n'
        need(self.event_bytes+len(line)<CAP//2 and
             sum(p.stat().st_size for p in self.path.iterdir())+len(line)<=CAP-1024**2 and
             shutil.disk_usage(self.path).free>=FLOOR+len(line),'durable event cap/final reserve')
        with (self.path/'events.jsonl').open('ab') as f:
            need(f.write(line)==len(line),'complete event');f.flush();os.fsync(f.fileno())
        self.event_bytes+=len(line);self.events.append(e)
    def put(self,name,value):
        import shutil
        b=host.box.raw(value)
        need(Path(name).name==name and len(b)<=CAP//2 and
             sum(p.stat().st_size for p in self.path.iterdir())+len(b)<=CAP and
             shutil.disk_usage(self.path).free>=FLOOR+len(b),'bounded own metadata')
        path=self.path/name
        with path.open('xb') as f:
            need(f.write(b)==len(b),'complete metadata');f.flush();os.fsync(f.fileno())
        host.old.fsync_dir(self.path)


def capture(tensor):
    a=tensor.detach().cpu().numpy()
    need(str(a.dtype) in ('float32','float64'),'float ABI')
    return dict(dtype=str(a.dtype),shape=list(a.shape),stride=list(tensor.stride()),
                raw_hex=a.tobytes(order='C').hex())


def check_blob(v):
    need(set(v)=={'dtype','shape','stride','raw_hex'} and v['dtype'] in ('float32','float64')
         and isinstance(v['shape'],list) and isinstance(v['stride'],list)
         and len(v['shape'])==len(v['stride']) and all(type(n) is int and 0<=n<=576 for n in v['shape'])
         and all(type(n) is int and n>=0 for n in v['stride']),'tensor descriptor ABI')
    b=bytes.fromhex(v['raw_hex']);size=4 if v['dtype']=='float32' else 8
    import math
    need(len(b)==math.prod(v['shape'])*size,'tensor exact byte length');return b


def compare(a,b):
    need(set(a)==set(b),'matching comparison fields')
    result={}
    for key in sorted(a):
        left,right=a[key],b[key];check_blob(left);check_blob(right)
        result[key]=dict(equal=left==right,bytes_equal=left['raw_hex']==right['raw_hex'],
                         metadata_equal={k:left[k] for k in ('dtype','shape','stride')}==
                                        {k:right[k] for k in ('dtype','shape','stride')})
    return result


def packet_view(events):
    """Keep actual source events, including the full native module payloads."""
    wanted={'forward_attempt','module_attempt','module_return','predict_return','forward_return'}
    selected=[e for e in events if e.get('kind') in wanted]
    packets=[];current=None
    for e in selected:
        kind=e['kind']
        if kind=='forward_attempt':
            need(current is None and e['index']==len(packets),'packet attempt order')
            current=dict(attempt=e,modules=[],prediction=None,returned=None)
        elif kind=='module_attempt':
            need(current is not None and len(current['modules'])<3,'module attempt order')
            current['modules'].append(dict(attempt=e,returned=None))
        elif kind=='module_return':
            need(current is not None and current['modules'] and current['modules'][-1]['returned'] is None,
                 'module return follows attempt')
            current['modules'][-1]['returned']=e
        elif kind=='predict_return':
            need(current is not None and len(current['modules'])==3 and
                 all(x['returned'] is not None for x in current['modules']),'complete module sequence')
            current['prediction']=e
        else:
            need(current is not None and current['prediction'] is not None,'packet complete')
            current['returned']=e;packets.append(current);current=None
    need(current is None and len(packets)==3,'three complete packets')
    for p in packets:
        need([x['attempt']['resource'] for x in p['modules']]==['action_encoder','predictor','pred_proj']
             and all(x['returned']['resource']==x['attempt']['resource'] for x in p['modules']),'module ABI order')
    return packets


def validate_packed(value):
    if isinstance(value,dict):
        if set(value)=={'shape','dtype','hex','sha256'}:
            host.grasp.unpack(value)
        else:
            for v in value.values():validate_packed(v)
    elif isinstance(value,list):
        for v in value:validate_packed(v)


def replay_host(task, side):
    """Rebuild paid HOST packet operations against recorded native tensors."""
    import numpy as np
    def array(d):
        raw=check_blob(d);return np.frombuffer(raw,dtype=d['dtype']).reshape(d['shape']).copy()
    cursor=host.Cursor(side['events']);tape=host.SavedTape(0,'grasp',cursor)
    need(tape.admit('joint:0'),'saved joint admission')
    b=host.SavedGrasp(task,cursor,tape,{'neural':side['neural']})
    z=array(side['inputs']['z']);v=array(side['inputs']['v']);x=array(side['inputs']['x'])
    need(np.array_equal(x,np.zeros(30,np.float64)),'fixed zero logits')
    h=np.asarray(task['root_context']['action_history'],np.float32)[None]
    goal=np.asarray(task['goal'],np.float32)[None]
    need(np.array_equal(z,np.asarray(task['root_context']['latent_history'],np.float32)[None]) and
         np.array_equal(array(side['inputs']['goal']),goal),'bound task root and goal')
    fraction=np.asarray([1/3,2/3],np.float32).reshape(1,2,1)
    need(np.array_equal(v,z[:,-1,None]+fraction*(goal[:,None]-z[:,-1,None])),
         'fixed source initial interpolation')
    actions=host.grasp.chart(x).astype(np.float32).reshape(3,5,2)
    for k in range(3):
        z,h=b.step(z,h,actions[k],'lifted/'+str(k))
        if k<2:z=np.concatenate((z[:,:-1],v[:,k:k+1]),axis=1)
    tape.call('score',lambda:None,trajectories=1,scope='lifted_loss')
    def backward():
        e=cursor.get('chart_backward');validate_packed(e)
    tape.call('backward',backward);b.check();cursor.complete()
    need(b.n==3 and b.module_index==len(side['neural'])==9,'all HOST packets/modules consumed')
    host.equal(tape.ledger.summary(),side['accounting'],'full reconstructed HOST ledger')


def execute_side(t,model,device,task,store,kind):
    events=[]
    layouts=[]
    def emit(e):events.append(e);store.event(dict(side=kind,event=e))
    row=dict(neural=[])
    class Layout:
        def step(self,z,h,a,label):
            def describe(x):return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),device=str(x.device))
            item=dict(label=label,z=describe(z),h=describe(h),action=describe(a))
            result=super().step(z,h,a,label)
            item.update(z_after=describe(result[0]),h_after=describe(result[1]));layouts.append(item)
            return result
    if kind=='original':
        class Original(Layout,host.grasp.Native):pass
        b=Original(model,device,task,emit,store)
    else:
        def clock():t.cuda.synchronize(device);return time.monotonic()
        tape=host.Tape(0,'grasp',emit,clock);need(tape.admit('joint:0'),'actual joint unit admission')
        class Host(Layout,host.GraspBackend):pass
        b=Host(model,device,task,emit,store,tape,row)
    z,_=b.root();goal=b.tensor(task['goal'])[None]
    frac=t.linspace(0,1,4,device=device)[1:-1][None,:,None]
    v=(z[:,-1,None]+frac*(goal[:,None]-z[:,-1,None])).detach().requires_grad_(True)
    x=t.zeros(30,device=device,dtype=t.float64,requires_grad=True)
    inputs=dict(x=capture(x),v=capture(v),z=capture(z),goal=capture(goal))
    with b.observe():
        loss,dyn,dense=b.soft(v,x)
        if kind=='host':tape.call('backward',lambda:loss.backward())
        else:loss.backward()
        b.check()
    need(x.grad is not None and v.grad is not None,'both actual gradients')
    out=dict(inputs=inputs,loss=capture(loss),dyn=capture(dyn),dense=capture(dense),gx=capture(x.grad),gv=capture(v.grad),events=events,packet_layouts=layouts)
    need(b.n==3,'three actual packet calls')
    if kind=='host':out['accounting']=tape.ledger.summary();out['neural']=row['neural']
    return out


def run(path,digest,output,authorization):
    need(authorization==AUTH,'MAIN authorization required before I/O')
    p=verify(path,digest);store=Store(output);model=before=after=None;errors=[];sides={};precision0=precision1=None
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.gpu_preflight import require_cuda_allocation
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,before=host.old.loader.loaded_model(p['input'],torch,device,store,digest)
        task=host.common.join_tasks(p['input']['slots'])[0]
        with torch.enable_grad(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            precision0=host.grasp.precision(torch)
            for kind in ('original','host'):
                sides[kind]=execute_side(torch,model,device,task,store,kind)
                store.put(kind+'.json',sides[kind])
            precision1=host.grasp.precision(torch)
    except BaseException as e:errors.append(dict(type=type(e).__name__,message=str(e)))
    finally:
        if model is not None:
            try:
                host.old.loader.enc.frozen(model);after=host.old.loader.enc.tensor_hash(model)
                need(before==after==p['input']['expected_model_hash'],'model unchanged')
                host.old.parent.check_files(p['input'])
            except BaseException as e:errors.append(dict(type=type(e).__name__,message=str(e)))
        report=dict(schema=NAME+'.report',prepare=str(Path(path).resolve()),prepare_sha256=digest,
            status='complete' if not errors and len(sides)==2 else 'failed_partial',errors=errors,
            model_before=before,model_after=after,precision_before=precision0,precision_after=precision1,
            artifacts={x.name:host.sha(x) for x in store.path.iterdir()},prediction_calls=6 if len(sides)==2 else None,
            backward_calls=2 if len(sides)==2 else None,scientific_admission=False)
        store.put('report.json',report)
    return dict(report=str(store.path/'report.json'),sha256=host.sha(store.path/'report.json'),errors=errors)


def consume(output,digest):
    output=Path(output);need(host.sha(output/'report.json')==digest,'report SHA')
    r=json.loads((output/'report.json').read_bytes());need(r['schema']==NAME+'.report' and r['status']=='complete' and not r['errors'],'complete only')
    p=verify(r['prepare'],r['prepare_sha256']);need(r['model_before']==r['model_after']==p['input']['expected_model_hash'],'model identity')
    guard=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,MATH=True,FLASH=False,MEM_EFFICIENT=False,CUDNN=False)
    need(r['precision_before']==r['precision_after']==guard and r['prediction_calls']==6 and r['backward_calls']==2,
         'precision and fixed actual calls')
    need(set(r['artifacts'])=={'loader.json','events.jsonl','original.json','host.json'} and
         {f.name for f in output.iterdir()}==set(r['artifacts'])|{'report.json'},'artifact census')
    for name,h in r['artifacts'].items():need(host.sha(output/name)==h,'artifact SHA')
    loader=json.loads((output/'loader.json').read_bytes());host.old.loader.validate_loader(loader,p['input'],r['prepare_sha256'])
    sides={k:json.loads((output/(k+'.json')).read_bytes()) for k in ('original','host')}
    events=[]
    for line in (output/'events.jsonl').read_bytes().splitlines(keepends=True):
        need(line.endswith(b'\n'),'complete durable event');e=json.loads(line)
        need(host.box.raw(e)+b'\n'==line,'canonical event');events.append(e)
    load_events=[]
    for f in loader['files']:
        load_events.extend([dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']),
                            dict(kind='consumed_buffer_frozen',receipt=f)])
    need(events==load_events+[dict(side=k,event=e) for k in ('original','host') for e in sides[k]['events']],
         'complete loader and ordered side event stream')
    for k in sides:
        need([e['event'] for e in events if e.get('side')==k]==sides[k]['events'],'complete side event subsequence')
        ev=sides[k]['events'];need(sum(e.get('kind')=='forward_attempt' for e in ev)==3 and
            sum(e.get('kind')=='forward_return' for e in ev)==3,'three full packet events')
        need(sum(e.get('kind')=='module_attempt' for e in ev)==9 and sum(e.get('kind')=='module_return' for e in ev)==9,'nine module pairs')
        need(len(sides[k]['packet_layouts'])==3 and [x['label'] for x in sides[k]['packet_layouts']]==['lifted/0','lifted/1','lifted/2'],'three actual packet layouts')
    need(sides['host']['accounting']['used']['predict_API']==3 and
         sides['host']['accounting']['used']['backward']==1,'HOST actual ledger counts')
    replay_host(host.common.join_tasks(p['input']['slots'])[0],sides['host'])
    a,b=sides['original'],sides['host'];metrics=compare({k:a[k] for k in ('loss','dyn','dense','gx','gv')},{k:b[k] for k in ('loss','dyn','dense','gx','gv')})
    inputs=compare(a['inputs'],b['inputs'])
    ap,bp=packet_view(a['events']),packet_view(b['events'])
    validate_packed(ap);validate_packed(bp)
    packets=[]
    for i,(x,y) in enumerate(zip(ap,bp,strict=True)):
        checks=dict(attempt=x['attempt']==y['attempt'],prediction=x['prediction']==y['prediction'],
                    returned=x['returned']==y['returned'],
                    packet_layout=a['packet_layouts'][i]==b['packet_layouts'][i],
                    modules=[dict(attempt=u['attempt']==v['attempt'],returned=u['returned']==v['returned'])
                             for u,v in zip(x['modules'],y['modules'],strict=True)])
        packets.append(dict(index=i,checks=checks))
    return dict(schema=NAME+'.consume',report_sha256=digest,inputs=inputs,metrics=metrics,
        packets=packets,all_exact=all(v['equal'] for v in list(inputs.values())+list(metrics.values())) and
          all(z['checks']['attempt'] and z['checks']['prediction'] and z['checks']['returned'] and z['checks']['packet_layout'] and
              all(m['attempt'] and m['returned'] for m in z['checks']['modules']) for z in packets),
        packet_state_action_strides_recorded=True,module_io_strides_recorded=False,independent_gradient_replay=False,scientific_admission=False)


def main():
    q=argparse.ArgumentParser();q.add_argument('command',choices=('prepare','verify','run','consume'))
    for key in ('protocol','protocol-sha','prepared','prepare-sha','output','authorization','report-sha','receipt'):q.add_argument('--'+key)
    a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.protocol,a.protocol_sha))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v={'verified':True}
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:
        v=consume(a.output,a.report_sha)
        if a.receipt:v=write(a.receipt,v)
    print(json.dumps(v,sort_keys=True))


if __name__=='__main__':main()
