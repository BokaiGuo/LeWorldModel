"""Saved-only, actual-report-specific GRASP v2 interpolation repair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.spectral_reachability import agent_p2_intersection_host_cuda_20261029_v2 as host
from experiments.spectral_reachability import agent_p2_grasp_interleave_audit_20261029_v1 as audit

NAME = 'agent_p2_grasp_interleave_cuda_20261029_v2'
REPAIR = 'agent_p2_grasp_interleave_saved_repair_20261029_v1'
ACTUAL_REPORT = '491367c06c8665c76c4aa3051e4f22db2554d0e44ded068ddb7b3d00d0448f87'
AUTH = 'MAIN_P2_GRASP_INTERLEAVE_V2_CUDA_ONLY'
ROOT = host.ROOT
CAP = 64 * 1024**2
FLOOR = 200 * 1024**2
FINAL = 2 * 1024**2
PINS = {
    ROOT/'experiments/spectral_reachability/agent_p2_intersection_host_cuda_20261029_v2.py':
        '4744fe156c547d53d75750d485473e31e027d51eac70ad83029c5b9e1f48d1f2',
    ROOT/'experiments/spectral_reachability/agent_p2_grasp_interleave_audit_20261029_v1.py':
        'd18b1b0cb362f71409665e534af21204e4a8033f637da4224168fb2c8b3d50d8',
}
need = host.need


def own():
    return [ROOT/'experiments/spectral_reachability'/(NAME+'.py'), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def build(protocol, digest):
    audit.inspect_sources()
    for path,h in PINS.items():need(host.sha(path)==h,'fixed v2/audit source SHA')
    p = host.common.build(protocol, digest)
    host.common.join_tasks(p['slots'])
    p['bindings_sha256'].update({str(f):host.sha(f) for f in own()})
    for path, sha in audit.SOURCES.values():
        p['bindings_sha256'][str(ROOT/path)] = sha
    p['bindings_sha256'].update({str(k):v for k,v in PINS.items()})
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
             sum(p.stat().st_size for p in self.path.iterdir())+len(line)<=CAP-FINAL and
             shutil.disk_usage(self.path).free>=FLOOR+len(line),'durable event cap/final reserve')
        with (self.path/'events.jsonl').open('ab') as f:
            need(f.write(line)==len(line),'complete event');f.flush();os.fsync(f.fileno())
        self.event_bytes+=len(line);self.events.append(e)
    def put(self,name,value,*,final=False):
        import shutil
        b=host.box.raw(value)
        need(Path(name).name==name and len(b)<=(FINAL if final else CAP//2) and
             sum(p.stat().st_size for p in self.path.iterdir())+len(b)<=CAP-(0 if final else FINAL) and
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


def resource_prefix(events):
    """Only durable observations; missing dispatch visibility remains unknown."""
    out={k:dict(packet_attempt=0,packet_return=0,predict_attempt=0,predict_return=0,
                module_attempt=0,module_return=0,backward_attempt=0,backward_dispatch=0,backward_return=0,
                module_dispatch=None,module_capture=None) for k in ('original','host')}
    out['host']['module_dispatch']=0;out['host']['module_capture']=0
    out['host']['ledger_debit_axes']=host.box.zeros()
    for event in events:
        if 'side' in event:
            side=event['side'];need(side in out,'resource side');e=event['event'];kind=e.get('kind')
            mapping={'forward_attempt':'packet_attempt','forward_return':'packet_return',
                     'predict_attempt':'predict_attempt','predict_return':'predict_return',
                     'module_attempt':'module_attempt','module_return':'module_return'}
            if kind in mapping:out[side][mapping[kind]]+=1
            if side=='host' and kind=='budget' and e['event'].get('kind')=='debit':
                debit=e['event'];out['host']['ledger_debit_axes']=host.box.add(out['host']['ledger_debit_axes'],debit['delta'])
                if debit['operation_kind']=='module_dispatch':out['host']['module_dispatch']+=1
                if debit['operation_kind']=='module_capture':out['host']['module_capture']+=1
        elif 'resource_side' in event:
            side=event['resource_side'];kind=event['kind']
            need(side in out and kind in ('backward_attempt','backward_dispatch','backward_return'),'resource marker')
            out[side][kind]+=1
    return out


def fixed_initialization(z,v,goal):
    """Check the exact MAIN01 diagnostic reconstruction, not CUDA fraction identity."""
    import numpy as np
    fraction=np.asarray([0x3eaaaaab,0x3f2aaaaa],np.uint32).view(np.float32).reshape(1,2,1)
    need(np.array_equal(v,z[:,-1,None]+fraction*(goal[:,None]-z[:,-1,None])),
         'exact fixed MAIN01 reconstructed interpolation')


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
    # Fixed diagnostic bits, not proof of CUDA's unsaved linspace output.
    fixed_initialization(z,v,goal)
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


def consume(output,digest):
    need(digest==ACTUAL_REPORT,'only externally pinned actual MAIN01 report')
    repair_files=[Path(__file__).resolve(),ROOT/'tests'/('test_'+REPAIR+'.py'),ROOT/'innerdocs'/(REPAIR+'.md')]
    closure={str(p):host.sha(p) for p in repair_files}
    output=Path(output);need(host.sha(output/'report.json')==digest,'report SHA')
    r=json.loads((output/'report.json').read_bytes());need(r['schema']==NAME+'.report' and r['status']=='complete' and not r['errors'],'complete only')
    p=verify(r['prepare'],r['prepare_sha256']);need(r['model_before']==r['model_after']==p['input']['expected_model_hash'],'model identity')
    guard=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,MATH=True,FLASH=False,MEM_EFFICIENT=False,CUDNN=False)
    need(r['precision_before']==r['precision_after']==guard and r['prediction_calls']==6 and r['backward_calls']==2,
         'precision and fixed actual calls')
    need(type(r['wall_seconds']) in (int,float) and 0<=r['wall_seconds'] and
         r['wall_scope']=='cooperative Store start to before report serialization','cooperative wall scope')
    for peak in (r['peak_before'],r['peak_after']):
        need(isinstance(peak,dict) and set(peak)=={'allocated','reserved'} and
             all(type(peak[k]) is int and peak[k]>=0 for k in peak) and
             peak['allocated']<=4*1024**3 and peak['reserved']<=6*1024**3,'actual CUDA peak')
    need(all(r['peak_before'][k]<=r['peak_after'][k] for k in ('allocated','reserved')),'monotonic process peaks')
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
    need([e for e in events if 'resource_side' not in e]==load_events+
         [dict(side=k,event=e) for k in ('original','host') for e in sides[k]['events']],
         'complete loader and ordered side event stream')
    markers=[e for e in events if 'resource_side' in e]
    need(markers==[dict(resource_side=k,kind='backward_'+phase) for k in ('original','host')
                   for phase in ('attempt','dispatch','return')],'fixed backward markers')
    for k in ('original','host'):
        positions=[i for i,e in enumerate(events) if e.get('resource_side')==k]
        packets=[i for i,e in enumerate(events) if e.get('side')==k and e['event'].get('kind')=='forward_return']
        need(len(packets)==3 and len(positions)==3 and packets[-1]<positions[0]<positions[1]<positions[2],
             'backward after completed packet prefix')
        if k=='original':
            other=[i for i,e in enumerate(events) if e.get('side')=='host']
            need(other and positions[-1]<other[0],'original complete before HOST starts')
    host.equal(resource_prefix(events),r['resource_prefix'],'full durable resource prefix')
    for counts in r['resource_prefix'].values():
        need(all(counts[k]==v for k,v in dict(packet_attempt=3,packet_return=3,predict_attempt=3,predict_return=3,
             module_attempt=9,module_return=9,backward_attempt=1,backward_dispatch=1,backward_return=1).items()),'complete observed resources')
    for k in sides:
        need([e['event'] for e in events if e.get('side')==k]==sides[k]['events'],'complete side event subsequence')
        ev=sides[k]['events'];need(sum(e.get('kind')=='forward_attempt' for e in ev)==3 and
            sum(e.get('kind')=='forward_return' for e in ev)==3,'three full packet events')
        need(sum(e.get('kind')=='module_attempt' for e in ev)==9 and sum(e.get('kind')=='module_return' for e in ev)==9,'nine module pairs')
        need(len(sides[k]['packet_layouts'])==3 and [x['label'] for x in sides[k]['packet_layouts']]==['lifted/0','lifted/1','lifted/2'],'three actual packet layouts')
    need(sides['host']['accounting']['used']['predict_API']==3 and
         sides['host']['accounting']['used']['backward']==1,'HOST actual ledger counts')
    host.equal(r['resource_prefix']['host']['ledger_debit_axes'],sides['host']['accounting']['used'],
               'all HOST debit prefix axes equal saved ledger')
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
    need(all(host.sha(p)==h for p,h in closure.items()),'repair sources unchanged during consume')
    return dict(schema=REPAIR+'.consume',report_sha256=digest,inputs=inputs,metrics=metrics,repair_sources=closure,
        packets=packets,all_exact=all(v['equal'] for v in list(inputs.values())+list(metrics.values())) and
          all(z['checks']['attempt'] and z['checks']['prediction'] and z['checks']['returned'] and z['checks']['packet_layout'] and
              all(m['attempt'] and m['returned'] for m in z['checks']['modules']) for z in packets),
        packet_state_action_strides_recorded=True,module_io_strides_recorded=False,independent_gradient_replay=False,scientific_admission=False,
        initialization_fraction_capture='not_saved_by_producer',
        initialization_reconstruction='f32 bit pattern 3eaaaaab,3f2aaaaa reproduces all saved v exactly; not proof of actual CUDA linspace bits',
        producer_modified=False)


def main():
    q=argparse.ArgumentParser();q.add_argument('command',choices=('consume','verify'))
    for key in ('output','report-sha','receipt','receipt-sha'):q.add_argument('--'+key)
    a=q.parse_args()
    v=consume(a.output,a.report_sha)
    if a.command=='consume' and a.receipt:
        p=Path(a.receipt).absolute();b=host.box.raw(v)
        need(p.name.startswith(REPAIR+'_') and not p.exists() and not p.is_symlink() and len(b)<=CAP,'exclusive repair receipt')
        with p.open('xb') as f:need(f.write(b)==len(b),'complete repair receipt');f.flush();os.fsync(f.fileno())
        host.old.fsync_dir(p.parent);v=dict(path=str(p),sha256=host.sha(p))
    elif a.command=='verify':
        need(a.receipt and a.receipt_sha and host.sha(a.receipt)==a.receipt_sha,'external repair receipt SHA')
        b=Path(a.receipt).read_bytes();stored=json.loads(b)
        need(b==host.box.raw(stored) and stored==v,'complete saved recomputation');v=dict(verified=True,sha256=a.receipt_sha)
    print(json.dumps(v,sort_keys=True))


if __name__=='__main__':main()
