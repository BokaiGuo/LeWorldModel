"""Ten independent native U/Z2 continuations; MAIN-only run, saved-only preparation.

Does not import stageB or invoke any inherited fullstrict. Instance native hooks
are the original hooks; all new transport, guards, accounting and consume are owned.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import time

import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_p3_actual_cp10_full_witness_20261029_v1 as fw

NAME = 'agent_p3_same_time_capability_probe_physical_20261029_v2'
B = 'agent_p3_same_time_capability_probe_cuda_20261029_v1'
ROOT = fw.ROOT
AUTH = 'MAIN_P3_SAME_TIME_10_NATIVE_PROBES_V2'
CAP, RESERVE, FLOOR, WALL = 64*1024**2, 8*1024**2, 200*1024**2, 900
QUOTAS = {'binding.json':2*1024**2, 'journal.jsonl':20*1024**2, 'report.json':14*1024**2,
          'completion.json':1024**2, 'failure.json':1024**2}
COUNTS = fw.NEXT_JOB_BUDGET['resource_caps']
ORDER = tuple((i, root, role, probe) for i, (root, role, _, _) in enumerate(fw.ORDER) for probe in ('U', 'Z2'))
FLAGS = dict(P1_admission=False, P3_admission=False, scientific_admission=False,
             full_C3_demonstrated=False, physical_release=False, freshness=False,
             equal_FLOPs=False, equal_total_compute=False, independent_confirmation=False, training=False)
FW_SHA = '2351ba5b05c91fb192ae4542e6839ab58b56bcd974170b94994246f61981f346'
FW_SOURCE_SHA = 'd493cf81126563dab3de57e7d5fd9904dbc1ff5881dfe11bd3b284898338f0f3'
B_PINS = dict(prepare='90b3a1a6284c5605594d4341a2f994870e4b8db9c1721ba40f0160244082e870',
    report='a64fbf31cf2601cf6ea26d100ef5cb5c01efbe8b63be6cf33bc4d91d735c25fc',
    completion='2d4f8364867ccfba4bb15f4794f0f665af34483cf4bdb1adf57af4efb1f27574',
    receipt='682a78366b10a60895c4530f2fc6cc401e6439b625f4ba0a8aa996dc2fed79a7')
U_PINS = ('6029b37c7a02543a58ef962c4369002939239d3f34bda34bced155204eb3124a',
    '011d086c47cfd2ffd33fc16a895960d83e6cb0308c63168ecfd9bfe41abb8b21',
    '5cd3933ffab7ed42b0124f26414bcba8de3c15075f6fe0d27815e5a4b8c3caa3',
    'e1e2bf9124070256da21b064e2c1316045c9b3dc22190c665f44b9c9a3146422',
    'ebf2b284e92c80905c2da561fc7b69f669d4cfd69bc5cd6f213ef34627ab7c55')
HELPER = 'p1_moving_overlay_recompile_physical_20261027_v1'
HELPER_SHA = 'd6ee57d4d902c2d8c865c8372718a3bbbea3964a23973da535f106e415229570'
SOURCE_REPORT = ROOT/'results/p1_moving_target_overlay_source_20261027_v1_main01/report.json'
SOURCE_SHA = '85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a'
TRANSPORT = ('index', 'previous_sha256', 'event_sha256', 'branch', 'arm', 'selection_digest')
require, same, digest, raw, read, sha = fw.require, fw.same, fw.digest, fw.raw, fw.read, fw.sha


def compact_digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def references():
    base = ROOT/'results'; out = base/(B+'_cuda_MAIN01')
    return dict(full_witness=dict(path=str(base/(fw.NAME+'_MAIN01.json')),sha256=FW_SHA),
        stageB={k:dict(path=str(base/(B+'_prepare_MAIN01.json') if k=='prepare' else
                      base/(B+'_consume_MAIN01.json') if k=='receipt' else out/(k+'.json')),sha256=h)
                for k,h in B_PINS.items()},
        U=[dict(path=str(base/(B+f'_witness{i}_MAIN01.json')),sha256=h) for i,h in enumerate(U_PINS)])


def bits(actions, n):
    values = fw.f32bits(actions,n)
    require(all(-1 <= x <= 1 for r in actions for x in r), 'native raw domain without clip')
    return values


def numeric_bound(value):
    """Conservative bound for the same source-defined JSON structure, finite numbers.

    Unknown source fields are retained, not projected out. Runtime is additionally
    byte-capped, so an unforeseen shape/string increase fails before publication.
    """
    if isinstance(value,dict):
        return 2+sum(len(raw(k))+2+numeric_bound(v)+2 for k,v in value.items())
    if isinstance(value,list):return 2+sum(numeric_bound(v)+2 for v in value)
    if type(value) in (int,float):return 32
    if type(value) is str:return max(100,len(raw(value)))
    return len(raw(value))


def png_budget():
    # One filter byte per RGB8 scanline. Generic zlib deflateBound, not
    # compressBound's tighter default-parameter assumption. Integer ceilings.
    scan=224*(224*3+1)
    compressed=scan+(scan+7)//8+(scan+63)//64+11
    # Pillow ImageFile uses max(MAXBLOCK=65536, width*4) here. Allow one
    # extra final encoder write (including an empty final IDAT), and verify
    # the actual stream's chunk grammar/count below rather than assuming it.
    chunks=(compressed+65535)//65536+1
    return dict(scanline_bytes=scan,deflate_upper_bytes=compressed,idat_block_bytes=65536,
        idat_chunks_upper=chunks,png_fixed_bytes=8+25+12,
        png_upper_bytes=compressed+8+25+12+12*chunks)


def check_png_stream(body):
    b=png_budget();require(body[:8]==b'\x89PNG\r\n\x1a\n','PNG signature')
    pos=8;chunks=[]
    while pos<len(body):
        require(pos+12<=len(body),'complete PNG chunk')
        n=struct.unpack('>I',body[pos:pos+4])[0];kind=body[pos+4:pos+8]
        require(pos+12+n<=len(body),'complete PNG payload')
        chunks.append((kind,n));pos+=12+n
    require(chunks[0]==(b'IHDR',13) and chunks[-1]==(b'IEND',0)
            and all(k==b'IDAT' and n<=65536 for k,n in chunks[1:-1])
            and 1<=len(chunks)-2<=b['idat_chunks_upper']
            and sum(n for _,n in chunks[1:-1])<=b['deflate_upper_bytes']
            and len(body)<=b['png_upper_bytes'],'minimal Pillow PNG generic deflate/chunk bound')


def census(w):
    png=png_budget()['png_upper_bytes']
    # Complete real event template: no invented rng/public_state envelope.
    observation=max(numeric_bound(e) for r in w['rows'] for e in r['observations'])
    operation=8192;guard=512*1024
    journal_upper=748*operation+81*observation+2*guard+128*1024
    report_upper=374*operation+81*observation+2*guard+256*1024
    total=81*png+sum(QUOTAS.values())+RESERVE
    require(journal_upper<=QUOTAS['journal.jsonl'] and report_upper<=QUOTAS['report.json']
            and total<=CAP, 'actual field budget exceeds original caps; report MAIN, never expand')
    return dict(PNG=81,operations=374,operation_events=748,observation_events=81,guards=2,
        lifecycle=3,total_events=834,png_upper_bytes=png,png_budget=png_budget(),observation_upper_bytes=observation,
        operation_event_upper_bytes=operation,guard_upper_bytes=guard,
        journal_upper_bytes=journal_upper,report_upper_bytes=report_upper,quotas=QUOTAS,
        conservative_total_bytes=total,cap_bytes=CAP,reserve_bytes=RESERVE,
        disk_floor_bytes=FLOOR,wall_seconds=WALL,counts=COUNTS)


def assemble(w,bp,br,bc,bs,exports,refs,*,fixture=False):
    """Explicit actual saved ABI join, not a call to stageB.witness/consume."""
    require(w['schema']==fw.NAME+'.witness' and len(w['rows'])==5
            and (w['actual_MAIN_authority'] is True or fixture), 'complete actual full witness')
    require(bs['schema']==B+'.consume' and bs['status']=='saved_complete_NOT_admitted'
            and bs['fixed_signals']==7 and bs['fixed_probes']==5
            and bs['verification']['full_native_closure'] is True
            and bs['verification']['native_calls']==17, 'full actual stageB native receipt')
    require(br['schema']==B+'.report' and bp['schema']==B+'.prepare' and bc['schema']==B+'.completion'
            and br['status']==br['result']['status']=='complete_NOT_admitted' and br['errors']==[]
            and br['sources_verified'] is True and br['model_before']==br['model_after'], 'stageB actual runtime closure')
    a=refs['stageB']
    require(br['prepared']==bc['prepared']==bs['prepared']==a['prepare']['path']
            and br['prepare_sha256']==bc['prepare_sha256']==bs['prepare_sha256']==a['prepare']['sha256']
            and bc['report_sha256']==bs['report_sha256']==a['report']['sha256']
            and bs['completion_sha256']==a['completion']['sha256'], 'stageB four exact authorities')
    require(bp['bridge_authority']==bs['bridge_authority'] and
            all(bs['bridge_authority'][k]==w['authorities'][k] for k in fw.PINS), 'two independent authority domains same stageA')
    for key in ('signals','probes','metrics','resource_receipt'):
        require(same(bs[key],br['result'][key]), 'entire stageB receipt/report '+key)
    require(len(bs['signals'])==7 and len(bs['probes'])==len(bs['plans'])==len(exports)==5, 'all7/5/5')
    paid=bs['resource_receipt']
    require(paid['attempted']==paid['completed']==paid['caps'], 'no dropped stageB paid costs')
    for k,n in dict(jacobian_API=7,VJP_directions=1344,predict_API=24,solve_calls=7,replay_calls=10,metric_calls=22).items():
        require(paid['completed'][k]==n,'stageB complete resource '+k)
    jobs=[]
    for i,(r,plan,export) in enumerate(zip(w['rows'],bs['plans'],exports,strict=True)):
        root,role,_,_=fw.ORDER[i]
        require((r['index'],r['root_slot'],r['role'])==(i,root,role) and len(r['observations'])==66,
                'fixed full witness order')
        require(export['schema']==B+'.witness' and export['receipt']==a['receipt']['path']
                and export['receipt_sha256']==a['receipt']['sha256'] and export['plan']==plan
                and export['evidence_kind']=='MAIN_SHA_bound_saved_receipt', 'whole exported U witness exact join')
        body={k:v for k,v in plan.items() if k!='witness_sha256'}
        require(compact_digest(body)==plan['witness_sha256'] and plan['index']==i
                and plan['root_slot']==root and plan['role']==role
                and plan['checkpoint']==10 and plan['probe_checkpoint']==15, 'U content/time identity')
        require(plan['task']==r['stageA_task']['task'] and plan['endpoint_identity']==r['stageA_task']['identity']
                and plan['source']==r['stageA_row']['source']
                and plan['causal_raw10']==r['control_raw10']
                and plan['whole']==plan['segmented'] and len(plan['whole'])==1, 'same actual endpoint and full dual context')
        require(plan['native_seal_sha256']==br['result']['native_seal_sha256']
                and plan['native_call_ids']==dict(signal=i+2,whole=7+2*i,segmented=8+2*i), 'double replay identity')
        require(plan['authority']==dict(prepared=a['prepare']['path'],prepare_sha256=a['prepare']['sha256'],
            report=a['report']['path'],report_sha256=a['report']['sha256'],completion=a['completion']['path'],completion_sha256=a['completion']['sha256']),
            'U authority report/prepare/completion exact')
        b=bits(plan['raw5'],5)
        require(''.join(b)==plan['raw_float32_bits'] and hashlib.sha256(bytes.fromhex(''.join(b))).hexdigest()==plan['raw_float32_sha256'],
                'literal certified U, no second chart')
        for probe in ('U','Z2'):
            tail=plan['raw5'] if probe=='U' else [[0.0,0.0] for _ in range(5)]
            job=dict(index=len(jobs),endpoint=i,root_slot=root,source_role=role,probe=probe,
                     warm10=deepcopy(r['warm_raw10']),raw15=deepcopy(r['control_raw10']+tail),
                     source_witness_sha256=refs['full_witness']['sha256'],U_witness=refs['U'][i],
                     unknown_semantics=deepcopy(r['unknown_semantics']))
            bits(job['warm10'],10);bits(job['raw15'],15)
            job['selection_digest']=digest(job);jobs.append(job)
    return dict(schema=NAME+'.prepare',authority=deepcopy(refs),jobs=jobs,census=census(w),
        inherited_stageB_resource_receipt=deepcopy(paid),fixture_only=fixture,known_roots=2,fresh_roots=0,
        stageB_connected=True,full_witness_stageB_flag_unchanged=w['stageB_ready'],**FLAGS)


def material(refs):
    require(refs==references(), 'only actual MAIN fixed full-witness/stageB authorities')
    require(sha(fw.__file__)==FW_SOURCE_SHA, 'frozen extractor source')
    fw.verify(refs['full_witness']['path'],refs['full_witness']['sha256'])
    sources=fw.Sources();w=sources.load(refs['full_witness'])
    bp,br,bc,bs=(sources.load(refs['stageB'][k]) for k in ('prepare','report','completion','receipt'))
    exports=[sources.load(ref) for ref in refs['U']]
    p=assemble(w,bp,br,bc,bs,exports,refs)
    # Bind only source/metadata needed here, never load/hash model tensor files.
    for f,h in bp['bindings'].items():
        if f.endswith('.py'):
            sources.bind(f,h)
    for r in w['rows']:
        old=sources.load(r['source_authorities']['prepare'])
        for f,h in old['bindings'].items():
            if f.endswith('.py'):
                sources.bind(f,h)
        require(old['bindings'].get(str(SOURCE_REPORT))==SOURCE_SHA, 'original paired source report bound')
    source=sources.load(dict(path=str(SOURCE_REPORT),sha256=SOURCE_SHA))
    goals=[]
    for root in (0,1):
        item=source['blind_exports'][root]['artifacts']['off']
        f=str(SOURCE_REPORT.parent/item['path'])
        require(Path(item['path']).name==item['path'], 'original flat blind NPZ')
        sources.bind(f,item['sha256']);goals.append(dict(path=f,sha256=item['sha256'],pixel_sha256=item['goal_pixel_sha256']))
    hp=ROOT/'experiments/spectral_reachability'/(HELPER+'.py')
    sources.bind(str(hp),HELPER_SHA)
    own=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    for f in own:sources.bind(str(f),sha(f))
    p.update(bindings=sources.bindings,goal_payloads=goals)
    sources.finish();return p,w


def write_new(p,v):
    return fw.write_new(p,v)


def prepare(output):
    require(not Path(output).exists(), 'exclusive prepare before inputs')
    p,_=material(references());return write_new(output,p)


def verify(prepared,expected):
    require(sha(prepared)==expected,'own prepare SHA')
    p=read(prepared);rebuilt,w=material(p['authority'])
    require(same(p,rebuilt) and sha(prepared)==expected,'whole prepare reconstruction')
    return p,w


def response(goal,root_state,step,result):
    s=np.asarray(result['state'],np.float64);g=np.asarray(goal,np.float64);d=s-g
    angle=float(abs(d[4]));native_angle=min(angle,2*np.pi-angle)
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))));xy=float(np.linalg.norm(d[:4]))
    metrics=dict(native_success=bool(xy<20 and native_angle<np.pi/9),native_distance=float(np.linalg.norm(d)),
        native_angle_term=float(native_angle),negative_native_angle=bool(native_angle<0),
        wrapped_allxy_success=bool(xy<20 and wrapped<np.pi/9),wrapped_object_angle_error=wrapped,
        object_position_error=float(np.linalg.norm(d[2:4])),object_pose=s[2:5].tolist(),object_success=None)
    delta=s-np.asarray(root_state,np.float64)
    require(result['terminated'] is metrics['native_success'] and result['reward']==-metrics['native_distance'], 'native reward/success contract')
    return dict(step=step,state=s.tolist(),delta_native7=delta.tolist(),
        delta_angle_wrapped=float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))),reward=result['reward'],
        native_success=metrics['native_success'],terminated=result['terminated'],truncated=result['truncated'],native_metrics=metrics)


def projected(e,source,job):
    require(set(e)==set(source) and e['kind']==source['kind']=='fresh_observation'
            and e['branch']==job['index'] and e['arm']==job['probe'] and e['root_slot']==job['root_slot']
            and e['condition']=='off' and e['selection_digest']==job['selection_digest'], 'exact observation scope/envelope')
    a=deepcopy(e);mapping=[]
    for k in TRANSPORT:
        mapping.append(dict(pointer='/'+k,actual=a[k],source=source[k]));a[k]=source[k]
    mapping.append(dict(pointer='/receipt/png',actual=a['receipt']['png'],source=source['receipt']['png']))
    a['receipt']['png']=source['receipt']['png']
    require(same(a,source), 'whole original observation, only enumerated transport mapping')
    return mapping


def guard_proof(job,reference,observations,ops,n):
    require(n in (51,66) and len(observations)==n, 'all51/66 before next public intent')
    mapping=[projected(e,old,job) for e,old in zip(observations,reference['observations'][:n],strict=True)]
    public=[op for op in ops if op['resource']=='public_step']
    count=15 if n==51 else 20
    require(len(public)==count and all(x['status']=='returned' for x in public), 'all actual prior public returns')
    commands=job['warm10']+job['raw15'][:count-10]
    require([op['raw_bits'] for op in public]==bits(commands,count), 'complete actual incoming raw bytes')
    for op in public:
        result=op['result'];require(result['actual_bits']==op['raw_bits'] and result['actual_dtype']=='<f4', 'actual latest_action raw')
    resets=[x for x in ops if x['resource']=='reset']
    recipe=reference['placement_events'][0];slot=reference['source_slot_metadata']
    require(len(resets)==2 and all(x['status']=='returned' for x in resets)
            and resets[0]['seed']==slot['reset_seed'] and resets[0]['role']=='seed'
            and resets[1]['seed']==0 and resets[1]['role']=='placement'
            and same(resets[0]['result']['goal_state'],recipe['native_goal'])
            and same(resets[1]['result']['snapshot'],observations[0]['pre']['snapshot'])
            and same(resets[1]['requested_state'],recipe['requested_state']), 'exact original reset/placement input and actual warm0')
    last=public[-1]
    if n==66:
        got=response(reference['source_row']['goal'],reference['source_row']['root_state'],10,last['result'])
        require(same(got,reference['actual_response10']), 'actual response10 entire original record')
    t=reference['source_row']['plan']['model_task'] if n==51 else reference['stageA_task']['task']
    incoming=np.asarray(commands[-10:],dtype='<f4')
    tokens=((incoming-np.asarray(t['action_mean'],np.float32))/np.asarray(t['action_std'],np.float32)).reshape(2,10)
    require(tokens.tobytes()==np.asarray(t['root_context']['action_history'],dtype='<f4').tobytes(), 'actual raw causal token history')
    return dict(n=n,reference_observations_sha256=digest(reference['observations'][:n]),
        actual_observations_sha256=digest(observations),actual_public_receipts_sha256=digest(public),
        raw_float32_sha256=hashlib.sha256(incoming.tobytes()).hexdigest(),
        tokens_float32_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),transport_mappings=mapping,
        unknown_semantics=reference['unknown_semantics'])


class Store:
    def __init__(self,output,binding,census_value,clock=time.monotonic):
        self.path=Path(output);self.binding=binding;self.limits=census_value;self.clock=clock;self.started=clock()
        require(self.path.is_absolute() and self.path.parent.is_dir() and not self.path.exists()
                and not self.path.is_symlink() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR, 'exclusive64MiB job +200MiB floor')
        self.path.mkdir();fsync_directory(self.path.parent)
        self.index=0;self.previous=None;self.observations=[];self.broken=False
        self.put('binding.json',binding)

    def check(self,size=0,final=False):
        files=list(self.path.iterdir())
        require(all(f.is_file() and not f.is_symlink() for f in files),'flat regular output')
        require(sum(f.stat().st_size for f in files)+size<=CAP-(0 if final else RESERVE)
                and shutil.disk_usage(self.path).free>=FLOOR+size,'job cap/reserve/floor')
        if not final:require(self.clock()-self.started<WALL,'whole per-job900s')

    def write(self,name,body,*,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..'),'flat artifact')
        p=self.path/name;require(not p.is_symlink(),'no symlink output')
        if name.endswith('.png'):
            require(not append and len(body)<=self.limits['png_upper_bytes']
                    and len(list(self.path.glob('*.png')))<81,'81 bounded PNG')
        else:
            require(name in QUOTAS and (not append or name=='journal.jsonl'),'owned metadata names')
            require((p.stat().st_size if p.exists() else 0)+len(body)<=QUOTAS[name],'all serialized copies byte quota')
        self.check(len(body),final)
        with p.open('ab' if append else 'xb') as f:
            require(f.write(body)==len(body),'complete write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return sha(str(p))

    def put(self,name,value,final=False):return self.write(name,raw(value)+b'\n',final=final)

    def emit(self,kind,**fields):
        e=dict(index=self.index,previous_sha256=self.previous,kind=kind,**deepcopy(fields));e['event_sha256']=digest(e)
        if kind.startswith('operation_'):
            require(len(raw(e))+1<=self.limits['operation_event_upper_bytes'],'full operation event byte bound')
        if kind=='guard':require(len(raw(e))+1<=self.limits['guard_upper_bytes'],'entire guard proof byte bound')
        self.write('journal.jsonl',raw(e)+b'\n',append=True,final=kind in ('branch_closed','run_finished','operation_failed'))
        self.index+=1;self.previous=e['event_sha256']
        if kind=='fresh_observation':
            require(len(raw(e))+1<=self.limits['observation_upper_bytes'],'actual observation source-derived byte bound')
            self.observations.append(e)
        return e

    def frame(self,name,value):
        require(value.shape==(224,224,3) and value.dtype==np.uint8,'native RGB224 uint8')
        b=io.BytesIO();Image.fromarray(value).save(b,format='PNG');body=b.getvalue()
        check_png_stream(body)
        return self.write(name,body)


class Meter:
    def __init__(self,store,job,reference,public_encoder=lambda x:x,reset_encoder=lambda x:x):
        self.store=store;self.job=job;self.reference=reference;self.public_encoder=public_encoder;self.reset_encoder=reset_encoder
        self.where=dict(branch=job['index'],root_slot=job['root_slot'],arm=job['probe'],condition='off',selection_digest=job['selection_digest'])
        self.counts={k:dict(attempted=0,completed=0) for k in COUNTS};self.active={k:0 for k in COUNTS}
        self.receipts=[];self.guarded=set();self.secondary=[]

    def call(self,resource,fn,*,cleanup=False,**fields):
        require(resource in COUNTS and (not cleanup or resource=='close'),'known resource/close only cleanup')
        if not cleanup:self.store.check()
        if resource=='public_step' and fields.get('phase')=='plan' and fields['step'] in (6,11):
            n=51 if fields['step']==6 else 66
            try:proof=guard_proof(self.job,self.reference,self.store.observations,self.receipts,n)
            except BaseException as e:
                self.store.emit('guard',n=n,accepted=False,proof=None,error=error(e));raise
            self.store.emit('guard',n=n,accepted=True,proof=proof,error=None)
            self.guarded.add(n)  # Only after durable write/fsync.
        if resource=='public_step' and fields.get('phase')=='plan':
            require(fields['step']<=5 or 51 in self.guarded,'no control6 before51guard')
            require(fields['step']<=10 or 66 in self.guarded,'no control11 before66guard')
        require(self.counts[resource]['attempted']<COUNTS[resource],'resource attempt cap')
        op=dict(index=len(self.receipts),resource=resource,**self.where,**fields,status='pending',result=None,error=None)
        self.store.emit('operation_dispatch_intent',receipt=op)
        self.receipts.append(op);self.counts[resource]['attempted']+=1;self.active[resource]+=1
        try:
            value=fn()
            op['result']=self.public_encoder(value) if resource=='public_step' else self.reset_encoder(value) if resource=='reset' else None
            op['status']='returned';self.counts[resource]['completed']+=1
        except BaseException as e:
            op.update(status='failed',error=error(e));self.store.emit('operation_failed',receipt=op);raise
        finally:self.active[resource]-=1
        self.store.emit('operation_returned',receipt=op)
        return value


def error(e):return dict(type=type(e).__name__,message=str(e))


def read_events(output):
    p=Path(output)/'journal.jsonl';require(p.stat().st_size<=QUOTAS[p.name],'bounded entire journal')
    events=[];tail=None
    with p.open('rb') as f:
        for i,line in enumerate(f):
            e=fw.decode(line)
            require(len(line)<=1024**2 and line==raw(e)+b'\n' and e['index']==i and type(e['index']) is int
                    and e['previous_sha256']==tail and e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}), 'whole event hashchain')
            tail=e['event_sha256'];events.append(e)
    return events,tail


def reconstruct(binding,reference,events):
    job=binding['job'];counts={k:dict(attempted=0,completed=0) for k in COUNTS}
    ops=[];stack=[];observations=[];guards=[];closed=None;finished=None
    require(events and set(events[0])=={'index','previous_sha256','event_sha256','kind','binding_sha256'}
            and events[0]['kind']=='run_started' and events[0]['binding_sha256']==digest(binding),'initial binding')
    for e in events[1:]:
        k=e['kind']
        base={'index','previous_sha256','event_sha256','kind'}
        extra={'operation_dispatch_intent':{'receipt'},'operation_returned':{'receipt'},'operation_failed':{'receipt'},
               'fresh_observation':{'branch','root_slot','arm','condition','selection_digest','receipt','pre','post'},
               'guard':{'n','accepted','proof','error'},'branch_closed':{'result'},'run_finished':{'result'}}
        require(k in extra and set(e)==base|extra[k],'exact owned full event fields')
        if k=='operation_dispatch_intent':
            op=e['receipt'];r=op['resource'];require(r in counts and op['index']==len(ops),'operation identity')
            require(op['status']=='pending' and op['result'] is op['error'] is None,'original intent')
            require(op['branch']==job['index'] and op['root_slot']==job['root_slot'] and op['arm']==job['probe']
                    and op['selection_digest']==job['selection_digest'] and op['condition']=='off','all operation job scope')
            if r=='public_step':
                previous=[x for x in ops if x['resource']=='public_step']
                i=len(previous);phase='warmup' if i<10 else 'plan';step=i+1 if i<10 else i-9
                expected=job['warm10']+job['raw15']
                require(i<25 and (op['phase'],op['step'])==(phase,step) and op['raw_bits']==bits(expected,25)[i],'literal all25 dispatch order')
                require(not any(x['status']=='returned' and (x['result']['terminated'] or x['result']['truncated']) for x in previous),'no dispatch after native stop')
                require(i<15 or 51 in guards,'first51 before control6')
                require(i<20 or 66 in guards,'all66 before control11')
            counts[r]['attempted']+=1;require(counts[r]['attempted']<=COUNTS[r],'no resource refunds/overbudget')
            active=[ops[i]['resource'] for i in stack]
            if r=='set_state':require('reset' in active,'state setter only reset, never hydrate')
            if r=='goal_setter':require('reset' in active or 'goal_bind' in active,'goal setter only original reset/attachment')
            if r=='physics_step':require(not set(active)&{'explicit_render','goal_render','goal_bind'},'no hidden render/attachment physics')
            stack.append(op['index']);ops.append(deepcopy(op))
        elif k in ('operation_returned','operation_failed'):
            op=e['receipt'];require(stack and stack.pop()==op['index'],'nested actual terminal')
            old=ops[op['index']];v=dict(op,status='pending',result=None,error=None)
            require(same(v,old) and op['status']==('returned' if k=='operation_returned' else 'failed'),'operation terminal exact')
            if op['resource']=='public_step' and op['status']=='returned':
                require(op['result']['actual_bits']==op['raw_bits'] and op['result']['actual_dtype']=='<f4','actual returned raw identity')
            ops[op['index']]=deepcopy(op);counts[op['resource']]['completed']+=op['status']=='returned'
        elif k=='fresh_observation':
            require(closed is None and e['branch']==job['index'] and e['root_slot']==job['root_slot']
                    and e['arm']==job['probe'] and e['selection_digest']==job['selection_digest'],'complete observation scope')
            f=e['receipt'];n=len(observations)
            expected=[(ph,st,ov) for ph,stop in (('warmup',10),('plan',15)) for st in range(stop+1) for ov in range(3)]
            require(n<81 and (f['phase'],f['step'],f['overlay_index'])==expected[n],'full81 order no padding')
            require(same(e['pre'],e['post']) and same(f['snapshot'],e['post']['snapshot'])
                    and same(f['state'],f['snapshot']['state']) and len(f['state'])==7
                    and struct.pack('<7d',*f['state']).hex()==f['snapshot']['state_float64_bits']
                    and f['condition']==('on' if f['overlay_index']==1 else 'off'),'full inert snapshot/state64/overlay')
            if f['overlay_index']==2:
                require(same(e['pre'],observations[-2]['pre']) and same(e['pre'],observations[-1]['pre'])
                        and f['file_sha256']==observations[-2]['receipt']['file_sha256']
                        and f['pixel_sha256']==observations[-2]['receipt']['pixel_sha256'],'off/on/off full invariant and pixel replay')
            if f['step']:
                prior=[x for x in ops if x['resource']=='public_step' and x.get('phase')==f['phase'] and x.get('step')==f['step']]
                require(len(prior)==1 and prior[0]['status']=='returned' and all(same(f[key],prior[0]['result'][key]) for key in ('state','snapshot')),'fresh RGB tied to actual public return')
            observations.append(e)
        elif k=='guard':
            n=e['n'];require(n not in guards and n in (51,66),'unique two guard nodes')
            if e['accepted']:
                require(e['error'] is None and same(e['proof'],guard_proof(job,reference,observations,ops,n)),'entire durable guard reconstruction')
                guards.append(n)
            else:
                require(e['proof'] is None and e['error'] is not None,'failed guard remains explicit')
                failed=False
                try:guard_proof(job,reference,observations,ops,n)
                except (ValueError,KeyError):failed=True
                require(failed,'rejected guard reproducible')
        elif k=='branch_closed':
            require(closed is None and set(e['result'])=={'status','error','cleanup_errors'},'one full closure');closed=e['result']
        elif k=='run_finished':
            require(e is events[-1] and closed is not None,'final terminal ordering');finished=e['result']
        else:raise ValueError('unknown stageC event '+k)
    require(closed is not None and finished==closed,'full failure/terminal disposition')
    # Pending intents are unknown, not fabricated failures or refunded work.
    public=[x for x in ops if x['resource']=='public_step' and x['status']=='returned']
    responses=[response(reference['source_row']['goal'],reference['source_row']['root_state'],x['step'],x['result']) for x in public if x['phase']=='plan']
    checkpoints=[deepcopy(e['receipt']) for e in observations if e['receipt']['phase']=='plan'
                 and e['receipt']['step'] in (0,5,10,15) and e['receipt']['overlay_index']==0]
    complete=closed['status']=='horizon_completed'
    if complete:
        require(not stack and not closed['error'] and closed['cleanup_errors']==[] and len(observations)==81
                and guards==[51,66] and all(c['attempted']==c['completed']==COUNTS[k] for k,c in counts.items()), 'complete all374/81 actual census')
    elif closed['status']=='native_stopped':
        require(public and (public[-1]['result']['terminated'] or public[-1]['result']['truncated'])
                and closed['error'] is None,'native early stop not invented')
    else:require(closed['status']=='engineering_failed' and closed['error'] is not None,'explicit engineering failure')
    return dict(job=job,status=closed['status'],error=closed['error'],cleanup_errors=closed['cleanup_errors'],
        counts=counts,operations=ops,pending_operation_indices=stack,observations=observations,guards=guards,
        responses=responses,checkpoints=checkpoints,missing_checkpoints=[n for n in (0,5,10,15) if n not in [x['step'] for x in checkpoints]],
        endpoint15=next((x for x in checkpoints if x['step']==15),None),
        unknowns=dict(rng=reference['unknown_semantics']['rng'],public_state=reference['unknown_semantics']['public_state'],
                      contact_cache=reference['unknown_semantics']['contact_cache'],FLOPs=None,CUDA_ns=None),
        new_neural_calls=0,fixture_only=binding['fixture_only'],**FLAGS)


def publish(store,reference):
    events,tail=read_events(store.path);row=reconstruct(store.binding,reference,events)
    report=dict(schema=NAME+'.report',binding=store.binding,row=row,journal_tail=tail,
        journal_sha256=sha(str(store.path/'journal.jsonl')),elapsed_seconds=store.clock()-store.started)
    store.put('report.json',report,final=True)
    files=list(store.path.iterdir())
    c=dict(schema=NAME+'.completion',report_sha256=sha(str(store.path/'report.json')),
        artifact_sha256={f.name:sha(str(f)) for f in files},bytes_before_completion=sum(f.stat().st_size for f in files),
        inclusive_wall_seconds=store.clock()-store.started,wall_scope='through report hash; completion self-write excluded')
    store.put('completion.json',c,final=True)
    if store.clock()-store.started>WALL:
        store.put('failure.json',dict(stage='completion_publication',error='whole900s exceeded'),final=True)
        raise ValueError('final publication wall failed, artifacts retained')
    return dict(output=str(store.path),report_sha256=sha(str(store.path/'report.json')),completion_sha256=sha(str(store.path/'completion.json')))


def consume_job(output,report_sha,completion_sha,binding,reference,*,permit_fixture=False):
    require(not binding['fixture_only'] or permit_fixture,'RAM never production evidence')
    out=Path(output);require(out.is_absolute() and not out.is_symlink(),'absolute job directory')
    require(sha(str(out/'report.json'))==report_sha and sha(str(out/'completion.json'))==completion_sha,'actual report/completion SHA')
    files=list(out.iterdir());require(all(f.is_file() and not f.is_symlink() for f in files)
        and not (out/'failure.json').exists() and sum(f.stat().st_size for f in files)<=CAP,'bounded all artifacts no failed final')
    c=read(out/'completion.json');r=read(out/'report.json')
    require(set(r)=={'schema','binding','row','journal_tail','journal_sha256','elapsed_seconds'}
            and set(c)=={'schema','report_sha256','artifact_sha256','bytes_before_completion','inclusive_wall_seconds','wall_scope'},'all report/completion fields')
    require(c['schema']==NAME+'.completion' and r['schema']==NAME+'.report'
        and c['report_sha256']==report_sha and 0<=r['elapsed_seconds']<=c['inclusive_wall_seconds']<=WALL
        and c['wall_scope']=='through report hash; completion self-write excluded','exact publication fields')
    require(c['artifact_sha256']=={f.name:sha(str(f)) for f in files if f.name!='completion.json'}
            and c['bytes_before_completion']==sum(f.stat().st_size for f in files if f.name!='completion.json'),'all file bytes/SHA including PNG')
    require(same(read(out/'binding.json'),binding) and same(r['binding'],binding),'whole job input binding')
    for name,quota in QUOTAS.items():
        if (out/name).exists():require((out/name).stat().st_size<=quota,'metadata quota '+name)
    events,tail=read_events(out);require(tail==r['journal_tail'] and r['journal_sha256']==sha(str(out/'journal.jsonl')),'report tail')
    row=reconstruct(binding,reference,events);require(same(row,r['row']),'entire row/resource/unknown/null reconstruction')
    png=set()
    for e in row['observations']:
        f=e['receipt'];name=f['png'];require(Path(name).name==name and name not in png,'each actual PNG unique')
        png.add(name);p=out/name
        require(p.stat().st_size<=census({'rows':[reference]})['png_upper_bytes'] and sha(str(p))==f['file_sha256'],'every PNG file')
        check_png_stream(p.read_bytes())
        with Image.open(p) as im:
            require(im.mode=='RGB' and im.size==(224,224) and im.format=='PNG','native RGB224 PNG')
            pixels=np.asarray(im).copy()
        require(hashlib.sha256(pixels.tobytes()).hexdigest()==f['pixel_sha256'],'all original decoded pixels')
    require({f.name for f in files}==png|{'binding.json','journal.jsonl','report.json','completion.json'},'no unconsumed artifacts')
    require(sha(str(out/'completion.json'))==completion_sha and
            all(sha(str(out/n))==h for n,h in c['artifact_sha256'].items()),'stable output bytes after full reconstruction')
    return dict(schema=NAME+'.job_consume',output=str(out),report_sha256=report_sha,completion_sha256=completion_sha,row=row,
                event_count=len(events),fixture_only=binding['fixture_only'],**FLAGS)


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def inert_constructor(source):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PushT')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    require(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
        {'reset','step','render','_set_state','_set_goal_state','_setup'} for n in ast.walk(init)),
        'constructor inert before native hooks; failed constructor callbacks unknown')


def native_helpers(p):
    # Import only after MAIN authorization and verified prepare.
    import importlib
    for f,h in p['bindings'].items():require(sha(f)==h,'source closure before runtime imports')
    require(p['bindings'].get(str(ROOT/'experiments/spectral_reachability'/(HELPER+'.py')))==HELPER_SHA,'pinned helper')
    helper=importlib.import_module('experiments.spectral_reachability.'+HELPER)
    env_source=Path(helper.moving.ENV).resolve()
    require(p['bindings'].get(str(env_source))==helper.moving.ENV_SHA
            and sha(str(env_source))==helper.moving.ENV_SHA,'exact native constructor source closure')
    inert_constructor(env_source.read_text())
    return helper


def native_job(p,w,job,output,prepared,prepare_sha,helper):
    ref=w['rows'][job['endpoint']]
    binding=dict(schema=NAME+'.job_binding',prepared=str(prepared),prepare_sha256=prepare_sha,
        job=job,fixture_only=False,authority=p['authority'],guard_transport_fields=list(TRANSPORT)+['receipt/png'])
    store=Store(output,binding,p['census']);store.emit('run_started',binding_sha256=digest(binding))
    meter=Meter(store,job,ref);env=None;scratch=dict(index=job['index'],condition='off',PNG=0,trajectory=[],
        checkpoints=[],missing_checkpoints=[0,5,10],last_observation=None)
    closed=dict(status='engineering_failed',error=None,cleanup_errors=[])
    def return_summary(ret):
        obs,reward,t,tr,info=ret;actual=np.asarray(env.unwrapped.latest_action)
        return dict(state=np.asarray(obs['state'],np.float64).tolist(),snapshot=helper.moving.snapshot(env.unwrapped),
            reward=float(reward),terminated=bool(t),truncated=bool(tr),goal_state=np.asarray(info['goal_state']).tolist(),
            actual_bits=actual.tobytes().hex(),actual_dtype=actual.dtype.str)
    def reset_summary(ret):
        obs,info=ret
        return dict(state=np.asarray(obs['state'],np.float64).tolist(),goal_state=np.asarray(info['goal_state']).tolist(),
                    snapshot=helper.moving.snapshot(env.unwrapped))
    meter.public_encoder=return_summary;meter.reset_encoder=reset_summary
    try:
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        env=meter.call('constructor',lambda:gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False))
        helper.validate_env(env.unwrapped)
        require(env.unwrapped.render_action is False and env.unwrapped.render_size==224,'native render contract')
        goal_payload=p['goal_payloads'][job['root_slot']]
        require(sha(goal_payload['path'])==goal_payload['sha256'],'bound original goal NPZ')
        with np.load(goal_payload['path'],allow_pickle=False) as z:
            require(set(z.files)=={'rgb_history','raw_action_history','goal_rgb'},'original blind numerical keys')
            goal_rgb=z['goal_rgb'].copy()
        require(goal_rgb.shape==(224,224,3) and goal_rgb.dtype==np.uint8
                and hashlib.sha256(goal_rgb.tobytes()).hexdigest()==ref['source_root']['goal_pixels']['off'],'original task goal pixels')
        with helper.native_hooks(env,meter,closed['cleanup_errors']):
            slot=ref['source_slot_metadata'];recipe=ref['placement_events'][0]
            _,info=meter.call('reset',lambda:env.reset(seed=slot['reset_seed']),role='seed',seed=slot['reset_seed'])
            require(same(np.asarray(info['goal_state']).tolist(),recipe['native_goal']),'original reset native goal')
            # Exact original placement reset input; NOT cp10 hydrate or new sampler.
            obs,info=meter.call('reset',lambda:env.reset(seed=0,options=dict(
                state=np.asarray(recipe['requested_state'],np.float64),goal_state=np.asarray(recipe['native_goal'],np.float64))),
                role='placement',seed=0,requested_state=deepcopy(recipe['requested_state']))
            for phase,commands in (('warmup',job['warm10']),('plan',job['raw15'])):
                if phase=='plan':
                    body=env.unwrapped;before=helper.invariant(body)
                    goal=np.asarray(ref['source_root']['goal'],np.float64)
                    meter.call('goal_bind',lambda:helper.moving.attach_goal(body,goal,goal_rgb))
                    after=helper.invariant(body)
                    require(all(same(before[k],after[k]) for k in ('snapshot','renderer','goal_pose')),'goal attach cannot alter dynamics/render pose')
                helper.capture(env,obs,phase,0,scratch,meter,store)
                for step,command in enumerate(commands,1):
                    a=np.asarray(command,dtype='<f4')
                    obs,reward,t,tr,info=meter.call('public_step',lambda:env.step(a.copy()),phase=phase,step=step,raw_bits=a.tobytes().hex())
                    helper.capture(env,obs,phase,step,scratch,meter,store)
                    expected_goal=np.asarray(recipe['native_goal'] if phase=='warmup' else ref['source_root']['goal'])
                    helper.paired.check_return(env,a,np.asarray(env.unwrapped.latest_action),info,expected_goal,float(reward),bool(t))
                    if t or tr:
                        closed['status']='native_stopped';raise NativeStop()
            closed['status']='horizon_completed'
    except NativeStop:
        closed['error']=None
    except BaseException as e:
        closed.update(status='engineering_failed',error=error(e))
    finally:
        if env is not None:
            try:meter.call('close',env.close,cleanup=True)
            except BaseException as e:closed['cleanup_errors'].append(error(e))
        if closed['cleanup_errors']:
            closed.update(status='engineering_failed',error=closed['error'] or dict(type='CleanupFailure',message='native hook/close failure'))
        try:
            for f,h in p['bindings'].items():require(sha(f)==h,'source/input final closure')
            require(sha(str(prepared))==prepare_sha,'prepare final bytes')
        except BaseException as e:closed.update(status='engineering_failed',error=error(e))
        try:
            store.emit('branch_closed',result=closed);store.emit('run_finished',result=closed)
            return publish(store,ref)
        except BaseException as e:
            return failed_publication(store,meter,closed,e)


class NativeStop(Exception):
    """Actual terminated/truncated; never pad to cp15."""


def failed_publication(store,meter,closed,exc):
    """Last reserve record; never substitutes unknown callbacks with successes."""
    journal_file=store.path/'journal.jsonl'
    v=dict(schema=NAME+'.publication_failure',binding_sha256=digest(store.binding),
        error=error(exc),closed=closed,observed_meter_counts=deepcopy(meter.counts),
        observed_operation_receipts=deepcopy(meter.receipts),
        journal_sha256=sha(str(journal_file)) if journal_file.exists() else None,
        artifact_sha256={f.name:sha(str(f)) for f in store.path.iterdir()},
        elapsed_seconds=store.clock()-store.started,status='transport_failed_NOT_verified',
        unknown_final_callbacks=True,endpoint15=None,**FLAGS)
    store.put('failure.json',v,final=True)
    return dict(output=str(store.path),status=v['status'],failure_sha256=sha(str(store.path/'failure.json')),
                report_sha256=None,completion_sha256=None)


def consume_failure(output,expected,binding):
    out=Path(output);f=out/'failure.json'
    require(sha(str(f))==expected and f.stat().st_size<=QUOTAS['failure.json'],'actual failure artifact')
    v=read(f);require(v['schema']==NAME+'.publication_failure' and v['status']=='transport_failed_NOT_verified'
        and v['binding_sha256']==digest(binding) and v['unknown_final_callbacks'] is True and v['endpoint15'] is None,'no complete authority from failed publication')
    files=list(out.iterdir());require(all(p.is_file() and not p.is_symlink() for p in files)
        and sum(p.stat().st_size for p in files)<=CAP and same(read(out/'binding.json'),binding),'failed flat bounded input identity')
    require(v['artifact_sha256']=={p.name:sha(str(p)) for p in files if p.name!='failure.json'},'all failed partial bytes sealed')
    require(v['journal_sha256']==v['artifact_sha256'].get('journal.jsonl'),'failed journal byte identity')
    return dict(schema=NAME+'.failure_consume',output=str(out),failure_sha256=expected,
        status=v['status'],declarations=v,counts=None,endpoint15=None,unknown_final_callbacks=True,**FLAGS)


def run(prepared,prepare_sha,output,authorization):
    require(authorization==AUTH,'MAIN explicit reviewed production authority')
    p,w=verify(prepared,prepare_sha);require(p['fixture_only'] is False,'no RAM dispatch')
    root=Path(output);require(root.is_absolute() and not root.exists() and root.parent.is_dir(),'exclusive batch directory')
    helper=native_helpers(p);root.mkdir();fsync_directory(root.parent);results=[]
    for job in p['jobs']:
        # Sequential independent constructors; no sharing live env/contact caches.
        result=native_job(p,w,job,root/f'job{job["index"]:02}',prepared,prepare_sha,helper)
        results.append(result)
    return write_new(str(root/'batch.json'),dict(schema=NAME+'.batch',prepared=str(prepared),prepare_sha256=prepare_sha,jobs=results,**FLAGS))


def consume(output,prepared,prepare_sha,batch_sha):
    p,w=verify(prepared,prepare_sha);root=Path(output);batch=read(root/'batch.json')
    require(sha(str(root/'batch.json'))==batch_sha,'external MAIN actual batch SHA')
    require(batch['schema']==NAME+'.batch' and batch['prepared']==str(prepared) and batch['prepare_sha256']==prepare_sha
            and len(batch['jobs'])==10,'full10 batch receipt')
    require({f.name for f in root.iterdir()}=={'batch.json'}|{f'job{i:02}' for i in range(10)},'all ten independent job directories')
    receipts=[]
    for job,seals in zip(p['jobs'],batch['jobs'],strict=True):
        out=root/f'job{job["index"]:02}';binding=read(out/'binding.json')
        require(same(binding,dict(schema=NAME+'.job_binding',prepared=str(prepared),prepare_sha256=prepare_sha,
            job=job,fixture_only=False,authority=p['authority'],guard_transport_fields=list(TRANSPORT)+['receipt/png'])),'full job/schema/domain binding')
        require(seals['output']==str(out),'no reused branch directory')
        if seals.get('status')=='transport_failed_NOT_verified':
            receipts.append(consume_failure(str(out),seals['failure_sha256'],binding))
        else:receipts.append(consume_job(str(out),seals['report_sha256'],seals['completion_sha256'],binding,w['rows'][job['endpoint']]))
    require(sha(str(root/'batch.json'))==batch_sha and sha(str(prepared))==prepare_sha,'stable batch/prepare after consume')
    for f,h in p['bindings'].items():require(sha(f)==h,'input/source final consume closure')
    partial=any(x.get('status')=='transport_failed_NOT_verified' for x in receipts)
    return dict(schema=NAME+'.consume',status='saved_partial_transport_NOT_verified' if partial else 'all10_saved_native_jobs_reconstructed_NOT_admitted',
        prepared=str(prepared),prepare_sha256=prepare_sha,fixed_denominator=10,jobs=receipts,
        source_authorities=p['authority'],inherited_stageB_resource_receipt=p['inherited_stageB_resource_receipt'],
        new_neural_calls=0,FLOPs=None,**FLAGS)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);sub=q.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    for cmd in ('verify','run','consume'):
        a=sub.add_parser(cmd);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if cmd in ('run','consume'):a.add_argument('--output',required=True)
        if cmd=='run':a.add_argument('--authorization',required=True)
        if cmd=='consume':a.add_argument('--receipt',required=True);a.add_argument('--batch-sha',required=True)
    a=q.parse_args(argv)
    if a.command=='prepare':value=prepare(a.output)
    elif a.command=='verify':p,_=verify(a.prepared,a.prepare_sha);value=dict(verified=True,jobs=10,census=p['census'])
    elif a.command=='run':value=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:value=write_new(a.receipt,consume(a.output,a.prepared,a.prepare_sha,a.batch_sha))
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
