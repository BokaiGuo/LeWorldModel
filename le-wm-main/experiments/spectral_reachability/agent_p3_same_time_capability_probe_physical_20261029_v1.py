"""Stage-C transport/schema RAM seam only. Native production deliberately unavailable."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time

from PIL import Image

NAME='agent_p3_same_time_capability_probe_physical_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'innerdocs/agent_p3_secant_same_time_capability_protocol_20261029_v1.md'
PROTOCOL_SHA='b35af390fc74e6ebd44a9879784308ec74947fb28f640ce5e795cc263e65dccf'
CAP,RESERVE,FLOOR,WALL=64*1024**2,8*1024**2,200*1024**2,900
QUOTAS={'binding.json':2*1024**2,'journal.jsonl':20*1024**2,'report.json':14*1024**2,
        'completion.json':1024**2,'failure.json':1024**2}
ORDER=((0,'C'),(0,'Z'),(0,'P'),(1,'Z'),(1,'P'))
COUNTS=dict(constructor=1,close=1,reset=2,set_state=4,goal_render=2,goal_bind=1,goal_setter=3,
            public_step=25,physics_step=254,explicit_render=81)
BRIDGE='agent_p3_same_time_cp10_rgb_bridge_20261029_v1'
AVAILABLE_STAGE_A={
 'prepare':{'path':'results/'+BRIDGE+'_prepare_MAIN01.json','sha256':'f2fac71516c36bdb654a9a84bbc0115d6155c04ed10231bc6b87b99ed0bda846'},
 'report':{'path':'results/'+BRIDGE+'_cuda_MAIN01/report.json','sha256':'b5744f311d3f412cfc5016bf37c8854cc7b5272a6c37e62055c7f3aa68f2c657'},
 'completion':{'path':'results/'+BRIDGE+'_cuda_MAIN01/completion.json','sha256':'37588f050ed9381b191673e27c0b6fcab3ddf0c726e63a54bb75d9579963facf'},
 'receipt':{'path':'results/'+BRIDGE+'_consume_MAIN01.json','sha256':'af67caf8536a9637d709fcfb11c947d993e566bd4de5914725f4640a75de4d6f'}}
BLOCKERS=('final frozen seven-signal/five-U producer source/schema and actual complete dual-replay authority',
          'audited native operation/observation extractor and full public state/RNG/goal/raw mapping')


def need(ok,msg):
    if not ok:raise ValueError(msg)


def raw(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def same(a,b):return raw(a)==raw(b)
def digest(b):return hashlib.sha256(b).hexdigest()
def fields(v,keys):need(type(v) is dict and set(v)==set(keys.split()),'exact fields '+keys)


def decode(b):
    def unique(pairs):
        d={}
        for k,v in pairs:need(k not in d,'duplicate JSON key');d[k]=v
        return d
    v=json.loads(b,object_pairs_hook=unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    raw(v);return v


def sha(p):
    p=Path(p);need(p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,'bounded regular file')
    return digest(p.read_bytes())


def shape_census():
    scan=224*(224*3+1);png=scan+(scan>>12)+(scan>>14)+(scan>>25)+13+1024
    metadata=sum(QUOTAS.values());total=81*png+metadata+RESERVE
    need(total<CAP,'transport proof must fit64MiB; no automatic expansion')
    return dict(jobs=10,per_job_PNG=81,png_upper_bytes=png,PNG_total_upper_bytes=81*png,
        operation_receipts=374,operation_events=748,observation_events=81,guard_events=2,
        lifecycle_events=3,total_events=834,metadata_file_quotas=QUOTAS,metadata_all_copies_upper_bytes=metadata,
        final_reserve_bytes=RESERVE,conservative_total_bytes=total,cap_bytes=CAP,
        quota_scope='all actual serialized copies incl binding, journal, row/report, completion and failure',
        scientific_shape_proof=False,native_extractor_certified=False)


def contract():
    return dict(schema=NAME+'.RAM_contract',fixture_only=True,production_ready=False,
        order=[dict(job=2*i+j,endpoint=i,root=root,source_role=role,probe=probe)
               for i,(root,role) in enumerate(ORDER) for j,probe in enumerate(('U','Z2'))],
        source_control_step=10,source_absolute_step=20,endpoint_control_step=15,endpoint_absolute_step=25,
        warm10=10,role_prefix10=10,probe5=5,independent_env_per_job=True,natural_replay_only=True,
        guard51_before_control6=True,guard66_before_control11=True,CP15_missing_no_padding=True,
        raw='original float32 [-1,1]; no clipping/tanh/reselect; U already mapped by producer exactly once',
        counts_per_complete_job=COUNTS,transport=shape_census(),wall_seconds=WALL,disk_floor_bytes=FLOOR,
        available_stage_A=deepcopy(AVAILABLE_STAGE_A),stage_A_authentication='MAIN supplied; no nested consume executed here',
        blockers=list(BLOCKERS),scientific_admission=False,P1_admission=False,P3_admission=False,full_C3_demonstrated=False)


def f32_bits(actions,n):
    need(type(actions) is list and len(actions)==n and all(type(a) is list and len(a)==2 for a in actions),'raw Nx2')
    flat=[x for a in actions for x in a]
    need(all(type(x) in (int,float) and math.isfinite(x) and -1<=x<=1 for x in flat),'raw finite native domain')
    packed=struct.pack('<'+'f'*len(flat),*flat)
    need(all(float(x)==v for x,v in zip(flat,struct.unpack('<'+'f'*len(flat),packed),strict=True)),'original lossless f32')
    return packed.hex()


def validate_binding(b):
    fields(b,'schema fixture_only job endpoint root source_role probe warm10 raw15 reference66 U_dual_replay')
    need(b['schema']==NAME+'.fixture_binding' and b['fixture_only'] is True,'fixture binding only, never production authority')
    need(type(b['job']) is int and 0<=b['job']<10 and b['endpoint']==b['job']//2,'all10 fixed slots')
    root,role=ORDER[b['endpoint']]
    need((b['root'],b['source_role'],b['probe'])==(root,role,('U','Z2')[b['job']%2]),'fixed endpoint/U-Z2 order')
    f32_bits(b['warm10'],10)
    need(len(b['reference66'])==66,'full66 reference, not just endpoint')
    need([tuple(f[k] for k in ('phase','step','overlay')) for f in b['reference66']]==frame_keys()[:66],'reference full natural chronology')
    for f in b['reference66']:validate_observation(f)
    if b['probe']=='U' and b['U_dual_replay'] is None:
        need(b['raw15'] is None,'unavailable U is null, never zero fallback')
        return
    bits=f32_bits(b['raw15'],15)
    if b['probe']=='Z2':
        need(f32_bits(b['raw15'][10:],5)=='00'*40 and b['U_dual_replay'] is None,'literal positive zero continuation')
    else:
        w=b['U_dual_replay'];fields(w,'raw5_bits whole_IO_sha256 segmented_IO_sha256')
        need(w['raw5_bits']==bits[160:] and w['whole_IO_sha256']==w['segmented_IO_sha256'] and
             type(w['whole_IO_sha256']) is str and len(w['whole_IO_sha256'])==64,'fixture dual full IO equality; NOT producer certification')


def frame_keys():
    return [(phase,step,overlay) for phase,end in (('warmup',10),('plan',15)) for step in range(end+1) for overlay in range(3)]


def png_identity(data):
    need(type(data) is bytes and len(data)<=shape_census()['png_upper_bytes'],'bounded full RGB PNG')
    with Image.open(io.BytesIO(data)) as im:
        need(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'native RGB224 ABI')
        im.load();return digest(im.tobytes())


def validate_observation(f):
    fields(f,'phase step overlay png file_sha256 pixel_sha256 state snapshot raw_history_bits goal')
    need(type(f['state']) is list and type(f['goal']) is list and len(f['state'])==len(f['goal'])==7 and
         all(type(x) in (int,float) and math.isfinite(x) for x in f['state']+f['goal']),'full native7/goal')
    need(type(f['snapshot']) is dict and same(f['snapshot']['state'],f['state']) and
         f['snapshot']['state_float64_bits']==struct.pack('<7d',*f['state']).hex(),'full snapshot state bits')
    need({'rng','public_state','goal'}<=set(f['snapshot']) and same(f['snapshot']['goal'],f['goal']),'explicit public/RNG/goal source witnesses')
    need(Path(f['png']).name==f['png'] and f['png'].endswith('.png'),'flat PNG name')


def comparable(f):return {k:v for k,v in f.items() if k not in ('png','file_sha256')}


class Replay:
    """Pure event consumer/state machine; no model/env callbacks or imports."""
    def __init__(self,b):
        validate_binding(b);self.b=deepcopy(b);self.ops=[];self.pending={};self.frames=[];self.guards=[];self.errors=[]
        self.counts={k:dict(attempted=0,completed=0,failed=0) for k in COUNTS};self.started=False;self.closed=False;self.finished=False
        self.native_stop=False;self.used_renders=set();self.guard_attempts=[]

    def proof(self,n):
        need(len(self.frames)==n and same([comparable(x) for x in self.frames],
             [comparable(x) for x in self.b['reference66'][:n]]),'complete exact prefix/RNG/goal/public/history witness')
        step=5 if n==51 else 10
        returns=[x for x in self.ops if x['dispatch']['resource']=='public_step' and x['dispatch']['phase']=='plan' and x['dispatch']['step']==step and x['status']=='returned']
        need(len(returns)==1 and same(returns[0]['result']['state'],self.frames[-1]['state']),'guard actual native return/state')
        need(self.frames[-1]['raw_history_bits']==f32_bits(self.b['warm10'],10)+f32_bits(self.b['raw15'][:step],step),'actual ordered raw history')
        return dict(observations=n,control_step=step,absolute_step=10+step,exact=True,
            reference_sha256=digest(raw(self.b['reference66'][:n])),actual_sha256=digest(raw(self.frames)))

    def feed(self,kind,v):
        need(not self.finished,'no events after terminal')
        if kind=='run_started':
            fields(v,'binding_sha256');need(not self.started and v['binding_sha256']==digest(raw(self.b)),'single bound start');self.started=True;return
        need(self.started,'start first')
        if kind=='operation_dispatch':
            fields(v,'id resource phase step overlay action');k=v['resource']
            need(not self.closed and k in COUNTS and v['id']==len(self.ops) and v['id'] not in self.pending,'ordered unique operation intent')
            need(self.b['raw15'] is not None,'unverified U cannot construct env or dispatch')
            need(self.counts[k]['attempted']<COUNTS[k],'fixed paid per-resource cap; partial not refunded')
            if k=='constructor':need(v['id']==0,'fresh constructor first, never shared/hydrated env')
            else:need(self.counts['constructor']['completed']==1,'original env construction must return')
            if k in ('reset','set_state','goal_render','goal_bind','goal_setter'):
                need(v['phase']=='setup' and not self.frames and self.counts['public_step']['attempted']==0,'no poststate reset/hydration or goal mutation')
            if k=='public_step':
                need(not self.native_stop and not self.errors,'no native stop/failure suppression')
                need(v['phase'] in ('warmup','plan') and type(v['step']) is int,'actual public step')
                prior=[o for o in self.ops if o['dispatch']['resource']=='public_step' and o['dispatch']['phase']==v['phase']]
                need(v['step']==len(prior)+1,'no skipped/repeated/padded public action')
                actions=self.b['warm10'] if v['phase']=='warmup' else self.b['raw15']
                need(1<=v['step']<=len(actions) and f32_bits([v['action']],1)==f32_bits([actions[v['step']-1]],1),'exact dispatched raw action')
                if v['phase']=='plan' and v['step']>=6:need(51 in self.guards,'durable51 guard BEFORE control6 intent')
                if v['phase']=='plan' and v['step']>=11:need(66 in self.guards,'durable66 guard BEFORE control11 intent')
            else:need(v['action'] is None,'no hidden action')
            self.counts[k]['attempted']+=1;self.pending[v['id']]=deepcopy(v)
            self.ops.append(dict(dispatch=deepcopy(v),status='pending',result=None,error=None));return
        if kind in ('operation_returned','operation_failed'):
            fields(v,'id result error');need(v['id'] in self.pending,'matched original intent')
            dispatch=self.pending.pop(v['id']);op=self.ops[v['id']];op['result']=deepcopy(v['result']);op['error']=deepcopy(v['error'])
            need((v['error'] is None)==(kind=='operation_returned'),'exact returned/failed semantics')
            key='completed' if kind=='operation_returned' else 'failed';self.counts[dispatch['resource']][key]+=1
            op['status']='returned' if kind=='operation_returned' else 'failed'
            if kind=='operation_returned' and dispatch['resource']=='public_step':
                result=v['result'];need(type(result) is dict and {'state','terminated','truncated'}<=set(result) and
                    type(result['terminated']) is type(result['truncated']) is bool,'actual native public return flags')
                self.native_stop=result['terminated'] or result['truncated']
            if kind=='operation_failed':self.errors.append(deepcopy(v['error']))
            return
        if kind=='observation':
            validate_observation(v);need(len(self.frames)<81 and tuple(v[k] for k in ('phase','step','overlay'))==frame_keys()[len(self.frames)],'full81 ordered observations')
            expected=f32_bits(self.b['warm10'][:v['step']],v['step']) if v['phase']=='warmup' else f32_bits(self.b['warm10'],10)+f32_bits(self.b['raw15'][:v['step']],v['step'])
            need(v['raw_history_bits']==expected,'each actual frame ordered raw history')
            renders=[o for o in self.ops if o['status']=='returned' and o['dispatch']['resource']=='explicit_render' and
                all(o['dispatch'][k]==v[k] for k in ('phase','step','overlay')) and o['dispatch']['id'] not in self.used_renders]
            need(len(renders)==1,'one actual render return per observation')
            need(type(renders[0]['result']) is dict and renders[0]['result'].get('pixel_sha256')==v['pixel_sha256'],'render result pixel identity')
            self.used_renders.add(renders[0]['dispatch']['id'])
            if v['step']:
                returns=[o for o in self.ops if o['status']=='returned' and o['dispatch']['resource']=='public_step' and
                    o['dispatch']['phase']==v['phase'] and o['dispatch']['step']==v['step']]
                need(len(returns)==1 and same(returns[0]['result']['state'],v['state']),'actual return/frame same time state')
            self.frames.append(deepcopy(v));return
        if kind=='guard':
            fields(v,'n accepted proof error');n=v['n']
            need(n in (51,66) and n not in self.guard_attempts and (n!=66 or 51 in self.guards),'both ordered guards, no retry')
            self.guard_attempts.append(n)
            if v['accepted'] is True:
                need(v['error'] is None and same(v['proof'],self.proof(n)),'full rebuilt guard proof');self.guards.append(n)
            else:
                need(v['accepted'] is False and v['proof'] is None and v['error'] is not None,'guard failure retained');self.errors.append(v['error'])
            return
        if kind=='branch_closed':
            fields(v,'status reason');need(not self.closed and v['status'] in ('completed','native_stopped','engineering_failed','blocked'),'one closed branch')
            if v['status']=='completed':
                need(len(self.frames)==81 and self.guards==[51,66] and not self.pending and not self.errors and
                     all(self.counts[k]==dict(attempted=n,completed=n,failed=0) for k,n in COUNTS.items()),'complete unpadded full native accounting')
            else:need(v['reason'] is not None,'noncomplete explicit reason')
            self.closed=deepcopy(v);return
        if kind=='run_finished':
            fields(v,'closed');need(self.closed and same(v['closed'],self.closed),'exact terminal branch');self.finished=True;return
        raise ValueError('unknown event kind')

    def report(self):
        endpoint=next((f for f in self.frames if (f['phase'],f['step'],f['overlay'])==('plan',15,0)),None)
        return dict(schema=NAME+'.fixture_report',fixture_only=True,production_ready=False,binding_sha256=digest(raw(self.b)),
            job=self.b['job'],counts=deepcopy(self.counts),operations=deepcopy(self.ops),observations=deepcopy(self.frames),
            guards=list(self.guards),errors=deepcopy(self.errors),pending=[dict(id=k,dispatch=deepcopy(v)) for k,v in sorted(self.pending.items())],closed=deepcopy(self.closed),finished=self.finished,
            guard_attempts=list(self.guard_attempts),
            native_stopped=self.native_stop,
            endpoint_control=15,endpoint_absolute=25,endpoint=deepcopy(endpoint),missing_endpoint_reason=None if endpoint else ('not_finished' if not self.closed else self.closed['status']),
            native_metrics=None,unknown='no native metric or scientific claim generated by transport schema',
            scientific_admission=False,P1_admission=False,P3_admission=False,FLOPs=None)


class Store:
    """Exclusive bounded durable RAM-fixture spool. Never a native runner."""
    def __init__(self,path,b,clock=time.monotonic):
        validate_binding(b);self.path=Path(path);self.clock=clock;self.start=clock();self.tail=None;self.seq=0
        need(self.path.name.startswith(NAME+'_') and not self.path.exists() and self.path.parent.is_dir() and
             shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new job64MiB +floor200MiB')
        self.path.mkdir();self._fsync_dir();self.put('binding.json',raw(b));self.replay=Replay(b)

    def _fsync_dir(self):
        fd=os.open(self.path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

    def put(self,name,data,*,append=False,final=False):
        need(Path(name).name==name and type(data) is bytes,'flat exact bytes')
        files=list(self.path.iterdir());need(all(p.is_file() and not p.is_symlink() for p in files),'no foreign subtrees/symlinks')
        png=name.endswith('.png')
        if png:need(not append and len(data)<=shape_census()['png_upper_bytes'] and sum(p.suffix=='.png' for p in files)<81,'all81 bounded PNG')
        else:
            need(name in QUOTAS and append==(name=='journal.jsonl'),'exact metadata transport formats')
            prior=(self.path/name).stat().st_size if (self.path/name).exists() else 0
            need(prior+len(data)<=QUOTAS[name],'aggregate per-file metadata quota')
        need(sum(p.stat().st_size for p in files)+len(data)<=CAP-(0 if final else RESERVE) and
             shutil.disk_usage(self.path).free>=FLOOR+len(data),'job cap/reserve/floor')
        need(final or self.clock()-self.start<=WALL,'whole900 wall')
        with (self.path/name).open('ab' if append else 'xb') as f:need(f.write(data)==len(data),'full write');f.flush();os.fsync(f.fileno())
        self._fsync_dir()

    def emit(self,kind,payload):
        # Validate against a copy; accepted guard becomes usable only AFTER fsync returns.
        trial=deepcopy(self.replay);trial.feed(kind,payload)
        e=dict(seq=self.seq,previous=self.tail,kind=kind,payload=payload);e['sha256']=digest(raw(e))
        self.put('journal.jsonl',raw(e),append=True)
        self.replay=trial;self.tail=e['sha256'];self.seq+=1

    def observation(self,frame,data):
        need(digest(data)==frame['file_sha256'] and png_identity(data)==frame['pixel_sha256'],'full PNG/pixel identity')
        self.put(frame['png'],data);self.emit('observation',frame)

    def finalize(self):
        report=self.replay.report();self.put('report.json',raw(report),final=True)
        inventory={p.name:sha(p) for p in self.path.iterdir()}
        elapsed=self.clock()-self.start
        completion=dict(schema=NAME+'.fixture_completion',fixture_only=True,inventory=inventory,
            journal_tail=self.tail,event_count=self.seq,wall_through_report_hash=elapsed,within_wall=elapsed<=WALL,
            closed=self.replay.closed,finished=self.replay.finished)
        self.put('completion.json',raw(completion),final=True)
        end=self.clock()-self.start
        if end>WALL:
            self.put('failure.json',raw(dict(stage='completion_publication',wall=end,receipt=None)),final=True)
            raise ValueError('completion900 wall failed; artifacts retained')
        return dict(report_sha256=sha(self.path/'report.json'),completion_sha256=sha(self.path/'completion.json'),fixture_only=True)


def consume_fixture(path,report_sha,completion_sha):
    path=Path(path);files=list(path.iterdir());need(all(p.is_file() and not p.is_symlink() for p in files),'flat fixture artifacts')
    need(sum(p.stat().st_size for p in files)<=CAP and 'failure.json' not in {p.name for p in files},'cap/no publication failure')
    need(sha(path/'report.json')==report_sha and sha(path/'completion.json')==completion_sha,'actual external two SHA')
    completion=decode((path/'completion.json').read_bytes())
    fields(completion,'schema fixture_only inventory journal_tail event_count wall_through_report_hash within_wall closed finished')
    need(completion['schema']==NAME+'.fixture_completion' and completion['fixture_only'] is True and completion['within_wall'] is True and
         0<=completion['wall_through_report_hash']<=WALL,'fixture complete publication bound')
    inventory={p.name:sha(p) for p in files if p.name!='completion.json'}
    need(same(inventory,completion['inventory']),'entire file inventory exact')
    for name in inventory:
        need(name.endswith('.png') or name in QUOTAS,'no hidden format')
        if name in QUOTAS:need((path/name).stat().st_size<=QUOTAS[name],'whole metadata quota')
    b=decode((path/'binding.json').read_bytes());replay=Replay(b);tail=None;events=0
    with (path/'journal.jsonl').open('rb') as stream:
        for line in stream:
            e=decode(line);fields(e,'seq previous kind payload sha256')
            need(raw(e)==line and e['seq']==events and e['previous']==tail,'full canonical sequence/chain')
            unsigned={k:v for k,v in e.items() if k!='sha256'};need(e['sha256']==digest(raw(unsigned)),'event full-field SHA')
            replay.feed(e['kind'],e['payload']);tail=e['sha256'];events+=1
    pngs=[]
    for frame in replay.frames:
        p=path/frame['png'];need(sha(p)==frame['file_sha256'] and png_identity(p.read_bytes())==frame['pixel_sha256'],'all saved full PNGs')
        pngs.append(frame['png'])
    need(len(set(pngs))==len(pngs) and set(pngs)=={p.name for p in files if p.suffix=='.png'},'no missing/extra/duplicate PNG')
    report=decode((path/'report.json').read_bytes());need(same(report,replay.report()),'entire report all fields independently rebuilt')
    need(completion['journal_tail']==tail and completion['event_count']==events and same(completion['closed'],replay.closed) and
         completion['finished'] is replay.finished,'entire terminal metadata reconstructed')
    status='partial_unfinished' if not replay.finished else replay.closed['status']
    return dict(schema=NAME+'.fixture_consume',fixture_only=True,production_ready=False,status=status,report_sha256=report_sha,
        completion_sha256=completion_sha,report=report,event_count=events,PNG=len(pngs),scientific_admission=False,
        new_neural_calls=0,new_physics_calls=0,independent_fullstrict=False)


def consume_batch_fixture(entries):
    """All ten slots, including blocked/failed/null; never an actual batch authority."""
    need(type(entries) is list and len(entries)==10,'all10 job denominator required')
    rows=[];outputs=set()
    for i,e in enumerate(entries):
        fields(e,'job output report_sha256 completion_sha256')
        need(e['job']==i and str(Path(e['output']).resolve()) not in outputs,'ordered ten independent job outputs')
        outputs.add(str(Path(e['output']).resolve()))
        row=consume_fixture(e['output'],e['report_sha256'],e['completion_sha256'])
        need(row['report']['job']==i,'same job/endpoint/probe, no old trajectory relabel')
        rows.append(row)
    totals={k:{axis:sum(r['report']['counts'][k][axis] for r in rows) for axis in ('attempted','completed','failed')} for k in COUNTS}
    return dict(schema=NAME+'.fixture_batch',fixture_only=True,production_ready=False,fixed_denominator=10,
        rows=rows,counts=totals,missing_endpoints=sum(r['report']['endpoint'] is None for r in rows),
        independent_roots=None,scientific_admission=False,new_neural_calls=0,new_physics_calls=0)


def refuse_production():
    raise ValueError('production prepare/run unavailable; no artifact written: '+'; '.join(BLOCKERS))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('describe','prepare','run','consume-fixture'))
    p.add_argument('--output',type=Path);p.add_argument('--report-sha');p.add_argument('--completion-sha');a=p.parse_args(argv)
    if a.command in ('prepare','run'):refuse_production()
    elif a.command=='describe':print(raw(contract()).decode())
    else:
        r=consume_fixture(a.output,a.report_sha,a.completion_sha)
        compact={k:r[k] for k in ('schema','fixture_only','production_ready','status','report_sha256','completion_sha256','event_count','PNG')}
        compact['counts']=r['report']['counts'];print(raw(compact).decode())


if __name__=='__main__':main()
