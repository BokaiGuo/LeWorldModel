"""Fixed 2x164 normalizer numerical ablation. MAIN HOST CUDA only, never deployment."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_observed_raw_coordinate_fd_20261028_v1 as FD

NAME='agent_raw_normalizer_ablation_20261028_v1'
AUTH='MAIN_FIXED328_NORMALIZER_ABLATION_HOST_CUDA_ONLY'
ROOT=FD.ROOT
F,core,signal,transport=FD.F,FD.core,FD.signal,FD.transport
require,sha,digest,raw=FD.require,FD.sha,FD.digest,FD.raw
ARMS=('public','archive_sample_stats')
PREP=ROOT/'results'/f'{FD.NAME}_prepare_MAIN01.json'
PREP_SHA='a1a87327d230af7e02e2379909305c9fbbfb8a05768ba85eb4fcbdac85f95ef5'
UP=ROOT/'results'/f'{FD.NAME}_cuda_MAIN01'
UP_SHA='df21e0cd0a235a6e6f43572cc02614f8c62b4704bac2aabb8ec4e2886104aa15'
STRICT=ROOT/'results'/f'{FD.NAME}_consume_MAIN01.json'
STRICT_SHA='c2c454cf51b11e5f28df12e2c2f9ef12c8c7cd1eba7bf99b9a9cf9c97977e2b0'
STATS=ROOT/'results/pusht_d2p_archive_action_statistics_v1_20261026.json'
STATS_SHA='c816c535e08a946a6f7464c659b23e98022b0375359bf5a7add4edd6e018cdda'
PROTOCOL=ROOT.parent/'docs/D2P_FIXED_NORMALIZER_ABLATION_PROTOCOL_MAIN_20261028_v1.md'
PROTOCOL_SHA='8f602e5228c7f4de0d5aa693ff669ec0c9e91b8b78e8269f8ffe1b1665689fdc'
REAL=ROOT/'results/agent_observed_raw_coordinate_rgb_20261028_v1_consume_MAIN01.json'
REAL_SHA='d6b6ed08732d7595e740f98e0dc2450adf439e726aa3078854badb78f7d8b4b2'
FD_SOURCE_SHA='83bf2420f5555f892ec1222bd158dde0634b096fa19f8a4bc626a2e07347970e'
META=64*1024**2


def contract():
    return dict(schema=NAME+'.contract',arms=list(ARMS),fixed_denominator_per_arm=164,forwards=328,pairs_per_arm=80,
        order='arm public then archive_sample_stats; panel0..3; all41 original job order',
        max_new_compute=F.caps(328,0),new_encoder=0,new_AD=0,new_physics=0,new_training=0,new_selection=0,
        incoming='regenerate raw10 chronological float32 (raw-mean)/std reshape(2,10); never old token renormalization',
        outgoing='literal raw suffix5 float32 (raw-mean)/std; original native predict/full3 shift',
        public_controls='ALL164 fullcontexts AND native objective arrays exact against complete original output; no tolerance',
        archive_identity='UNAUTHENTICATED_TRAINING_LINEAGE_DIAGNOSTIC_ONLY',
        stats_choice='preexisting action-only statistics.mean/sample_std; one float32 cast; not fitted',
        real_input='post-run consumer ONLY; pinned entire RGB strict receipt; absent from forward/prepare semantic inputs',
        wall_seconds=1800,output_bytes=128*1024**2,final_reserve_bytes=16*1024**2,disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        CUDA_only=True,CPU_fallback=False,TF32=False,autocast=False,SDP='MATH',retries=0,
        P1_admission=False,freshness=False,training_provenance=False,deployment_allowed=False,scientific_admission=False,
        physical_release=False,equal_FLOPs=False,total_FLOPs=None)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
def same(a,b):return raw(a)==raw(b)
def eq(a,b):
    require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'exact float32 shape/dtype/bits')


def arm_specs(task,stats):
    public=[FD.packet(task[k],(2,)).tolist() for k in ('action_mean','action_std')]
    st=stats['statistics'];archive=[]
    for key in ('mean','sample_std'):
        a=np.asarray(st[key],np.float64);require(a.shape==(2,) and np.isfinite(a).all(),'finite fixed archive statistic')
        b=a.astype(np.float32);require(np.isfinite(b).all(),'archive float32 cast once');archive.append(b.tolist())
    require(np.all(np.asarray(public[1])>0) and np.all(np.asarray(archive[1])>0),'positive std')
    return [dict(name=name,mean=values[0],std=values[1],training_provenance=False,deployment_allowed=False,
        identity='public_literal_values_not_training_certificate' if i==0 else 'UNAUTHENTICATED_TRAINING_LINEAGE_DIAGNOSTIC_ONLY')
        for i,(name,values) in enumerate(zip(ARMS,(public,archive),strict=True))]


def packet_for(job,arm):
    require(arm['name'] in ARMS,'only two fixed arms')
    mean,std=(FD.packet(arm[k],(2,)) for k in ('mean','std'));require(np.all(std>0),'positive stats')
    incoming=FD.packet(job['observed_raw10'],(10,2));a=FD.packet(job['raw_suffix5'])
    require(np.all(abs(a)<=1),'literal legal suffix box, never clip')
    history=((incoming-mean)/std).reshape(2,10);token=((a-mean)/std).reshape(10)
    return dict(latent=FD.packet(job['observed_context']['latent_history'],(3,192)),incoming_raw10=incoming,
        history=history,raw=a,mean=mean,std=std,token=token,outgoing=np.concatenate((history,token[None]),axis=0))


def input_context(job,arm):
    c=deepcopy(job['observed_context']);c['action_history']=packet_for(job,arm)['history'].tolist();return c


def original_values(report):
    require(len(report['rows'])==4,'all4 original panels')
    values=[]
    for i,row in enumerate(report['rows']):
        require(row['index']==i and row['status']=='recorded' and len(row['values'])==41,'all original41 recorded')
        for k,value in enumerate(row['values']):
            require(value['local_index']==k and value['status']=='returned','original actual-return roster')
            values.append(deepcopy(value))
    return values


def validate_roster(p):
    require(p['schema']==NAME+'.prepare' and same(p['contract'],contract()) and len(p['jobs'])==len(p['reference'])==164
        and len(p['arms'])==2 and p['manifest_sha256']==digest(p['jobs']),'complete immutable164x2 prepare')
    require(same(p['arms'],arm_specs(p['jobs'][0]['task'],p['statistics_report'])),'only preexisting literal stats')
    schedule=[(None,None,0)]+[(c,h,s) for h in FD.STEPS for c in range(10) for s in (-1,1)]
    for i,(j,ref) in enumerate(zip(p['jobs'],p['reference'],strict=True)):
        panel,local=divmod(i,41);coord,h,sign=schedule[local]
        require((j['index'],j['slot'],j['local_index'])==(i,panel,local) and (j['coord'],j['h'],j['sign'])==(coord,h,sign)
            and j['job_sha256']==digest({k:v for k,v in j.items() if k!='job_sha256'}) and j['legal'] is True
            and j['missing_reason'] is None and ref['job_id']==j['id'],'all jobs order/hash/legal/original references')
        require(same(arm_specs(j['task'],p['statistics_report']),p['arms']),'same normalizers all fixed tasks')
        core.validate_task(j['task']);packet_for(j,p['arms'][1])
        require(same(input_context(j,p['arms'][0]),j['observed_context']),'public BOTH incoming tokens exact original layout/bits')
        # Explicit actual raw5/input/output shape check, not a native route substitute.
        FD.packet(ref['context']['latent_history'],(3,192));FD.packet(ref['context']['action_history'],(2,10))
        FD.scalar_fd.check_objective(ref['objective']['arrays'],ref['context']['latent_history'][-1],j['task']['goal'])


def build():
    for f,h in ((PREP,PREP_SHA),(UP/'report.json',UP_SHA),(STRICT,STRICT_SHA),(STATS,STATS_SHA),
                (PROTOCOL,PROTOCOL_SHA),(Path(FD.__file__).resolve(),FD_SOURCE_SHA)):
        require(sha(f)==h,'fixed actual authority SHA '+str(f))
    accepted=FD.consume(UP,UP_SHA)
    require(accepted['complete'] is True and same(accepted,signal.read(STRICT,META))
        and accepted['actual_new_compute']['attempted']==accepted['actual_new_compute']['completed']==F.caps(164,0),
        'entire original164 strict reconstructed and persisted receipt exact')
    f=signal.read(PREP,META);r=signal.read(UP/'report.json',META);stats=signal.read(STATS,META)
    require(stats['schema']=='pusht_d2p_archive_action_statistics.v1' and stats['status']=='completed','actual fixed action-only stats')
    bindings=deepcopy(f['bindings'])
    def add(items):
        for path,h in items.items():require(path not in bindings or bindings[path]==h,'source closure collision');bindings[path]=h
    add({str(x):h for x,h in ((PREP,PREP_SHA),(UP/'report.json',UP_SHA),(STRICT,STRICT_SHA),(STATS,STATS_SHA),(PROTOCOL,PROTOCOL_SHA))})
    add({str(UP/n):h for n,h in r['artifact_sha256'].items()})
    add({str(x):sha(x) for x in own_files()})
    installed=F.installed_class_sources();add({v['path']:v['sha256'] for v in installed.values()});F.check_bindings(bindings)
    p=dict(schema=NAME+'.prepare',contract=contract(),jobs=deepcopy(f['jobs']),manifest_sha256=f['manifest_sha256'],
        arms=arm_specs(f['jobs'][0]['task'],stats),statistics_report=stats,reference=original_values(r),bindings=bindings,
        policy=f['policy'],expected_model_hash=f['expected_model_hash'],original_model_before=r['model_before'],
        original_model_after=r['model_after'],installed_class_sources=installed,
        upstream=dict(prepared=str(PREP),prepare_sha256=PREP_SHA,output=str(UP),report_sha256=UP_SHA,
            receipt=str(STRICT),receipt_sha256=STRICT_SHA),
        protocol=dict(path=str(PROTOCOL),sha256=PROTOCOL_SHA),
        deferred_real_RGB_authority=dict(path=str(REAL),sha256=REAL_SHA),
        inherited_compute=accepted['actual_new_compute'],inherited_resources=f['inherited_resources'])
    validate_roster(p);return p


def verify(prepared,expected):
    core.hash_value(expected);require(sha(prepared)==expected,'own prepare SHA');p=signal.read(prepared,META)
    require(same(p,build()) and sha(prepared)==expected,'full original authority/manifest/stats/protocol reconstruction');return p


def write(path,value):
    path=Path(path).absolute();body=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve()==path and not path.exists()
        and len(body)<=META and shutil.disk_usage(path.parent).free>=contract()['disk_floor_bytes']+len(body),'exclusive bounded metadata')
    with path.open('xb') as f:require(f.write(body)==len(body),'full metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


class Store(transport.Store):
    def __init__(self,path,started):
        self.path=Path(path).absolute();self.started=started;self.finalizing=False
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and self.path.resolve()==self.path
            and not self.path.exists() and shutil.disk_usage(self.path.parent).free>=contract()['output_bytes']+contract()['disk_floor_bytes'],
            'new128MiB output +200MiB floor')
        self.path.mkdir();signal.fsync_dir(self.path.parent)
    def namespace(self,name):
        require(name in {f'a{a}_p{i}' for a in range(2) for i in range(4)},'fixed8 native journals')
        path=self.path/name;path.mkdir(exist_ok=False);signal.fsync_dir(self.path);return path
    def check(self,size=4096):
        require(self.finalizing or time.monotonic()-self.started<contract()['wall_seconds'],'whole1800s')
        files=[f for f in self.path.rglob('*') if f.is_file()]
        require(sum(f.stat().st_size for f in files)+size<=contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
            and shutil.disk_usage(self.path).free>=contract()['disk_floor_bytes']+size,'128MiB cap/reserve/200MiB floor')
    def put(self,name,value,final=False):return self.write(name,raw(value),final=final)


class Backend:
    """Source-faithful full3 forward; original task untouched; local IO hooks only."""
    def __init__(self,model,device,task,arm,journal,ledger,check):
        require(getattr(device,'type',None)=='cuda','HOST CUDA only')
        import torch
        self.torch=torch;self.device=device;self.task=task;self.arm=arm;self.check=check;self.serial=0;self.y=None
        self.o=F.native.NativeProgram(model,device,task,journal);self.o.ledger=ledger
    @contextmanager
    def observe(self):
        with self.o.observe():yield self
    def tensor(self,value):return self.torch.tensor(value,device=self.device,dtype=self.torch.float32)
    def step(self,job):
        t=self.torch;o=self.o;self.check();c=input_context(job,self.arm);p=packet_for(job,self.arm)
        o.phase=f'f{self.serial:03}';self.serial+=1;o.step=0;o.ledger.scope=dict(forward=o.phase,label=job['id'])
        o.store.event(dict(kind='forward_started',phase=o.phase,label=job['id'],input=deepcopy(c),commands=deepcopy(job['raw_suffix5'])))
        z,h=(self.tensor(c[k])[None] for k in ('latent_history','action_history'))
        a=self.tensor(job['raw_suffix5']);mean,std=(self.tensor(self.arm[k]) for k in ('mean','std'))
        incoming=self.tensor(job['observed_raw10'])
        # History was regenerated once in NumPy float32, exactly the original input route.
        # Suffix arithmetic is the original CUDA expression, not a new host kernel.
        token=((a-mean)/std).reshape(10);outgoing=t.cat((h,token[None,None]),1)
        o.arrays('program_packet',latent=z,history=h,raw=a,mean=mean,std=std,token=token,outgoing=outgoing,incoming_raw10=incoming)
        emb=o.model.action_encoder(outgoing)
        predicted=o.call('predict',lambda:o.model.predict(z,emb),dict(latent=z,act_emb=emb),shape=list(z.shape))
        require(tuple(predicted.shape)==(1,3,192) and bool(t.isfinite(predicted).all()),'original native full3 output, no shape adaptation')
        z,h=t.cat((z[:,1:],predicted[:,-1:]),1),outgoing[:,1:]
        o.arrays('program_boundary',latent=z,history=h)
        after=dict(c,latent_history=z[0].detach().cpu().tolist(),action_history=h[0].detach().cpu().tolist())
        self.y=z[0,-1];o.store.event(dict(kind='forward_finished',phase=o.phase,context=after));self.check();return after
    def objective(self,after):
        t=self.torch
        require(self.y is not None and self.y.dtype==t.float32 and tuple(self.y.shape)==(192,),'native terminal tensor')
        eq(self.y.detach().cpu().numpy(),FD.packet(after['latent_history'][-1],(192,)))
        goal=self.tensor(self.task['goal']);residual=self.y-goal;square=residual**2;loss=square.mean()
        require(not loss.requires_grad and loss.ndim==0,'native float32 scalar no AD')
        return {k:signal.blob(v.detach().cpu().numpy()) for k,v in
            dict(endpoint=self.y,goal=goal,residual=residual,square=square,loss=loss.reshape(1)).items()}


def new_row(arm,panel):
    return dict(arm=arm,panel=panel,status='not_started',values=[],resources=None,
        journal_file=None,journal_sha256=None,journal_tail=None)


def public_exact(job,value,reference):
    require(value['job_id']==reference['job_id']==job['id'] and same(value['context'],reference['context'])
        and same(value['objective'],reference['objective']),'ALL164 public original fullcontext/objective EXACT; no relaxation')


def evaluate(jobs,arm,backend,row,journal,reference):
    require(len(jobs)==len(reference)==41,'whole41 fixed roster')
    row['values']=[dict(index=j['index'],local_index=j['local_index'],job_id=j['id'],job_sha256=j['job_sha256'],
        status='not_started',context=None,objective=None) for j in jobs]
    row['status']='attempted'
    for j,v,ref in zip(jobs,row['values'],reference,strict=True):
        journal.event(dict(kind='request_intent',arm=arm['name'],index=j['index'],job_id=j['id'],
            job_sha256=j['job_sha256'],input_sha256=digest(input_context(j,arm)),stats_sha256=digest(arm)))
        v['status']='entered';after=backend.step(j);arrays=backend.objective(after)
        diagnostic=FD.scalar_fd.check_objective(arrays,after['latent_history'][-1],j['task']['goal'])
        v.update(status='returned',context=after,objective=dict(arrays=arrays,diagnostic=diagnostic))
        journal.event(dict(kind='request_return',arm=arm['name'],index=j['index'],value_sha256=digest(v)))
        if arm['name']=='public':public_exact(j,v,ref)
    row['status']='recorded'


def dispatch(p,store,rows,check,factory):
    validate_roster(p);require(rows==[new_row(a,i) for a in range(2) for i in range(4)],'fresh full8 row roster')
    for row in rows:
        a,i=row['arm'],row['panel'];jobs=p['jobs'][i*41:(i+1)*41];arm=p['arms'][a]
        journal=transport.NativeJournal(store,f'a{a}_p{i}',check);ledger=core.Ledger(F.caps(41,0),sink=journal.event)
        row['status']='attempted'
        try:
            backend=factory(jobs[0]['task'],arm,journal,ledger)
            with backend.observe():evaluate(jobs,arm,backend,row,journal,p['reference'][i*41:(i+1)*41])
            require(ledger.attempted==ledger.completed==F.caps(41,0),'actual41 predict/AE/predictor/predproj, zeroAD/encoder')
        finally:
            if row['status']!='recorded':row['status']='failed_partial'
            row['resources']=ledger.receipt();journal.close()
            row.update(journal_file=f'a{a}_p{i}/events.jsonl',journal_sha256=sha(journal.path),journal_tail=journal.tail)
            store.put(f'a{a}_p{i}_row.json',row,final=True)


def audit_forward(job,arm,value,events,phase,receipt):
    """Independent saved full3 wiring proof; no task/stat identity substitution."""
    local=[e for e in events if e.get('phase')==phase]
    def only(kind):
        es=[e for e in local if e['kind']==kind];require(len(es)==1,'exact native phase record '+kind);return es[0]
    st,en=only('forward_started'),only('forward_finished');c=input_context(job,arm);p=packet_for(job,arm)
    require(st['label']==job['id'] and same(st['input'],c) and same(st['commands'],job['raw_suffix5'])
        and same(en['context'],value['context']),'all actual call input/output contexts and literal raw')
    packet,boundary=only('program_packet'),only('program_boundary')
    require(set(packet['arrays'])==set(p) and set(boundary['arrays'])=={'latent','history'},'whole packet/boundary exact array set')
    for k,v in p.items():
        want=v[None] if k in ('latent','history','outgoing') else v
        eq(signal.unblob(packet['arrays'][k]),want)
    scope=dict(forward=phase,label=job['id'])
    ops=[op for op in receipt['events'] if op['scope']==scope]
    require([op['resource'] for op in ops]==['action_encoder','predict','predictor','pred_proj']
        and all(op['status']=='returned' for op in ops),'exact native forward module roster')
    pos={id(e):i for i,e in enumerate(events)}
    linked={}
    for op in ops:
        es=[e for e in local if e['kind'] in ('io_input','io_return') and e['operation_index']==op['index']]
        require([e['kind'] for e in es]==['io_input','io_return'],'each actual module has original buffers')
        for e in es:
            require(e['scope']==scope and e['step']==0,'actual IO scope/step')
            for b in e['arrays'].values():require(signal.unblob(b).dtype==np.float32,'native float32 IO only')
        linked[op['resource']]=es
        attempt=[e for e in events if e['kind']=='resource_attempt' and e['entry']['index']==op['index']]
        ret=[e for e in events if e['kind']=='resource_return' and e['entry']['index']==op['index']]
        require(len(attempt)==len(ret)==1 and pos[id(st)]<pos[id(attempt[0])]<pos[id(es[0])]<pos[id(es[1])]<pos[id(ret[0])]<pos[id(en)],
            'module actual entry/IO/return between call bookends')
    def arr(resource,which,key):
        arrays=linked[resource][which]['arrays']
        return signal.unblob(arrays[resource+':'+key])
    for resource,keys in (('action_encoder',('0',)),('predict',('latent','act_emb')),('predictor',('0','1')),('pred_proj',('0',))):
        require(set(linked[resource][0]['arrays'])=={resource+':'+k for k in keys}
            and set(linked[resource][1]['arrays'])=={resource+':output'},'exact full native ABI input/output key sets')
    z=p['latent'][None];out=p['outgoing'][None]
    eq(arr('action_encoder',0,'0'),out)
    emb=arr('action_encoder',1,'output')
    for resource,key in (('predict','latent'),('predictor','0')):eq(arr(resource,0,key),z)
    for resource,key in (('predict','act_emb'),('predictor','1')):eq(arr(resource,0,key),emb)
    pred=arr('predictor',1,'output')
    require(pred.shape==(1,3,192) and emb.shape==(1,3,192),'original predictor/AE full3 shape, no structural adaptation')
    eq(arr('pred_proj',0,'0'),pred.reshape(3,192))
    projected=arr('pred_proj',1,'output');require(projected.shape==(3,192),'original predproj flat rows')
    predicted=arr('predict',1,'output');eq(predicted,projected.reshape(1,3,192))
    shifted=np.concatenate((z[:,1:],predicted[:,-1:]),axis=1);history=out[:,1:]
    eq(signal.unblob(boundary['arrays']['latent']),shifted);eq(signal.unblob(boundary['arrays']['history']),history)
    expected=dict(c,latent_history=shifted[0].tolist(),action_history=history[0].tolist())
    require(same(en['context'],expected),'exact full3 shifted latent/action context')
    for e in (packet,boundary):
        require(e['scope']==scope and e['step']==0 and e['operation_index'] is None,'native program packet scope')
    require(pos[id(st)]<pos[id(packet)]<pos[id(linked['action_encoder'][0])]
        and pos[id(linked['predict'][1])]<pos[id(boundary)]<pos[id(en)],'packet/native/boundary chronology')
    # Exact complete phase sequence includes nested predict -> predictor -> pred_proj.
    want=['forward_started','program_packet','resource_attempt','io_input','io_return','resource_return',
        'resource_attempt','io_input','resource_attempt','io_input','io_return','resource_return',
        'resource_attempt','io_input','io_return','resource_return','io_return','resource_return','program_boundary','forward_finished']
    related=[e for e in events if e.get('phase')==phase or e['kind'].startswith('resource_') and e['entry']['scope']==scope]
    require([e['kind'] for e in related]==want,'full native route exact chronology, no unassigned extra IO')
    check=FD.scalar_fd.check_objective(value['objective']['arrays'],expected['latent_history'][-1],job['task']['goal'])
    require(same(check,value['objective']['diagnostic']),'exact objective arithmetic diagnostic')


def audit_row(p,row,events):
    a,i=row['arm'],row['panel'];jobs=p['jobs'][41*i:41*(i+1)];arm=p['arms'][a]
    require(type(a) is type(i) is int and a in (0,1) and i in range(4)
        and row['status'] in ('recorded','failed_partial'),'fixed native row')
    allowed={'request_intent','request_return','resource_attempt','resource_return','resource_failure','resource_blocked',
        'forward_started','forward_finished','program_packet','program_boundary','io_input','io_return'}
    require(all(e['kind'] in allowed for e in events),'unknown/AD/encoder extra events rejected')
    transport.audit_resources(row['resources'],events)
    require(row['resources']['caps']==F.caps(41,0),'exact per-panel caps, no resource refunds')
    controls=[e for e in events if e['kind'] in ('request_intent','request_return')]
    if not row['values']:
        require(row['status']=='failed_partial' and not events and not row['resources']['events'],'factory failure retains empty ledger');return
    require(len(row['values'])==41,'all41 slots including failed/unstarted retained')
    want=[];active=0;entered=False
    for k,(j,v) in enumerate(zip(jobs,row['values'],strict=True)):
        require((v['index'],v['local_index'],v['job_id'],v['job_sha256'])==(j['index'],k,j['id'],j['job_sha256']),
            'all41 literal identities; no subset replacement')
        if v['status']=='not_started':
            require(v['context'] is v['objective'] is None,'unknown remains null');entered=True;continue
        require(not entered and v['status'] in ('entered','returned'),'contiguous attempts, no postfailure requests')
        active+=1
        want.append(dict(kind='request_intent',arm=arm['name'],index=j['index'],job_id=j['id'],job_sha256=j['job_sha256'],
            input_sha256=digest(input_context(j,arm)),stats_sha256=digest(arm)))
        if v['status']=='entered':
            require(row['status']=='failed_partial' and v['context'] is v['objective'] is None,'one partial invocation preserved')
            entered=True;continue
        want.append(dict(kind='request_return',arm=arm['name'],index=j['index'],value_sha256=digest(v)))
        audit_forward(j,arm,v,events,f'f{k:03}',row['resources'])
        if a==0:public_exact(j,v,p['reference'][j['index']])
    require(same(controls,want),'complete request/return disposition and no hidden retries')
    scopes={f'f{k:03}':j['id'] for k,j in enumerate(jobs[:active])}
    for e in events:
        if e['kind'] in ('request_intent','request_return'):continue
        phase=e['entry']['scope']['forward'] if e['kind'].startswith('resource_') else e['phase']
        require(phase in scopes,'all native events assigned to actual attempted job')
        if e['kind'].startswith('resource_'):require(e['entry']['scope']==dict(forward=phase,label=scopes[phase]),'native exact job label')
    pos={id(e):k for k,e in enumerate(events)}
    for k,v in enumerate(row['values'][:active]):
        intent=next(e for e in controls if e['kind']=='request_intent' and e['index']==v['index'])
        phase=f'f{k:03}';native=[e for e in events if e.get('phase')==phase or e['kind'].startswith('resource_') and e['entry']['scope']['forward']==phase]
        require(all(pos[id(intent)]<pos[id(e)] for e in native),'intent before any actual native call')
        if v['status']=='returned':
            ret=next(e for e in controls if e['kind']=='request_return' and e['index']==v['index'])
            require(all(pos[id(e)]<pos[id(ret)] for e in native),'native fully returned before durable result')
            if k+1<active:
                nxt=next(e for e in controls if e['kind']=='request_intent' and e['index']==v['index']+1)
                require(pos[id(ret)]<pos[id(nxt)],'no interleaving or outcome-dependent scheduling')
    if row['status']=='recorded':
        require(active==41 and all(v['status']=='returned' for v in row['values'])
            and row['resources']['attempted']==row['resources']['completed']==F.caps(41,0),'full41 exact forward resource census')

def precision():return dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN HOST CUDA authority before reads/output/model import')
    started=time.monotonic();p=verify(prepared,expected);store=Store(output,started)
    rows=[new_row(a,i) for a in range(2) for i in range(4)]
    errors=[];model=None;before=after=flags=pb=pa=peaks=host=None;sources_ok=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def check():
            store.check();require(FD.source.precision(torch)==precision() and not torch.is_grad_enabled(),'original precision/no AD')
            require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes']
                and torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'actual per-call CUDA allocation/reservation caps')
        with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);check()
            before=signal.model_hash(model)
            require(before==p['original_model_before']==p['original_model_after']
                and before['original_state_dict']==p['expected_model_hash'],'same entire original frozen model/buffers')
            for module in model.modules():
                require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'loaded model classes source sealed')
            pb=FD.source.precision(torch)
            dispatch(p,store,rows,check,lambda task,arm,j,l:Backend(model,device,task,arm,j,l,check))
            pa=FD.source.precision(torch);torch.cuda.synchronize(device);check()
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        try:
            if model is not None:
                FD.source.midpoint.bridge.native.frozen(model);after=signal.model_hash(model)
                require(before==after,'all original model parameters and buffers unchanged')
                peaks=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
            F.check_bindings(p['bindings']);require(sha(prepared)==expected,'prepare/source/stats/protocol unchanged');sources_ok=True
        except BaseException as exc:errors.append(dict(stage='closure',error=repr(exc)))
        wall=time.monotonic()-started
        if wall>=contract()['wall_seconds']:errors.append(dict(stage='wall',error='whole1800s including verify exceeded'))
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        good=not errors and sources_ok and before is not None and before==after and all(r['status']=='recorded' for r in rows)
        report=dict(schema=NAME+'.report',status='normalizer328_recorded_diagnostic_only' if good else 'partial_failclosed',
            contract=contract(),prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,host=host,
            fixed_denominator_per_arm=164,source_verified=sources_ok,source_manifest_sha256=digest(p['bindings']),
            protocol=p['protocol'],deferred_real_RGB_authority=p['deferred_real_RGB_authority'],
            model_before=before,model_after=after,precision_before=pb,precision_after=pa,CUDA_peak_bytes=peaks,wall_seconds=wall,
            artifact_sha256={str(f.relative_to(store.path)):sha(f) for f in store.path.rglob('*') if f.is_file()},
            P1_admission=False,freshness=False,training_provenance=False,deployment_allowed=False,scientific_admission=False,
            physical_release=False)
        store.put('report.json',report,final=True)
    return dict(output=str(store.path),report_sha256=sha(store.path/'report.json'),status=report['status'],errors=errors)


def signed(x):return 1 if x>0 else (-1 if x<0 else 0)


def comparison(predicted,actual):
    if predicted is None or actual is None:return dict(category='missing',predicted_sign=None,actual_sign=None)
    ps,rs=signed(predicted),signed(actual)
    label='both_tie' if ps==rs==0 else ('actual_tie' if rs==0 else ('predicted_tie' if ps==0 else ('same_sign' if ps==rs else 'strict_reversal')))
    return dict(category=label,predicted_sign=ps,actual_sign=rs)


def native_numpy_Q(z,goal):
    a=z if isinstance(z,np.ndarray) else FD.packet(z,(192,))
    require(a.dtype==np.float32 and a.shape==(192,) and np.isfinite(a).all(),'already validated actual latent float32')
    d=a-FD.packet(goal,(192,));return float(np.mean(d*d,dtype=np.float32))


def diagnostics(p,rows,real):
    """Post-run only: every80 pair per arm retained; no model or selection."""
    require(real['schema']=='agent_observed_raw_coordinate_rgb_20261028_v1.consume'
        and real['status']=='raw164_RGB_verified_not_admitted' and real['fixed_denominator']==len(real['rows'])==164
        and real['baseline_indices']==[0,41,82,123] and real['known_roots']==2 and real['fresh_roots']==0
        and all(real[k] is False for k in ('P1_admission','scientific_admission','freshness')),'fixed complete RGB receipt')
    actual=[]
    for i,(j,r) in enumerate(zip(p['jobs'],real['rows'],strict=True)):
        require((r['index'],r['panel'],r['local_index'],r['probe_id'],r['producer_job_sha256'],r['task_id'])==
            (i,j['slot'],j['local_index'],j['id'],j['job_sha256'],j['task_id'])
            and same(r['model_task_binding'],j['task']['binding']) and same(r['goal192'],j['task']['goal'])
            and (r['coord'],r['h'],r['sign'],r['actual_delta'])==(j['coord'],j['h'],j['sign'],j['actual_delta']),
            'real RGB same job/task/goal/raw probe/coordinate/time authority')
        require(type(r['latent_missing']) is bool,'literal missing flag')
        z=None if r['latent_missing'] else FD.packet(r['actual192'],(192,))
        require((r['actual192'] is None)==(z is None),'no substituted missing actual latent')
        if z is not None:
            q=float(np.mean((z.astype(float)-np.asarray(j['task']['goal'],float))**2))
            require(q==r['actual_Q'],'original RGB identity192 float64 Q exact')
        actual.append(z)
    arms=[]
    for a,arm in enumerate(p['arms']):
        values=[None]*164
        for row in rows:
            if row['arm']!=a:continue
            for v in row['values']:
                if v['status']=='returned':values[v['index']]=v
        endpoints=[];panels=[]
        for i,(j,v,z) in enumerate(zip(p['jobs'],values,actual,strict=True)):
            y=None if v is None else FD.packet(v['context']['latent_history'][-1],(192,))
            endpoints.append(dict(index=i,panel=i//41,local_index=i%41,baseline=i%41==0,
                predicted_native_Q=None if v is None else v['objective']['diagnostic']['native_float32_Q'],
                actual_Q_float64=real['rows'][i]['actual_Q'],actual_Q_numpy_float32=None if z is None else native_numpy_Q(z,j['task']['goal']),
                endpoint192_MSE=None if y is None or z is None else float(np.mean((y.astype(float)-z.astype(float))**2)),
                endpoint192_L2=None if y is None or z is None else float(np.linalg.norm(y.astype(float)-z.astype(float)))))
        for panel in range(4):
            steps=[]
            for hi,h in enumerate(FD.STEPS):
                pairs=[]
                for coord in range(10):
                    minus=41*panel+1+20*hi+2*coord;plus=minus+1;j=p['jobs'][minus];k=p['jobs'][plus]
                    require(j['coord']==k['coord']==coord and j['sign']==-1 and k['sign']==1,'fixed central pair')
                    spacing=float(FD.packet(k['raw_suffix5']).flat[coord])-float(FD.packet(j['raw_suffix5']).flat[coord])
                    require(spacing>0 and spacing==j['actual_pair_spacing']==k['actual_pair_spacing'],'actual float32 signed spacing')
                    vm,vp=values[minus],values[plus];zm,zp=actual[minus],actual[plus]
                    dq=None if vm is None or vp is None else vp['objective']['diagnostic']['native_float32_Q']-vm['objective']['diagnostic']['native_float32_Q']
                    rq=None if zm is None or zp is None else native_numpy_Q(zp,j['task']['goal'])-native_numpy_Q(zm,j['task']['goal'])
                    rq64=None if zm is None or zp is None else real['rows'][plus]['actual_Q']-real['rows'][minus]['actual_Q']
                    pred=None if vm is None or vp is None else (FD.packet(vp['context']['latent_history'][-1],(192,)).astype(float)
                        -FD.packet(vm['context']['latent_history'][-1],(192,)).astype(float))
                    act=None if zm is None or zp is None else zp.astype(float)-zm.astype(float)
                    vector=None
                    if pred is not None and act is not None:
                        vector=dict(predicted_secant192=(pred/spacing).tolist(),actual_secant192=(act/spacing).tolist(),
                            predicted_nominal_secant192=(pred/(2*h)).tolist(),actual_nominal_secant192=(act/(2*h)).tolist(),
                            exact_actual_zero_response=bool(np.all(act==0)),exact_predicted_zero_response=bool(np.all(pred==0)),
                            actual_zero_coordinates=int(np.count_nonzero(act==0)),predicted_zero_coordinates=int(np.count_nonzero(pred==0)),
                            response_metrics=FD.vector_metrics(pred/spacing,act/spacing))
                    pairs.append(dict(coordinate=coord,minus_index=minus,plus_index=plus,actual_float32_spacing=spacing,nominal_spacing=2*h,
                        predicted_delta_native_Q=dq,actual_delta_numpy_float32_Q=rq,actual_delta_float64_Q=rq64,
                        predicted_scalar_secant=None if dq is None else dq/spacing,actual_scalar_secant_numpy_float32=None if rq is None else rq/spacing,
                        actual_scalar_secant_float64=None if rq64 is None else rq64/spacing,
                        predicted_scalar_nominal=None if dq is None else dq/(2*h),actual_scalar_nominal=None if rq is None else rq/(2*h),
                        scalar_signs=comparison(dq,rq),scalar_signs_actual_float64=comparison(dq,rq64),vector=vector))
                categories=('same_sign','strict_reversal','both_tie','actual_tie','predicted_tie','missing')
                steps.append(dict(h=h,pairs=pairs,pair_denominator=10,
                    signs={c:sum(x['scalar_signs']['category']==c for x in pairs) for c in categories},
                    actual_zero_response_pairs=sum(x['vector'] is not None and x['vector']['exact_actual_zero_response'] for x in pairs)))
            panels.append(dict(panel=panel,baseline=endpoints[41*panel],steps=steps,pair_denominator=20))
        arms.append(dict(arm=arm['name'],identity=arm['identity'],panels=panels,endpoints=endpoints,pair_denominator=80,
            endpoint_denominator=164,no_winning_arm_or_h=True,training_provenance=False,deployment_allowed=False))
    return arms


def consume(output,expected):
    output=Path(output).absolute();core.hash_value(expected)
    require(output.resolve()==output and sha(output/'report.json')==expected,'own actual report SHA/path')
    r=signal.read(output/'report.json',META);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and same(r['contract'],contract()) and len(r['rows'])==8
        and r['fixed_denominator_per_arm']==164 and r['status'] in ('normalizer328_recorded_diagnostic_only','partial_failclosed')
        and r['protocol']==p['protocol'] and r['deferred_real_RGB_authority']==p['deferred_real_RGB_authority']
        and all(r[k] is False for k in ('P1_admission','freshness','training_provenance','deployment_allowed','scientific_admission','physical_release')),
        'full own328 diagnostic authority/no admission')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'exact run binding')
    allpaths=list(output.rglob('*'));files={str(f.relative_to(output)) for f in allpaths if f.is_file()}
    require(all(not f.is_symlink() for f in allpaths) and files==set(r['artifact_sha256'])|{'report.json'}
        and sum(f.stat().st_size for f in allpaths if f.is_file())<=contract()['output_bytes'],'full output census/cap/no extras')
    for name,h in r['artifact_sha256'].items():
        q=Path(name);require(not q.is_absolute() and 1<=len(q.parts)<=2 and all(s not in ('.','..') for s in q.parts)
            and (output/q).resolve()==output/q and sha(output/q)==h,'each actual artifact path/SHA')
    receipts=[];complete=r['status']=='normalizer328_recorded_diagnostic_only' and not r['errors']
    stopped=False;public=0
    for n,row in enumerate(r['rows']):
        a,i=divmod(n,4);require(type(row['arm']) is type(row['panel']) is int and (row['arm'],row['panel'])==(a,i),'fixed2x4 order')
        if row['resources'] is None:
            require(row==new_row(a,i),'not started remains null');stopped=True;complete=False;continue
        require(not stopped and row['journal_file']==f'a{a}_p{i}/events.jsonl'
            and same(signal.read(output/f'a{a}_p{i}_row.json',META),row),'durable exact contiguous dispatch row')
        es=F.load_journal(output,row);audit_row(p,row,es);receipts.append(row['resources'])
        if a==0:public+=sum(v['status']=='returned' for v in row['values'])
        if row['status']!='recorded':stopped=True;complete=False
    counts={kind:{k:sum(v[kind][k] for v in receipts) for k in core.AXES} for kind in ('attempted','completed')}
    require(all(counts['attempted'][k]<=F.caps(328,0)[k] for k in core.AXES),'global328 actual cap/no refunds')
    if complete:
        require(public==164 and counts['attempted']==counts['completed']==F.caps(328,0),'ALL164 exact public controls +328 actual forward resources')
        require(r['source_verified'] is True and r['source_manifest_sha256']==digest(p['bindings'])
            and r['model_before']==r['model_after']==p['original_model_before']==p['original_model_after']
            and r['precision_before']==r['precision_after']==precision(),'original complete model/precision/source closure')
        require(type(r['wall_seconds']) in (int,float) and 0<=r['wall_seconds']<contract()['wall_seconds']
            and len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=contract()[k] for v,k in
                zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'actual wall/GPU peak caps')
    else:require(r['errors'] and r['status']=='partial_failclosed','no hidden failure cast as complete')
    # Deferred evaluation dependency is intentionally NOT part of forward bindings.
    require(p['deferred_real_RGB_authority']==dict(path=str(REAL),sha256=REAL_SHA) and sha(REAL)==REAL_SHA,'user-pinned entire real RGB strict receipt')
    real=signal.read(REAL,META);panels=diagnostics(p,r['rows'],real)
    F.check_bindings(p['bindings']);require(sha(output/'report.json')==expected and sha(REAL)==REAL_SHA,'all final source/report/real seals')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'all saved artifacts unchanged after read')
    require({str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}==files,'final inventory unchanged')
    return dict(schema=NAME+'.consume',status='normalizer328_verified_diagnostic_only' if complete else 'partial_failclosed',
        complete=complete,report_sha256=expected,fixed_denominator_per_arm=164,pair_denominator_per_arm=80,arms=panels,
        public_exact_controls=public,baseline_public_indices=[0,41,82,123],actual_new_compute=counts,errors=r['errors'],
        upstream=p['upstream'],protocol=p['protocol'],real_RGB_authority=p['deferred_real_RGB_authority'],
        inherited_compute=p['inherited_compute'],inherited_resources=p['inherited_resources'],known_roots=2,fresh_roots=0,
        P1_admission=False,freshness=False,training_provenance=False,deployment_allowed=False,scientific_admission=False,
        physical_release=False,total_FLOPs=None,no_winning_arm_or_h=True,consume_new_neural_calls=0,consume_new_physics=0)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','receipt','authorization'):q.add_argument('--'+k)
    a=q.parse_args(argv)
    if a.command=='prepare':v=write(a.output,build())
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',forwards=328)
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:require(a.receipt,'own durable receipt path');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),flush=True);return 2 if v.get('status')=='partial_failclosed' else 0


if __name__=='__main__':raise SystemExit(main())




