"""Forty sealed teacher-library slots -> available terminal RGB/native192.

MAIN CUDA only. Full40 fails before model/output creation; partial is explicit.
No environment construction, rendering, prediction, fitting or replacement.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import stat
import time
from types import ModuleType
import numpy as np
from experiments.spectral_reachability import agent_p1_teacher_cross_library_20261028_v1 as teacher

NAME='agent_p1_teacher_library40_encode_20261028_v2'
ROOT=Path(__file__).resolve().parents[2]
AUTH='MAIN_PARTIAL39_TEACHER_LIBRARY_TERMINALS_CUDA_ONLY'
CAP,FINAL,FLOOR,WALL=64*1024**2,16*1024**2,200*1024**2,600
PREP_CAP,JOURNAL_CAP,REPORT_CAP=64*1024**2,32*1024**2,8*1024**2
require,sha,digest,canonical=teacher.require,teacher.sha,teacher.digest,teacher.canonical
INPUTS={
 'physical':('results/agent_p1_teacher_cross_library_physical_20261028_v1_MAIN02/report.json','464656755bb83cec14b7acdd457641592b11be6d335f285166687df15368bc01'),
 'physical_receipt':('results/agent_p1_teacher_cross_library_physical_20261028_v1_consume_MAIN02.json','44acd71c60c8edc7fbfa196c2f46f7d8692ef487ec0c3a01b6b8f7e5071e9212'),
 'native_audit':('results/audit_teacher_cross_physical_MAIN_20261028_v1_MAIN02.json','3dfc301c40471e0620caf8660baf4d8ad6534f25055e546aadc7466f61e79b78'),
 'compiler':('results/agent_p1_teacher_cross_library_20261028_v1_cuda_MAIN01/report.json','d5ab99d55bacd73c8d9cf0fb774819bb46af63100053198858e639d8fa746738'),
 'compiler_receipt':('results/consume_teacher_cross_MAIN_20261028_v1_MAIN01.json','d82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213'),
 'protocol':('innerdocs/agent_p1_teacher_cross_library_rgb_protocol_20261028_v1.md','0b9148dba5211723e6fb670658a71acd9de73487c2140affa78fbc28f2ace8aa'),
 'correction':('innerdocs/agent_p1_teacher_cross_library_rgb_protocol_correction_20261028_v2.md','06bba2e5af4992003549a6ea3817d40162fc8179617753d69cf67622024049f9')}
IDENTITY=('index','job_digest','anchor_id','task_id','blind_sha256','root_slot','condition','role','library_index','library_label','plan_digest')
# Native durable resource names, NOT physical contract.caps aliases. No zero-fill.
PHYSICAL_PAID=dict(close=40,constructor=40,explicit_render=2638,goal_bind=40,
    goal_render=80,goal_setter=120,internal_physics_step=8160,public_step=800,reset=80,set_state=160)


def contract():
    return dict(fixed_denominator=40,eligible=39,missing=[12],encoder=39,projector=39,batch=1,
        predict=0,action_encoder=0,pred_proj=0,AD=0,physics=0,training=False,Qfit=False,retries=0,dedup=False,
        CUDA_only=True,MATH=True,TF32=False,autocast=False,native_shape=[1,257,192],latent_dimension=192,
        Q='mean((identity192 @ (native192-goal192))**2), original compiler float64 arithmetic',
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,wall_seconds=WALL,
        full40_ready=False,full40_released=False,P1_admission=False,known_roots=2,fresh_roots=0,
        equal_FLOPs=False,equal_total_compute=False)


def decode(raw):
    def unique(items):
        d={}
        for k,v in items:require(k not in d,'duplicate JSON key');d[k]=v
        return d
    v=json.loads(raw,object_pairs_hook=unique);canonical(v);return v


def read(path,h=None,cap=PREP_CAP):
    p=Path(path).absolute();require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,'bounded canonical regular input')
    b=p.read_bytes();require(len(b)<=cap and (h is None or hashlib.sha256(b).hexdigest()==h),'input byte SHA')
    return decode(b)


def bind(bindings,path,h):
    p=Path(path).absolute();require(p.resolve()==p and p.is_file() and not p.is_symlink(),'canonical nonsymlink closure')
    require(str(p) not in bindings or bindings[str(p)]==h,'conflicting source binding')
    bindings[str(p)]=h


def verify_bindings(bindings):
    for path,h in bindings.items():
        require(Path(path).resolve()==Path(path) and sha(path)==h,'full input/model/source closure '+path)


def own_files():
    seen={};todo=[teacher]
    while todo:
        m=todo.pop()
        if m.__name__ in seen:continue
        require(len(seen)<192,'bounded imported source closure');seen[m.__name__]=Path(m.__file__).resolve()
        todo.extend(v for v in vars(m).values() if isinstance(v,ModuleType) and v.__name__.startswith('experiments.') and getattr(v,'__file__',None))
    site=ROOT/'.venv/lib/python3.12/site-packages'
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
            ROOT/'experiments/checkpoint_compat.py',ROOT/'experiments/gpu_preflight.py',
            ROOT/'experiments/spectral_reachability/preflight_pusht_d2p_p0.py',
            ROOT/'experiments/run_pusht_counterfactual.py',site/'stable_pretraining/data/dataset_stats.py',*seen.values()]


def terminal_events(path,expected,tail):
    """Stream whole sealed journal; retain only actual suffix5/checkpoint10 events."""
    p=Path(path);require(p.stat().st_size<=128*1024**2 and sha(p)==expected,'physical journal byte binding')
    out=[];previous=None;count=0;last=None
    with p.open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n') and len(line)<=2*1024**2,'bounded complete physical event')
            e=decode(line);require(e['index']==count and e['previous_sha256']==previous and
                e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}) and line==canonical(e)+b'\n','physical actual journal chain')
            previous=e['event_sha256'];count+=1;last=e
            if (e['kind']=='fresh_observation' and e['receipt']['phase']=='suffix' and e['receipt']['step']==5) or (e['kind']=='checkpoint_recorded' and e['checkpoint']['step']==10):out.append(e)
    require(last and last['kind']=='run_finished' and previous==tail and sha(p)==expected,'terminal journal closure')
    return out


def saved_headers(v):
    """All consumed input-header fields checked against the actual pinned schemas.

    Protocol/correction are Markdown, native_audit is deliberately unversioned
    JSON. Do not demand a fictitious audit.schema or rewrite native resource keys.
    Row/frame/array/plan fields are joined by roster/build, not inferred here.
    """
    required={
        'physical':'schema status fixed_denominator inputs_final_verified rows counts errors execution_binding artifact_sha256 journal journal_tail',
        'physical_receipt':'schema status report_sha256 prepare_sha256 rows counts errors fixed_denominator P1_admission compiler_report_sha256 compiler_consume_sha256',
        'compiler':'schema status prepared prepare_sha256 model_unchanged model_hash_before model_hash_after errors resources rows artifact_sha256 source_verified',
        'compiler_receipt':'schema status report_sha256 prepare_sha256 plans resources P1_admission new_neural_calls new_physics_calls',
        'native_audit':'status failed_observation full40_RGB_release independent_roots known_roots native_successes_at_actual_last_return no_new_model_or_physics scientific_admission rows'}
    for key,fields in required.items():
        require(type(v[key]) is dict and set(fields.split())<=v[key].keys(),'actual input fields '+key)
    r,w,cr,cw,a=[v[k] for k in ('physical','physical_receipt','compiler','compiler_receipt','native_audit')]
    require(cr['schema']==teacher.NAME+'.report' and cw['schema']==teacher.NAME+'.consume' and
        cr['source_verified'] is True and cw['P1_admission'] is False and cw['new_neural_calls']==cw['new_physics_calls']==0 and
        cr['resources']==cw['resources'] and len(cr['rows'])==8 and len(cw['plans'])==40,'original compiler schema/resources/full denominator')
    require(set(r['execution_binding'])=={'prepared','prepare_sha256','compiler_report_sha256','compiler_consume_sha256'} and
        set(r['journal'])=={'file','sha256'} and r['journal']['file']=='journal.jsonl','physical native execution/journal header ABI')
    require(a['status']=='all40_actual_raw_returns_audited_not_full_RGB_release' and a['full40_RGB_release'] is a['scientific_admission'] is False and
        a['independent_roots']==0 and a['known_roots']==2 and a['no_new_model_or_physics'] is True and len(a['rows'])==40 and
        a['native_successes_at_actual_last_return']==sum(row['last_return']['native_success'] for row in r['rows'])==0 and
        a['failed_observation']==dict(index=12,agent_xy=r['rows'][12]['last_state'][:2],canonical_blue_pixels=8,canvas_violation=True),
        'actual schema-less audit/canvas incident; no fabricated complete40')
    return {key:sorted(v[key]) for key in required}


def roster(report,receipt,package,release,events,physical_sha,receipt_prepare):
    require(report['schema']=='agent_p1_teacher_cross_library_physical_20261028_v1.report' and
        report['status']=='partial' and report['fixed_denominator']==40 and report['inputs_final_verified'] is True,'fixed actual physical partial40')
    require(receipt['schema']=='agent_p1_teacher_cross_library_physical_20261028_v1.consume' and
        receipt['status']=='partial_teacher_cross_library40_preserved_not_admitted' and receipt['report_sha256']==physical_sha and
        receipt['prepare_sha256']==receipt_prepare and receipt['rows']==report['rows'] and receipt['counts']==report['counts'] and
        receipt['errors']==report['errors'] and receipt['fixed_denominator']==40 and receipt['P1_admission'] is False,'MAIN accepted partial strict receipt exact full40')
    require(len(report['rows'])==len(package['jobs'])==len(release['plans'])==40,'all40 literal identities')
    require(report['counts']=={k:dict(attempted=n,completed=n) for k,n in PHYSICAL_PAID.items()},
        'physical actual all10 native paid axes retained, not contract.caps physics_step')
    rows=[]
    for i,(row,job,plan) in enumerate(zip(report['rows'],package['jobs'],release['plans'],strict=True)):
        ti,rem=divmod(i,10);ri,li=divmod(rem,5);slot,condition=((0,'off'),(0,'on'),(1,'off'),(1,'on'))[ti]
        require(row['index']==job['index']==plan['index']==i and (row['task'],row['root_slot'],row['condition'],row['role'],row['library_index'],row['library_label'])==
            (ti,slot,condition,('Z','S')[ri],li,teacher.LIBRARY[li]),'fixed task/role/library order, never outcome filter')
        require(all(row[k]==job[k] for k in IDENTITY) and all(row[k]==plan[k] for k in IDENTITY if k!='job_digest'),'full physical/compiler identity join')
        require(job['job_digest']==digest({k:v for k,v in job.items() if k!='job_digest'}) and
                plan['plan_digest']==digest({k:v for k,v in plan.items() if k!='plan_digest'}),'original plan and job seals')
        a=np.asarray(row['commands'],dtype='<f4');b=np.asarray(plan['commands'],dtype='<f4')
        require(a.shape==b.shape==(10,2) and a.tobytes()==b.tobytes() and np.isfinite(a).all(),'literal raw10, no rescore or replacement')
        require(row['public']==dict(warmup=dict(attempted=10,completed=10),prefix=dict(attempted=5,completed=5),suffix=dict(attempted=5,completed=5)) and
            row['reference_frames_verified']==51 and row['reference_raw_verified']==15 and
            all(row[k] is True for k in ('initial_exact','history_exact','start_verified','teacher_history_verified')),'actual warm/history and all suffix5 returns')
        require(row['last_return']['phase']=='suffix' and row['last_return']['step']==5,'actual terminal return not reconstructed state')
        scoped=[e for e in events if e.get('job_digest')==row['job_digest']]
        f=[e for e in scoped if e['kind']=='fresh_observation'];cp=[e for e in scoped if e['kind']=='checkpoint_recorded']
        item={k:deepcopy(row[k]) for k in IDENTITY};item.update(physical_status=row['status'],public=deepcopy(row['public']),
            goal_binding=dict(state=row['goal'],pixel_sha256=row['goal_pixel_sha256'],pose=row['goal_pose']),
            raw10_digest=hashlib.sha256(a.tobytes()).hexdigest(),checkpoint_requested=10,absolute_step_requested=20,
            native_success=row['last_return']['native_success'],actual_latent=None,actual_Q=None,actual_model_error=None)
        if i==12:
            require(row['status']=='failed' and row['missing_checkpoints']==[10] and row['end_state'] is None and
                row['rgb_guard_accepted'] is False and len(f)==1 and not cp and f[0]['receipt']['overlay_index']==0 and
                f[0]['receipt']['png']=='b12_suffix_05_0.png' and row['last_state'][1]>512,'missing on remains unknown despite actual off incident')
            item.update(frame_status='missing_condition_terminal',missing_reason='guard_rejected_off_before_required_on_capture',eligible_for_encoding=False,
                png=None,png_sha256=None,pixel_sha256=None,guard_failure='canvas_out_of_bounds',canonical_blue_pixels=8,
                incident_png=f[0]['receipt']['png'],incident_png_sha256=f[0]['receipt']['file_sha256'])
        else:
            require(row['status']=='horizon_completed' and row['rgb_guard_accepted'] is True and row['missing_checkpoints']==[] and
                len(f)==3 and [e['receipt']['overlay_index'] for e in f]==[0,1,2] and len(cp)==1,'full off/on/off terminal and original guard accepted')
            require(f[0]['receipt']['file_sha256']==f[2]['receipt']['file_sha256'] and f[0]['receipt']['pixel_sha256']==f[2]['receipt']['pixel_sha256'],'off replay exact bytes')
            selected=f[1 if condition=='on' else 0];q=selected['receipt'];checkpoint=row['checkpoints'][-1];frame=checkpoint['frame']
            require(cp[0]['checkpoint']==checkpoint and checkpoint['step']==10 and checkpoint['absolute_step']==20 and
                q['condition']==condition and q['phase']=='suffix' and q['step']==5 and frame['step']==5 and
                frame['file']==q['png'] and frame['sha256']==q['file_sha256'] and frame['pixel_sha256']==q['pixel_sha256'] and
                frame['pre']==frame['post']==selected['pre']==selected['post'] and frame['state']==q['state']==q['snapshot']['state']==row['end_state'],
                'same-condition actual checkpoint/native snapshot/PNG journal join')
            require(frame['pre']['snapshot']==q['snapshot'] and frame['pre']['goal_state']==row['goal'] and
                frame['pre']['goal_pose']==row['goal_pose'] and frame['pre']['goal_rgb_sha256']==row['goal_pixel_sha256'],'native goal identity unchanged')
            s=np.asarray(q['state'],dtype=np.float64);require(s.shape==(7,) and np.isfinite(s).all() and ((s[:4]>=0)&(s[:4]<=512)).all(),'original canvas no relaxation')
            item.update(frame_status='available_same_condition_terminal',missing_reason=None,eligible_for_encoding=True,
                        png=q['png'],png_sha256=q['file_sha256'],pixel_sha256=q['pixel_sha256'],guard_failure=None,canonical_blue_pixels=None)
        rows.append(item)
    require(sum(r['eligible_for_encoding'] for r in rows)==39,'exact availability, no silent extra missing');return rows


def metrics(z,goal,predicted):
    z,goal,predicted=[np.asarray(v,dtype=np.float64) for v in (z,goal,predicted)]
    require(all(v.shape==(192,) and np.isfinite(v).all() for v in (z,goal,predicted)),'original native192 scalar arithmetic')
    return dict(actual_Q=float(np.mean((z-goal)**2)),actual_model_error=float(np.mean((predicted-z)**2)))


def preprocess_pixels(pixels):
    require(pixels.dtype==np.uint8 and pixels.shape==(224,224,3),'original uint8 RGB224')
    x=pixels.transpose(2,0,1).astype(np.float32)*np.float32(1/255)
    return ((x-np.asarray([.485,.456,.406],np.float32)[:,None,None])/np.asarray([.229,.224,.225],np.float32)[:,None,None])[None,None]


def npy_bytes(a):
    b=BytesIO();np.save(b,np.ascontiguousarray(a),allow_pickle=False);return b.getvalue()


def capacity(rows,load_rgb):
    # Exact lossless known pixel/preprocessing sizes; unpredictable native tensors
    # use gzip compressBound-style n+n/1000+64 (larger than zlib bound).
    total=0
    for r in rows:
        if not r['eligible_for_encoding']:continue
        a=load_rgb(r);pre=preprocess_pixels(a)
        total+=len(gzip.compress(npy_bytes(a),mtime=0))+len(gzip.compress(npy_bytes(pre),mtime=0))+len(gzip.compress(npy_bytes(pre.reshape(1,3,224,224)),mtime=0))
        for shape in ((1,257,192),(1,192),(1,192),(192,)):
            n=int(np.prod(shape))*4+256;total+=n+n//1000+64
    # 8MiB reserved for all bounded event/row JSON, below32MiB journal cap.
    total+=8*1024**2+65536
    require(total<=CAP-FINAL,'conservative full39 tensor/metadata capacity before CUDA')
    return dict(pre_final_upper_bound_bytes=total,available_pre_final_bytes=CAP-FINAL,assumed_metadata_upper_bound=8*1024**2+65536)


def build(mode='full40'):
    require(mode in ('full40','partial_observed_only'),'explicit encoding mode')
    require(mode!='full40','full40 blocked: index12 same-condition final-on absent; model/output not created')
    bindings={};v={}
    for key,(path,h) in INPUTS.items():
        p=ROOT/path;bind(bindings,p,h)
        if key not in ('protocol','correction'):v[key]=read(p,h)
    input_fields=saved_headers(v)
    r,w,cr,cw=[v[k] for k in ('physical','physical_receipt','compiler','compiler_receipt')]
    b=r['execution_binding'];package=read(b['prepared'],b['prepare_sha256']);bind(bindings,b['prepared'],b['prepare_sha256'])
    require(package['schema']=='agent_p1_teacher_cross_library_physical_20261028_v1.prepare','original physical prepare schema')
    require(b['compiler_report_sha256']==INPUTS['compiler'][1] and b['compiler_consume_sha256']==INPUTS['compiler_receipt'][1] and
        w['compiler_report_sha256']==INPUTS['compiler'][1] and w['compiler_consume_sha256']==INPUTS['compiler_receipt'][1],'accepted physical/compiler reports match')
    compiled=read(cr['prepared'],cr['prepare_sha256']);bind(bindings,cr['prepared'],cr['prepare_sha256'])
    require(package['compiler_prepared']==cr['prepared'] and package['compiler_prepare_sha256']==cr['prepare_sha256'] and
        package['anchors']==compiled['anchors'] and
        cw['report_sha256']==INPUTS['compiler'][1] and cw['prepare_sha256']==cr['prepare_sha256'] and
        cw['status']=='teacher_cross_library_commands_verified_not_admitted' and
        cr['status']=='common_library_sealed_not_admitted' and cr['model_unchanged'] is True and cr['errors']==[] and
        cr['model_hash_before']==cr['model_hash_after']==compiled['expected_model_hash']==teacher.enc.MODEL_TENSOR_SHA,'frozen accepted compiler model and prepare')
    for material in (package,compiled):
        for path,h in material['bindings'].items():bind(bindings,path,h)
    physical=(ROOT/INPUTS['physical'][0]).parent;compiler=(ROOT/INPUTS['compiler'][0]).parent
    for base,report in ((physical,r),(compiler,cr)):
        require({p.name for p in base.iterdir()}==set(report['artifact_sha256'])|{'report.json'},'original durable artifact full census')
        for name,h in report['artifact_sha256'].items():bind(bindings,teacher.flat(base,name),h)
    require(read(physical/'execution_binding.json')==b,'physical durable execution binding')
    ev=terminal_events(physical/'journal.jsonl',r['journal']['sha256'],r['journal_tail'])
    rows=roster(r,w,package,cw,ev,INPUTS['physical'][1],b['prepare_sha256'])
    for row in rows:
        i=row['index'];ai,li=divmod(i,5);a=compiled['anchors'][ai];ca=cr['rows'][ai];e=ca['evaluations'][li]
        require(all(a[k]==row[k] for k in ('anchor_id','task_id','blind_sha256','condition','role','root_slot')) and
            np.array_equal(np.asarray(a['q']),np.eye(192)),'same-condition original anchor/identity192')
        goal=teacher.load_array(compiler,ca['arrays']['goal']);prediction=teacher.load_array(compiler,e['latents'])
        require(goal.dtype==prediction.dtype==np.float32 and goal.shape==(192,) and prediction.shape==(3,192) and
            np.array_equal(goal,np.asarray(a['goal'],np.float32)) and np.array_equal(prediction,np.asarray(e['context']['latent_history'],np.float32)) and
            e['stage']==f'evaluate_{li}' and e['library_index']==li,'saved native goal192/prediction actual arrays')
        predicted=prediction[-1].astype(np.float64);original_goal=np.asarray(a['goal'],np.float64)
        pq=metrics(predicted,original_goal,predicted)['actual_Q']
        require(pq==e['cost']==cw['plans'][i]['model_Q'],'original compiler identity192 Q exact normalization')
        row.update(goal192=a['goal'],predicted192=predicted.tolist(),predicted_Q=pq,
                   goal_array=deepcopy(ca['arrays']['goal']),prediction_array=deepcopy(e['latents']))
        if row['eligible_for_encoding']:row['png']=str(teacher.flat(physical,row['png']).resolve())
    incident=rows[12];incident_event=next(e for e in ev if e.get('job_digest')==r['rows'][12]['job_digest'] and e['kind']=='fresh_observation')
    pix=teacher.rgb(teacher.flat(physical,incident['incident_png']),incident['incident_png_sha256'],incident_event['receipt']['pixel_sha256'])
    require(np.count_nonzero(np.all(pix==[65,105,225],axis=-1))==8,'incident8 blue pixels do not waive canvas failure')
    for path in own_files():bind(bindings,path,sha(path))
    for path,h in teacher.enc.MODELS.items():bind(bindings,path,h)
    config=read(teacher.enc.POLICY/'config.json');require(teacher.overlay.encoder_output_shape(config)==[1,257,192],'frozen patch14 tokens')
    verify_bindings(bindings)
    cap=capacity(rows,lambda row:load_pixels(row))
    return dict(schema=NAME+'.prepare',mode=mode,contract=contract(),rows=rows,bindings=bindings,
                input_fields=input_fields,physical_count_fields=sorted(PHYSICAL_PAID),
                policy=str(teacher.enc.POLICY),expected_model_hash=teacher.enc.MODEL_TENSOR_SHA,capacity=cap,
                compiler_output=str(compiler),physical_output=str(physical),physical_report_sha256=INPUTS['physical'][1],
                physical_receipt_sha256=INPUTS['physical_receipt'][1],inherited_compiler_resources=cr['resources'],inherited_physical_counts=r['counts'])


def load_pixels(row):
    require(row['eligible_for_encoding'] is True and row['png'] is not None,'never load missing/off incident as on')
    a=teacher.rgb(row['png'],row['png_sha256'],row['pixel_sha256'])
    require(np.count_nonzero(np.all(a==[65,105,225],axis=-1))>0,'original canonical agent presence guard');return a


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_new(path,raw):
    p=Path(path).absolute();require(p.resolve()==p and p.name.startswith(NAME+'_') and not p.exists() and len(raw)<=PREP_CAP and
        shutil.disk_usage(p.parent).free>=FLOOR+len(raw),'new own prepare, full64MiB closure and200MiB floor')
    with p.open('xb') as f:require(f.write(raw)==len(raw),'short prepare write');f.flush();os.fsync(f.fileno())
    sync_dir(p.parent)


def prepare(path,mode):
    p=build(mode);write_new(path,canonical(p));return dict(prepared=str(Path(path).absolute()),prepare_sha256=sha(path),full40_released=False)


def verify(path,h):
    p=read(path,h);require(p==build(p['mode']),'complete own prepare reconstruction');return p


class Store:
    """Lossless64MiB durable O(1) ledger, partial writes charged; exclusive files."""
    def __init__(self,path):
        self.output=Path(path).absolute();require(self.output.resolve()==self.output and self.output.name.startswith(NAME+'_') and
            not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'own fresh output,64MiB+floor')
        self.output.mkdir();sync_dir(self.output.parent);self.used=0;self.files={};self.index=0;self.previous=None;self.broken=False
    def check(self,size=0,final=False):
        require(self.used+size<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+size,'output/final reserve/floor')
    def write(self,name,raw,append=False,final=False):
        p=teacher.flat(self.output,name);old=self.files.get(name)
        require(type(raw) is bytes and (old is None or append) and (not append or old is not None or name=='journal.jsonl'),'owned exclusive file')
        self.check(len(raw),final)
        if name=='journal.jsonl':require((old[2] if old else 0)+len(raw)<=8*1024**2,'capacity-bounded8MiB journal (below32MiB cap)')
        fd=os.open(p,os.O_WRONLY|os.O_NOFOLLOW|(os.O_APPEND if old else os.O_CREAT|os.O_EXCL),0o600)
        try:
            s=os.fstat(fd);require(stat.S_ISREG(s.st_mode) and (old is None or (s.st_dev,s.st_ino,s.st_size)==old),'owned inode/size')
            with os.fdopen(fd,'ab' if old else 'wb',closefd=False) as f:
                require(f.write(raw)==len(raw),'short artifact write');f.flush();os.fsync(f.fileno())
        finally:
            s=os.fstat(fd);os.close(fd);self.used+=s.st_size-(old[2] if old else 0);self.files[name]=(s.st_dev,s.st_ino,s.st_size)
        if old is None:sync_dir(self.output)
    def put(self,name,value,final=False):
        b=canonical(value);require(name!='report.json' or len(b)<=REPORT_CAP,'8MiB report cap');self.write(name,b,final=final)
    def event(self,value):
        require(not self.broken,'no retry broken journal');e=dict(index=self.index,previous_sha256=self.previous,payload=deepcopy(value));e['event_sha256']=digest(e)
        try:self.write('journal.jsonl',canonical(e)+b'\n',append=True)
        except BaseException:self.broken=True;raise
        self.previous=e['event_sha256'];self.index+=1
    def array(self,name,a):
        a=np.ascontiguousarray(a);require(a.dtype in (np.dtype('uint8'),np.dtype('float32')) and np.isfinite(a).all(),'lossless native precision')
        b=npy_bytes(a);raw=gzip.compress(b,mtime=0);name+='.npy.gz';self.write(name,raw)
        return dict(file=name,file_sha256=hashlib.sha256(raw).hexdigest(),npy_sha256=hashlib.sha256(b).hexdigest(),
                    array_sha256=hashlib.sha256(a.tobytes()).hexdigest(),dtype=str(a.dtype),shape=list(a.shape))
    def census(self):
        actual={}
        for p in self.output.iterdir():
            s=p.lstat();require(stat.S_ISREG(s.st_mode),'no unknown directory/symlink');actual[p.name]=(s.st_dev,s.st_ino,s.st_size)
        require(actual==self.files and sum(s[2] for s in actual.values())==self.used,'whole terminal inode/size/byte census')
        self.check(0,final=True);return {n:sha(self.output/n) for n in sorted(actual)}


class Trace(teacher.overlay.NativeTrace):
    """Own39-call gate; never reuse old8/16 caps or modify their globals."""
    def __enter__(self):
        def pre(kind):
            def hook(module,args):
                self.check();require(kind in ('encoder','projector'),'no predictor/action encoder/pred_proj allowed')
                require(self.scope is not None and module not in self.pending and sum(e['resource']==kind for e in self.entries)<39 and
                    not any(e['scope']==self.scope and e['resource']==kind for e in self.entries),'actual independent39 batch1 cap')
                e=dict(index=len(self.entries),resource=kind,scope=deepcopy(self.scope),completed=False)
                self.entries.append(e);self.pending[module]=e;self.store.event(dict(kind='tensor_module_attempt',entry=deepcopy(e)))
                require(len(args)==1 and tuple(args[0].shape)==((1,3,224,224) if kind=='encoder' else (1,192)),'original native module input dimensions')
                e['inputs']=[self.save(args[0],f'module{e["index"]}_input')]
            return hook
        def post(module,args,out):
            e=self.pending.pop(module)
            if hasattr(out,'last_hidden_state'):out=out.last_hidden_state
            require(tuple(out.shape)==((1,257,192) if e['resource']=='encoder' else (1,192)),'fixed checkpoint output,197 rejected')
            e['output']=self.save(out,f'module{e["index"]}_output');self.torch.cuda.synchronize();self.check();e['completed']=True
            self.store.event(dict(kind='tensor_module_completed',entry=deepcopy(e)))
        try:
            for kind in ('encoder','projector','action_encoder','predictor','pred_proj'):
                module=getattr(self.model,kind);self.hooks.append(module.register_forward_pre_hook(pre(kind)));self.hooks.append(module.register_forward_hook(post))
            return self
        except BaseException as e:self.__exit__(type(e),e,e.__traceback__);raise


def execute(p,store,encode,rows,check=lambda:None):
    require(p['mode']=='partial_observed_only' and len(rows)==len(p['rows'])==40,'partial40-slot mode sealed before first call')
    for spec,row in zip(p['rows'],rows,strict=True):
        if not spec['eligible_for_encoding']:continue
        check();row['status']='attempted';store.event(dict(kind='image_attempt',index=row['index'],identity={k:spec[k] for k in IDENTITY}))
        pix=load_pixels(spec);row['pixels']=store.array(f'b{row["index"]}_pixels',pix)
        z=encode(pix,row);require(z.dtype==np.float32 and z.shape==(192,) and np.isfinite(z).all(),'actual native192 only')
        row['latent']=store.array(f'b{row["index"]}_latent',z);row.update(status='encoded',actual_latent=deepcopy(row['latent']),**metrics(z,spec['goal192'],spec['predicted192']))
        store.event(dict(kind='image_return',entry=deepcopy(row)))


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'explicit MAIN partial39 CUDA authorization')
    started=time.monotonic();p=verify(prepared,expected)  # before output or model load
    rows=[dict(**deepcopy(s),status='not_started' if s['eligible_for_encoding'] else 'missing_condition_terminal') for s in p['rows']]
    store=Store(output);model=None;trace=None;before=after=host=None;errors=[];source_ok=False
    def check():require(time.monotonic()-started<=WALL,'600s whole job no retry');store.check()
    def err(stage,e):errors.append(dict(stage=stage,type=type(e).__name__,message=str(e)))
    try:
        check();store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected,mode=p['mode']))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0');require(device.type=='cuda','no CPU neural fallback')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        teacher.enc.frozen(model);before=teacher.enc.tensor_hash(model);require(before==p['expected_model_hash'],'frozen native model')
        trace=Trace(model,torch,store,check,[1,257,192]);transform=p0.image_transform()
        def encode(pixels,row):
            with teacher.overlay.traced_scope(trace,dict(image_index=row['index']),row),torch.inference_mode():
                x=p0._image_tensor(pixels,transform=transform,device=device)
                require(tuple(x.shape)==(1,1,3,224,224) and x.dtype==torch.float32 and x.device.type=='cuda','real CUDA preprocessing')
                row['preprocessed']=trace.save(x,f'b{row["index"]}_preprocessed')
                require(np.array_equal(x.detach().cpu().numpy(),preprocess_pixels(pixels)),'exact original preprocessing arithmetic')
                z=model.encode({'pixels':x})['emb'];require(tuple(z.shape)==(1,1,192),'native projected embedding')
                return z[0,0].detach().cpu().numpy().copy()
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False),trace:execute(p,store,encode,rows,check)
    except BaseException as e:
        err('execution',e)
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:err('error_journal',e)
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:teacher.enc.frozen(model);after=teacher.enc.tensor_hash(model)
            except BaseException as e:err('final_model',e)
        try:verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');check();source_ok=True
        except BaseException as e:err('final_input_or_wall',e)
        hashes={}
        try:hashes=store.census()
        except BaseException as e:err('final_census',e)
        modules=deepcopy(trace.entries) if trace is not None else []
        counts={k:dict(attempted=sum(e['resource']==k for e in modules),completed=sum(e['resource']==k and e['completed'] for e in modules)) for k in ('encoder','projector','action_encoder','predictor','pred_proj')}
        good=not errors and source_ok and before==after==p['expected_model_hash'] and sum(r['status']=='encoded' for r in rows)==39
        report=dict(schema=NAME+'.report',status='partial39_terminals_encoded_full40_blocked_not_admitted' if good else 'partial_or_failed_full40_blocked',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,mode=p['mode'],contract=contract(),rows=rows,modules=modules,counts=counts,
            image_attempted=sum(r['status'] in ('encoded','failed_partial') for r in rows),image_completed=sum(r['status']=='encoded' for r in rows),
            expected40=40,eligible39=39,missing=[12],full40_ready=False,full40_released=False,P1_admission=False,
            model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,
            journal=dict(events=store.index,last_sha256=store.previous),elapsed_seconds=time.monotonic()-started,host=host,
            inherited_compiler_resources=p['inherited_compiler_resources'],inherited_physical_counts=p['inherited_physical_counts'],
            new_predict=0,new_AD=0,new_physics=0,training=False,equal_FLOPs=False)
        try:store.put('report.json',report,final=True);store.census()
        except BaseException as e:
            err('final_report',e);good=False
            try:store.put('failure.json',dict(schema=NAME+'.failure',errors=errors,full40_released=False),final=True)
            except BaseException as e:err('failure_receipt',e)
    require(good,'partial retained, no retry: '+str(errors));return report


def consume(output,expected):
    base=Path(output).absolute();r=read(base/'report.json',expected,REPORT_CAP);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['mode']==p['mode']=='partial_observed_only' and r['contract']==contract() and
        r['status']=='partial39_terminals_encoded_full40_blocked_not_admitted' and r['errors']==[] and r['source_verified'] is True and
        r['model_unchanged'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and r['elapsed_seconds']<=WALL,'completed partial engineering receipt only')
    require(r['full40_ready'] is r['full40_released'] is r['P1_admission'] is False and r['expected40']==40 and r['eligible39']==39 and r['missing']==[12] and
        len(r['rows'])==40 and r['image_attempted']==r['image_completed']==39,'fixed40/null no admission')
    require(r['inherited_compiler_resources']==p['inherited_compiler_resources'] and r['inherited_physical_counts']==p['inherited_physical_counts'] and
        r['new_predict']==r['new_AD']==r['new_physics']==0 and r['training'] is r['equal_FLOPs'] is False,'inherited versus actual new paid axes; never equal FLOPs')
    require({f.name for f in base.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and sum(f.stat().st_size for f in base.iterdir())<=CAP,'exact64MiB artifact census')
    for n,h in r['artifact_sha256'].items():require(sha(teacher.flat(base,n))==h,'whole actual artifact SHA')
    require(read(base/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256'],mode=p['mode']),'execution prepare/mode binding')
    events=[];prev=None
    path=base/'journal.jsonl';require(path.stat().st_size<=8*1024**2,'bounded journal')
    raw=path.read_bytes();require(raw.endswith(b'\n'),'complete terminal journal newline')
    for line in raw.splitlines(keepends=True):
        e=decode(line);require(line==canonical(e)+b'\n','canonical actual event bytes')
        h=e.pop('event_sha256');require(e['index']==len(events) and e['previous_sha256']==prev and digest(e)==h,'actual durable journal chain')
        prev=h;events.append(e['payload'])
    require(r['journal']==dict(events=len(events),last_sha256=prev),'journal terminal binding')
    validate_encoded(p,r,events,lambda receipt:teacher.load_array(base,receipt))
    verify_bindings(p['bindings']);require(sha(base/'report.json')==expected,'unchanged consume input')
    return dict(schema=NAME+'.consume',status='partial39_actual_RGB_native192_verified_full40_blocked_not_admitted',report_sha256=expected,
                fixed_denominator=40,encoded=39,missing=[12],full40_released=False,P1_admission=False,new_neural_calls=0,new_physics_calls=0,
                rows=deepcopy(r['rows']),counts=r['counts'],known_roots=2,independent_confirmation=False)


def validate_encoded(p,r,events,load):
    modules=r['modules'];require([m['index'] for m in modules]==list(range(78)) and all(m['completed'] is True for m in modules),'all78 actual visual module returns')
    attempted=[e['entry'] for e in events if e['kind']=='tensor_module_attempt'];completed=[e['entry'] for e in events if e['kind']=='tensor_module_completed']
    require(completed==modules and len(attempted)==78,'actual predispatch attempts/returns full module census')
    for a,b in zip(attempted,completed,strict=True):require(a=={k:b[k] for k in ('index','resource','scope','completed')}|{'completed':False},'module intent before array/actual completion')
    available=[s['index'] for s in p['rows'] if s['eligible_for_encoding']]
    require([e['index'] for e in events if e['kind']=='image_attempt']==available and [e['entry'] for e in events if e['kind']=='image_return']==[r['rows'][i] for i in available],'all39 image attempts/actual returns, no dedup')
    expected_events=[]
    for i in available:
        expected_events.append(dict(kind='image_attempt',index=i,identity={k:p['rows'][i][k] for k in IDENTITY}))
        for m in r['rows'][i]['modules']:
            expected_events.extend([dict(kind='tensor_module_attempt',entry={k:m[k] for k in ('index','resource','scope')}|dict(completed=False)),
                                    dict(kind='tensor_module_completed',entry=m)])
        expected_events.append(dict(kind='image_return',entry=r['rows'][i]))
    require(events==expected_events,'literal image->encoder entry/return->projector entry/return->image return chronology')
    nested=[]
    for s,row in zip(p['rows'],r['rows'],strict=True):
        require(all(row[k]==v for k,v in s.items() if k not in ('actual_latent','actual_Q','actual_model_error')),'original full row source/goal/prediction/missing identity')
        if not s['eligible_for_encoding']:
            require(row['status']=='missing_condition_terminal' and row['actual_latent'] is row['actual_Q'] is row['actual_model_error'] is None and
                not any(k in row for k in ('modules','pixels','preprocessed','latent')),'missing slot never encoded or zero filled');continue
        require(row['status']=='encoded','every available slot encoded once')
        pixels=load(row['pixels']);require(np.array_equal(pixels,load_pixels(s)),'actual PNG bytes to full pixels')
        pre=load(row['preprocessed']);require(pre.dtype==np.float32 and np.array_equal(pre,preprocess_pixels(pixels)),'independent exact preprocessing')
        mods=row['modules'];require([m['resource'] for m in mods]==['encoder','projector'] and all(m['scope']==dict(image_index=row['index']) for m in mods),'one scoped native pair')
        nested+=mods;en,proj=mods;z=load(row['latent'])
        require(len(en['inputs'])==len(proj['inputs'])==1 and z.shape==(192,),'single native inputs and exact latent192')
        arrays=[load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z]
        require(all(a.dtype==np.float32 for a in arrays),'original full float32 native precision')
        teacher.overlay.visual_bridge(pre,*arrays,[1,257,192])
        require(row['actual_latent']==row['latent'] and metrics(z,s['goal192'],s['predicted192'])=={k:row[k] for k in ('actual_Q','actual_model_error')},'saved exact original Q/model error arithmetic')
    require(nested==modules and r['counts']=={k:dict(attempted=39 if k in ('encoder','projector') else 0,completed=39 if k in ('encoder','projector') else 0) for k in ('encoder','projector','action_encoder','predictor','pred_proj')},'no hidden/doubled modules, no prediction')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('manifest','prepare','verify','run','consume'))
    parser.add_argument('--mode',choices=('full40','partial_observed_only'),default='full40')
    for key in ('output','prepared','prepare-sha','report-sha'):parser.add_argument('--'+key)
    parser.add_argument('--authorize-main-cuda-partial39',action='store_true');a=parser.parse_args()
    if a.command=='manifest':out=build(a.mode)
    elif a.command=='prepare':out=prepare(a.output,a.mode)
    elif a.command=='verify':out=dict(schema=verify(a.prepared,a.prepare_sha)['schema'],verified=True,full40_released=False)
    elif a.command=='consume':out=consume(a.output,a.report_sha)
    else:
        require(a.authorize_main_cuda_partial39,'explicit MAIN partial39 flag');out=run(a.prepared,a.prepare_sha,a.output,AUTH)
    print(canonical(out).decode())


if __name__=='__main__':main()
