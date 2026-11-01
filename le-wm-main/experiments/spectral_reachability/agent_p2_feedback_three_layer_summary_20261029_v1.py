"""Saved-only O/I/F 84-cell three-layer joins, independent stdlib fsum arithmetic.

MAIN external complete receipt SHAs are trust anchors, NOT rerun fullstrict.
No model/physics/imported producer math. No actual output without all authorities.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import zlib

NAME='agent_p2_feedback_three_layer_summary_20261029_v1'
RGB='agent_p2_full30_feedback_rgb_20261029_v1'
ORG='agent_p2_two_program_full30_rgb_encode_20261029_v1'
MODEL='agent_p2_full30_control15_feedback_20261029_v2'
MS='agent_p2_full30_control15_feedback_summary_20261029_v1'
NS='agent_p2_full30_feedback_native_summary_20261029_v2'
OLD='agent_p2_two_program_full30_physical_20261029_v1'
NEW='agent_p2_full30_control15_feedback_physical_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CP=(0,5,10,15,20,25,30)
POL=('O','I','F')
ARMS=('two_program_transfer','joint_two_round')
KEYS=('if_rgb','O_rgb','feedback_prepare','feedback_receipt','model_summary','native_summary')
LIMIT=16*1024**2
FLAGS={k:False for k in ('P1_admission','P2_admission','P2_mechanism','C3','scientific_admission',
    'physical_release','equal_FLOPs','equal_total_compute','independent_confirmation','freshness')}
PAID=dict(predict=216,J=8,VJP=1536,J_elements=46080,AD_primal=24)
KINDS=('encoder','projector','action_encoder','predict_API','predictor','pred_proj')


def need(ok,label):
    if not ok:raise ValueError(label)


def raw(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def digest(x):return hashlib.sha256(raw(x)).hexdigest()
def equal(x,y):return raw(x)==raw(y)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absolute(path):
    need(type(path) is str,'literal path')
    p=Path(path);need(p.is_absolute() and str(p.resolve())==path and not p.is_symlink(),'canonical absolute path')
    return p


def hash_string(h):need(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'SHA256 literal')


def decode(body):
    def unique(items):
        out={}
        for k,v in items:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    def bad(_):raise ValueError('nonfinite JSON')
    x=json.loads(body,object_pairs_hook=unique,parse_constant=bad);raw(x);return x


def read(ref):
    need(type(ref) is dict and set(ref)=={'path','sha256'},'exact path/SHA authority')
    hash_string(ref['sha256']);p=absolute(ref['path'])
    need(p.is_file() and p.stat().st_size<=LIMIT,'bounded regular saved JSON')
    b=p.read_bytes();need(len(b)<=LIMIT and hashlib.sha256(b).hexdigest()==ref['sha256'],'actual saved bytes SHA')
    return decode(b)


def vector(v,n,f32=False):
    need(type(v) is list and len(v)==n and all(type(x) in (int,float) and math.isfinite(x) for x in v),
        'finite literal vector/no bool')
    if f32:
        try:roundtrip=list(struct.unpack('<'+str(n)+'f',struct.pack('<'+str(n)+'f',*v)))
        except (OverflowError,struct.error) as e:raise ValueError('float32 range') from e
        need(roundtrip==v,'literal float32/no repair')
    return v


def mse(a,b):
    vector(a,192,True);vector(b,192,True)
    d=[float(x)-float(y) for x,y in zip(a,b,strict=True)]
    return math.fsum(x*x for x in d)/192


def native(state,goal):
    vector(state,7);vector(goal,7)
    d=[float(x)-float(y) for x,y in zip(state,goal,strict=True)];q=[x*x for x in d]
    need(all(math.isfinite(x) for x in q),'finite squared native residuals')
    xy=math.fsum(q[:4]);obj=math.fsum(q[2:4]);vel=math.fsum(q[5:]);total=math.fsum(q)
    angle=min(abs(d[4]),2*math.pi-abs(d[4]))
    return dict(axis_residual=d,axis_squared=q,agent_xy_squared=math.fsum(q[:2]),
        object_xy_squared=obj,joint_xy_squared=xy,velocity_squared=vel,
        native7_squared=total,joint_xy_error=math.sqrt(xy),object_xy_error=math.sqrt(obj),
        velocity_error=math.sqrt(vel),native7_distance=math.sqrt(total),native7_reward=-math.sqrt(total),
        raw_angle_squared=q[4],wrapped_angle_error=abs(math.atan2(math.sin(d[4]),math.cos(d[4]))),
        native_angle_term=angle,native_success=math.sqrt(xy)<20 and angle<math.pi/9)


def sign(v):
    if v is None:return 'null'
    need(type(v) in (int,float) and math.isfinite(v),'finite scalar sign')
    return 'positive' if v>0 else 'negative' if v<0 else 'tie'


def raw30(a):
    need(type(a) is list and len(a)==30,'literal raw30')
    v=[]
    for row in a:
        vector(row,2,True);need(all(-1<=x<=1 for x in row),'legal raw domain');v.extend(row)
    return struct.pack('<60f',*v)


def latent(ref,directory,bindings):
    need(set(ref)=={'file','sha256','array_sha256','shape','dtype'} and
        Path(ref['file']).name==ref['file'] and ref['shape']==[192] and ref['dtype']=='float32','native192 descriptor')
    p=directory/ref['file'];need(p.is_file() and not p.is_symlink() and p.stat().st_size<=4096,'bounded saved native192')
    b=p.read_bytes();need(hashlib.sha256(b).hexdigest()==ref['sha256'],'native192 compressed SHA')
    z=zlib.decompressobj(31);data=z.decompress(b,4097)
    need(len(data)<=4096 and z.eof and not z.unused_data and not z.unconsumed_tail,'one bounded gzip member')
    need(data[:8]==b'\x93NUMPY\x01\x00','literal NPY1 native192')
    size=struct.unpack('<H',data[8:10])[0]
    header=ast.literal_eval(data[10:10+size].decode('ascii'))
    need(header==dict(descr='<f4',fortran_order=False,shape=(192,)) and len(data)==10+size+768,'NPY shape/dtype/body')
    body=data[10+size:];need(hashlib.sha256(body).hexdigest()==ref['array_sha256'],'native192 raw SHA')
    v=list(struct.unpack('<192f',body));vector(v,192,True)
    bindings[str(p)]=ref['sha256'];return v


def counts(n):
    return {k:dict(attempted=n if k in ('encoder','projector') else 0,
        completed=n if k in ('encoder','projector') else 0) for k in KINDS}


def summarize(v,original,jobs):
    f,o,p,m,ms,ns=[v[k] for k in KEYS]
    need(type(f['encoded']) is int and type(f['missing']) is int and
        f['encoded']==sum(r['actual192'] is not None for r in f['rows']) and
        f['missing']==sum(r['actual192'] is None for r in f['rows']),'all24 actual/missing encoding census')
    need(f['schema']==RGB+'.consume' and f['status']=='all_available24_IF_RGB_verified_NOT_admitted'
        and f['fixed_denominator']==24 and f['trajectory_denominator']==8 and len(f['rows'])==24
        and f['own_native_encoding_saved_reconstructed_here'] is True and f['full_upstream_rebuilt_here'] is False
        and f['counts']==counts(f['encoded']) and f['encoded']+f['missing']==24,'complete external IF RGB receipt')
    need(o['schema']==ORG+'.consume' and o['status']=='all28_full30_RGB_verified_NOT_admitted'
        and len(o['rows'])==o['fixed_denominator']==28 and o['counts']==counts(sum(r['actual192'] is not None for r in o['rows'])),
        'complete O RGB receipt')
    need(p['schema']==MODEL+'.prepare' and m['schema']==MODEL+'.consume' and
        m['status']=='control15_feedback_verified_NOT_admitted' and m['errors']==[] and len(m['plans'])==len(p['jobs'])==8
        and m['actual_new']==PAID and p['generator_labels_max_control']==15,'full216/8J feedback source authority')
    need(ms['schema']==MS+'.summary' and ms['status']=='saved_scalar_verified_NOT_admitted'
        and ms['actual_new']==PAID and len(ms['jobs'])==8,'independent model summary source')
    need(ns['schema']==NS+'.summary' and len(ns['summary']['panel'])==84 and len(ns['summary']['rows'])==12,
        'independent native v2 summary source')
    for obj in (f,o,m,ms,ns['summary']):
        need(all(obj[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','physical_release')),'no gate admission')
    need(original['schema']==OLD+'.consume' and len(original['rows'])==4 and len(jobs)==4,'all native source receipts')
    rows=[]
    for base in range(4):
        old=original['rows'][base];parent=old['plan']['planner_plan'];commands0=raw30(old['plan']['commands'])
        need(old['root_slot']==base//2 and old['arm']==ARMS[base%2] and old['task_id']==parent['task_id'],'original base task')
        need(raw30(parent['commands'])==commands0 and parent['raw_float32_bits']==commands0.hex()
            and parent['witness_sha256']==digest({k:x for k,x in parent.items() if k!='witness_sha256'}),
            'original full30 action/model witness')
        for pi,policy in enumerate(POL):
            actual=old if pi==0 else jobs[base]['rows'][pi-1]
            plan=actual['plan'];commands=raw30(plan['commands'])
            need(actual['task_id']==old['task_id'] and actual['root_slot']==base//2 and actual['arm']==ARMS[base%2]
                and actual['condition']=='off' and equal(actual['goal'],old['goal'])
                and equal(plan['planner_plan'],parent) and commands[:120]==commands0[:120],'same root/goal/prefix authority')
            need(actual['selection_digest']==plan['selection_digest'] and
                plan['selection_digest']==hashlib.sha256(json.dumps({k:x for k,x in plan.items() if k!='selection_digest'},
                    sort_keys=True,allow_nan=False).encode()).hexdigest(),'own execution digest, not cross-policy equal')
            modeljob=None if pi==0 else m['plans'][2*base+pi-1]
            if modeljob is not None:
                sj=ms['jobs'][2*base+pi-1];job=p['jobs'][2*base+pi-1]
                need(equal(plan['new_model_plan'],modeljob) and modeljob['policy']==policy and modeljob['base_index']==base
                    and modeljob['raw_float32_bits']==commands.hex() and modeljob['task_id']==old['task_id']
                    and modeljob['input_context']==job['input_context'] and modeljob['goal']==parent['goal']
                    and sj['new_witness_sha256']==modeljob['new_witness_sha256']
                    and sj['input_context_sha256']==digest(job['input_context']),'same feedback/model-summary witness')
                need(modeljob['new_witness_sha256']==digest({k:x for k,x in modeljob.items() if k!='new_witness_sha256'}),
                    'new compact model witness')
                q=mse(modeljob['new_contexts'][-1]['latent_history'][-1],modeljob['goal'])
                need(q==sj['candidates'][sj['original_selected']]['Q_fsum'],'independent terminal fsum model-summary join')
            need(type(actual['own_returned']) is int and 0<=actual['own_returned']<=30,'actual eventtime')
            for ci,cp in enumerate(CP):
                oldrgb=o['rows'][7*base+ci]
                need((oldrgb['plan_index'],oldrgb['root_slot'],oldrgb['arm'],oldrgb['checkpoint'])==
                    (base,base//2,ARMS[base%2],cp) and oldrgb['task_id']==old['task_id']
                    and oldrgb['goal192']==parent['goal'],'O RGB task/time/goal join')
                inherited=pi>0 and cp<=15
                rgb=oldrgb if pi==0 or inherited else f['rows'][6*base+(pi-1)*3+(ci-4)]
                if pi>0 and not inherited:
                    need((rgb['base_index'],rgb['policy'],rgb['root_slot'],rgb['arm'],rgb['checkpoint'])==
                        (base,policy,base//2,ARMS[base%2],cp) and rgb['task_id']==actual['task_id']
                        and rgb['new_witness_sha256']==modeljob['new_witness_sha256']
                        and rgb['raw30_float32_bits']==commands.hex() and rgb['goal192']==parent['goal']
                        and rgb['selection_digest']==actual['selection_digest'],'own I/F RGB not O suffix alias')
                checkpoints=[c for c in actual['checkpoints'] if c['step']==cp]
                present=cp<=actual['own_returned']
                need(len(checkpoints)==int(present),'actual reached cp versus missing')
                frame=checkpoints[0] if present else None
                z=rgb['actual192'] if present else None
                need((rgb['actual192'] is not None)==present or inherited,'missing actual latent no padding')
                if present:
                    fields=set(frame)-({'png'} if inherited else set())
                    need(z is not None and rgb['source'] is not None and all(equal(rgb['source'][k],frame[k]) for k in fields)
                        and frame['state']==frame['snapshot']['state'],'same native7/RGB/time source')
                    vector(frame['state'],7)
                    need(struct.pack('<7d',*frame['state']).hex()==frame['snapshot']['state_float64_bits'],
                        'literal native7 snapshot float64 bits')
                if inherited:
                    need(commands[:120]==commands0[:120] and (not present or
                        equal(frame['state'],old['checkpoints'][ci]['state'])),'inherited prefix, not future O latent')
                pred=input_context=scope=None
                if policy=='O' and cp>0:
                    pred=parent['contexts'][ci-1]['latent_history'][-1]
                    need(equal(oldrgb['predicted192'],pred),'O same-action native prediction')
                    input_context=parent['root_context'] if cp<=15 else parent['contexts'][2]
                    scope='original_O_forecast'
                elif policy!='O' and cp>15:
                    pred=modeljob['new_contexts'][ci-4]['latent_history'][-1]
                    input_context=modeljob['input_context'];scope='new_I_or_F_suffix_forecast'
                qmodel=None if pred is None else mse(pred,parent['goal'])
                qactual=None if z is None else mse(z,parent['goal'])
                state=None if frame is None else frame['state']
                nm=None if state is None else native(state,actual['goal'])
                native_row=ns['summary']['panel'][base*21+pi*7+ci]
                need((native_row['base'],native_row['policy'],native_row['control_step'],native_row['absolute_step'])==
                    (base,policy,cp,10+cp) and equal(native_row['state7'],state)
                    and equal(native_row['goal7'],actual['goal']),'native-v2 identity/state join')
                need((native_row['metrics'] is None)==(nm is None),'native null agreement')
                if nm is not None:
                    need(all(equal(native_row['metrics'][k],value) for k,value in nm.items()),'independent fsum all native axes')
                rows.append(dict(index=len(rows),base=base,root=base//2,arm=ARMS[base%2],policy=policy,
                    control_step=cp,absolute_step=10+cp,task_id=actual['task_id'],selection_digest=actual['selection_digest'],
                    actual_returned_control=actual['own_returned'],
                    recorded_transition_native_success=native_row['recorded_transition_native_success'],
                    raw30_sha256=hashlib.sha256(commands).hexdigest(),goal192_sha256=digest(parent['goal']),
                    raw_prefix_to_cp_bits=commands[:cp*8].hex(),
                    raw_interval5_bits=None if cp==0 else commands[(cp-5)*8:cp*8].hex(),
                    model_Q=qmodel,actual_encoded_Q=qactual,model_minus_actual_Q=None if qmodel is None or qactual is None else qmodel-qactual,
                    prediction_MSE=None if pred is None or z is None else mse(pred,z),native=nm,
                    model_input_context_sha256=None if input_context is None else digest(input_context),
                    model_scope=scope,model_null_reason=('no_forecast_anchor' if cp==0 else
                        'no_new_I_F_prefix_prediction' if inherited else None),
                    inherited_O_prefix_model_Q=None if not inherited or cp==0 else mse(parent['contexts'][ci-1]['latent_history'][-1],parent['goal']),
                    actual_encoding_origin='inherited_O_prefix_exact_guard' if inherited else 'O_original' if pi==0 else 'new_IF_actual_image',
                    missing_reason=None if present else 'native_checkpoint_not_reached',source=deepcopy(frame)))
    pairs=[]
    for base in range(4):
        for left,right in ((0,1),(0,2),(1,2)):
            for ci,cp in enumerate(CP):
                a,b=rows[21*base+7*left+ci],rows[21*base+7*right+ci]
                same_input=a['model_input_context_sha256'] is not None and a['model_input_context_sha256']==b['model_input_context_sha256']
                md=None if not same_input or a['model_Q'] is None or b['model_Q'] is None else b['model_Q']-a['model_Q']
                aq=None if a['actual_encoded_Q'] is None or b['actual_encoded_Q'] is None else b['actual_encoded_Q']-a['actual_encoded_Q']
                nd=None if a['native'] is None or b['native'] is None else {
                    k:b['native'][k]-a['native'][k] for k in ('native7_squared','joint_xy_squared','object_xy_squared','velocity_squared','wrapped_angle_error')}
                pairs.append(dict(base=base,pair=POL[left]+'-'+POL[right],control_step=cp,direction=POL[right]+' minus '+POL[left],
                    model_Q_delta=md,model_delta_reason=None if md is not None else 'missing_prediction_or_different_input_context',
                    same_model_input=same_input,same_full_raw=a['raw30_sha256']==b['raw30_sha256'],
                    same_raw_prefix_to_cp=a['raw_prefix_to_cp_bits']==b['raw_prefix_to_cp_bits'],
                    same_raw_interval5=None if cp==0 else a['raw_interval5_bits']==b['raw_interval5_bits'],
                    actual_encoded_Q_delta=aq,native_delta=nd,model_sign=sign(md),actual_sign=sign(aq),
                    native_signs=None if nd is None else {k:sign(value) for k,value in nd.items()},
                    causal_gain_claim=False))
    need(len(rows)==len(pairs)==84,'whole12x7 denominator')
    return dict(rows=rows,pairs=pairs,denominator=dict(bases=4,policies=12,positions=84,pairs=84,known_roots=2,fresh_roots=0),
        resources=dict(feedback_model_new=PAID,model_independent_census=deepcopy(ms['actual_census']),
            IF_encoding_new=deepcopy(f['counts']),O_encoding_inherited=deepcopy(o['counts']),
            native=deepcopy(ns['summary']['resources']),FLOPs=None),
        interpretation='within-row model error only; cross-policy model delta requires exact same model input; not causal gain',
        new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0,full_upstream_rebuilt_here=False,**FLAGS)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def analyze(manifest_path,manifest_sha):
    manifest=read(dict(path=manifest_path,sha256=manifest_sha))
    need(set(manifest)=={'schema','trust','inputs'} and manifest['schema']==NAME+'.inputs'
        and manifest['trust']=='MAIN_complete_external_receipt_SHAs_not_rebuilt'
        and set(manifest['inputs'])==set(KEYS),'complete explicit authority manifest; no future default SHA')
    refs=manifest['inputs'];bindings={manifest_path:manifest_sha};v={}
    def get(ref):
        x=read(ref);bindings[ref['path']]=ref['sha256'];return x
    for key in KEYS:v[key]=get(refs[key])
    f,o,p,m,ms,ns=[v[k] for k in KEYS]
    need(f['source_authorities']['O_receipt']==refs['O_rgb']
        and ms['sources']['prepare']==refs['feedback_prepare']
        and ms['sources']['receipt']==refs['feedback_receipt'],'same explicit MAIN authorities, not similar schema')
    need(ms['sources']['prepare']['sha256']==m['prepare_sha256'] and
        ms['sources']['report']['sha256']==m['report_sha256'] and
        ms['sources']['completion']['sha256']==m['completion_sha256'],'model summary source join')
    nr=ns['arguments'];need(len(nr['jobs'])==len(nr['job_shas'])==4,'native-v2 all4 sources')
    original_ref=dict(path=nr['original'],sha256=nr['original_sha'])
    original=get(original_ref);jobs=[]
    need(o['physical_receipt_sha256']==original_ref['sha256'],'O RGB/native actual authority')
    for i in range(4):
        ref=dict(path=nr['jobs'][i],sha256=nr['job_shas'][i])
        need(f['source_authorities'][f'job{i}']==ref,'I/F RGB/native exact source authority')
        job=get(ref)
        need(job['schema']==NEW+'.consume' and job['status']=='feedback_two_native_recorded_NOT_admitted'
            and job['job_index']==i and job['errors']==[] and len(job['rows'])==2
            and job['own_native_saved_reconstructed_here'] is True,'complete actual source native receipt')
        jobs.append(job)
    need(original['status']=='full30_four_native_recorded_NOT_admitted'
        and original['errors']==[] and original['own_native_saved_reconstructed_here'] is True,
        'complete original native receipt')
    report_paths=[path for path,h in f['bindings'].items() if h==f['report_sha256']]
    need(len(report_paths)==1,'unique actual encoded report path')
    fr=get(dict(path=report_paths[0],sha256=f['report_sha256']))
    co=f['completion']
    fp=get(dict(path=co['prepared'],sha256=f['prepare_sha256']))
    need(fr['schema']==RGB+'.report' and fr['status']=='all24_IF_actual_RGB_sealed_NOT_admitted'
        and fp['schema']==RGB+'.prepare' and fr['prepared']==co['prepared']
        and fr['prepare_sha256']==f['prepare_sha256'] and fp['authorities']==f['source_authorities']
        and co['report_sha256']==f['report_sha256'] and len(fr['rows'])==len(fp['rows'])==24
        and f['counts']==fr['counts'] and fr['errors']==fr['blocked']==[],'same complete actual IF encoding report/prepare')
    od=f['source_authorities']['O_report'];oreport=get(od)
    need(o['report_sha256']==od['sha256'] and oreport['schema']==ORG+'.report' and len(oreport['rows'])==28,
        'original O encoding report/receipt')
    for rows,report,directory in ((f['rows'],fr,Path(report_paths[0]).parent),
                                 (o['rows'],oreport,Path(od['path']).parent)):
        for receipt_row,report_row in zip(rows,report['rows'],strict=True):
            need(receipt_row['index']==report_row['index'],'whole actual RGB row order')
            actual=receipt_row['actual192'];desc=report_row['latent']
            need((actual is None)==(desc is None),'actual missing array agreement')
            if actual is not None:
                need(desc['sha256']==report['artifact_sha256'][desc['file']],'actual latent in complete encoding report')
                need(equal(latent(desc,directory,bindings),actual),'independent literal NPY/native192 receipt vector')
    for spec,row in zip(fp['rows'],f['rows'],strict=True):
        need(all(equal(row[k],x) for k,x in spec.items() if k not in ('status','latent')),
            'literal original24 input/source recipe, no task construction')
    result=summarize(v,original,jobs)
    for path,h in bindings.items():need(sha(path)==h,'saved input unchanged after independent arithmetic')
    return dict(schema=NAME+'.summary',arguments=dict(manifest=manifest_path,manifest_sha256=manifest_sha),
        inputs=deepcopy(refs),checked_saved_bytes=bindings,summary=result,
        own_sha256={str(path):sha(path) for path in own_files()},
        trust='external complete MAIN receipts inherited, local joins/arithmetic/192 arrays independently rebuilt; NOT upstream fullstrict')


def write(path,result):
    path=absolute(path);body=raw(result)
    need(path.name.startswith(NAME+'_') and len(body)<=LIMIT and not path.exists(),'own exclusive bounded summary')
    with path.open('xb') as f:need(f.write(body)==len(body),'complete summary write');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),sha256=sha(path))


def verify(path,h):
    old=read(dict(path=path,sha256=h));args=old['arguments']
    now=analyze(args['manifest'],args['manifest_sha256'])
    need(equal(old,now),'whole own saved summary reconstruction')
    return dict(verified=True,path=path,sha256=h)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);s=q.add_subparsers(dest='command',required=True)
    a=s.add_parser('analyze');a.add_argument('--inputs',required=True);a.add_argument('--inputs-sha',required=True)
    a.add_argument('--output',required=True)
    a=s.add_parser('verify');a.add_argument('--summary',required=True);a.add_argument('--summary-sha',required=True)
    a=q.parse_args(argv)
    r=write(a.output,analyze(a.inputs,a.inputs_sha)) if a.command=='analyze' else verify(a.summary,a.summary_sha)
    print(raw(r).decode(),end='',flush=True)


if __name__=='__main__':main()
