"""Independent RGBv2 saved audit: stdlib/NumPy/PIL only, no producer imports.

External report -> binding/prepare -> task exports + parent physical -> PNG/NPY.
Recompute identity192 mean-square diagnostics; predictions MUST stay unknown.
No model loading, neural/physics execution, sourcepack/certificate/witness reads.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import zlib
import numpy as np
from PIL import Image

NAME='p1_constructive_selected_rgb_saved_audit_20261027_v2'
PRODUCER='p1_constructive_selected_rgb_cuda_20261027_v2'
PHYSICAL='p1_constructive_plan_physical_20261027_v1'
TASK='p1_constructive_blind_encode_cuda_20261027_v1'
CAP=2*1024**2


def need(ok,message):
    if not ok:raise ValueError(message)
def bytes_sha(value):return hashlib.sha256(value).hexdigest()
def digest(value):return bytes_sha(json.dumps(value,sort_keys=True,allow_nan=False).encode())
def data(path,expected=None,limit=8*1024**2):
    path=Path(path);need(path.is_file() and not path.is_symlink() and path.stat().st_size<=limit,'bounded nonsymlink file: '+str(path))
    value=path.read_bytes();need(expected is None or bytes_sha(value)==expected,'SHA mismatch: '+str(path));return value
def decode(value):
    def unique(items):
        result={}
        for k,v in items:need(k not in result,'duplicate JSON key');result[k]=v
        return result
    v=json.loads(value,object_pairs_hook=unique);json.dumps(v,allow_nan=False);return v
def read(path,expected=None,limit=8*1024**2):
    value=data(path,expected,limit)
    if str(path).endswith('.gz'):
        d=zlib.decompressobj(31);value=d.decompress(value,limit+1)
        need(len(value)<=limit and d.eof and not d.unused_data and not d.unconsumed_tail,'one bounded complete gzip')
    return decode(value)
def flat(parent,name):
    need(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat artifact name');return Path(parent)/name
def finite(value,shape):
    a=np.asarray(value,dtype=np.float64);need(a.shape==shape and np.isfinite(a).all(),'finite array '+str(shape));return a
def native192(value):
    a=np.asarray(value);need(a.shape==(192,) and a.dtype==np.dtype('float32') and np.isfinite(a).all(),'native float32[192]');return a
def raw10(value):
    need(np.asarray(value).dtype.kind in 'fi','numeric nonboolean raw commands')
    a=finite(value,(10,2));need((abs(a)<=1).all() and np.array_equal(a,a.astype(np.float32)),'literal raw10x2');return a
def chain(value):
    need(value.endswith(b'\n'),'complete physical journal lines');ev=[decode(l) for l in value.splitlines()];prev=None
    for index,e in enumerate(ev):
        need(type(e['index']) is int and e['index']==index and e['previous_sha256']==prev and e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}),'physical hashchain');prev=e['event_sha256']
    return ev
def npy(value):
    f=io.BytesIO(value);z=np.load(f,allow_pickle=False);native192(z);need(f.tell()==len(value),'no trailing NPY bytes');return z
def png(value,receipt):
    need(bytes_sha(value)==receipt['file_sha256'],'physical PNG bytes SHA')
    with Image.open(io.BytesIO(value)) as im:
        need(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'native RGB224 PNG');rgb=np.asarray(im).copy()
    need(bytes_sha(rgb.tobytes())==receipt['pixel_sha256'],'physical pixel SHA')
    need(np.any(np.all(rgb==[65,105,225],axis=-1)),'actual agent visible');return rgb


def load_task(path,expected,slot):
    e=read(path,expected,2*1024**2);t=e['task'];c=e['task_contract']
    need(e['schema']==TASK+'.task_export' and e['task_id']==digest(t) and t['task_contract_sha256']==digest(c),'actual task/contract semantic SHA')
    need(type(e['source_slot']) is int and e['source_slot']==slot and type(c['source_slot']) is int and c['source_slot']==slot,'strict task source slot')
    need(e['blind_sha256']==e['blind_source_sha256']==c['blind_source_sha256'] and e['native_root_sha256'] is None and
        e['with_target'] is False and c['with_target'] is False and e['P1_admission'] is False and e['new_independent_confirmation'] is False,'opaque no-target task provenance')
    need(t['representation']=='native_projected_cls' and t['latent_dim']==192 and t['horizon']==2 and t['packet_steps']==5 and
        t['control_dim']==2 and np.array_equal(t['projection'],np.eye(192)) and c['metric']=='identity_native192_no_fit','same native192 identity Q')
    need(t['metric_source_sha256']==digest(dict(metric='identity_native192',rank=192,no_fit=True,with_target=False)),'no fitted/replaced task metric')
    model_map={Path(k).name:v for k,v in c['model_bindings'].items()}
    need(t['model_sha256']==model_map['weights.pt'] and t['model_config_sha256']==model_map['config.json'],'task checkpoint metadata binding')
    ctx=t['root_context'];h=finite(ctx['latent_history'],(3,192));goal=finite(t['goal'],(192,))
    need(np.array_equal(h,h.astype(np.float32)) and np.array_equal(goal,goal.astype(np.float32)),'lossless native root/goal values')
    need(ctx['layout']=='states_with_between_state_actions_v1' and ctx['context_length']==3 and ctx['latent_dim']==192 and ctx['action_dim']==10 and
        c['observed_steps']==[0,5,10] and type(c['raw_between_steps']) is int and c['raw_between_steps']==10,'full observed H2 chronology')
    actions=raw10(t['root_raw_action_history']);need(not actions.any(),'actual raw zero warmup history')
    mean=np.tile(finite(t['action_mean'],(2,)).astype(np.float32),5);std=np.tile(finite(t['action_std'],(2,)).astype(np.float32),5)
    need(np.array_equal(t['action_mean'],c['normalization']['mean']) and np.array_equal(t['action_std'],c['normalization']['std']),'task same public normalizer')
    need((std>0).all() and np.array_equal(finite(ctx['action_history'],(2,10)),(actions.astype(np.float32).reshape(2,10)-mean)/std),'normalized history exact original scale')
    return e


def expected_contract(arms):
    need(isinstance(arms,list) and 1<=len(arms)<=4 and len(set(arms))==len(arms) and all(type(a) is str and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}',a) for a in arms),'fixed safe arms')
    n=4*len(arms)
    return dict(roots=2,arms=arms,branches=2*len(arms),images=n,encoder=n,projector=n,predict=0,action_encoder=0,AD=0,physics=0,
        training=False,Q='identity192_no_fit',selection=False,no_dedup=True,P1_admission=False,P2_admission=False,
        prediction='unknown; no model rollout or schema-dependent imagined-prefix substitution',
        inherited_costs='physical/compiler/original encoding costs remain separately paid; not refunded',
        numerics=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),output_cap_bytes=CAP,disk_floor_bytes=200*1024**2)


def load_parent(arguments,tasks):
    path=Path(arguments['physical_report']);r=read(path,arguments['physical_sha']);binding=read(path.parent/'execution_binding.json')
    p=read(binding['prepared'],binding['prepare_sha256'],256*1024);m=read(p['manifest'],p['manifest_sha256'],256*1024)
    ev=chain(data(path.parent/'journal.jsonl',r['journal_sha256']))
    arms=arguments['arms'];expected=[(i,a) for i in range(2) for a in arms];n=len(expected)
    need(r['schema']==PHYSICAL+'.report' and r['status']=='physical_calibration_recorded' and r['errors']==[] and r['P1_admission'] is False,'complete actual parent physical')
    need(p['schema']==PHYSICAL+'.prepare' and r['contract']==p['contract'] and r['contract']['with_target'] is False and r['contract']['own_horizon']==10 and r['contract']['branches']==n,'parent physical protocol')
    need(m['arms']==arms and m['plans']==p['plans'] and m['roots']==p['source_slots'] and m['sourcepack_sha256']==p['sourcepack_sha256']==r['sourcepack_sha256'],'parent manifest/source selection closure')
    need(ev[0]['kind']=='run_started' and ev[0]['prepare']==p and ev[-1]['kind']=='run_finished' and ev[-1]['counts']==r['counts'] and
        ev[-1]['status']==r['status'] and ev[-1]['errors']==[],'parent actual durable terminal')
    need([(x['root_slot'],x['arm']) for x in r['rows']]==expected and [(x['root_slot'],x['arm']) for x in p['plans']]==expected and
        all(type(x['root_slot']) is int for x in r['rows']+p['plans']),'parent full denominator')
    per=dict(reset=2,public_step=20,explicit_render=22,goal_render=2,set_state=4,physics_step=204,goal_bind=1)
    counts={k:dict(attempted=0,completed=0) for k in per};stack=[]
    for e in ev:
        if e['kind'] not in ('operation_attempted','operation_completed'):continue
        resource=e['resource'];need(resource in per,'known parent resource');signature={k:v for k,v in e.items() if k not in ('kind','index','previous_sha256','event_sha256')}
        if e['kind']=='operation_attempted':stack.append(signature);counts[resource]['attempted']+=1
        else:need(bool(stack) and stack.pop()==signature,'parent nested operation receipts');counts[resource]['completed']+=1
    need(not stack and counts==r['counts']=={k:dict(attempted=n*v,completed=n*v) for k,v in per.items()},'independent actual full parent physical counts')
    images=[]
    for branch,(row,plan) in enumerate(zip(r['rows'],p['plans'])):
        slot=row['root_slot'];task=tasks[slot];t=task['task'];raw10(plan['commands'])
        need(plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})==row['selection_digest'],'parent own raw plan digest')
        need(row['status'] in ('horizon_completed','native_success') and row['initial_exact'] is True and row['full_horizon'] is True and
            row['first_error'] is None and row['secondary_errors']==[] and row['own_attempted']==row['own_returned']==row['warmup_returned']==10 and not row['truncated'],'parent actually completed10, no early endpoint substitution')
        be=[e for e in ev if e.get('branch')==branch]
        need(all(e['root_slot']==slot and e['arm']==row['arm'] and e['selection_digest']==row['selection_digest'] for e in be),'parent branch identities')
        starts=[e for e in be if e['kind']=='branch_started'];closed=[e for e in ev if e['kind']=='branch_closed' and e['receipt']['selection_digest']==row['selection_digest']]
        need(len(starts)==len(closed)==1 and starts[0]['plan']==plan and closed[0]['receipt']==row,'parent actual starts/closed row')
        ts=[e for e in be if e['kind']=='returned_transition' and e['phase']=='plan'];ats=[e for e in be if e['kind']=='operation_attempted' and e['resource']=='public_step' and e.get('phase')=='plan']
        need([e['step'] for e in ts]==list(range(1,11)) and [e['step'] for e in ats]==list(range(1,11)) and
            [e['command'] for e in ts]==[e['raw_action'] for e in ats]==plan['commands'],'parent all own literal commands')
        need(not any(e['terminated'] or e['truncated'] for e in ts[:-1]),'parent no dispatch after early native stop')
        root_hash=bytes_sha(np.ascontiguousarray(np.r_[finite(row['root_state'],(7,)),finite(row['goal'],(7,))],dtype=np.float64).tobytes())
        need(p['source_slots'][slot]==dict(root_slot=slot,root_sha256=root_hash),'parent root + constructed goal identity')
        attached=[e for e in be if e['kind']=='goal_attached'];need(len(attached)==1 and attached[0]['goal']==row['goal'] and attached[0]['goal_rgb_sha256']==t['goal_source_sha256'] and
            all(e['returned_goal_state']==row['goal'] and e['returned_goal_rgb_sha256']==t['goal_source_sha256'] for e in ts),'own task actual physical goalRGB binding')
        cps={}
        for step in (0,5,10):
            cs=[c for c in row['checkpoints'] if c['phase']=='plan' and c['step']==step];need(len(cs)==1,'one actual physical checkpoint')
            c=cs[0];obs=[e for e in be if e['kind']=='fresh_observation' and e['receipt']==c];gs=[e for e in be if e['kind']=='guard_receipt' and e['phase']=='plan' and e['step']==step]
            need(len(obs)==len(gs)==1 and obs[0]['index']<gs[0]['index'] and gs[0]['accepted'] is True and gs[0]['png']==c['png'] and gs[0]['file_sha256']==c['file_sha256'] and
                c['state']==c['observation_state']==c['snapshot']['state'],'actual checkpoint observed before accepted guard')
            need(c['state']==(row['root_state'] if step==0 else ts[step-1]['state']),'parent checkpoint actual returned state');cps[step]=c
        need(cps[10]['state']==row['final_state'] and attached[0]['snapshot']==cps[0]['snapshot'],'actual terminal/root dynamics')
        for step in (5,10):
            c=cps[step];f=flat(path.parent,c['png']);png(data(f,c['file_sha256']),c)
            images.append(dict(index=len(images),branch=branch,root_slot=slot,source_slot=slot,arm=row['arm'],task_id=task['task_id'],
                task_export=arguments['task_exports'][slot],task_sha256=arguments['task_shas'][slot],blind_sha256=task['blind_sha256'],
                role='prefix5' if step==5 else 'terminal10',checkpoint=c,root=t['root_context']['latent_history'][-1],goal=t['goal'],
                Q='identity192',predicted=None,selection_digest=row['selection_digest'],png_path=str(f)))
    need(r['native_successes']==sum(row['status']=='native_success' for row in r['rows']),'parent native success count')
    return dict(report_path=str(path),report_sha256=arguments['physical_sha'],prepare_sha256=binding['prepare_sha256'],counts=counts,
        branches=n,native_successes=r['native_successes'],sourcepack_sha256=r['sourcepack_sha256']),images


def recompute(item,z):
    z=native192(z).astype(np.float64);root=finite(item['root'],(192,));goal=finite(item['goal'],(192,))
    aq=float(np.mean(np.square(z-goal)));rq=float(np.mean(np.square(root-goal)));mse=float(np.mean(np.square(z-root)))
    return dict(actual_Q=aq,root_Q=rq,actual_Q_change=aq-rq,actual_vs_root_mse=mse,
        predicted_Q=None,predicted_vs_real_mse=None,Q_improvement_sign_reversed=None,prediction_status='unknown_not_recomputed')


def inspect(report,prepared,events,expected_images,load_latent):
    """Pure array/event contract. Task and parent are independently rebuilt by audit()."""
    arms=prepared['arguments']['arms'];contract=expected_contract(arms);n=contract['images']
    need(report['schema']==PRODUCER+'.report' and report['status']=='selected_rgb_encoded_not_admitted' and report['errors']==[] and report['first_error'] is None,'complete RGBv2 release, never partial')
    need(prepared['schema']==PRODUCER+'.prepare' and report['inputs']==prepared['arguments'] and report['contract']==prepared['contract']==contract,'independent fixed RGB protocol')
    need(report['source_verified'] is True and report['model_unchanged'] is True and report['model_hash_before']==report['model_hash_after']==prepared['expected_model_hash'],'saved frozen-model before/after attestation')
    need(report['new_predict']==report['new_AD']==report['new_physics']==0 and report['training'] is False and report['P1_admission'] is False and report['P2_admission'] is False,'no extra neural/physics/admission claims')
    need(len(expected_images)==len(prepared['images'])==len(report['rows'])==n and prepared['images']==expected_images,'ALL actual parent images; no subset/substitution')
    need(len(events)==6*n,'exact complete per-image event denominator')
    results=[];counts={k:dict(attempted=0,completed=0) for k in ('encoder','projector')}
    for index,(item,row) in enumerate(zip(expected_images,report['rows'])):
        slot=index//(2*len(arms));branch=index//2;arm=arms[branch%len(arms)];role='prefix5' if index%2==0 else 'terminal10'
        need((item['index'],item['branch'],item['root_slot'],item['source_slot'],item['arm'],item['role'])==(index,branch,slot,slot,arm,role),'fixed root-major arm/role order')
        keys=('index','branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256')
        need(all(row[k]==item[k] for k in keys) and all(type(row[k]) is int for k in ('index','branch','root_slot','source_slot')) and row['status']=='encoded' and row['provenance']==item,'own complete RGB row identity/provenance')
        need(item['predicted'] is None and item['Q']=='identity192','unknown prediction must not be invented')
        ev=events[6*index:6*index+6]
        need(ev[0]==dict(kind='image_attempt',provenance=item) and ev[-1]==dict(kind='image_saved',receipt=row),'durable attempt and saved receipt exact')
        for offset,resource in enumerate(('encoder','projector')):
            entry=dict(index=2*index+offset,image=index,resource=resource)
            need(ev[1+2*offset]==dict(kind='module_attempt',entry=dict(entry,completed=False)) and
                ev[2+2*offset]==dict(kind='module_return',entry=dict(entry,completed=True)),'actual module attempt/return order no retry/dedup')
            counts[resource]['attempted']+=1;counts[resource]['completed']+=1
        latent=row['latent'];need(set(latent)=={'file','sha256','dtype','shape'} and latent['file']==f'image_{index:02d}.native192.npy' and latent['dtype']=='float32' and latent['shape']==[192],'own literal lossless latent file')
        m=recompute(item,load_latent(latent));need(row['metrics']==m,'independent actualQ/rootQ/MSE incl unknown prediction fields')
        results.append(dict(index=index,root_slot=slot,arm=arm,role=role,PNG_sha256=item['checkpoint']['file_sha256'],latent_sha256=latent['sha256'],metrics=m))
    need(report['resources']==counts=={k:dict(attempted=n,completed=n) for k in counts},'independently recounted RGB encoder/projector budget')
    return dict(schema=NAME+'.audit',status='saved_rgb_v2_bindings_metrics_verified',findings=[],images=n,branches=contract['branches'],resources=counts,rows=results,
        prediction_status='unknown_not_recomputed',new_model_calls=0,new_physics=0,P1_admission=False,P2_admission=False,new_independent_samples=0,global_freshness='unknown_calibration_only',
        limitations=['saved module receipts are not an independent rerun/proof of neural execution',
            'model before/after equality is the sealed producer attestation, not a newly measured tensor hash',
            'parent bindings/checkpoints are audited, not sourcepack/certificate scientific validity or simulator replay',
            'unknown predictions are null, not zero error or evidence of sign reversal'])


def audit(report_path,report_sha):
    path=Path(report_path).absolute();r=read(path,report_sha,CAP);art=r['artifact_sha256']
    n=len(r['rows']);expected={'binding.json','events.jsonl'}|{f'image_{i:02d}.native192.npy' for i in range(n)}
    need(set(art)==expected and {f.name for f in path.parent.iterdir()}==expected|{'report.json'},'exact saved artifact census')
    blobs={name:data(flat(path.parent,name),h,CAP) for name,h in art.items()}
    need(sum(f.stat().st_size for f in path.parent.iterdir())<=CAP,'actual2MiB output cap')
    b=decode(blobs['binding.json']);need(b['prepare_sha256']==r['prepare_sha256'],'binding/report actual prepare SHA')
    p=read(b['prepared'],b['prepare_sha256'],256*1024);args=p['arguments']
    need(len(args['task_exports'])==len(args['task_shas'])==2 and len(set(args['task_exports']))==len(set(args['task_shas']))==2,'two explicit unique approved tasks')
    tasks=[load_task(f,h,i) for i,(f,h) in enumerate(zip(args['task_exports'],args['task_shas']))]
    need(tasks[0]['task_contract']['model_bindings']==tasks[1]['task_contract']['model_bindings'] and
        all(p['bindings'][k]==v for k,v in tasks[0]['task_contract']['model_bindings'].items()),'same task/RGB declared model file SHA metadata')
    parent,images=load_parent(args,tasks);need(blobs['events.jsonl'].endswith(b'\n'),'complete RGB event lines')
    events=[decode(line) for line in blobs['events.jsonl'].splitlines()]
    def load_latent(receipt):
        value=blobs[receipt['file']];need(bytes_sha(value)==receipt['sha256']==art[receipt['file']],'latent artifact/row SHA');return npy(value)
    result=inspect(r,p,events,images,load_latent)
    need(bytes_sha(data(path))==report_sha and bytes_sha(data(b['prepared']))==b['prepare_sha256'],'final report/prepare stability')
    result.update(report=str(path),report_sha256=report_sha,prepare_sha256=b['prepare_sha256'],parent_physical=parent,
        task_bindings=dict(zip(args['task_exports'],args['task_shas'])),auditor_source_sha256=bytes_sha(data(__file__)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--report-sha',required=True);p.add_argument('--output');a=p.parse_args()
    result=audit(a.report,a.report_sha);value=(json.dumps(result,sort_keys=True,allow_nan=False)+'\n').encode()
    if a.output:
        out=Path(a.output);need(len(value)<=100*1024 and not out.exists() and not out.is_symlink() and shutil.disk_usage(out.parent).free>=200*1024**2+len(value),'NEW small100KiB +200MiB floor')
        need(out.parent.resolve() not in (Path(a.report).parent.resolve(),Path(result['parent_physical']['report_path']).parent.resolve()),'no writing sealed RGB/physical directories')
        with out.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
        fd=os.open(out.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        print(json.dumps(dict(output=str(out.absolute()),sha256=bytes_sha(value),bytes=len(value),status=result['status']),sort_keys=True))
    else:print(value.decode(),end='')


if __name__=='__main__':main()
