"""Saved H1 JEPA/native-callback pair-set intersection. No neural/physics imports."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import numpy as np

NAME='agent_p5_jepa_callback_zero_sets_20261029_v1'
JEPA='agent_p5_raw164_jepa_cuda_20261029_v1'
CALLBACK='agent_raw_native_callback_capture_20261029_v2'
ROOT=Path(__file__).resolve().parents[2]
META=256*1024**2
FIELDS=('slot','root_slot','condition','h','coord','minus_index','plus_index')
LAYERS=('integer_primitives_equal','surface512_equal','rgb224_equal')
ORDER=((0,'off'),(0,'on'),(1,'off'),(1,'on'))


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def unique(items):
    out={}
    for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
    return out


def path(value):
    p=Path(value);require(p.is_absolute() and p.resolve()==p and p.is_file() and not p.is_symlink(),'canonical regular input')
    return p


def checked(value,expected,used):
    p=path(value)
    require(type(expected) is str and len(expected)==64 and all(c in '0123456789abcdef' for c in expected)
        and sha(p)==expected,'external or inherited actual SHA256')
    require(str(p) not in used or used[str(p)]==expected,'consistent source binding');used[str(p)]=expected
    return p


def read(value,expected,used):
    p=checked(value,expected,used);require(p.stat().st_size<=META,'bounded saved metadata')
    v=json.loads(p.read_bytes(),object_pairs_hook=unique);canonical(v);return v


def schedule():
    return [dict(slot=s,root_slot=ORDER[s][0],condition=ORDER[s][1],h=h,coord=c,
        minus_index=41*s+1+20*hi+2*c,plus_index=41*s+2+20*hi+2*c)
        for s in range(4) for hi,h in enumerate((.01,.001)) for c in range(10)]


def identity(v):
    if type(v) is not dict or any(k not in v for k in FIELDS):return None
    if any(type(v[k]) is not int for k in ('slot','root_slot','coord','minus_index','plus_index')):return None
    if type(v['h']) not in (int,float) or not np.isfinite(v['h']) or type(v['condition']) is not str:return None
    return tuple(v[k] for k in FIELDS)


def index_pairs(values):
    require(type(values) is list,'pair list, never numeric cardinality as membership')
    indexed={};invalid=[]
    for i,v in enumerate(values):
        key=identity(v)
        if key is None:invalid.append(i);continue
        indexed.setdefault(key,[]).append(v)
    return indexed,invalid


def rawbits(v,n):
    require(type(v) is list and len(v)==n and all(type(r) is list and len(r)==2
        and all(type(x) in (int,float) for x in r) for r in v),'literal raw dimensions/types')
    a=np.asarray(v,np.float64);b=a.astype(np.float32)
    require(np.isfinite(a).all() and np.all(abs(b)<=1) and np.array_equal(a,b.astype(np.float64)),'lossless raw float32 box')
    return b.tobytes()


def selected_captures(report):
    result={}
    for shot in report['captures']:
        ob=shot.get('original_observation')
        if ob is None or ob.get('phase')!='plan' or ob.get('step')!=10:continue
        b=shot['branch'];require(type(b) is int and 0<=b<164,'capture branch')
        condition=ORDER[b//41][1]
        if ob.get('condition')!=condition or ob.get('overlay_index')!=(1 if condition=='on' else 0):continue
        require(b not in result,'unique original terminal selected capture');result[b]=shot
    return result


def load_patch(directory,descriptor,inventory,used):
    require(type(descriptor) is dict and descriptor['shape']==[1,256,384] and descriptor['dtype']=='float32',
        'exact factual terminal label patch, no feature re-encoding')
    name=descriptor['file'];require(Path(name).name==name and name in inventory and
        inventory[name]['sha256']==descriptor['sha256'],'sealed native label artifact')
    p=checked(directory/name,descriptor['sha256'],used)
    require(p.stat().st_size==descriptor['file_bytes']==inventory[name]['file_bytes']<=4*1024**2,'native label file bytes')
    body=p.read_bytes();f=io.BytesIO(body);a=np.load(f,allow_pickle=False)
    require(f.tell()==len(body) and a.dtype==np.float32 and list(a.shape)==[1,256,384]
        and a.nbytes==descriptor['nbytes'] and np.isfinite(a).all(),'all original finite label values/shape/dtype/EOF')
    return a


def branch_link(i,sign,spec,requests,jrows,callback_rows,jobs,captures,jp,cp,jdir,cdir,used):
    """Return missing reason when a sealed record has no exact identity mapping."""
    req=requests[i];jr=jrows[i];cr=callback_rows[i];job=jobs[i];shot=captures.get(i)
    expected_id=f's{spec["slot"]}_h{(.01,.001).index(spec["h"])}_c{spec["coord"]}_{sign}'
    for record in (req,jr,cr):
        if (record.get('index'),record.get('root_slot'),record.get('condition'))!=(i,spec['root_slot'],spec['condition']):
            return 'branch_index_root_condition_not_mapped'
    if (job.get('index'),job.get('slot'),job.get('root_slot'),job.get('condition'),job.get('h'),
        job.get('coord'),job.get('sign'),job.get('id'))!=(i,spec['slot'],spec['root_slot'],spec['condition'],
            spec['h'],spec['coord'],-1 if sign=='minus' else 1,expected_id):return 'original_job_identity_not_mapped'
    plan=cr.get('plan')
    if not isinstance(plan,dict) or plan.get('probe_id')!=expected_id or plan.get('producer_job_sha256')!=job.get('job_sha256'):
        return 'callback_plan_original_job_not_mapped'
    if not all(req[k]==jr[k]==cr[k]==plan[k] for k in ('task_id','selection_digest')):return 'task_selection_identity_not_mapped'
    if rawbits(req['commands'][-10:],10)!=rawbits(plan['commands'],10) or rawbits(job['full10'],10)!=rawbits(plan['commands'],10):
        return 'literal_raw10_not_mapped'
    if req.get('context_global_frames')!=[5,10,15] or req.get('successor_global_frame')!=20:return 'not_factual_H1_time_contract'
    if shot is None:return 'callback_selected_terminal_capture_missing'
    label=req.get('label');ob=shot['original_observation'];rgb=shot['rgb224'];surf=shot['surface512']
    if (not isinstance(label,dict) or (label.get('phase'),label.get('step'),label.get('condition'),label.get('overlay_index'))!=
        ('plan',10,spec['condition'],1 if spec['condition']=='on' else 0)):return 'JEPA_terminal_label_identity_missing'
    if (label['file_sha256']!=ob['file_sha256'] or label['pixel_sha256']!=ob['pixel_sha256'] or
        rgb['sha256']!=label['file_sha256'] or rgb['pixel_sha256']!=label['pixel_sha256']):
        return 'replay224_not_exact_original_JEPA_label'
    for rec in (label,*req['images']):
        if jp['bindings'].get(rec['path'])!=rec['file_sha256'] or cp['bindings'].get(rec['path'])!=rec['file_sha256']:
            return 'original_RGB_not_bound_by_both_prepares'
        checked(rec['path'],rec['file_sha256'],used)
    for rec in (rgb,surf):
        name=rec['file'];require(Path(name).name==name,'flat callback capture')
        if cp.get('_output_artifacts',{}).get(name)!=rec['sha256']:return 'callback_capture_not_in_report_inventory'
        checked(cdir/name,rec['sha256'],used)
    return None


def compare_sets(rows):
    zero=[r['pair_id'] for r in rows if r['observed_response_zero'] is True]
    unknown_zero=[r['pair_id'] for r in rows if r['observed_response_zero'] is None]
    result={}
    for layer in LAYERS:
        equal=[r['pair_id'] for r in rows if r[layer] is True]
        missing=[r['pair_id'] for r in rows if r[layer] is None or r['observed_response_zero'] is None]
        es,zs=set(equal),set(zero)
        result[layer]=dict(equal_members=equal,zero_members=zero,unknown=missing,
            intersection=sorted(es&zs),equal_only_known=[r['pair_id'] for r in rows if r[layer] is True and r['observed_response_zero'] is False],
            zero_only_known=[r['pair_id'] for r in rows if r[layer] is False and r['observed_response_zero'] is True],
            full_sets_equal=None if missing else es==zs,
            contingency={f'equal_{a}_zero_{b}':sum(r[layer] is a and r['observed_response_zero'] is b for r in rows)
                for a in (False,True) for b in (False,True)})
    return dict(observed_zero_members=zero,observed_zero_unknown=unknown_zero,layers=result)


def connect(jr,cr,ca,jp,cp,original,jdir,cdir,used):
    requests=jp['bridge']['requests'];jobs=original['jobs']
    require(len(requests)==len(jr['rows'])==len(cr['rows'])==len(jobs)==164,'all164 branches including baselines')
    jmap,jinvalid=index_pairs(jr['pairs']);cmap,cinvalid=index_pairs(ca['pairs']);captures=selected_captures(cr)
    cp=deepcopy(cp);cp['_output_artifacts']=cr['artifacts'];rows=[]
    for spec in schedule():
        key=identity(spec);hi=(.01,.001).index(spec['h']);pid=f's{spec["slot"]}_h{hi}_c{spec["coord"]}'
        out=dict(spec,pair_id=pid,missing=[],observed_response_zero=None,observed_response_l2=None,
            integer_primitives_equal=None,surface512_equal=None,rgb224_equal=None,callbacks_equal=None,
            historical_callback_membership=None,replay_to_original224_exact=None,first_loss_stage=None)
        js=jmap.get(key,[]);cs=cmap.get(key,[])
        if len(js)!=1:out['missing'].append('JEPA_pair_identity_missing_or_ambiguous')
        if len(cs)!=1:out['missing'].append('callback_pair_identity_missing_or_ambiguous')
        if not out['missing']:
            for sign in ('minus','plus'):
                reason=branch_link(spec[sign+'_index'],sign,spec,requests,jr['rows'],cr['rows'],jobs,captures,jp,cp,jdir,cdir,used)
                if reason:out['missing'].append(sign+':'+reason)
        if not out['missing']:
            j,c=js[0],cs[0]
            if j.get('missing') or c.get('missing'):out['missing'].append('upstream_pair_members_missing')
            else:
                arrays=[load_patch(jdir,jr['rows'][spec[s+'_index']]['label_patch'],jr['artifacts'],used) for s in ('minus','plus')]
                norm=float(np.linalg.norm((arrays[1].astype(np.float64)-arrays[0].astype(np.float64)).ravel()))
                require(type(j['observed_response_l2']) in (int,float) and np.isfinite(j['observed_response_l2'])
                    and norm==j['observed_response_l2'],'independent original fullpatch response norm, no epsilon')
                for layer in (*LAYERS,'callbacks_equal'):
                    require(type(c[layer]) is bool,'actual complete callback equality flags');out[layer]=c[layer]
                out.update(observed_response_zero=bool(norm==0),observed_response_l2=norm,
                    replay_to_original224_exact=True,first_loss_stage=c['first_loss_stage'])
        rows.append(out)
    expected={identity(s) for s in schedule()}
    extra=dict(JEPA=[i for i,v in enumerate(jr['pairs']) if identity(v) is not None and identity(v) not in expected],
        callback=[i for i,v in enumerate(ca['pairs']) if identity(v) is not None and identity(v) not in expected])
    complete=(len(jr['pairs'])==len(ca['pairs'])==80 and not jinvalid and not cinvalid
        and not extra['JEPA'] and not extra['callback'] and all(not r['missing'] for r in rows))
    return dict(rows=rows,sets=compare_sets(rows),invalid_identity_records=dict(JEPA=jinvalid,callback=cinvalid),
        unmapped_identity_records=extra,input_pair_counts=dict(JEPA=len(jr['pairs']),callback=len(ca['pairs'])),
        status='all80_exact_identity_joined' if complete else 'blocked_missing_pair_mapping')


def analyze(jepa_output,jepa_report_sha,jepa_receipt,jepa_receipt_sha,
            callback_output,callback_report_sha,callback_receipt,callback_receipt_sha):
    used={};jdir=Path(jepa_output);cdir=Path(callback_output)
    jr=read(jdir/'report.json',jepa_report_sha,used);ja=read(jepa_receipt,jepa_receipt_sha,used)
    cr=read(cdir/'report.json',callback_report_sha,used);ca=read(callback_receipt,callback_receipt_sha,used)
    require(jr['schema']==JEPA+'.report' and ja['schema']==JEPA+'.consume'
        and ja['status']=='all164_factual_H1_and80_pairs_verified' and ja['report_sha256']==jepa_report_sha
        and ja['requests']==164 and ja['pairs']==80 and ja['counts']==jr['counts']
        and jr['status']=='fixed164_actual15_conditioned_factual_H1_recorded' and jr['errors']==[]
        and jr['source_errors']==[] and jr['model_unchanged'] is True,'complete externally certified H1, not H2')
    require(cr['schema']==CALLBACK+'.report' and ca['schema']==CALLBACK+'.consume'
        and ca['status']=='all164_native_callback_capture_verified' and ca['report_sha256']==callback_report_sha
        and ca['fixed_jobs']==164 and ca['fixed_pairs']==80 and ca['counts']==cr['counts']
        and ca['full_replay_exact'] is cr['full_replay_exact'] is True and cr['errors']==[]
        and ca['historical_callback_membership'] is cr['historical_callback_membership'] is None
        and ca['sufficient_original_cause_certified'] is False,'actual certified v2 new replay, not historical callbacks')
    require(ja['P1_admission'] is ja['P5_admission'] is ca['P1_admission'] is False,'closed science gates')
    jb=jr['execution_binding'];cb=cr['execution_binding']
    jp=read(jb['prepared'],jb['prepare_sha256'],used);cp=read(cb['prepared'],cb['prepare_sha256'],used)
    require(jp['schema']==JEPA+'.prepare' and cp['schema']==CALLBACK+'.prepare'
        and jp['protocol']['pairs']==schedule() and cr['contract']['pairs_schedule']==schedule(), 'fixed original80 schedule')
    require(jp['bindings'].get(cp['original_prepare'])==cp['original_prepare_sha256']
        and jp['bindings'].get(cp['original_report'])==cp['original_report_sha256']
        and cb['original_report_sha256']==cp['original_report_sha256'],'same historical physical prepare/report authority')
    original=read(cp['original_prepare'],cp['original_prepare_sha256'],used)
    checked(cp['original_report'],cp['original_report_sha256'],used)
    result=connect(jr,cr,ca,jp,cp,original,jdir,cdir,used)
    for f,h in used.items():require(sha(path(f))==h,'all actually used saved evidence unchanged after joining')
    for f in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):used[str(f)]=sha(f)
    result.update(schema=NAME+'.analysis',fixed_pairs=80,fixed_jobs=164,known_roots=2,fresh_roots=0,
        authority=dict(jepa_report_sha256=jepa_report_sha,jepa_receipt_sha256=jepa_receipt_sha,
            callback_report_sha256=callback_report_sha,callback_receipt_sha256=callback_receipt_sha),bindings=used,
        full_upstream_consumers_rebuilt_here=False,new_neural_calls=0,new_physics_calls=0,
        historical_callback_membership=None,sufficient_original_cause_certified=False,
        P1_admission=False,P5_admission=False,matched_cross_model_performance=False,
        interpretation='exact sets of new replay raster equalities vs historical-label JEPA zero responses; source/state/raw/224 matched, old callback/cache history unknown')
    return result


def write_new(value,output):
    p=Path(output);b=canonical(value)+b'\n'
    require(p.is_absolute() and p.parent.resolve()==p.parent and p.name.startswith(NAME+'_') and not p.exists()
        and not p.is_symlink() and len(b)<=32*1024**2 and shutil.disk_usage(p.parent).free>=200*1024**2+len(b),
        'exclusive bounded own analysis')
    with p.open('xb') as f:require(f.write(b)==len(b),'full saved analysis');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(p),sha256=sha(p))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    names=('jepa-output','jepa-report-sha','jepa-receipt','jepa-receipt-sha',
        'callback-output','callback-report-sha','callback-receipt','callback-receipt-sha')
    for n in (*names,'output'):ap.add_argument('--'+n,required=True)
    a=ap.parse_args();v=analyze(*(getattr(a,n.replace('-','_')) for n in names))
    print(json.dumps(write_new(v,a.output),sort_keys=True))


if __name__=='__main__':main()
