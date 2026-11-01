"""Independent stdlib/NumPy audit; never imports a neural/physics consumer."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

NAME='p5_jepa_fresh_fidelity_audit_20261027_v1'
PRIMARY='p5_jepa_fresh_fidelity_main_20261027_v11'
CONTRAST='p5_jepa_fresh_fidelity_main_20261027_v10'
FACTUAL='p5_jepa_factual_nonzero_20261027_v2'
ROOT=Path(__file__).resolve().parents[2]
MANIFEST_FILE_SHA='09f5f61cf9eb9e9d65b21a2aec8262e699d662c97e77afb5e0ca7e9c4d49ef70'
REPORT_FILE_SHA='c77ff194fc8510093b1602930b6f7d041896a67affef734358d935eee64d0b36'
CONTRAST_MANIFEST_SHA='fb697d9ae90898cc7c87753ebc07174163075fe025df7df5356a6ed6820d9619'
CONTRAST_REPORT_SHA='873f4e2c3accdc9d944c461b71190392ab70217563500f76bd8cafd9c50642a2'
FACTUAL_MANIFEST_SHA='4decc107d6e8ce71263f996ff99a5b177c99bd71391d06d29f909365a1c44210'
FRAMES=[0,5,10,15]
ARMS=['actual','normalized_zero','negative_normalized','reverse_steps']
ATOL=1e-12
RTOL=1e-10


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()


def sha(path):
    path=Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)),'symlink/non-file rejected: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read_json(path):
    require(Path(path).stat().st_size<=4*1024**2,'JSON size bound')
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def array_sha(x):
    a=np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def load_array(path,expected_sha,shape,dtype,allowed_dir):
    """Check bounded header/payload BEFORE read-only mmap; no pickle/trailing data."""
    path=Path(path);dtype=np.dtype(dtype);size=math.prod(shape)*dtype.itemsize
    require(path.is_absolute() and path.parent==Path(allowed_dir),'array path/parent rejected')
    require(sha(path)==expected_sha,'array SHA rejected')
    actual=path.stat().st_size
    require(size<actual<=size+1024,'array byte bound rejected')
    with path.open('rb') as f:
        version=np.lib.format.read_magic(f)
        require(version in ((1,0),(2,0)),'npy version rejected')
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        stored,fortran,dt=reader(f,max_header_size=1024)
        require(stored==shape and dt==dtype and not dt.hasobject and not fortran,'npy header contract rejected')
        require(f.tell()+size==actual,'trailing/truncated npy payload rejected')
    a=np.load(path,allow_pickle=False,mmap_mode='r',max_header_size=1024)
    require(np.isfinite(a).all(),'nonfinite array rejected')
    return a


def source_stats(path):
    result={}
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            k=n.targets[0].id
            if k in ('ACTION_MEAN','ACTION_STD','PROPRIO_MEAN','PROPRIO_STD'):
                require(isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.tensor','stats not literal tensor')
                result[k]=np.asarray(ast.literal_eval(n.value.args[0]),np.float32)
    require(set(result)=={'ACTION_MEAN','ACTION_STD','PROPRIO_MEAN','PROPRIO_STD'},'native stats missing')
    require(np.all(result['ACTION_STD']>0) and np.all(result['PROPRIO_STD']>0),'nonpositive std')
    return result


def normalize(u,proprio,stats):
    require(u.shape==(15,2) and u.dtype==np.float64,'native u64 contract')
    require(proprio.shape==(4,4) and proprio.dtype==np.float32,'native proprio contract')
    pixels=(100*u).astype(np.float32)
    packets=((pixels/np.float32(100)-stats['ACTION_MEAN'])/stats['ACTION_STD']).reshape(3,10)
    prop=(proprio-stats['PROPRIO_MEAN'])/stats['PROPRIO_STD']
    controls=np.repeat(packets[None],4,axis=0)
    controls[1,-1]=0;controls[2,-1]=-packets[-1]
    controls[3,-1]=packets[-1].reshape(5,2)[::-1].reshape(10)
    return pixels,packets,prop,controls


def recompute(features,successors,reported):
    require(features.shape==(4,256,384) and np.isfinite(features).all(),'four patch-feature frames required')
    require(set(successors)==set(ARMS) and set(reported)==set(ARMS),'exact four arms required')
    target=np.asarray(features[3:4],np.float64)
    rows={}
    for arm in ARMS:
        a=np.asarray(successors[arm],np.float64)
        require(a.shape==target.shape and np.isfinite(a).all(),'successor contract')
        cost=float(np.mean(np.square(a-target),dtype=np.float64))
        value=reported[arm]
        require(type(value) in (float,int) and np.isfinite(value),'finite reported cost required')
        rows[arm]=dict(mse=cost,reported_mse=value,absolute_receipt_error=abs(cost-value),
            receipt_consistent=bool(np.isclose(cost,value,atol=ATOL,rtol=RTOL)))
    for arm in ARMS[1:]:
        delta=rows['actual']['mse']-rows[arm]['mse']
        rows[arm].update(actual_minus_control=delta,ordering='actual_lower' if delta<0 else 'actual_higher' if delta>0 else 'exact_tie')
    return dict(arms=rows,rank_low_to_high=sorted(ARMS,key=lambda a:rows[a]['mse']),
        all_receipts_consistent=all(r['receipt_consistent'] for r in rows.values()),
        target_frame=15,coordinates=98304,tolerance_scope='CPU/GPU float64 reduction receipts only; NOT an accuracy/effectiveness threshold')


def journal_records(path,report):
    require(sha(path)==report['journal_sha256'],'factual journal SHA')
    require(path.stat().st_size<2*1024**2,'journal byte bound')
    previous=None;pending=None;records={};intents=Counter();entries=[]
    for i,line in enumerate(path.read_text().splitlines()):
        row=json.loads(line);payload={k:v for k,v in row.items() if k!='entry_sha256'}
        require(row['sequence']==i and row['previous_sha256']==previous and digest(payload)==row['entry_sha256'],'journal chain rejected')
        previous=row['entry_sha256'];e=row['event'];entries.append(e)
        if e['phase']=='intent':
            require(pending is None,'overlapping native intents')
            pending=(e['role'],e['operation']);intents[pending]+=1
        elif e['phase']=='returned':
            require(pending==(e['role'],e['operation']),'return without matching intent')
            pending=None
        elif e['phase']=='assets_durable':
            require(pending is None,'assets before native return')
            key=(e['role'],e['record']['frame']);require(key not in records,'duplicate frame receipt');records[key]=e['record']
    require(pending is None and entries[-1]['phase']=='terminal','incomplete journal')
    require(entries[-1]['summary_sha256']==digest({k:v for k,v in report.items() if k!='journal_sha256'}),'terminal report binding')
    require(set(records)=={(r,f) for r in ('a','b') for f in range(16)},'all32 frame receipts required')
    for role in ('a','b'):
        require({k[1]:v for k,v in intents.items() if k[0]==role}==dict(construct=1,seed=1,reset=1,step=15,close=1),'journal operation census')
    return records,entries


def protocol():
    return dict(schema=NAME,primary=PRIMARY,contrast=CONTRAST,scope='posthoc independent saved-array/input provenance arithmetic audit',
        array_policy='numeric bounded npy header+payload+SHA, mmap read-only, allow_pickle=False; no symlinks/escapes',
        target='saved official_patch_features[3:4], factual frame15; no counterfactual ground truth',
        reduction='independent NumPy float64 MSE over98304 coordinates, all four saved successors',
        arithmetic_tolerance=dict(atol=ATOL,rtol=RTOL,scientific_accuracy_threshold=False),
        ordering='exact sign of reconstructed MSE difference, no tolerance-based relabeling',
        normalization='native source literals; (100*u64).astype(float32)/float32(100), then native float32 mean/std',
        outcome_selection='all four arms/32factual frames retained, no replacement',
        feature_provenance_limit='hash-bound CUDA source/load receipts only; cannot independently regenerate RGB->features without neural execution',
        version_boundary='v11 failure retention/attempted-count repair is NOT asserted to cause ordering; compare all saved arrays/costs to v10',
        neural_calls=0,physics_calls=0,prospective_control=False,control_benefit_claim=False,P5_admission=False)


def prepare(root):
    top=root/f'results/{PRIMARY}';mp=top/'manifest_frozen.json';rp=top/'cuda_main_v1/report.json'
    require(sha(mp)==MANIFEST_FILE_SHA and sha(rp)==REPORT_FILE_SHA,'exact reviewed CUDA result rejected')
    m=read_json(mp);r=read_json(rp);f=root/f'results/{FACTUAL}/factual_main_v1'
    require(m['factual']==str(f),'factual root differs')
    files={str(mp):MANIFEST_FILE_SHA,str(rp):REPORT_FILE_SHA,**m['bindings']}
    files[str(f/'schedule_env_u.npy')]=sha(f/'schedule_env_u.npy')
    for k,a in r['artifacts'].items():
        p=root/a['path'] if not Path(a['path']).is_absolute() else Path(a['path'])
        require(p==top/'cuda_main_v1'/(k+'.npy'),'artifact path differs');files[str(p)]=a['sha256']
    contrast=root/f'results/{CONTRAST}';cm=contrast/'manifest_frozen.json';cr=contrast/'cuda_main_v1/report.json'
    require(sha(cm)==CONTRAST_MANIFEST_SHA and sha(cr)==CONTRAST_REPORT_SHA,'v10 contrast identity rejected')
    files[str(cm)]=CONTRAST_MANIFEST_SHA;files[str(cr)]=CONTRAST_REPORT_SHA
    for p,h in read_json(cm)['bindings'].items():
        require(p not in files or files[p]==h,'v10/v11 shared binding conflict');files[p]=h
    for k,rec in read_json(cr)['artifacts'].items():
        expected=contrast/'cuda_main_v1'/(k+'.npy')
        require(root/rec['path']==expected,'contrast artifact path rejected');files[str(expected)]=rec['sha256']
    for p in [Path(__file__).resolve(),root/'tests'/f'test_{NAME}.py',root.parent/'docs'/f'{NAME}.md']:files[str(p)]=sha(p)
    return dict(schema=NAME,status='prepared_array_only',protocol=protocol(),seals=files,neural_calls=0,physics_calls=0,P5_admission=False)


def audit(root,frozen):
    require(frozen==prepare(root),'audit manifest/source differs')
    for p,h in frozen['seals'].items():require(sha(p)==h,'frozen file SHA differs: '+p)
    top=root/f'results/{PRIMARY}';m=read_json(top/'manifest_frozen.json');r=read_json(top/'cuda_main_v1/report.json')
    checks={}
    def check(name,condition):checks[name]=bool(condition);require(condition,'audit mismatch: '+name)
    check('cuda_report_protocol_and_canonical_manifest_sha',r['schema']==PRIMARY and r['status']=='fresh_native_fidelity_recorded' and r['protocol']==m['protocol'] and r['manifest_sha256']==digest(m))
    check('noncontrol_scope',not r['P5_admission'] and not r['protocol']['prospective_control'] and not r['protocol']['target_optimized'] and not r['protocol']['training'] and r['new_physical_steps']==0)
    check('cuda_counts_receipt',r['counts']==dict(official_encoder=1,proprio_encoder=1,predictor=4) and r['counts_match'] and r['cuda_executed'])
    check('v11_attempted_completed_all_arms',r['attempted_counts']==r['counts'] and r['arm_status']=={a:'completed' for a in ARMS} and not r['source_check_errors'] and not r['finalization_errors'])
    check('frozen_model_receipt',r['model_before']==r['model_after'] and r['model_unchanged'] and all(all(v[k] for k in ('all_cuda_finite','eval','frozen')) for v in r['model_after'].values()))
    check('native_strict_loading_receipts',all(v['exact_match'] and not any(v[k] for k in ('missing_keys','unexpected_keys','shape_mismatches','dtype_mismatches')) for v in r['strict_loads'].values()))
    check('all_imported_sources_bound',all(p in m['bindings'] for p in r['imported_sources'].values()) and r['sources_unchanged'])
    f=Path(m['factual']);fm=f.parent/'manifest_frozen.json';fr=read_json(f/'report.json');fp=read_json(fm)
    check('fixed_physical_manifest',sha(fm)==FACTUAL_MANIFEST_SHA and fr['manifest_file_sha256']==FACTUAL_MANIFEST_SHA and fr['protocol']==fp['protocol'])
    check('complete_factual_pair',fr['status']=='fresh_native_nonzero_factual_recorded' and fr['error'] is None and fr['trajectory_frames']=={'a':16,'b':16} and fr['source_hashes_unchanged'])
    records,events=journal_records(f/'journal.jsonl',fr)
    schedule=np.array(fp['protocol']['schedule'],np.float64).reshape(15,2)
    a=load_array(f/'schedule_env_u.npy',events[0]['schedule_file_sha256'],(15,2),np.float64,f)
    check('schedule_durable_before_construct',np.array_equal(a,schedule) and events[0]['phase']=='preflight_intent' and events[0]['protocol']==fp['protocol'])
    dt=fr['native_call_receipts']['space_step_events'];census=Counter((e['context']['role'],e['context']['operation'],e['context']['frame']) for e in dt)
    check('302_source_native_substeps',len(dt)==302 and all(e['dt']==.01 for e in dt) and census==Counter({**{(role,'reset',None):1 for role in ('a','b')},**{(role,'step',frame):10 for role in ('a','b') for frame in range(1,16)}}))
    stats=source_stats(root/'external/jepa-wms/app/plan_common/datasets/pusht_dset.py')
    check('literal_native_stats',all(np.array_equal(np.array(m['stats'][k],np.float32),v) for k,v in stats.items()))
    selected={};consumer_results={}
    for role in ('a','b'):
        rgb=[];prop=[];states=[]
        for frame in range(16):
            raw=f/role/f'frame_{frame:02d}';files=records[role,frame]['files']
            for name,rec in files.items():check(f'{role}_frame{frame:02d}_{name}_hash',sha(raw/name)==rec['sha256'] and (raw/name).stat().st_size==rec['bytes'])
            s=load_array(raw/'state.npy',files['state.npy']['sha256'],(7,),np.float32,raw)
            p=load_array(raw/'proprio.npy',files['proprio.npy']['sha256'],(4,),np.float32,raw)
            image=load_array(raw/'rgb.npy',files['rgb.npy']['sha256'],(224,224,3),np.uint8,raw)
            full=read_json(raw/'full_public_state.json');ag=full['bodies']['agent'];block=full['bodies']['block']
            full_state=np.array(ag['position']+block['position']+[block['angle']%(2*np.pi)]+ag['velocity'],np.float32)
            check(f'{role}_frame{frame:02d}_proprio_full_public_state',np.array_equal(p,s[[0,1,5,6]]) and np.array_equal(s,full_state))
            if frame in FRAMES:rgb.append(np.array(image));prop.append(np.array(p));states.append(np.array(s))
            if frame:
                cmd=read_json(f/role/f'command_{frame:02d}.json');before=read_json(f/role/f'frame_{frame-1:02d}/full_public_state.json')
                u=np.array(cmd['raw_env_argument'],np.float64);pos=np.array(before['bodies']['agent']['position'],np.float64)
                check(f'{role}_command{frame:02d}_time_units_target',cmd['frame']==frame and cmd['action_index']==frame-1 and cmd['packet_index']==(frame-1)//5 and cmd['step_within_packet']==(frame-1)%5 and np.array_equal(u,schedule[frame-1]) and np.array_equal(cmd['pixel_displacement'],100*u) and np.array_equal(cmd['agent_xy_before'],pos) and np.array_equal(cmd['absolute_target'],pos+100*u) and np.array_equal(full['env']['latest_action'],pos+100*u) and cmd['pre_public_sha256']==sha(f/role/f'frame_{frame-1:02d}/full_public_state.json'))
        rgb=np.stack(rgb);prop=np.stack(prop);states=np.stack(states);pixels,packets,norm_prop,controls=normalize(schedule,prop,stats)
        selected[role]=dict(rgb=rgb,normalized_proprio=norm_prop,packets=packets)
        dest=f/role/'consumer';im=read_json(dest/'inputs_manifest.json')
        check(role+'_consumer_manifest_bound',sha(dest/'inputs_manifest.json')==fr['consumer_inputs'][role]['sha256'] and im['frame_order']==FRAMES and im['arms']==ARMS)
        expected=dict(raw_env_commands=schedule,raw_pixel_displacements=100*schedule,dataset_pixel_displacements=pixels,normalized_packets=packets,controls_packets=controls,states_frames=states,raw_proprio_frames=prop,normalized_proprio_frames=norm_prop,context_proprio=norm_prop[:3])
        for key,wanted in expected.items():
            filename=key+'.npy';receipt=im['file_hashes'][filename]
            saved=load_array(dest/filename,receipt['sha256'],wanted.shape,wanted.dtype,dest)
            check(role+'_consumer_'+key,np.array_equal(saved,wanted))
        for arm,bank in zip(ARMS,controls):check(role+'_control_'+arm,np.array_equal(np.array(m['controls'][arm],np.float32),bank) and np.array_equal(bank[:2],packets[:2]))
        hashes={k:array_sha(v) for k,v in selected[role].items()}
        check(role+'_cuda_input_hashes',hashes==m['input_sha256'])
        consumer_results[role]=dict(recomputed_input_sha256=hashes,all_exported_arrays_exact=True,
            direct_u32_vs_dataset_projection_max_abs=float(np.max(np.abs(((schedule.astype(np.float32)-stats['ACTION_MEAN'])/stats['ACTION_STD']).reshape(3,10)-packets))))
    check('all32_repeat_assets_exact',all(records['a',i]['files'][name]['sha256']==records['b',i]['files'][name]['sha256'] for i in range(16) for name in records['a',i]['files']))
    check('physical_report_rows_exact',len(fr['rows'])==16 and all(row['frame']==i and row['all_exact'] and all(row['exact'].values()) for i,row in enumerate(fr['rows'])))
    artifacts={}
    expected_names=['official_patch_features']+[a+'_successor' for a in ARMS]
    check('five_complete_saved_arrays',set(r['artifacts'])==set(expected_names))
    for key in expected_names:
        shape=(4,256,384) if key=='official_patch_features' else (1,256,384)
        rec=r['artifacts'][key];check(key+'_receipt_shape',rec['shape']==list(shape))
        artifacts[key]=load_array(top/'cuda_main_v1'/(key+'.npy'),rec['sha256'],shape,np.float32,top/'cuda_main_v1')
    arithmetic=recompute(artifacts['official_patch_features'],{a:artifacts[a+'_successor'] for a in ARMS},r['costs'])
    check('all_four_mse_receipts',arithmetic['all_receipts_consistent'])
    for arm in ARMS[1:]:check(arm+'_delta_receipt',np.isclose(arithmetic['arms'][arm]['actual_minus_control'],r['actual_minus_control'][arm],atol=ATOL,rtol=RTOL))
    contrast=root/f'results/{CONTRAST}';cm=read_json(contrast/'manifest_frozen.json');cr=read_json(contrast/'cuda_main_v1/report.json')
    check('v10_contrast_manifest_status',cr['schema']==CONTRAST and cr['status']=='fresh_native_fidelity_recorded' and cr['manifest_sha256']==digest(cm) and cr['protocol']==cm['protocol'])
    check('v10_v11_identical_input_controls_model_receipts',cm['input_sha256']==m['input_sha256'] and cm['controls']==m['controls'] and cr['model_before']==r['model_before'] and cr['model_after']==r['model_after'])
    old_arrays={}
    for key in expected_names:
        shape=artifacts[key].shape;rec=cr['artifacts'][key]
        old_arrays[key]=load_array(contrast/'cuda_main_v1'/(key+'.npy'),rec['sha256'],shape,np.float32,contrast/'cuda_main_v1')
        check('v10_v11_bitwise_'+key,rec['sha256']==r['artifacts'][key]['sha256'] and np.array_equal(old_arrays[key],artifacts[key]))
    old_arithmetic=recompute(old_arrays['official_patch_features'],{a:old_arrays[a+'_successor'] for a in ARMS},cr['costs'])
    check('v10_v11_cost_and_order_exact',cr['costs']==r['costs'] and cr['actual_minus_control']==r['actual_minus_control'] and old_arithmetic['all_receipts_consistent'] and old_arithmetic['rank_low_to_high']==arithmetic['rank_low_to_high'])
    for p,h in frozen['seals'].items():require(sha(p)==h,'sealed bytes changed during audit: '+p)
    return dict(schema=NAME,status='saved_fresh_fidelity_arithmetic_consistent',protocol=protocol(),checks=checks,
        check_count=len(checks),all_checks=all(checks.values()),consumer_inputs=consumer_results,arithmetic=arithmetic,
        v10_contrast=dict(all_five_array_files_identical=True,costs_and_order_identical=True,
            cause_boundary='failure recording/attempted-count fix did NOT change saved numerical outputs here; no causal explanation of reversal is established'),
        source_data_result_hashes_unchanged=True,manifest_file_sha256=MANIFEST_FILE_SHA,manifest_canonical_sha256=digest(m),
        report_file_sha256=REPORT_FILE_SHA,neural_calls=0,physics_calls=0,
        limitation='saved neural features/successors and runtime receipts are authenticated to supplied hashes, NOT independently regenerated; one factual target, no counterfactual ground truth or prospective control',P5_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','audit'])
    p.add_argument('--manifest',type=Path,default=ROOT/f'results/{NAME}/manifest_frozen.json');args=p.parse_args()
    value=prepare(ROOT) if args.command=='prepare' else audit(ROOT,read_json(args.manifest))
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
