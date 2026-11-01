"""Independent saved-array/JSON panel audit. No env, model or checkpoint decoding."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

NAME = 'p5_jepa_panel_audit_20261027_v1'
COLLECTOR = 'p5_jepa_factual_panel_20261027_v3'
CUDA = 'p5_jepa_fresh_fidelity_main_20261027_v12'
ROOT = Path(__file__).resolve().parents[2]
SEEDS = (2026102702,2026102703,2026102704)
COLLECTOR_SHA = '67b719d332a9405b6f4f6dd1adfdf2f5bdbd5c2e8bede1e16d73360a6470d516'
HELPER = 'p5_jepa_fresh_fidelity_audit_20261027_v1'
HELPER_SHA = '42dfe3b7df0eb973c61980b8f49c245f9c24b546eb483faf99f3ac34b965031f'
ARMS = ['actual','normalized_zero','negative_normalized','reverse_steps']
FRAMES = [0,5,10,15]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'non-file/symlink rejected')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()


def strict_pairs(pairs):
    result = {}
    for k,v in pairs:
        require(k not in result,'duplicate JSON key')
        result[k] = v
    return result


def read_json(path):
    require(Path(path).stat().st_size <= 4*1024**2,'JSON byte bound')
    sha(path)
    return json.loads(Path(path).read_text(),object_pairs_hook=strict_pairs,
                      parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def helpers(root):
    """Reuse only SHA-bound non-neural numeric/header/journal audit utilities."""
    source = root/f'experiments/spectral_reachability/{HELPER}.py'
    require(sha(source)==HELPER_SHA,'numeric audit helper identity rejected')
    spec = importlib.util.spec_from_file_location(NAME+'_numeric_helpers',source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_rows(rows):
    require(len(rows)==3 and [r['seed'] for r in rows]==list(SEEDS),'all three fixed seeds in order; no subset/replacement')
    return {r['seed']:r for r in rows}


def protocol():
    return dict(schema=NAME,seeds=list(SEEDS),physics='all96 saved frame assets,48 exact a/b pairs,906 receipt substeps; no integration',
                trajectory='all16 native float64 public-body positions/angles per role, agent and block kept separate',
                coverage='stored source-bound native coverage/reward/contact receipts, NOT independent polygon intersection',
                normalization='AST native stats, pixel32=(100*u64).astype(float32); normalize pixel32/100; all9 exports/role',
                fidelity='optional v12 only: all3 roots, all5 saved arrays/root, float64 MSE over98304 coordinates, exact-sign ordering',
                arithmetic_tolerance=dict(atol=1e-12,rtol=1e-10,scientific_accuracy_threshold=False),
                selection='all three fixed roots retained; incomplete/failed/not_run evidence rejected, never silently filtered',
                claim='agent movement is NOT object response/C2; factual target is NOT counterfactual/control effectiveness',
                neural_calls=0,physics_calls=0,checkpoint_decoding=False,C2_admission=False,P5_admission=False)


def prepare(root, cuda_manifest=None, cuda_report=None):
    cm = root/f'results/{COLLECTOR}/manifest_frozen.json'
    require(sha(cm)==COLLECTOR_SHA,'collector manifest identity rejected')
    collector = read_json(cm)
    require(collector['protocol']['seeds']==list(SEEDS),'fixed collector cohort differs')
    panel = root/f'results/{COLLECTOR}/factual_main_v1'
    seals = dict(collector['seals'])
    seals[str(cm)] = COLLECTOR_SHA
    for path in (Path(__file__).resolve(),root/f'tests/test_{NAME}.py',root.parent/f'docs/{NAME}.md',root/f'experiments/spectral_reachability/{HELPER}.py'):
        seals[str(path)] = sha(path)
    require(seals[str(root/f'experiments/spectral_reachability/{HELPER}.py')]==HELPER_SHA,'helper SHA')
    files = list(panel.rglob('*'))
    require(len(files)<1024 and not any(p.is_symlink() for p in files),'panel node/symlink bound')
    require(sum(p.stat().st_size for p in files if p.is_file())<40_000_000,'panel payload byte bound')
    require((panel/'report.json').is_file(),'complete terminal panel report required')
    for path in files:
        if path.is_file():
            seals[str(path)] = sha(path)
    mode = 'physics'
    require((cuda_manifest is None)==(cuda_report is None),'both CUDA manifest/report required')
    if cuda_manifest is not None:
        mode = 'fidelity'
        cuda_manifest,cuda_report = Path(cuda_manifest),Path(cuda_report)
        m,r = read_json(cuda_manifest),read_json(cuda_report)
        require(m['schema']==r['schema']==CUDA and r['manifest_sha256']==digest(m),'v12 source/result identity')
        fixed_rows(m['roots']);fixed_rows(r['rows'])
        require(m['collector_manifest']==str(cm),'same collector source required')
        for p,h in m['bindings'].items():
            require(p not in seals or seals[p]==h,'CUDA/physics binding conflict')
            seals[p] = h
        for path in (cuda_manifest,cuda_report):
            seals[str(path)] = sha(path)
        for row in r['rows']:
            require(set(row['artifacts'])=={'official_patch_features',*[a+'_successor' for a in ARMS]},'all15 CUDA arrays required')
            for key,rec in row['artifacts'].items():
                path = Path(rec['path'])
                require(path==cuda_report.parent/f"seed_{row['seed']}_{key}.npy",'exact CUDA artifact path')
                seals[str(path)] = rec['sha256']
    for p,h in seals.items():
        require(sha(p)==h,'sealed bytes differ: '+p)
    return dict(schema=NAME,status='prepared_array_only',mode=mode,protocol=protocol(),panel=str(panel),seals=seals,
                cuda_manifest=str(cuda_manifest) if cuda_manifest else None,cuda_report=str(cuda_report) if cuda_report else None,
                neural_calls=0,physics_calls=0,C2_admission=False,P5_admission=False)


def vector(value,shape,dtype):
    require(value['shape']==list(shape) and value['dtype']==np.dtype(dtype).str,'typed JSON vector contract')
    a = np.asarray(value['values'],dtype)
    require(a.shape==shape and np.isfinite(a).all(),'finite typed vector required')
    return a


def trajectory_summary(full, transitions):
    require(len(full)==len(transitions)==16,'complete16-frame trajectory required')
    agent = np.array([x['bodies']['agent']['position'] for x in full],np.float64)
    block = np.array([x['bodies']['block']['position'] for x in full],np.float64)
    angles = np.array([x['bodies']['block']['angle'] for x in full],np.float64)
    coverage = [t['info']['final_coverage'] for t in transitions[1:]]
    contacts = [x['env']['n_contact_points'] for x in full]
    def travel(x):
        return dict(path_length_px=float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum()),net_displacement_px=float(np.linalg.norm(x[-1]-x[0])),
                    max_displacement_from_root_px=float(np.linalg.norm(x-x[0],axis=1).max()),xy_frames=x.tolist())
    stationary = bool(np.array_equal(block,np.repeat(block[:1],16,axis=0)) and np.array_equal(angles,np.repeat(angles[:1],16)))
    return dict(agent=travel(agent),object=travel(block),object_angle_native_rad=angles.tolist(),object_angle_net_rad=float(angles[-1]-angles[0]),
                object_angle_path_abs_rad=float(np.abs(np.diff(angles)).sum()),object_pose_exactly_stationary=stationary,
                raw_contact_points_per_frame=contacts,total_recorded_contact_points=sum(contacts),
                coverage_frames1_to15=coverage,coverage_reset=None,coverage_first_step=coverage[0],coverage_last=coverage[-1],
                coverage_min=min(coverage),coverage_max=max(coverage),coverage_last_minus_first_step=coverage[-1]-coverage[0],
                coverage_exactly_constant=all(c==coverage[0] for c in coverage),rewards_frames1_to15=[t['reward'] for t in transitions[1:]],
                agent_only_motion=bool(stationary and np.any(agent!=agent[:1])),C2_admission=False,
                coverage_boundary='source-bound receipt only; reset coverage not returned, no geometry/env recomputation; contacts not attributed to a particular body pair')


def root_identity(seed,rgb,prop):
    def ah(a):
        a = np.ascontiguousarray(a)
        return hashlib.sha256(str(a.dtype).encode()+json.dumps(list(a.shape)).encode()+a.tobytes()).hexdigest()
    return digest(dict(seed=seed,initial_rgb_sha256=ah(rgb[0]),initial_proprio_sha256=ah(prop[0])))


def audit(root,frozen):
    require(frozen==prepare(root,frozen['cuda_manifest'],frozen['cuda_report']),'audit source/manifest differs')
    op = helpers(root)
    cm = read_json(root/f'results/{COLLECTOR}/manifest_frozen.json')
    panel = Path(frozen['panel'])
    report = read_json(panel/'report.json')
    checks = {}
    def check(name,condition):
        checks[name] = bool(condition)
        require(condition,'audit mismatch: '+name)
    check('complete_source_bound_panel',report['schema']==COLLECTOR and report['status']=='fresh_native_nonzero_panel_recorded' and report['manifest_file_sha256']==COLLECTOR_SHA and report['protocol']==cm['protocol'] and report['source_hashes_unchanged'] and report['runtime_error'] is None and report['source_error'] is None and not report['physics_execution_indeterminate'])
    rows = fixed_rows(report['rows'])
    stats = op.source_stats(root/'external/jepa-wms/app/plan_common/datasets/pusht_dset.py')
    check('native_literal_stats',all(np.array_equal(v,np.array(cm['native_stats'][k],np.float32)) for k,v in stats.items()))
    schedule = np.array(cm['protocol']['schedule'],np.float64).reshape(15,2)
    check('panel_schedule',np.array_equal(op.load_array(panel/'schedule_env_u.npy',frozen['seals'][str(panel/'schedule_env_u.npy')],(15,2),np.float64,panel),schedule))
    total = Counter();summaries=[];root_inputs={}
    for seed in SEEDS:
        prefix = str(seed)+'_'
        def ck(name,condition):
            check(prefix+name,condition)
        directory = panel/f'seed_{seed}'
        row = rows[seed]
        ck('panel_row_paths',row['root']==directory.name and row['report']==directory.name+'/report.json' and row['journal']==directory.name+'/journal.jsonl' and row['status']=='fresh_native_panel_root_recorded')
        r = read_json(directory/'report.json')
        ck('root_report_identity',sha(directory/'report.json')==row['report_sha256'] and sha(directory/'journal.jsonl')==row['journal_sha256']==r['journal_sha256'] and r['seed']==seed and r['schema']==COLLECTOR and r['protocol']==cm['protocol'] and r['manifest_file_sha256']==COLLECTOR_SHA and r['source_hashes_unchanged'] and not r['errors'] and r['role_status']=={'a':'completed','b':'completed'} and r['trajectory_frames']=={'a':16,'b':16})
        records,events = op.journal_records(directory/'journal.jsonl',r)
        ck('journal_preflight',events[0]==dict(phase='panel_registered_before_runtime',seed=seed,manifest_file_sha256=COLLECTOR_SHA) and events[1]['phase']=='preflight_intent' and events[1]['seed']==seed and events[1]['protocol']==cm['protocol'])
        ck('root_schedule',np.array_equal(op.load_array(directory/'schedule_env_u.npy',events[1]['schedule_file_sha256'],(15,2),np.float64,directory),schedule))
        intents = [e for e in events if e['phase']=='intent']
        ck('operation_order',[(e['role'],e['operation']) for e in intents]==[(role,operation) for role in ('a','b') for operation in ['construct','seed','reset',*['step']*15,'close']])
        ck('seed_before_reset',all(e['details']['seed']==seed for e in intents if e['operation'] in ('seed','reset')))
        commands = {(e['role'],e['details']['frame']):e['details'] for e in intents if e['operation']=='step'}
        counts = r['native_call_receipts']['python_call_entries'];total.update(counts)
        ck('call_census',all(counts.get(k)==v for k,v in cm['protocol']['per_root_expected'].items()))
        steps = r['native_call_receipts']['space_step_events']
        expected = Counter({**{(role,'reset',None):1 for role in ('a','b')},**{(role,'step',i):10 for role in ('a','b') for i in range(1,16)}})
        ck('302_substep_contexts',len(steps)==302 and all(e['dt']==.01 for e in steps) and Counter((e['context']['role'],e['context']['operation'],e['context']['frame']) for e in steps)==expected)
        role_metrics={};input_bank={}
        for role in ('a','b'):
            states=[];props=[];rgbs=[];full=[];transitions=[]
            for i in range(16):
                raw=directory/role/f'frame_{i:02d}';receipt=records[role,i]
                ck(f'{role}_{i:02d}_asset_closure',set(receipt['files'])=={'rgb.npy','state.npy','proprio.npy','full_public_state.json','transition.json'})
                for name,rec in receipt['files'].items():
                    ck(f'{role}_{i:02d}_{name}_identity',sha(raw/name)==rec['sha256']==records['a',i]['files'][name]['sha256'] and (raw/name).stat().st_size==rec['bytes'])
                state=op.load_array(raw/'state.npy',receipt['files']['state.npy']['sha256'],(7,),np.float32,raw)
                prop=op.load_array(raw/'proprio.npy',receipt['files']['proprio.npy']['sha256'],(4,),np.float32,raw)
                rgb=op.load_array(raw/'rgb.npy',receipt['files']['rgb.npy']['sha256'],(224,224,3),np.uint8,raw)
                f,t=read_json(raw/'full_public_state.json'),read_json(raw/'transition.json');full.append(f);transitions.append(t)
                ag,block=f['bodies']['agent'],f['bodies']['block']
                expected_state=np.array(ag['position']+block['position']+[block['angle']%(2*np.pi)]+ag['velocity'],np.float32)
                ck(f'{role}_{i:02d}_state_public',np.array_equal(state,expected_state) and np.array_equal(prop,state[[0,1,5,6]]) and state.tolist()==receipt['state'] and prop.tolist()==receipt['proprio'] and f['env']['seed']==seed)
                if i in FRAMES:
                    states.append(np.array(state));props.append(np.array(prop));rgbs.append(np.array(rgb))
                ck(f'{role}_{i:02d}_coverage_history',len(f['env']['coverage_arr'])==i and (not i or f['env']['coverage_arr'][:-1]==full[-2]['env']['coverage_arr']))
                if i:
                    cmd=read_json(directory/role/f'command_{i:02d}.json');pos=np.array(full[-2]['bodies']['agent']['position'],np.float64);u=schedule[i-1]
                    ck(f'{role}_{i:02d}_command_timing_units',cmd==commands[role,i] and cmd['frame']==i and cmd['action_index']==i-1 and cmd['packet_index']==(i-1)//5 and cmd['step_within_packet']==(i-1)%5 and np.array_equal(cmd['raw_env_argument'],u) and np.array_equal(cmd['pixel_displacement'],100*u) and np.array_equal(cmd['agent_xy_before'],pos) and np.array_equal(cmd['absolute_target'],pos+100*u) and np.array_equal(f['env']['latest_action'],pos+100*u) and cmd['pre_public_sha256']==sha(directory/role/f'frame_{i-1:02d}/full_public_state.json') and cmd['dt']==.01 and cmd['expected_substeps']==10)
                    info=t['info'];coverage=f['env']['coverage_arr'][-1]
                    ck(f'{role}_{i:02d}_transition_native_fields',t['kind']=='step' and t['done'] is False and np.array_equal(vector(info['state'],(7,),np.float32),state) and np.array_equal(vector(info['pos_agent'],(2,),np.float64),ag['position']) and np.array_equal(vector(info['vel_agent'],(2,),np.float64),ag['velocity']) and np.array_equal(vector(info['block_pose'],(3,),np.float64),block['position']+[block['angle']]) and info['goal_pose']==f['env']['goal_pose'])
                    ck(f'{role}_{i:02d}_coverage_reward_contact_receipt',0<=coverage<=1 and info['final_coverage']==coverage and info['max_coverage']==max(f['env']['coverage_arr']) and t['reward']==float(np.clip(coverage/f['env']['success_threshold'],0,1)) and info['n_contacts']==int(np.ceil(f['env']['n_contact_points']/10)))
                else:
                    ck(role+'_reset_transition',t=={'kind':'reset'})
            states,props,rgbs=np.stack(states),np.stack(props),np.stack(rgbs)
            pixels,packets,norm,controls=op.normalize(schedule,props,stats)
            dest=directory/role/'consumer';im=read_json(dest/'inputs_manifest.json')
            ck(role+'_consumer_source_identity',sha(dest/'inputs_manifest.json')==r['consumer_inputs'][role]['sha256'] and r['consumer_inputs'][role]['path']==f'{role}/consumer/inputs_manifest.json' and im['schema']==COLLECTOR and im['seed']==seed and im['role']==role and im['frame_order']==FRAMES and im['arms']==ARMS and im['context_frames']==[0,5,10] and im['target_frame']==15)
            wanted=dict(raw_env_commands=schedule,raw_pixel_displacements=100*schedule,dataset_pixel_displacements=pixels,normalized_packets=packets,controls_packets=controls,states_frames=states,raw_proprio_frames=props,normalized_proprio_frames=norm,context_proprio=norm[:3])
            ck(role+'_exact9_exports',set(im['file_hashes'])=={k+'.npy' for k in wanted})
            for key,value in wanted.items():
                rec=im['file_hashes'][key+'.npy'];path=dest/(key+'.npy')
                ck(role+'_export_'+key,path.stat().st_size==rec['bytes'] and np.array_equal(op.load_array(path,rec['sha256'],value.shape,value.dtype,dest),value))
            for j,frame in enumerate(FRAMES):
                rec=im['frames'][j]
                ck(f'{role}_consumer_frame{frame}',rec['frame']==frame and all(rec[k]==f'{role}/frame_{frame:02d}/{filename}' for k,filename in dict(rgb='rgb.npy',state='state.npy',proprio='proprio.npy',full_public_state='full_public_state.json').items()) and rec['files']==records[role,frame]['files'])
            input_bank[role]=dict(rgb=rgbs,normalized_proprio=norm,packets=packets,controls=controls,raw_proprio=props)
            role_metrics[role]=trajectory_summary(full,transitions)
        ck('metrics_repeat_exact',role_metrics['a']==role_metrics['b'])
        ck('reported16pairs',len(r['rows'])==16 and all(x['frame']==i and x['all_exact'] and all(x['exact'].values()) for i,x in enumerate(r['rows'])))
        inp=input_bank['a'];root_inputs[seed]=inp
        summaries.append(dict(seed=seed,identity=root_identity(seed,inp['rgb'],inp['raw_proprio']),roles=role_metrics,
                              input_sha256={k:op.array_sha(inp[k]) for k in ('rgb','normalized_proprio','packets')},all32_frames_exact=True,substeps=302))
    check('906_total_substeps',total['space_step']==906 and all(total[k]==v for k,v in cm['protocol']['panel_expected'].items()))
    fidelity=None
    if frozen['mode']=='fidelity':
        fidelity=audit_fidelity(frozen,summaries,root_inputs,op,check)
    for path,h in frozen['seals'].items():
        require(sha(path)==h,'bytes changed during audit: '+path)
    return dict(schema=NAME,status='saved_panel_fidelity_audited' if fidelity else 'saved_panel_physics_audited',protocol=protocol(),checks=checks,check_count=len(checks),all_checks=all(checks.values()),
                rows=summaries,frame_count=96,exact_frame_pairs=48,substeps=906,all3_object_poses_exactly_stationary=all(x['roles']['a']['object_pose_exactly_stationary'] for x in summaries),
                fidelity=fidelity,source_data_hashes_unchanged=True,audit_manifest_sha256=digest(frozen),neural_calls=0,physics_calls=0,C2_admission=False,P5_admission=False)


def audit_fidelity(frozen,physical,inputs,op,check):
    m,r=read_json(frozen['cuda_manifest']),read_json(frozen['cuda_report'])
    rows,prepared=fixed_rows(r['rows']),fixed_rows(m['roots'])
    check('v12_complete_bound_result',r['schema']==CUDA and r['status']=='fresh_native_panel_fidelity_recorded' and r['protocol']==m['protocol'] and r['manifest_sha256']==digest(m) and r['error'] is None and not r['finalization_errors'] and not r['source_check_errors'] and r['sources_unchanged'] and r['new_physical_steps']==0 and not r['P5_admission'])
    check('v12_model_receipts',r['model_unchanged'] and r['model_before']==r['model_after'] and all(all(v[k] for k in ('all_cuda_finite','eval','frozen')) for v in r['model_after'].values()))
    check('v12_strict_receipts',all(v['exact_match'] and not any(v[k] for k in ('missing_keys','unexpected_keys','shape_mismatches','dtype_mismatches')) for v in r['strict_loads'].values()))
    expected=dict(official_encoder=1,proprio_encoder=1,predictor=4);frames=dict(official_encoder=4,proprio_encoder=3,predictor=12)
    check('v12_total_counts',r['counts']==r['attempted_counts']=={k:3*v for k,v in expected.items()} and r['frame_counts']==r['attempted_frame_counts']=={k:3*v for k,v in frames.items()})
    result=[]
    for phys in physical:
        seed=phys['seed'];row=rows[seed];p=prepared[seed];inp=inputs[seed]
        def ck(name,condition):
            check(str(seed)+'_v12_'+name,condition)
        ck('source_input_identity',row['root_sha256']==p['root_sha256']==phys['identity'] and p['input_sha256']==phys['input_sha256'] and p['factual']==str(Path(frozen['panel'])/f'seed_{seed}') and all(np.array_equal(np.array(p['controls'][a],np.float32),inp['controls'][i]) for i,a in enumerate(ARMS)))
        ck('completed_frozen_counts',row['status']=='completed' and row['error'] is None and not row['finalization_errors'] and row['counts']==row['attempted_counts']==expected and row['frame_counts']==row['attempted_frame_counts']==frames and row['arm_status']=={a:'completed' for a in ARMS} and row['model_unchanged'] and row['model_after']==r['model_before'])
        arrays={}
        for key,rec in row['artifacts'].items():
            shape=(4,256,384) if key=='official_patch_features' else (1,256,384)
            ck(key+'_shape',rec['shape']==list(shape) and rec['dtype']=='float32' and Path(rec['path']).stat().st_size==rec['bytes'])
            arrays[key]=op.load_array(Path(rec['path']),rec['sha256'],shape,np.float32,Path(frozen['cuda_report']).parent)
        costs=op.recompute(arrays['official_patch_features'],{a:arrays[a+'_successor'] for a in ARMS},row['costs'])
        ck('all4_mse_receipts',costs['all_receipts_consistent'])
        for a in ARMS[1:]:
            ck(a+'_delta',np.isclose(costs['arms'][a]['actual_minus_control'],row['actual_minus_control'][a],atol=1e-12,rtol=1e-10))
        result.append(dict(seed=seed,arithmetic=costs,object_pose_exactly_stationary=phys['roles']['a']['object_pose_exactly_stationary']))
    return dict(rows=result,actual_lower_counts={a:sum(x['arithmetic']['arms'][a]['ordering']=='actual_lower' for x in result) for a in ARMS[1:]},
                actual_higher_counts={a:sum(x['arithmetic']['arms'][a]['ordering']=='actual_higher' for x in result) for a in ARMS[1:]},
                scope='all3 retained calibration roots against their actual targets only; no counterfactual ground truth, efficacy/control benefit or C2/P5 admission')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','audit'])
    parser.add_argument('--manifest',type=Path,default=ROOT/f'results/{NAME}/manifest_physics_frozen.json')
    parser.add_argument('--cuda-manifest',type=Path)
    parser.add_argument('--cuda-report',type=Path)
    args=parser.parse_args()
    value=prepare(ROOT,args.cuda_manifest,args.cuda_report) if args.command=='prepare' else audit(ROOT,read_json(args.manifest))
    print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
