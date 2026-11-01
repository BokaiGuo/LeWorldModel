"""Saved-only fixed beam/GN/D2P/matched-CEM/nominal-MPPI comparison.

No inference, physics, fitting, selection, author-value ranking or admission.
Identity partitions retain ALL ten branches, including unsuccessful tasks.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import os
import shutil
import numpy as np
from experiments.spectral_reachability import agent_kinodynamic_matched_rgb_20261028_v1 as beam
from experiments.spectral_reachability import moving_prediction_matched_sampling_rgb_cuda_20261027_v2 as sampling
from experiments.spectral_reachability import p1_constructive_moving_history_rgb_cuda_20261027_v1 as gn
from experiments.spectral_reachability import p1_moving_baseline_comparison_20261027_v2 as legacy

NAME='agent_baseline_matched_rgb_comparison_20261028_v1'
ROOT=beam.ROOT
arrays=beam.arrays
require,read,sha,canonical,digest=arrays.require,arrays.read,arrays.sha,arrays.canonical,arrays.digest
ARMS=(beam.ARM,'line_search_GN','full10_conditional_D2P','CEM','kernel_faithful_mppi_nominal')
EXPECTED=[(s,a) for s in (0,1) for a in ARMS]
CAP,FLOOR=2*1024**2,200*1024**2
SPECS={
 'beam':('agent_kinodynamic_matched_rgb_20261028_v1_cuda_MAIN01','52e6d7c025dcb836aed3ec28aefc9b249d449be2eb19618c8e6ebdae7ea4db49'),
 'gn':('p1_constructive_moving_history_rgb_cuda_20261027_v1_cuda_main01','7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71'),
 'sampling':('moving_prediction_matched_sampling_rgb_cuda_20261027_v2_MAIN01','63c48c16b91d16b8f20c5f6ba08010d366544558f6e7eb9dd7a9542a5e50fa2f')}
PINS={beam.__file__:'a3187726bc3fdc1e07dd6dfed10378d388794787824a744ade4db40b944498be',
 sampling.__file__:'dde4a59a92583161e488c906aaffe64f5f1e33c3be6bfcae9c46bd8172c83acc',
 gn.__file__:'ed9b80bebb212434d60ddd55d3a3cba01e5cc55306170a0ea6f9a167bb156469',
 legacy.__file__:'2464d72928056358d860ad68f7bdb855884a860efa8ab39f50c8773a3a3974aa'}


def fingerprint(value):return digest(canonical(value))


def vector(v,n):
    a=np.asarray(v,np.float64);require(a.shape==(n,) and np.isfinite(a).all(),'finite exact native vector'+str(n));return a


def key(row):
    s=row['root_slot'];require(type(s) is int and (s,row['arm']) in EXPECTED,'fixed literal slot/arm');return s,row['arm']


def domain(plan):
    """Check ORIGINAL logits, never infer a ball from inverse-tanh(raw)."""
    s=plan['source_slot'];a=plan['arm'];raw=np.asarray(plan['commands']);cap=float(.75*np.sqrt(5))
    epsilon=0.
    if a==beam.ARM:
        bank=beam.producer.bank(s);logits=np.asarray([bank['logits'][i] for i in plan['selected_path']],np.float64)
        require(plan['bank_sha256']==beam.producer.core.digest(bank),'original beam bank')
        support='fixed20 seeded packet bank; H1beam6/H2terminal120; finite subset, NOT RRT'
        precision='source float64 solid-ball points, float32 before tanh'
    else:
        logits=np.asarray(plan['logits'],np.float64)
        support={'line_search_GN':'four GN updates and fixed line trials; local AD-guided proposals',
            'full10_conditional_D2P':'fixed10 conditional prefix/own-suffix candidates; spectral+AD',
            'CEM':'72 H2 sampled candidates, 3x24, elite4 projected Gaussian updates',
            'kernel_faithful_mppi_nominal':'72 H2 sampled candidates, 3x24, box parameters then packet projection; local nominal-MPPI NOT official full MPPI'}[a]
        precision='float32 feasible/inward projected packet logits'
        if a in ('line_search_GN','full10_conditional_D2P'):
            # EXACT inherited check_prediction ABI, not a new scientific tolerance.
            epsilon=1e-12
        if a=='full10_conditional_D2P':precision='source float64 spectral radius projection; legacy check_prediction cap+1e-12'
    require(logits.shape==(2,10) and np.isfinite(logits).all() and np.all(np.linalg.norm(logits,axis=1)<=cap+epsilon),
        'selected ORIGINAL logits violate their sealed producer validation bound')
    require(raw.shape==(10,2) and np.isfinite(raw).all() and np.array_equal(raw,raw.astype(np.float32)) and
        np.array_equal(raw,np.tanh(logits.astype(np.float32)).reshape(10,2)),'literal raw10/tanh(logits), no rescaling')
    return dict(admissible=dict(packet_steps=5,packets=2,control_dim=2,logit_cap=cap,map='float32 tanh then time-major raw10x2'),
        selected_logit_norms=np.linalg.norm(logits,axis=1).tolist(),selected_raw_sha256=digest(raw.astype(np.float32).tobytes()),
        validation_epsilon=epsilon,epsilon_source='p1_conditional_task_admission_20261027_v1.check_prediction' if epsilon else 'producer strict cap',
        strictly_inside_closed_balls=bool(np.all(np.linalg.norm(logits,axis=1)<=cap)),
        logit_norm_excess=(np.linalg.norm(logits,axis=1)-cap).tolist(),search_support=support,logit_rounding=precision,equal_search_coverage=False)


def native(goal,state):
    d=vector(state,7)-vector(goal,7);angle=abs(float(d[4]));old=min(angle,2*np.pi-angle)
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))));xy=float(np.linalg.norm(d[:4]))
    return dict(native_distance=float(np.linalg.norm(d)),native_reward=-float(np.linalg.norm(d)),native_success=bool(xy<20 and old<np.pi/9),
        native_angle_term=old,wrapped_allxy_success=bool(xy<20 and wrapped<np.pi/9),object_xy_error=float(np.linalg.norm(d[2:4])),
        agent_xy_error=float(np.linalg.norm(d[:2])),velocity_error=float(np.linalg.norm(d[5:])),
        success_includes_agent_and_object=True,success_has_velocity_threshold=False)


def identity(plan,row,export,task,source_sha):
    require(plan['task_id']==export['task_id'] and plan['task_sha256']==gn.EXPORT_SHAS[plan['source_slot']] and
        plan['blind_sha256']==export['blind_sha256'] and export['source_slot']==plan['source_slot'],'original moving task/slot/SHA')
    cp=next(c for c in row['checkpoints'] if c['step']==0);tc=export['task_contract']
    require(cp['pixel_sha256']==tc['pixel_sha256'][2] and cp['state']==row['root_state'] and
        cp['snapshot']['state']==cp['observation_state']==cp['state'],'actual root/state/RGB bridge')
    root_hash=beam.saved.source.root_hash(vector(row['root_state'],7),vector(row['goal'],7))
    if 'selected_plan' in row:require(row['selected_plan']['root_sha256']==root_hash,'actual root+native goal digest')
    return dict(source_slot=plan['source_slot'],sourcepack_sha256=source_sha,root_sha256=root_hash,
        native_root=deepcopy(row['root_state']),native_goal=deepcopy(row['goal']),root_snapshot=deepcopy(cp['snapshot']),
        task_export=plan['task_export'],task_sha256=plan['task_sha256'],task_id=plan['task_id'],blind_sha256=plan['blind_sha256'],
        root_pixel_sha256=cp['pixel_sha256'],rgb_history_pixel_sha256=tc['pixel_sha256'][:3],goal_pixel_sha256=tc['pixel_sha256'][3],
        goal_latent=deepcopy(task['goal']),root_context=deepcopy(task['root_context']),Q_sha256=fingerprint(task['projection']),
        normalization=deepcopy(tc['normalization']),model_bindings=deepcopy(tc['model_bindings']),model_tensor_sha256=arrays.MODEL_TENSOR_SHA,
        raw_history=deepcopy(task['root_raw_action_history']),raw_history_sha256=tc['raw_action_sha256'],
        history_layout=task['root_context']['layout'],horizon=task['horizon'],packet_steps=task['packet_steps'],with_target=tc['with_target'])


def record(plan,physical_row,export,task,zs,resources,source_sha,refs,root_encoded=None):
    s,a=plan['source_slot'],plan['arm'];require((s,a) in EXPECTED and physical_row['root_slot']==s and physical_row['arm']==a,'same row/plan')
    require(physical_row['initial_exact'] is True and physical_row['full_horizon'] is True and physical_row['own_returned']==10 and
        physical_row['first_error'] is None and physical_row['secondary_errors']==[] and
        [c['step'] for c in physical_row['checkpoints']]==[0,5,10],'all real endpoints/negative rows retained')
    ident=identity(plan,physical_row,export,task,source_sha);goal=vector(task['goal'],192)
    root=vector(task['root_context']['latent_history'][-1],192);rootq=float(np.mean((root-goal)**2));cps=[]
    for step in (0,5,10):
        cp=next(c for c in physical_row['checkpoints'] if c['step']==step)
        if step==0:z=root if root_encoded is None else vector(root_encoded,192)
        else:z=vector(zs[step],192)
        metric=native(physical_row['goal'],cp['state'])
        if step in (0,10):
            old=physical_row['initial_metrics'] if step==0 else physical_row['final_metrics']
            require(all(old[k]==metric[k] for k in ('native_distance','native_success','native_angle_term','wrapped_allxy_success')),'saved full7 native metrics recomputed')
        if step==10:require(metric['native_success']==physical_row['native_success'],'native terminal success')
        q=float(np.mean((z-goal)**2))
        cps.append(dict(step=step,Q=q,Q_change_from_original_root=q-rootq,native=metric,
            latent_source='original_task_root_not_new_RGB_encode' if step==0 and root_encoded is None else 'actual_saved_RGB_encoding',
            latent_sha256=digest(z.astype(np.float32).tobytes()),pixel_sha256=cp['pixel_sha256'],PNG_sha256=cp['file_sha256']))
    return dict(root_slot=s,arm=a,identity=ident,identity_sha256=fingerprint(ident),domain=domain(plan),
        selection_digest=physical_row['selection_digest'],compiler_selection_sha256=plan['selection_sha256'],
        commands=deepcopy(plan['commands']),physical_status=physical_row['status'],own_returned=10,native_success=physical_row['native_success'],
        root_Q_original_task=rootq,model_terminal_Q=plan['model_terminal_cost'],checkpoints=cps,resources=deepcopy(resources),provenance=refs)


def load_gn(bind):
    output,h=SPECS['gn'];path=ROOT/'results'/output/'report.json';bind(path,h);r=read(path)
    binding=read(path.parent/'binding.json');p=gn.verify(binding['prepared'],binding['prepare_sha256']);bind(binding['prepared'],binding['prepare_sha256'])
    require(r['schema']==gn.NAME+'.report' and r['status']=='moving_selected_rgb_encoded_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['source_verified'] is r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']==arrays.MODEL_TENSOR_SHA and
        r['contract']==p['contract'] and r['inputs']==p['arguments'] and r['prepare_sha256']==binding['prepare_sha256'] and
        r['P1_admission'] is r['training'] is False and r['new_predict']==r['new_AD']==r['new_physics']==0,'actual original GN/D2P RGB scope')
    names={'binding.json','events.jsonl'}|{f'image_{i:02d}.native192.npy' for i in range(8)}
    require(set(r['artifact_sha256'])==names and {f.name for f in path.parent.iterdir()}==names|{'report.json'},'all10 original RGB artifacts')
    for name,expected in r['artifact_sha256'].items():bind(arrays.flat(path.parent,name),expected)
    raw=(path.parent/'events.jsonl').read_bytes();require(raw.endswith(b'\n') and len(raw)<=8*1024**2,'complete RGB events')
    events=[arrays.decode(line) for line in raw.splitlines()]
    zs=[legacy.latent_file(path.parent,x['latent']) for x in r['rows']]
    legacy.rgb_receipts(r,p,events,zs,gn.compare)  # Saved exact six-event ABI, no encoder/model execution.
    for f,expected in p['bindings'].items():bind(f,expected)
    ph=Path(p['arguments']['physical_report']);pr=read(ph)
    journal=gn.physical.journal(ph.parent/'journal.jsonl',pr['journal_chain_tail'])
    for e in journal:
        if e['kind']=='fresh_observation':
            c=e['receipt'];fp=arrays.flat(ph.parent,c['png']);bind(fp,c['file_sha256'])
            arrays.load_image(dict(kind='checkpoint',path=str(fp),file_sha256=c['file_sha256'],pixel_sha256=c['pixel_sha256']))
    selected=gn.planner.consume(Path(p['arguments']['planner_report']).parent,p['arguments']['planner_sha'])['plans']
    mr=gn.planner.read(p['arguments']['planner_report']);rows=[]
    for plan in selected:
        s,a=plan['source_slot'],plan['arm'];row=next(x for x in pr['rows'] if (x['root_slot'],x['arm'])==(s,a))
        z={5:None,10:None}
        for item,value in zip(p['images'],zs,strict=True):
            if (item['root_slot'],item['arm'])==(s,a):z[5 if item['role']=='prefix5' else 10]=value
        require(all(v is not None for v in z.values()),'both true endpoints')
        e=gn.moving.read(plan['task_export']);t=gn.moving.load_task(plan['task_export'],plan['task_sha256'])['gn_task']
        native_row=next(x for x in mr['rows'] if (x['source_slot'],x['arm'])==(s,a))
        resources=dict(original_names=deepcopy(native_row['resources']),AD_API_resources=deepcopy(native_row['AD_API_resources']),
            predict_forward_count=native_row['resources']['predict'],predictor_samples=None,
            samples_reason='legacy GN/D2P report has no separate sample meter; do not infer additional paid axis',RGB_cohort_resources=deepcopy(r['resources']))
        rows.append(record(plan,row,e,t,z,resources,pr['sourcepack_sha256'],dict(RGB_report=str(path),RGB_report_sha256=h,
            physical_report=str(ph),physical_report_sha256=p['arguments']['physical_sha'],planner_report=p['arguments']['planner_report'],planner_report_sha256=p['arguments']['planner_sha'])))
    return rows,dict(provider='gn',RGB_resources=r['resources'],physical_resources=pr['counts'],RGB_images=8,
        acquisition='original source8 shared once; this cohort RGB8 is separate paid cost',legacy_constructor_meter='not identical to new physical construct/close hooks')


def load_modern(kind,module,bind):
    output,h=SPECS[kind];path=ROOT/'results'/output/'report.json';bind(path,h)
    module.consume(path.parent,h)  # Saved own full module/seal/NPY consumer, not neural execution.
    r=read(path);b=read(path.parent/'execution_binding.json');p=module.verify(b['prepared'],b['prepare_sha256']);bind(b['prepared'],b['prepare_sha256'])
    for f,expected in p['bindings'].items():bind(f,expected)
    for name,expected in r['artifact_sha256'].items():bind(arrays.flat(path.parent,name),expected)
    ph=Path(p['arguments']['physical_report']);pr=read(ph)
    if kind=='beam':
        pp=read(pr['prepared']);m=read(pp['manifest']);selected,_=beam.physical.ownconsume(m['authorities'][0])
        # Model resources below come from the original compiler report, not task fields.
        modelpath=Path(pp['plans'][0]['compiler_report']);mr=beam.producer.read(modelpath);models=mr['rows']
    else:
        pp=read(pr['prepared']);m=read(pp['manifest']);selected=sampling.planner.consume(Path(m['authorities'][0]['output']),m['authorities'][0]['report_sha256'])['rows']
        modelpath=Path(m['authorities'][0]['output'])/'report.json';mr=sampling.planner.read(modelpath);models=mr['rows']
    rows=[]
    for plan in selected:
        s,a=plan['source_slot'],plan['arm'];row=next(x for x in pr['rows'] if (x['root_slot'],x['arm'])==(s,a))
        rr=next(x for x in r['rows'] if (x['root_slot'],x['arm'])==(s,a));z={}
        for cp in rr['checkpoints']:z[cp['step']]=arrays.load_array(cp['latent'],lambda name:arrays.flat(path.parent,name).read_bytes())
        e=gn.moving.read(plan['task_export']);t=gn.moving.load_task(plan['task_export'],plan['task_sha256'])['gn_task']
        resource=next(x for x in models if (x['source_slot'],x['arm'])==(s,a))
        rows.append(record(plan,row,e,t,{k:z[k] for k in (5,10)},dict(original_names=deepcopy(resource['resources']),
            AD_API_resources={},predict_forward_count=deepcopy(resource['resources']['predict_API']),predictor_samples=deepcopy(resource['predictor_samples']),
            zero_AD_explicit=True,RGB_cohort_resources=deepcopy(r['counts'])),pr['sourcepack_sha256'],
            dict(RGB_report=str(path),RGB_report_sha256=h,physical_report=str(ph),physical_report_sha256=p['arguments']['physical_sha'],
                planner_report=str(modelpath),planner_report_sha256=sha(modelpath)),root_encoded=z[0]))
    return rows,dict(provider=kind,RGB_resources=deepcopy(r['counts']),physical_resources=deepcopy(pr['counts']),
        RGB_images=len(r['images']),acquisition='original source8 shared once; this RGB cohort paid separately; no sums of overlapping API/modules')


def assemble(records,costs):
    require(len(records)==10 and len({key(r) for r in records})==10 and {key(r) for r in records}==set(EXPECTED),'fixed2x5, no favorable subset or duplicate')
    ordered=sorted(records,key=lambda r:EXPECTED.index(key(r)));groups={}
    for r in ordered:
        ident=r['identity'];require(r['identity_sha256']==fingerprint(ident) and ident['source_slot']==r['root_slot'],'exact identity fingerprint')
        require(len(r['checkpoints'])==3 and [c['step'] for c in r['checkpoints']]==[0,5,10] and r['own_returned']==10 and
            type(r['native_success']) is bool and r['checkpoints'][-1]['native']['native_success']==r['native_success'],'full denominator includes failures')
        group=groups.setdefault(r['identity_sha256'],dict(source_slot=r['root_slot'],identity_sha256=r['identity_sha256'],identity=deepcopy(ident),arms=[],rows=[]))
        group['arms'].append(r['arm']);group['rows'].append(dict(arm=r['arm'],checkpoints=deepcopy(r['checkpoints']),model_terminal_Q=r['model_terminal_Q'],native_success=r['native_success']))
    partitions=[]
    for g in groups.values():
        selected=[r for r in ordered if r['identity_sha256']==g['identity_sha256']]
        g.update(descriptive_joint_allowed=len(selected)>1,all_five_present=g['arms']==list(ARMS),
            same_admissible_packet_domain=len({fingerprint(r['domain']['admissible']) for r in selected})==1,
            all_selected_strictly_inside_closed_balls=all(r['domain']['strictly_inside_closed_balls'] for r in selected),
            identical_numeric_validation_bounds=len({r['domain']['validation_epsilon'] for r in selected})==1,
            same_search_support=False,equal_total_compute=False,equal_FLOPs=False)
        partitions.append(g)
    return dict(schema=NAME+'.report',status='saved_identity_partitioned_not_admitted',fixed_denominator=10,fixed_arms=list(ARMS),rows=ordered,
        identity_partitions=partitions,all_expected_same_identity=len(partitions)==2 and all(g['all_five_present'] for g in partitions),
        native_successes=sum(r['native_success'] for r in ordered),native_failures=sum(not r['native_success'] for r in ordered),
        cohort_costs=costs,new_neural_calls=0,new_physics=0,training=False,Qfit=False,P1_admission=False,scientific_admission=False,
        independent_confirmation=False,new_independent_samples=0,significance_test=False,cross_paper_ranking=False,rankings=None,
        equal_total_compute=False,equal_FLOPs=False,
        matched_axis='beam/CEM/nominal-MPPI each144 forward/root only; GN78/D2P66 plus AD/J/SVD are different',
        metric_warning='latent Q != native success; full native7 distance includes velocity, native success uses agent+object XY and original angle, not object-only')


def build():
    bindings={}
    def bind(path,expected=None):
        path=str(Path(path).absolute());actual=sha(path)
        require((expected is None or actual==expected) and (path not in bindings or bindings[path]==actual),'saved input SHA conflict '+path)
        bindings[path]=actual
    for f,h in PINS.items():bind(f,h)
    rows=[];costs=[]
    for kind,module in (('beam',beam),('sampling',sampling)):
        rr,cost=load_modern(kind,module,bind);rows+=rr;costs.append(cost)
    rr,cost=load_gn(bind);rows+=rr;costs.append(cost)
    result=assemble(rows,costs)
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):bind(f)
    for f,h in bindings.items():require(sha(f)==h,'final immutable saved closure')
    result['bindings_sha256']=bindings;return result


def write(output):
    result=build();raw=canonical(result);path=Path(output).absolute()
    require(len(raw)<=CAP and not path.exists() and not path.is_symlink() and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'NEW <=2MiB saved report +200MiB floor')
    with path.open('xb') as f:require(f.write(raw)==len(raw),'full saved report');f.flush();os.fsync(f.fileno())
    arrays.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),bytes=len(raw),fixed_denominator=10)


def consume(path,expected):
    arrays.hash_value(expected);require(sha(path)==expected,'external comparison SHA');r=read(path,CAP)
    require(r==build() and sha(path)==expected,'entire saved comparison/partition/metric/resource reconstruction')
    return dict(schema=NAME+'.consume',fixed_denominator=10,identity_partitions=len(r['identity_partitions']),
        all_expected_same_identity=r['all_expected_same_identity'],scientific_admission=False,new_neural_calls=0,new_physics=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','run','consume'))
    p.add_argument('--output');p.add_argument('--report-sha');a=p.parse_args()
    if a.command=='run':result=write(a.output)
    elif a.command=='consume':result=consume(a.output,a.report_sha)
    else:
        r=build();result={k:r[k] for k in ('schema','status','fixed_denominator','all_expected_same_identity','native_successes','native_failures')}
        result.update(bytes=len(canonical(r)),bindings=len(r['bindings_sha256']),identity_partitions=len(r['identity_partitions']))
    print(canonical(result).decode().strip())


if __name__=='__main__':main()
