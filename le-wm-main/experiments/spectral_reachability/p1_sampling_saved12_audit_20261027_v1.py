"""Independent saved12 raw-array audit; no model, environment or temporary files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_sampling_rgb_cuda_20261027_v1 as producer
from experiments.spectral_reachability import p1_sampling_physical_saved_audit_20261027_v1 as geometry

TAG='p1_sampling_saved12_audit_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/p1_sampling_rgb_cuda_20261027_v1_cuda_main01'
PREPARED=ROOT/'results/p1_sampling_rgb_cuda_20261027_v1_prepared01.json'
REPORT_SHA='ae0fe77b628c35a129bd6035ef59771876fe4791bd12774f1e9123e30a6ef34b'
PREPARE_SHA='485040cd82418b1f395de9dcc578611c78d152e89f146094c6261afeb966955a'
require=producer.require
CAP=100_000  # report + source/tests/doc remain below user's150KB total
FLOOR=200*1024**2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def keyed(rows):
    result={}
    for row in rows:
        key=(row['root_sha256'],row['arm'],row['step'])
        require(key not in result,'duplicate root/arm/step')
        result[key]=row
    return result


def recompute(frames,events,tasks):
    """Independent quadratic forms and exact frame-scoped attempt/array census."""
    roots=set(tasks)
    require(len(roots)==2 and len(frames)==12 and set(keyed(frames))=={
        (r,a,s) for r in roots for a in ('cem','iid_random') for s in (0,5,10)},'full12 panel')
    require(len(events)==60,'exact12 five-event groups; no extra/dropped calls')
    rows=[];latents={}
    kinds=['frame_started','native_image_attempted','native_image_attempted','native_image_completed','frame_completed']
    for order,f in enumerate(frames):
        group=events[order*5:order*5+5]
        require(f['order']==order and [e['kind'] for e in group]==kinds,'ordered frame events')
        scope=dict(frame_order=order,root_sha256=f['root_sha256'],arm=f['arm'],step=f['step'],pixel_sha256=f['pixel_sha256'])
        require(all(all(e.get(k)==v for k,v in scope.items()) for e in group),'all events bound to actual PNG frame')
        start,enc,proj,completed,end=group
        task=tasks[f['root_sha256']]
        require(start['order']==end['order']==order and start['task_id']==f['task_id']==producer.core.task_id(task),'original task identity')
        require(enc['resource']=='encoder' and proj['resource']=='projector' and enc['samples']==proj['samples']==1
            and completed['counts']==dict(encoder=1,projector=1)
            and completed['rgb_sha256']==f['pixel_sha256'],'paid12+12 no dedup or refund')
        z,p,g,q=(np.asarray(x,dtype=np.float64) for x in
            (completed['latent'],f['predicted_native192'],task['goal'],task['projection']))
        require(z.shape==p.shape==g.shape==(192,) and q.shape==(192,192)
            and all(np.isfinite(x).all() for x in (z,p,g,q)),'finite original native arrays')
        require(np.array_equal(z,np.asarray(end['raw_native192'])),'two durable raw latent copies agree')
        # Unlike producer's mean-square helper, compute explicit quadratic forms.
        ep=q@(p-g);ea=q@(z-g);error=p-z
        rows.append(dict(root_sha256=f['root_sha256'],arm=f['arm'],step=f['step'],
            predicted_Q_mse=float(ep@ep/192),actual_Q_mse=float(ea@ea/192),
            native_prediction_mse=float(error@error/192)))
        latents[f['root_sha256'],f['arm'],f['step']]=z
    initial=[];idx=keyed(frames)
    for root in sorted(roots):
        left,right=(latents[root,a,0] for a in ('cem','iid_random'))
        samepixel=idx[root,'cem',0]['pixel_sha256']==idx[root,'iid_random',0]['pixel_sha256']
        require(samepixel,'initial pixels differ')
        initial.append(dict(root_sha256=root,initial_pixels_identical=samepixel,
            initial_latents_bitwise_identical=bool(np.array_equal(left,right)),
            max_abs_initial_difference=float(np.max(np.abs(left-right)))))
    return rows,initial,latents


def native_decomposition(goal,state):
    metrics,wrapped=geometry.delta_metrics(goal,state)
    delta=np.asarray(state,dtype=np.float64)-np.asarray(goal,dtype=np.float64)
    terms=dict(all_xy_squared=float(delta[:4]@delta[:4]),raw_angle_squared=float(delta[4]**2),
        agent_velocity_squared=float(delta[5:]@delta[5:]))
    require(np.isclose(sum(terms.values()),metrics['native_state']**2,rtol=1e-12,atol=1e-9),'native reward decomposition')
    # Success depends on front4 position and angle, NOT these velocity terms.
    return dict(metrics=metrics,reward_distance_squared_terms=terms,
        native_reward=-metrics['native_state'],success_has_velocity_condition=False)


def audit():
    bindings={RUN/'report.json':REPORT_SHA,PREPARED:PREPARE_SHA}
    require(all(sha(p)==h for p,h in bindings.items()),'externally pinned actual bytes')
    report=read(RUN/'report.json')
    for name,field in (('analysis.json','analysis_json_sha256'),('image.events.jsonl','image_events_jsonl_sha256')):
        path=RUN/name;require(sha(path)==report[field],'actual report binds '+name);bindings[path]=report[field]
    require(report['status']=='all12_sampling_checkpoints_encoded' and report['errors']==[]
        and report['frames_completed']==12 and report['model_unchanged'] is True
        and report['model_hash_before'] is not None and report['model_hash_before']==report['model_hash_after']
        and report['prepare_sha256']==PREPARE_SHA and report['forbidden_predictor_action_encoder_attempts']==[]
        and all(report[k]==0 for k in ('new_predict_calls','new_AD_calls','new_physical_steps'))
        and report['resources']=={k:dict(attempted=12,completed=12) for k in ('encoder','projector')}
        and not report['training'] and not report['new_Q'] and not report['P1_admission'],'complete image-only frozen-model scope')
    p=producer.verify(PREPARED,PREPARE_SHA)  # pure transitive source/sampling/physical/PNG reconstruction
    pp=read(producer.PREPARED/'prepare.json');tasks=producer.task_table(pp)
    require(report['physical_report_sha256']==producer.RUN_SHA and report['physical_prepare_sha256']==producer.PREPARE_SHA,
        'exact physical producer identity')
    require(p['encoder_cap']==p['projector_cap']==12 and all(p[k]==0 for k in ('predict_cap','action_encoder_cap','AD_cap','physical_steps_cap')),'frozen image-only caps')
    rows,initial,latents=recompute(p['frames'],[json.loads(line) for line in (RUN/'image.events.jsonl').read_text().splitlines()],tasks)
    saved=keyed(read(RUN/'analysis.json')['rows']);actual=keyed(rows)
    require(saved.keys()==actual.keys(),'analysis census independent of array order')
    errors=[]
    for key,row in actual.items():
        for field in ('predicted_Q_mse','actual_Q_mse','native_prediction_mse'):
            difference=abs(row[field]-saved[key][field]);errors.append(difference)
            require(difference<=2e-12,'independent original-Q arithmetic '+field)
    physical=read(producer.RUN/'report.json');branches=geometry.keyed(physical['rows'])
    plans=geometry.keyed(pp['inputs']['plans']);frames=keyed(p['frames']);endpoints=[];png_count=0
    for key,branch in branches.items():
        task=tasks[key[0]];plan=plans[key]
        reference=producer.task_reference(key[0],task)
        require(reference in p['tasks'] and reference['task_id']==plan['task_id']
            and reference['goal_Q_context_sha256']==plan['goal_Q_context_sha256'],'same original goal/Q/context')
        for check in branch['checkpoints']:
            f=frames[(*key,check['step'])]
            require(f['selection_sha256']==branch['selection_sha256']==plan['selection_sha256']
                and f['file_sha256']==check['file_sha256'] and f['pixel_sha256']==check['rgb_sha256']
                and Path(f['path']).resolve()==Path(check['path']).resolve(),'actual physical PNG selection join')
            geometry.independent.png_array(check,producer.RUN);png_count+=1
            require(saved[(*key,check['step'])]['physical_diagnostic']==f['physical_diagnostic'],'saved physical diagnostic mapping')
        goal=geometry.consumer.source.load_raw_source(plan['raw_source_file'],pp['inputs']['source_plan'])['goal']
        diagnostic=native_decomposition(goal,branch['final_state'])
        geometry.independent.check_metric(goal,branch['final_state'],branch['pose_diagnostics'],branch['official_evaluation'])
        endpoints.append(dict(candidate_id=plan['candidate_id'],**actual[(*key,10)],**diagnostic))
    require(png_count==12,'all12 actual PNGs no thinning')
    # Original24 comparison is explicitly inherited, not a new encoding of its arrays.
    priorpath=ROOT/'results/p1_witness_saved24_audit_20261027_v1_audit01.json'
    prior_sha='23536377645defbb81bdac3cf16deb33040c677fe3163ab9d70223bda5a94914'
    require(sha(priorpath)==prior_sha,'frozen prior24 arithmetic report')
    bindings[priorpath]=prior_sha;prior=keyed(read(priorpath)['rows']);comparisons=[]
    oldtasks=ROOT/'results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01'
    for i in (0,1):
        path=oldtasks/f'task_{i}.json';old=read(path);task=tasks[old['root_sha256']]
        require(all(task[k]==old['task'][k] for k in ('goal','projection','root_context','action_mean','action_std','model_sha256','model_config_sha256')),
            'cross-cohort unchanged original Q/goal/model/context')
        bindings[path]=sha(path)
    for endpoint in endpoints:
        root=endpoint['root_sha256'];two=prior[root,'two_stage',10]
        comparisons.append(dict(root_sha256=root,arm=endpoint['arm'],comparator='original_two_stage_same_roots',
            predicted_Q_improvement_over_two_stage=two['predicted_Q_mse']-endpoint['predicted_Q_mse'],
            actual_Q_improvement_over_two_stage=two['actual_Q_mse']-endpoint['actual_Q_mse']))
    require(all(sha(path)==hashed for path,hashed in bindings.items()),'immutable during audit')
    return dict(schema=TAG+'.report',status='saved12_sameQ_PNG_hash_count_checks_completed',rows=rows,
        endpoints=endpoints,initial_duplicates=initial,comparisons_to_inherited24=comparisons,
        maximum_metric_recompute_error=max(errors),checked_png=png_count,resources=report['resources'],
        input_bindings={str(k):v for k,v in bindings.items()},transitive_manifest_binding_count=len(p['bindings']),
        image_completion_semantics=report['image_completion_semantics'],
        interpretation='Native reward includes agent velocity; native success has NO velocity condition. Original Q is not an official success certificate.',
        P1_admission=False,equal_total_compute=False,independent_confirmation=False,
        new_neural_calls=0,new_physical_steps=0,new_Q=False,
        limitation='Saved receipts/raw arrays validated, not independent model re-encoding. Same two calibration roots; no outcome tuning or capability probe.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    path=parser.parse_args().output.resolve()
    require(TAG in path.name and not path.exists(),'exclusive new output')
    result=audit();result['audit_source_sha256']=sha(__file__)
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(len(raw)<=CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'small output/200MiB floor')
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(path),sha256=sha(path),bytes=len(raw))))


if __name__=='__main__':main()
