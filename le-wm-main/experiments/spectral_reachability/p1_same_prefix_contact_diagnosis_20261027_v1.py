"""Saved-only contact/coverage diagnosis, never a new controller or admission.

No simulator, CUDA, torch, model inference, action generation for execution, or
semantic latent-Q attribution. Measurements below are descriptive, not gates.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_same_prefix_saved_audit_20261027_v1 as saved

NAME = 'p1_same_prefix_contact_diagnosis_20261027_v1'
ROOT = saved.ROOT
AUDIT = ROOT/'results/p1_same_prefix_saved_audit_20261027_v1_main01.json'
AUDIT_SHA = 'dcc18b35b5a4327d264a2c14a3d78edd88b3b021e6356cfa36177c3e92670f66'
require = saved.require


def span(v):
    a=np.asarray(v,dtype=float);require(a.size>0 and np.isfinite(a).all(),'finite descriptive range')
    return [float(a.min()),float(a.max())]


def goal_errors(state, goal):
    s,g=saved.vec(state),saved.vec(goal);d=s-g
    a=float(np.linalg.norm(d[:2]));o=float(np.linalg.norm(d[2:4]));denom=a*a+o*o
    return dict(agent_position=a,object_position=o,wrapped_object_angle=float(abs(saved.delta(s,g)[4])),
        agent_share_native_position_squared_error=None if denom==0 else a*a/denom,
        native_state7_distance=float(np.linalg.norm(d)),
        caveat='physical native positions only; NOT an attribution of latent model Q')


def agent_pixels(image):
    a=np.asarray(image);require(a.shape==(224,224,3) and a.dtype==np.uint8,'raw native RGB')
    ij=np.argwhere(np.all(a==[65,105,225],axis=-1))
    return dict(canonical_RoyalBlue_pixels=len(ij),centroid_xy=None if len(ij)==0 else ij[:,::-1].mean(0).tolist(),
        scope='saved color evidence, not a fitted semantic readout or planner input')


def group(role, plans, rows, goal, task, zero_q5):
    require(len(plans)==len(rows)==41 and [p['probe'] for p in plans]==saved.probes(),'full frozen probe census')
    require(all(p['role']==role['role'] and p['root_sha256']==role['root_sha256'] for p in plans),'stable role/root')
    initial=saved.vec(rows[0]['checkpoints'][0]['state']);mid=saved.vec(rows[0]['checkpoints'][1]['state'])
    baseline=saved.vec(rows[0]['final_state']);traces=[];contacts=[];signed=[];cross=[];before=[]
    centered=[];temporal=[];body_speed=[];body_omega=[]
    for p,r in zip(plans,rows,strict=True):
        require(r['full_horizon'] is True and r['status']=='completed' and len(r['trace'])==10,'no partial/full mixing')
        require(r['checkpoints'][1]['state']==mid.tolist() and r['body_dynamics']['5']==rows[0]['body_dynamics']['5'],'same recorded prefix')
        require(np.array_equal(np.asarray(p['commands'][:5]),np.asarray(role['prefix'])),'exact inherited prefix')
        raw=np.asarray(p['commands'][5:]);x=np.asarray(p['probe']['logits'],np.float32)
        require(np.array_equal(raw,np.tanh(x).reshape(5,2)) and np.linalg.norm(x)<=task['packet_logit_cap'],'no coordinate conversion/ball repair')
        require([t['command'] for t in r['trace']]==p['commands'],'actual dispatch follows raw time-major commands')
        allstates=np.asarray([mid.tolist()]+[t['actual_state'] for t in r['trace'][5:]],dtype=float)
        traces.append(allstates)
        contacts.extend(t['official_info']['n_contacts'] for t in r['trace'][5:])
        for state in allstates: temporal.append(saved.delta(state,mid))
        centered.append(saved.delta(r['final_state'],baseline))
        block=r['body_dynamics']['10']['block'];body_speed.append(np.linalg.norm(block['velocity']));body_omega.append(abs(block['angular_velocity']))
        if p['probe']['index']:
            coordinate=p['probe']['coordinate'];axis=coordinate%2;step=coordinate//2
            difference=np.asarray(r['final_state'])-baseline
            signed.append(float(difference[axis]*p['probe']['sign']));cross.append(float(abs(difference[1-axis])))
            before.append(all(r['trace'][5+k]['actual_state']==rows[0]['trace'][5+k]['actual_state'] for k in range(step)))
    states=np.asarray(traces);centered=np.asarray(centered);temporal=np.asarray(temporal)
    separation=np.linalg.norm(states[:,:,:2]-states[:,:,2:4],axis=2)
    delta_prefix=saved.delta(mid,initial)
    suffix_position_changes=[saved.metrics(goal,r['final_state'])['independent']['object_position_error']-
        saved.metrics(goal,baseline)['independent']['object_position_error'] for r in rows]
    suffix_angle_changes=[saved.metrics(goal,r['final_state'])['independent']['wrapped_angle_error']-
        saved.metrics(goal,baseline)['independent']['wrapped_angle_error'] for r in rows]
    return dict(root_sha256=role['root_sha256'],role=role['role'],bank_index=role['bank_index'],
        model_Q5=zero_q5 if role['role']=='zero' else role['model_Q5'],model_zero_prefix_Q5=zero_q5,
        initial_goal_errors=goal_errors(initial,goal),step5_goal_errors=goal_errors(mid,goal),
        prefix_agent_displacement=float(np.linalg.norm(delta_prefix[:2])),prefix_object_displacement=float(np.linalg.norm(delta_prefix[2:4])),
        prefix_object_wrapped_angle_delta=float(delta_prefix[4]),initial_agent_object_center_distance=float(np.linalg.norm(initial[:2]-initial[2:4])),
        step5_agent_object_center_distance=float(np.linalg.norm(mid[:2]-mid[2:4])),
        suffix_sampled_center_distance_range=span(separation),
        suffix_sampled_agent_displacement_from_step5_max=float(np.linalg.norm(states[:,:,:2]-mid[:2],axis=2).max()),
        suffix_sampled_object_displacement_from_step5_max=float(np.linalg.norm(temporal[:,2:4],axis=1).max()),
        suffix_sampled_object_wrapped_angle_from_step5_max=float(abs(temporal[:,4]).max()),
        endpoint_centered_object_displacement_max=float(np.linalg.norm(centered[:,2:4],axis=1).max()),
        endpoint_centered_object_wrapped_angle_max=float(abs(centered[:,4]).max()),
        endpoint_centered_object_position_error_change_range=span(suffix_position_changes),
        endpoint_centered_object_angle_error_change_range=span(suffix_angle_changes),
        endpoint_centered_agent_displacement_max=float(np.linalg.norm(centered[:,:2],axis=1).max()),
        signed_probed_axis_agent_response_range=span(signed),off_axis_agent_response_max=max(cross),
        nonzero_probes_with_positive_signed_agent_response=sum(s>0 for s in signed),
        no_agent_state_change_before_scheduled_probe=all(before),
        suffix_steps=205,suffix_steps_with_positive_native_contact_receipt=sum(c>0 for c in contacts),
        suffix_native_contact_receipt_values=sorted(set(contacts)),
        prefix_center_native_contact_receipts=[t['official_info']['n_contacts'] for t in rows[0]['trace'][:5]],
        step5_agent_velocity=mid[5:].tolist(),step5_block_dynamics=deepcopy(rows[0]['body_dynamics']['5']['block']),
        suffix_endpoint_block_speed_max=float(max(body_speed)),suffix_endpoint_block_angular_speed_max=float(max(body_omega)),
        raw_suffix_peak_abs=float(max(np.max(np.abs(p['commands'][5:])) for p in plans)))


def repetition(plans, rows):
    require(len(plans)==len(rows)==246,'complete repetition denominator')
    groups=[]
    for root in dict.fromkeys(p['root_sha256'] for p in plans):
        ix=[i for i,p in enumerate(plans) if p['root_sha256']==root]
        chunks=[ix[k:k+41] for k in range(0,123,41)]
        for ai,a in enumerate(chunks):
            for b in chunks[ai+1:]:
                if plans[a[0]]['commands'][:5]==plans[b[0]]['commands'][:5]:
                    groups.append(dict(root_sha256=root,roles=[plans[a[0]]['role'],plans[b[0]]['role']],
                        identical_commands=all(plans[i]['commands']==plans[j]['commands'] for i,j in zip(a,b)),
                        identical_actual_traces=all(rows[i]['trace']==rows[j]['trace'] for i,j in zip(a,b)),
                        identical_checkpoint_dynamics=all(rows[i]['body_dynamics']==rows[j]['body_dynamics'] for i,j in zip(a,b))))
    return dict(repeated_role_pairs=groups,root_prefix_groups=len({(p['root_sha256'],saved.digest(p['commands'][:5])) for p in plans}),
        unique_root_command_sequences=len({(p['root_sha256'],saved.digest(p['commands'])) for p in plans}),
        planned_branches=246,independent_new_roots=0)


def analyze():
    reader=saved.Reader();audited=reader.read(AUDIT,AUDIT_SHA)
    require(audited['status']=='saved_data_audited_not_admitted' and audited['actual_report_sha256']==saved.REPORT_SHA
        and audited['prepare_sha256']==saved.PREPARE_SHA and audited['branches']==246,'prior independent audit binding')
    for f,h in audited['audit_sources'].items():reader.bind(f,h)
    p,r,packets=saved.load_inputs(reader)
    require(audited['resources']['counts']==r['counts'],'same actual resource census')
    protocol=reader.read(ROOT/'results/p1_two_stage_witness_20261027_v1_prepared01/protocol.json')
    taskmap={v['root_sha256']:v['task'] for v in protocol['inputs']['tasks']}
    parent=reader.read(ROOT/'results/p1_witness_physical_20261027_v1_prepared01/prepare.json')
    zero_q5={}
    for old in parent['inputs']['plans']:
        if old['arm']=='two_stage':
            m=reader.read(old['origin']);zero_q5[old['root_sha256']]=saved.qcost(taskmap[old['root_sha256']],m['candidates'][0]['contexts'][0]['latent_history'][-1])
    groups=[];goal_evidence=[]
    for root,packet in packets.items():
        require(np.array_equal(taskmap[root]['projection'],np.eye(192)),'actual fixed identity192 Q')
        require(saved.core.sha(np.ascontiguousarray(packet['goal_rgb']).tobytes())==taskmap[root]['goal_source_sha256'],'same full goal RGB')
        goal_evidence.append(dict(root_sha256=root,projection='identity192_native_projected_CLS',
            initial_RGB=agent_pixels(packet['rgb_history'][0]),goal_RGB=agent_pixels(packet['goal_rgb']),
            action_mean=taskmap[root]['action_mean'],action_std=taskmap[root]['action_std'],
            caveat='latent Q cannot be decomposed into agent/object contributions from saved identity Q alone'))
    for i,role in enumerate(p['inputs']['roles']):
        ps=p['inputs']['plans'][41*i:41*(i+1)];rs=r['rows'][41*i:41*(i+1)];packet=packets[role['root_sha256']]
        for plan,row in zip(ps,rs):saved.check_row(plan,row,packet)
        groups.append(group(role,ps,rs,packet['goal'],taskmap[role['root_sha256']],zero_q5[role['root_sha256']]))
    caps={t['packet_logit_cap'] for t in taskmap.values()};require(len(caps)==1,'same original packet cap');cap=caps.pop()
    coverage=dict(absolute_logit_radii=[.25,.5],packet_logit_cap=cap,fraction_of_cap=[.25/cap,.5/cap],
        nonzero_coordinates_per_probe=1,packet_coordinates=10,time_order='5 consecutive steps, x/y within each step',
        raw_magnitudes=[float(np.tanh(np.float32(v))) for v in (.25,.5)],
        native_desired_target_offsets=[100*float(np.tanh(np.float32(v))) for v in (.25,.5)],
        norm_projection_applied=False,logit_center='raw-zero suffix; not winner-suffix or object-relative center',
        native_control='each public step target=current_agent_position+100*raw; PD kp100 kv20; dt.01,10substeps',
        note='desired target offsets are not achieved displacement or contact radii; other directions/full cap untested')
    reader.recheck()
    return dict(schema=NAME+'.report',status='saved_contact_diagnosis_not_admitted',actual_report_sha256=saved.REPORT_SHA,
        independent_audit_sha256=AUDIT_SHA,groups=groups,coverage=coverage,repetition=repetition(p['inputs']['plans'],r['rows']),
        goal_evidence=goal_evidence,input_bindings_sha256=saved.digest(reader.bindings),
        interpretation='finite noncontact sampled suffix responses after selected prefixes; no global uncontrollability claim',
        limitations=['center distances are not T-shape surface clearance',
            'native n_contacts is ceil(callback point accumulator/1000), not pair-resolved impulses/counts',
            'state samples at public steps do not reconstruct all substep trajectories or solver caches',
            'goal agent pixels and selected agent approach do not causally attribute latent Q',
            'posthoc descriptive analysis, no new scale/threshold/action chosen for execution'],
        new_neural_calls=0,new_physics_steps=0,P1_admission=False,
        diagnostic_sources={str(f):saved.sha(f) for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    require(not a.output.exists(),'exclusive new output');result=analyze();raw=(saved.core.canonical(result)+'\n').encode()
    require(len(raw)<=64*1024 and shutil.disk_usage(a.output.parent).free>=saved.FLOOR+len(raw),'bounded output plus200MiB floor')
    with a.output.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(saved.core.canonical(dict(status=result['status'],output=str(a.output),sha256=saved.sha(a.output),bytes=len(raw))))


if __name__=='__main__':main()
