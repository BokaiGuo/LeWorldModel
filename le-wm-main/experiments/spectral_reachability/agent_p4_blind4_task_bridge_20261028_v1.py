"""Future lawful blind4 -> task exports/GN-D2P input bridge. No acquisition/run."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import numpy as np
from experiments.spectral_reachability import agent_p4_fresh_source_saved_consume_20261028_v2 as saved
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as moving
from experiments.spectral_reachability import moving_root_exclusion_manifest_20261027_v1 as exclusion

NAME='agent_p4_blind4_task_bridge_20261028_v1'
ROOT=saved.ROOT
core=moving.core
need=saved.need
PRODUCER_SHA='5d2103a883b58db85e9877fb25697ef835e317c529771691a3bdec701a582d3a'
CONSUMER_SHA='dcdd2e76559f084a87435e9aaf551eac1ff1d9affdba7eac09ed888a322d29d5'
MOVING_SHA='246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63'
EXCLUSION_SHA='e6521ae535bdf26b6544868bf3c9018761332ee3b339936475ea54707c0db79c'
RECIPE=ROOT/'results/p1_constructive_reachable_source_20261027_v1_prepare01.json'
RECIPE_SHA='98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086'
PROTOCOL_SHA='42f1bcbf517dfd46b8ba05525663d82efac32dbb5fb24f87a8ca572bef507a2f'
ARMS=('line_search_GN','full10_conditional_D2P')
CAP=4*1024**2


def raw(v):return moving.canonical(v)
def sha(v):return hashlib.sha256(v).hexdigest()
def ref(path,h):
    core.hash_value(h)
    p=Path(path)
    need(p.is_absolute() and '..' not in p.parts,'explicit absolute asset path')
    return dict(path=str(p),sha256=h)


def slots():
    return [dict(index=i,root_seed=203027101+i,reset_seed=203027201+i,sign=1 if i%2==0 else -1) for i in range(4)]


def source_pins(reader):
    for rel,h in saved.PINS.items():reader.read(ROOT/rel,h)
    for path,h in ((Path(saved.__file__),CONSUMER_SHA),(Path(moving.__file__),MOVING_SHA),
                   (Path(exclusion.__file__),EXCLUSION_SHA)):
        reader.read(path,h)


def history_input(p,reader):
    """Bind existing approval, never create one or adjudicate original evidence."""
    need(p['schema']==saved.PRODUCER+'.prepare' and p['mode']=='fresh_candidate',
         'v2 fresh_candidate only; calibration_only forbidden for reserved four slots')
    cert=p['history_certificate']
    need(cert is not None,'complete external historical certificate missing; do not open blind4')
    c=reader.json(cert['path'],cert['sha256'])
    need(c['schema']=='moving_fresh_source_exclusions.v1' and c['protocol']['sha256']==PROTOCOL_SHA and
         c['source_run_preconditions_met'] is True and c['unresolved_sources']==[] and c['approval_errors']==[]
         and c['approval']['approved'] is True,'historical coverage incomplete/unknown')
    extra={'coverage_content_sha256','approval','approval_errors','source_run_preconditions_met','status',
           'request_sha256','projected_file_bytes','expanded_metadata_bytes'}
    need(exclusion.digest({k:v for k,v in c.items() if k not in extra})==c['coverage_content_sha256'],
         'existing history content SHA')
    approval=c['approval']['external_file']
    a=reader.json(approval['path'],approval['sha256'])
    need(a['schema']==exclusion.NAME+'.MAIN_coverage_approval' and a['authority']=='MAIN' and
         a['approved'] is True and a['manifest_content_sha256']==c['coverage_content_sha256'] and
         a['cutoff_utc']==c['cutoff_utc'] and set(a['coverage_scope'])>=set(exclusion.REQUIRED_FAMILIES)
         and set(c['approval']['coverage_scope'])==set(a['coverage_scope']), 'existing external MAIN approval binding')
    need(c['project_root']==str(ROOT) and c['observed_roots'] and p['historical_roots']==c['observed_roots'],
         'same nonempty original historical identities')
    need({e['family'] for e in c['evidence_files']}>=set(exclusion.REQUIRED_FAMILIES),'all original history families')
    need(c['source_sha256'].get(str(Path(exclusion.__file__).resolve()))==EXCLUSION_SHA,
         'original history projector binding, same v2 producer prerequisite')
    for path,h in c['source_sha256'].items():
        need(p['bindings'].get(path)==h,'history source closure in executed bindings')
        reader.read(path,h)
    for r in c['observed_roots']:
        identity=exclusion.state_identity(r['root_state'])
        need(identity['root_state_sha256']==r['root_state_sha256'] and identity['physical_pose_key']==r['physical_pose_key'],
             'historical state/pose actual identity')
        for k in ('root_snapshot_sha256','raw_history_sha256'):core.hash_value(r[k])
        need(len(r['history_pixels_sha256'])==3,'full historical RGB3')
        for h in r['history_pixels_sha256']:core.hash_value(h)
    for r in (cert,approval):need(p['bindings'].get(r['path'])==r['sha256'],'certificate/approval in executed source closure')
    return dict(history_certificate=deepcopy(cert),history_approval=deepcopy(approval),
                coverage_content_sha256=c['coverage_content_sha256'])


def prepare_material(source_dir,source_sha,prepared,prepare_sha,*,reader=None):
    reader=saved.Reader() if reader is None else reader
    folder=Path(source_dir);prepare_ref=ref(prepared,prepare_sha);pack_ref=ref(folder/'sourcepack.json',source_sha)
    p=reader.json(prepare_ref['path'],prepare_sha)
    history=history_input(p,reader)  # No sourcepack/blind/reserved assets read on missing cert/mode.
    source_pins(reader)
    need(p['slots']==slots() and p['fixed_denominator']==4 and p['with_target'] is False and
         p['warmup']==10 and p['horizon']==10 and p['replay_episodes']==2 and
         p['saved_telemetry']=='native_latest_action_float32_bits_and_goal_pose_float64_bits.v2','original v2 four-slot native contract')
    need(p['original_recipe']==ref(RECIPE,RECIPE_SHA),'original recipe reference, not v1/calibration substitution')
    original=reader.json(RECIPE,RECIPE_SHA)
    for k in ('recipe','witness','packet_logit_cap','environment'):
        need(p[k]==original[k],'original recipe/witness/action/environment: '+k)
    need(np.asarray(p['raw_warmup'],np.float32).tobytes()==moving.fixed_raw().tobytes(),'original moving raw10')
    receipt=saved.consume(folder,source_sha,prepared,prepare_sha,reader=reader)
    need(receipt['schema']==saved.NAME+'.receipt' and receipt['strict_complete'] is True and
         receipt['fixed_denominator']==4 and receipt['phase_denominator']==12,
         'latest v2 complete saved native consumer required')
    pack=reader.json(pack_ref['path'],source_sha)
    need(pack['sourcepack_complete'] is True and pack['mode']=='fresh_candidate' and pack['errors']==[],
         'sourcepack full terminal closure')
    rows=[]
    for i,row in enumerate(pack['rows']):
        need(row['slot']==slots()[i],'fixed slot-major source identity')
        identity=row['actual_root_identity']
        out=dict(order=i,source_slot=i,source_status=row['status'],ready=row['reachable'] is True)
        if out['ready']:
            packet=moving.decode_blind(reader.read(saved.flat(folder,row['blind_payload']),row['blind_payload_sha256'],2*1024**2))
            pixels=[sha(a.tobytes()) for a in [*packet['rgb_history'],packet['goal_rgb']]]
            need(identity['history_pixels_sha256']==pixels[:3] and
                 identity['raw_history_sha256']==sha(packet['raw_action_history'].tobytes()),'same actual root/history/raw identity')
            out.update(blind_npz=str(saved.flat(folder,row['blind_payload'])),blind_sha256=row['blind_payload_sha256'],
                       pixel_sha256=pixels,raw_action_sha256=identity['raw_history_sha256'],
                       identity={k:deepcopy(identity[k]) for k in ('root_state_sha256','physical_pose_key',
                           'root_snapshot_sha256','history_pixels_sha256','raw_history_sha256')},
                       root_goal_sha256=row['root_sha256'],source_certificate_sha256=row['certificate_sha256'])
        else:
            out.update(blind_npz=None,blind_sha256=None,pixel_sha256=None,raw_action_sha256=None,
                       identity=None,root_goal_sha256=None,source_certificate_sha256=None)
        rows.append(out)
    need(len(rows)==4,'all four source denominator retained')
    ready=all(r['ready'] for r in rows)
    deployment=dict(schema=NAME+'.blind4',fixed_denominator=4,slots=rows,ready=ready,
        sourcepack_sha256=source_sha,source_prepare_sha256=prepare_sha,
        history_certificate_sha256=history['history_certificate']['sha256'],
        history_approval_sha256=history['history_approval']['sha256'],recipe_sha256=RECIPE_SHA,
        normalization=moving.normalization(),model_bindings=deepcopy(moving.MODELS),
        expected_model_tensor_sha256=moving.low.MODEL_TENSOR_SHA,
        image_order=['observed0','observed5','observed10','goal'],
        future_encoding_caps=dict(encoder=16,projector=16,predict=0,AD=0,physics=0),
        actual_new_encoding=None,execution_authorized=False,freshness_certified=False,
        P1_admission=False,P4_admission=False,scientific_admission=False)
    return dict(schema=NAME+'.prepare',arguments=dict(source_dir=str(folder),source_sha=source_sha,
        prepared=prepare_ref['path'],prepare_sha=prepare_sha),deployment=deployment,
        admin=dict(sourcepack=pack_ref,source_prepare=prepare_ref,**history,source_receipt=receipt),
        input_sha256=dict(reader.inputs),own_sha256={str(p):moving.sha(p) for p in own_files()},
        status='blind4_prepared_NOT_authorized' if ready else 'source_k_of_4_downstream_BLOCKED')


def verify(path,expected):
    reader=saved.Reader();v=reader.json(ref(path,expected)['path'],expected)
    need(v==prepare_material(**v['arguments']),'entire source/cert/recipe/identity/source closure rebuild')
    return v


def task_export(deployment,prepare_sha,slot_index,packet,values):
    """MAIN encoder callback target; values MUST be caller-paid native z0/z5/z10/goal."""
    need(deployment['schema']==NAME+'.blind4' and deployment['ready'] is True and
         deployment['fixed_denominator']==len(deployment['slots'])==4,'full four legal blind inputs required')
    need(type(slot_index) is int and 0<=slot_index<4,'literal source slot0..3')
    s=deployment['slots'][slot_index]
    need(s['order']==s['source_slot']==slot_index and s['ready'] is True,'stable slot-major mapping')
    core.hash_value(prepare_sha);moving.check_packet(packet)
    need(len(values)==4,'four paid values in original image order, no dedup')
    for z in values:
        need(isinstance(z,np.ndarray) and z.dtype==np.float32 and z.shape==(192,) and np.isfinite(z).all(),
             'actual native float32 CLS192, no cast repair')
    need([sha(a.tobytes()) for a in [*packet['rgb_history'],packet['goal_rgb']]]==s['pixel_sha256'] and
         sha(packet['raw_action_history'].tobytes())==s['raw_action_sha256'],'exact blind pixels/raw, no chronology substitution')
    need(deployment['normalization']==moving.normalization() and deployment['model_bindings']==moving.MODELS,
         'original model/normalizer, no refit or defaults')
    tc=dict(schema=NAME+'.task_contract',prepare_sha256=prepare_sha,source_slot=slot_index,
        sourcepack_sha256=deployment['sourcepack_sha256'],source_prepare_sha256=deployment['source_prepare_sha256'],
        history_certificate_sha256=deployment['history_certificate_sha256'],history_approval_sha256=deployment['history_approval_sha256'],
        recipe_sha256=RECIPE_SHA,blind_sha256=s['blind_sha256'],source_certificate_sha256=s['source_certificate_sha256'],
        identity=s['identity'],root_goal_sha256=s['root_goal_sha256'],pixel_sha256=s['pixel_sha256'],
        raw_action_sha256=s['raw_action_sha256'],observed_steps=[0,5,10],with_target=False,
        normalization=moving.normalization(),model_bindings=moving.MODELS,metric='identity_native192_no_fit')
    t=dict(environment='swm/PushT-v1',task_contract_sha256=core.digest(tc),
        model_sha256=moving.MODELS[str(moving.POLICY/'weights.pt')],model_config_sha256=moving.MODELS[str(moving.POLICY/'config.json')],
        metric_source_sha256=core.digest(dict(metric='identity_native192',rank=192,no_fit=True,with_target=False)),
        goal_source_sha256=s['pixel_sha256'][3],representation='native_projected_cls',latent_dim=192,horizon=2,
        packet_steps=5,control_dim=2,action_mean=moving.normalization()['mean'],action_std=moving.normalization()['std'],
        packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only',projection=np.eye(192).tolist(),
        goal=values[3].astype(float).tolist(),root_raw_action_history=packet['raw_action_history'].astype(float).tolist(),
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=[a.astype(float).tolist() for a in values[:3]],
            action_history=core.normalized_tokens(dict(action_mean=moving.normalization()['mean'],
                action_std=moving.normalization()['std']),packet['raw_action_history']).astype(float).tolist()))
    core.validate_task(t)
    return dict(schema=NAME+'.task_export',source_slot=slot_index,task=t,task_id=core.task_id(t),task_contract=tc,
                encoding_provenance_verified=False,execution_authorized=False,freshness_certified=False,
                P1_admission=False,P4_admission=False)


def planner_inputs(exports):
    """Real core-compatible arrays, slot-major GN/D2P; never dispatch a planner."""
    need(type(exports) is list and len(exports)==4,'exact full four export denominator')
    requests=[];seen=set()
    cohort=None
    for i,e in enumerate(exports):
        core.fields(e,'schema source_slot task task_id task_contract encoding_provenance_verified execution_authorized freshness_certified P1_admission P4_admission','bridge task export')
        need(e['schema']==NAME+'.task_export' and type(e['source_slot']) is int and e['source_slot']==i,
             'ordered slots0/1/2/3, no selection/replacement')
        t=e['task'];tc=e['task_contract'];core.validate_task(t)
        core.fields(tc,'schema prepare_sha256 source_slot sourcepack_sha256 source_prepare_sha256 history_certificate_sha256 history_approval_sha256 recipe_sha256 blind_sha256 source_certificate_sha256 identity root_goal_sha256 pixel_sha256 raw_action_sha256 observed_steps with_target normalization model_bindings metric','bridge task contract')
        core.fields(tc['identity'],'root_state_sha256 physical_pose_key root_snapshot_sha256 history_pixels_sha256 raw_history_sha256','source identity hashes only')
        for key in ('prepare_sha256','sourcepack_sha256','source_prepare_sha256','history_certificate_sha256',
                    'history_approval_sha256','recipe_sha256','blind_sha256','source_certificate_sha256','root_goal_sha256','raw_action_sha256'):
            core.hash_value(tc[key])
        for key in ('root_state_sha256','physical_pose_key','root_snapshot_sha256','raw_history_sha256'):core.hash_value(tc['identity'][key])
        need(len(tc['pixel_sha256'])==4 and tc['identity']['history_pixels_sha256']==tc['pixel_sha256'][:3] and
             tc['raw_action_sha256']==tc['identity']['raw_history_sha256']==sha(moving.fixed_raw().tobytes()),'exact causal history identities')
        for h in tc['pixel_sha256']:core.hash_value(h)
        shared={k:tc[k] for k in ('prepare_sha256','sourcepack_sha256','source_prepare_sha256',
            'history_certificate_sha256','history_approval_sha256','recipe_sha256')}
        need(cohort is None or shared==cohort,'single full source cohort, not stitched tasks');cohort=shared
        need(core.task_id(t)==e['task_id'] and core.digest(tc)==t['task_contract_sha256'] and
             tc['source_slot']==i and tc['schema']==NAME+'.task_contract','real task/goal/Q/history/source seals')
        need(t['model_sha256']==moving.MODELS[str(moving.POLICY/'weights.pt')] and
             t['model_config_sha256']==moving.MODELS[str(moving.POLICY/'config.json')] and
             np.array_equal(t['projection'],np.eye(192)) and tc['normalization']==moving.normalization() and
             tc['model_bindings']==moving.MODELS and tc['observed_steps']==[0,5,10] and tc['with_target'] is False and
             tc['recipe_sha256']==RECIPE_SHA and tc['metric']=='identity_native192_no_fit' and
             t['goal_source_sha256']==tc['pixel_sha256'][3] and
             t['metric_source_sha256']==core.digest(dict(metric='identity_native192',rank=192,no_fit=True,with_target=False)) and
             np.array_equal(t['action_mean'],moving.normalization()['mean']) and
             np.array_equal(t['action_std'],moving.normalization()['std']) and
             t['packet_logit_cap']==float(.75*np.sqrt(5)) and
             np.array_equal(t['root_raw_action_history'],moving.fixed_raw().astype(float)), 'same frozen task contract')
        z=np.asarray(t['root_context']['latent_history']+[t['goal']])
        need(np.array_equal(z,z.astype(np.float32).astype(float)),'actual native float32 latents, no hidden extra precision')
        need(e['encoding_provenance_verified'] is e['execution_authorized'] is e['freshness_certified'] is
             e['P1_admission'] is e['P4_admission'] is False,'bridge cannot authorize or self-certify execution/encoding')
        need(e['task_id'] not in seen,'duplicate task identity');seen.add(e['task_id'])
        for arm in ARMS:
            requests.append(dict(source_slot=i,arm=arm,task_id=e['task_id'],gn_task=deepcopy(t),
                d2p_packet=dict(root_context=deepcopy(t['root_context']),goal=deepcopy(t['goal'])),q=deepcopy(t['projection']),
                execution_authorized=False,P1_admission=False,P4_admission=False))
    return requests


def own_files():return [Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def write(path,value):
    path=Path(path);need(path.is_absolute() and path.parent==ROOT/'results' and
        re.fullmatch(re.escape(NAME)+r'_[A-Za-z0-9_-]+\.json',path.name),'NEW exclusive bridge output')
    payload=raw(value);need(len(payload)<=CAP,'4MiB output cap')
    need(shutil.disk_usage(path.parent).free>=200*1024**2+len(payload),'200MiB floor')
    with path.open('xb') as f:need(f.write(payload)==len(payload),'complete prepare write');f.flush();os.fsync(f.fileno())
    return dict(path=str(path),sha256=sha(payload),status=value['status'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('prepare')
    for key in ('source-dir','source-sha','prepared','prepare-sha','output'):c.add_argument('--'+key,required=True)
    c=sub.add_parser('verify');c.add_argument('--input',required=True);c.add_argument('--sha256',required=True)
    a=p.parse_args(argv)
    if a.command=='prepare':
        value=prepare_material(a.source_dir,a.source_sha,a.prepared,a.prepare_sha)
        result=write(a.output,value)
    else:
        value=verify(a.input,a.sha256);result=dict(status=value['status'],verified=True,execution_authorized=False)
    print(raw(result).decode(),end='')


if __name__=='__main__':main()
