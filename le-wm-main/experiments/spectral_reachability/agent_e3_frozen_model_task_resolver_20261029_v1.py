"""Saved-only E3 known2 x goal2 x horizon2 x domain2 input resolver.

No optimizer, neural/physics entry, producer consumer, or upstream fullstrict.
MAIN complete external receipt SHAs are inherited authority, not re-certified.
"""
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
from PIL import Image

from experiments.spectral_reachability import agent_p3_actual_cp5_task_builder_20261029_v1 as cp5

NAME='agent_e3_frozen_model_task_resolver_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
PHYSICAL='agent_p3_h3_witness_physical_20261029_v1'
PROTOCOL=ROOT/'innerdocs/agent_e3_frozen_model_task_matrix_20261029_v1.md'
PROTOCOL_SHA='51321bd5a139f0a41b24078df5a58e47548560859d6e00597ab966bcae36bf45'
BUILDER_SHA='703f5f552c1911159b2144f72711c542ca48c408ed4e306b1fc17b019df889f1'
MODEL_SHA='4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546'
KNOWN_ROOTS=('a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7',
    'df038a122a660da57675fa281ec7dd358c9f9f67cb387aca9595f31711ca60ad')
MiB=1024**2
CAPS=dict(inputs=64*1024,prepared=16*MiB,report=256*MiB,receipt=32*MiB,
          native_receipt=32*MiB,native_report=16*MiB,export=8*MiB,PNG=256*1024)
OUTPUT_CAP=16*MiB
TOTAL_READ_CAP=512*MiB
FLOOR=200*MiB
PACKET_CAP=float(.75*np.sqrt(5))
PLANNER_FIELDS=('schema','root_context','goal','projection','action_mean','action_std',
    'model_sha256','root_sha256','horizon_packets','domain','identity_sha256',
    'feedback_boundary','task_deadline_control')
FLAGS=dict(P1_admission=False,P2_admission=False,P3_admission=False,P4_admission=False,
    P5_admission=False,E3_demonstrated=False,C3=False,scientific_admission=False,
    freshness=False,physical_release=False,production_authorized=False,equal_FLOPs=False)
require,canonical,digest,sha,array,same=cp5.require,cp5.canonical,cp5.digest,cp5.sha,cp5.array,cp5.same


class Blocked(ValueError):
    """An identified missing/unsupported qualification, NOT corrupt bytes."""


def absolute(value):
    require(type(value) is str,'literal canonical absolute path')
    p=Path(value)
    require(p.is_absolute() and str(p.resolve())==value and not p.is_symlink(),'canonical absolute path')
    return p


def descriptor(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'},'exact path/SHA descriptor')
    absolute(ref['path']);cp5.hash_value(ref['sha256']);return ref


class Reader:
    def __init__(self):self.bindings={};self.total=0
    def bytes(self,ref,cap):
        descriptor(ref);p=absolute(ref['path'])
        require(p.is_file() and p.stat().st_size<=cap,'bounded regular saved input')
        if ref['path'] in self.bindings:
            require(self.bindings[ref['path']]==ref['sha256'],'conflicting path authority')
        else:
            self.total+=p.stat().st_size
            require(self.total<=TOTAL_READ_CAP,'unique saved input aggregate cap')
        b=p.read_bytes()
        require(len(b)<=cap and hashlib.sha256(b).hexdigest()==ref['sha256'],'saved input SHA')
        self.bindings[ref['path']]=ref['sha256'];return b
    def read(self,ref,cap):return cp5.decode(self.bytes(ref,cap))
    def pin(self,path,expected):
        path=str(Path(path).resolve());cp5.hash_value(expected)
        require(sha(path)==expected,'source/protocol seal')
        self.bindings[path]=expected
    def finish(self):
        for path,h in self.bindings.items():require(sha(path)==h,'input/source changed during own reconstruction')


def parameters(value,K):
    require(type(K) is int and K in (1,3),'literal horizon packets 1/3')
    return array(value,(K,5,2),f32=False)


def mask_chart(logits,K,domain):
    """Chart only. Never project/optimize/clip an incumbent or change history."""
    require(domain in ('Dxy','Dx') and type(domain) is str,'fixed action domain')
    x=parameters(logits,K)
    require(np.all(np.linalg.norm(x.reshape(K,10),axis=1)<=PACKET_CAP),'original f64 parameter packet ball')
    if domain=='Dx':require(np.all(x[:,:,1]==0),'Dx parameters must already be masked')
    y=np.tanh(x.astype(np.float32)).astype(np.float32)
    if domain=='Dx':y[:,:,1]=np.float32(0)  # explicit raw positive-zero ABI
    require(np.isfinite(y).all() and (np.abs(y)<=1).all(),'finite original tanhf32 chart')
    return y.reshape(5*K,2)


def mask_gradient(gradient,K,domain):
    require(type(domain) is str and domain in ('Dxy','Dx'),'fixed action domain')
    g=parameters(gradient,K).copy()
    if domain=='Dx':g[:,:,1]=0.
    return g


def budget(K):
    require(type(K) is int and K in (1,3),'literal horizon packets 1/3')
    return dict(nodes=3,candidate_slots=99,non_AD_predict=108*K,AD_primal_predict=96*K,
        predict_API=204*K,predict_samples=204*K,predict_frame_samples=612*K,
        scalar_reverse=96,scalar_grad_API=96,encoder_new=2,projector_new=2,
        J=0,SVD=0,solve=0,warmup_public=10,prefix_public=5,task_public=15,
        all_public=30,complete_path_physics=304,complete_path_explicit_render=32,
        backward_engine_calls=None,FLOPs=None)


def roster():
    return [dict(index=i,cell_id=f'r{r}.g{g}.h{K}.d{d}',root=r,goal_index=g,
        horizon_packets=K,domain=d,budget=budget(K))
        for i,(r,g,K,d) in enumerate((r,g,K,d) for r in range(2) for g in range(2)
                                    for K in (1,3) for d in ('Dxy','Dx'))]


def complete_path_budget():
    return dict(cells=16,known_roots=2,fresh_roots=0,nodes=48,candidate_slots=1584,
        predict_API=6528,non_AD_predict=3456,AD_primal_predict=3072,predict_samples=6528,
        predict_frame_samples=19584,scalar_reverse=1536,encoder_new=32,projector_new=32,
        all_public=480,complete_path_physics=4864,complete_path_explicit_render=512,
        output_per_job=512*MiB,final_reserve_per_job=64*MiB,wall_per_job_seconds=1800,
        all16_output_cap=16*512*MiB,all16_job_wall_cap_sum_seconds=16*1800,
        serial_inclusive_wall_actual_seconds=None,
        GPU_allocated=3*1024**3,GPU_reserved=4*1024**3,disk_floor=FLOOR,
        numerical_runner_transport_proven=False,FLOPs=None)


def native7(frame):
    require(type(frame.get('step')) is int,'literal native cp time')
    v=array(frame['state'],(7,),f32=False)
    same(v,array(frame['snapshot']['state'],(7,),f32=False),'native7 snapshot literal')
    require(v.astype('<f8').tobytes().hex()==frame['snapshot']['state_float64_bits'],'native7 f64 bits')
    return v


def pixels(ref,expected,reader):
    b=reader.bytes(ref,CAPS['PNG'])
    with Image.open(io.BytesIO(b)) as im:
        require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'original RGB224 PNG ABI')
        z=np.asarray(im)
        require(z.shape==(224,224,3) and z.dtype==np.uint8,'raw RGB pixels')
        require(hashlib.sha256(z.tobytes()).hexdigest()==expected,'root RGB decoded pixel SHA')
    return dict(**ref,pixel_sha256=expected,shape=[224,224,3],dtype='uint8')


def domain_spec(domain):
    return dict(name=domain,raw_bounds=[-1,1],packet_steps=5,action_dim=2,
        packet_logit_cap=PACKET_CAP,parameter_cap_domain='original float64 before float32 chart',
        chart='numpy tanh(float32 logits), no second scale/clip',
        future_raw_mask=[True,domain=='Dxy'],Dx_y_bits='positive_zero_float32',history_masked=False)


def qualify_root(root,joined,p,export,native,report,report_ref,reader):
    """Check prefix-only native/RGB evidence. Never examine suffix outcome/score."""
    original=p['tasks'][root];prov=joined['provenance'][root];local=joined['tasks'][root]
    cp5.task_check(local);cp5.task_check(original)
    require(original['binding']['model_sha256']==MODEL_SHA,'original fixed checkpoint tensor identity')
    require(original['binding']['root_sha256']==KNOWN_ROOTS[root],'only original two known roots, no replacement')
    require(export['native_root_sha256'] is None and
        export['blind_sha256']==export['blind_source_sha256']==original['binding']['blind_sha256'] and
        export['task_contract']['pixel_sha256'] and export['task_contract']['with_target'] is False,
        'original BLIND export/visual identity; native root intentionally withheld')
    expected_pixels=export['task_contract']['pixel_sha256']
    require(type(expected_pixels) is list and len(expected_pixels)==4,'original three history plus goal pixels')
    for h in expected_pixels:cp5.hash_value(h)
    rows=[]
    for bank in ('endpoints','midpoints'):
        found=[x for x in native['rows'] if type(x.get('root_slot')) is int and x['root_slot']==root and x.get('arm')==bank]
        if len(found)!=1:raise Blocked('missing_or_duplicate_same_root_native_branch:'+bank)
        row=found[0];idx=2*root+int(bank=='midpoints')
        require(type(row['index']) is int and row['index']==idx and row['condition']=='off' and row['initial_exact'] is True and
            row['task_id']==original['binding']['task_id'] and row['root_sha256']==original['binding']['root_sha256'],
            'native branch/original root task')
        saved=report['rows'][idx]
        prefix_fields=('index','root_slot','arm','condition','initial_exact','task_id','root_sha256',
                       'root_state','goal','goal_pixel_sha256')
        require(all(canonical(saved[k])==canonical(row[k]) for k in prefix_fields),
            'native report/complete receipt prefix identities')
        plan=row['plan'];w=plan['source_witness']
        saved_w=saved['plan']['source_witness']
        require(all(canonical(saved_w[k])==canonical(w[k]) for k in
            ('root_slot','bank','boundary','task_id','prefix','task','request','actual_cp5')),
            'saved prefix-only source witness join')
        require(w['root_slot']==root and type(w['root_slot']) is int and w['bank']==bank and
            w['boundary']==5 and type(w['boundary']) is int and
            w['task_id']==original['binding']['task_id'] and
            canonical(w['prefix']['commands'])==canonical(prov['prefix5']) and
            w['prefix']['raw_float32_sha256']==hashlib.sha256(array(prov['prefix5'],(5,2)).tobytes()).hexdigest(),
            'native witness prefix/original task')
        # The authoritative root/context fields are checked, not the saved winner/result/future suffix.
        require(canonical(w['task'])==canonical(original),'source witness ORIGINAL control0 task')
        require(canonical(w['request']['task']['root_context'])==canonical(local['root_context']) and
            canonical(w['request']['task']['goal'])==canonical(local['goal']),
            'source witness request actual cp5 context and original goal')
        pts=[]
        for step in (0,5):
            cs=[c for c in row['checkpoints'] if type(c.get('step')) is int and c['step']==step]
            if len(cs)!=1:raise Blocked('missing_or_duplicate_native_cp'+str(step))
            other=[c for c in saved['checkpoints'] if type(c.get('step')) is int and c['step']==step]
            require(canonical(other)==canonical(cs),'saved prefix-only checkpoint join')
            native7(cs[0]);pts.append(cs[0])
        c0,c5=pts
        same(array(row['root_state'],(7,),f32=False),native7(c0),'root state is actual control0')
        require(c0['pixel_sha256']==expected_pixels[2] and
            row['goal_pixel_sha256']==plan['native_goal_pixel_sha256']==expected_pixels[3],
            'original root and G0 goal pixel join')
        array(row['goal'],(7,),f32=False)
        require(canonical(row['goal'])==canonical(plan['native_goal']),'G0 original native goal')
        aw=w['actual_cp5'];native7(aw)
        require(aw['step']==5 and aw['pixel_sha256']==c5['pixel_sha256'] and
            canonical(aw['snapshot'])==canonical(c5['snapshot']),'actual cp5 native observation join')
        matches=[b for b in prov['branch_witnesses'] if b['episode_id']==f's{root}_adaptive_{bank}']
        require(len(matches)==1 and matches[0]['observation']['pixel_sha256']==c5['pixel_sha256'],
            'native cp5 to original encoded observation')
        require(digest(w['request'])==matches[0]['request_sha256'],'full original saved cp5 request digest')
        same(array(plan['prefix_commands'],(5,2)),array(prov['prefix5'],(5,2)),'prefix action bits')
        rows.append(dict(index=idx,bank=bank,c0=deepcopy(c0),c5=deepcopy(c5),goal=deepcopy(row['goal'])))
    for key in ('state','snapshot','pixel_sha256'):
        require(canonical(rows[0]['c0'][key])==canonical(rows[1]['c0'][key]) and
            canonical(rows[0]['c5'][key])==canonical(rows[1]['c5'][key]),'both banks same actual root and cp5')
    require(canonical(rows[0]['goal'])==canonical(rows[1]['goal']),'both banks same native goal')
    g1_latent=array(original['root_context']['latent_history'],(3,192))[-1].tolist()
    goal_sources={0:dict(kind='G0_original_source_goal',latent=deepcopy(local['goal']),
        native_goal=rows[0]['goal'],goal_pixel_sha256=expected_pixels[3],RGB_file=None,
        RGB_evidence='original export/complete native receipt, not locally decoded G0 goal image'),
        1:dict(kind='G1_previously_observed_control0',latent=g1_latent,
        native_goal=deepcopy(rows[0]['c0']['state']),goal_pixel_sha256=expected_pixels[2],
        original_absolute_step=10,original_control_step=0,RGB_files=[])}
    for item in rows:
        c0=item['c0'];name=c0['png']
        require(type(name) is str and Path(name).name==name,'literal native PNG basename')
        path=str(Path(report_ref['path']).parent/name)
        require(native['bindings'].get(path)==c0['file_sha256'],'root PNG in accepted native closure')
        if not Path(path).is_file():
            goal_sources[1]=dict(blocked='missing_existing_root_RGB:'+path);break
        goal_sources[1]['RGB_files'].append(pixels(dict(path=path,sha256=c0['file_sha256']),expected_pixels[2],reader))
    return goal_sources,dict(original_task=deepcopy(original),actual_cp5_task=deepcopy(local),
        prefix5=deepcopy(prov['prefix5']),raw10=deepcopy(prov['raw10']),
        control5_absolute_history=[5,10,15],control5_relative_history=[-5,0,5],
        root_native_branches=rows,cp5_provenance=deepcopy(prov))


def make_task(base,goal,cell,authorities):
    identity=dict(schema=NAME+'.identity',cell=deepcopy(cell),authorities=deepcopy(authorities),
        actual_cp5_task_sha256=digest(base),goal_evidence=deepcopy(goal),protocol_sha256=PROTOCOL_SHA)
    body=dict(schema=NAME+'.task',root_context=deepcopy(base['root_context']),goal=deepcopy(goal['latent']),
        projection=deepcopy(base['projection']),action_mean=deepcopy(base['action_mean']),
        action_std=deepcopy(base['action_std']),model_sha256=base['binding']['model_sha256'],
        root_sha256=base['binding']['root_sha256'],horizon_packets=cell['horizon_packets'],
        domain=domain_spec(cell['domain']),identity_sha256=digest(identity),feedback_boundary=5,task_deadline_control=20)
    require(set(body)==set(PLANNER_FIELDS),'exact planner-only allowlist')
    # evaluator state/goal is deliberately NOT in this planner allowlist.
    task_id=digest([NAME,identity]);return dict(task_id=task_id,task_sha256=digest(body),body=body)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build(inputs,inputs_sha):
    reader=Reader();reader.pin(PROTOCOL,PROTOCOL_SHA);reader.pin(cp5.__file__,BUILDER_SHA)
    for path in own_files():reader.pin(path,sha(path))
    ref=dict(path=inputs,sha256=inputs_sha);manifest=reader.read(ref,CAPS['inputs'])
    require(type(manifest) is dict and set(manifest)=={'schema','planner','native_receipt'} and
        manifest['schema']==NAME+'.inputs','own explicit source authority schema')
    authorities=manifest['planner'];require(type(authorities) is dict and set(authorities)==set(cp5.CAPS),
        'planner three authority fields')
    for ref in [*authorities.values(),manifest['native_receipt']]:
        if ref is not None:descriptor(ref)
    cells=roster();root_provenance=[];join=None
    def block(c,why):c.update(status='blocked',blocked_reason=why,task=None,evaluator=None)
    # Explicit nulls are missing qualification, not zero-cost or permission to guess.
    if any(v is None for v in authorities.values()) or manifest['native_receipt'] is None:
        for c in cells:block(c,'explicit_authority_missing')
    else:
        pv={k:reader.read(authorities[k],CAPS[k]) for k in cp5.CAPS}
        nr=manifest['native_receipt'];native=reader.read(nr,CAPS['native_receipt'])
        if (any(pv[k].get('schema')!=cp5.PLANNER+'.'+suffix for k,suffix in
                (('prepared','prepare'),('report','report'),('receipt','consume'))) or
                pv['receipt'].get('complete') is not True or
                pv['receipt'].get('status')!='saved_shared_panel_verified_NOT_admitted'):
            for c in cells:block(c,'unsupported_or_incomplete_planner_ABI')
        elif native.get('schema')!=PHYSICAL+'.consume' or native.get('status')!='h3_witness4_control20_native_recorded_not_admitted':
            for c in cells:block(c,'unsupported_or_incomplete_native_receipt_ABI')
        else:
            require(native['errors']==[] and native['all_four_physically_executed'] is True and
                len(native['rows'])==4,'complete external four-branch physical authority')
            pa=native['planner_authority']
            require(all(pa[k+'_sha256']==authorities[a]['sha256'] for k,a in
                (('prepare','prepared'),('report','report'),('receipt','receipt'))),'same exact planner/native authority')
            join=cp5.resolve_actual_cp5_tasks(authorities)  # saved join ONLY; no producer/fullstrict
            for path,h in join['bindings'].items():reader.pin(path,h)
            p=reader.read(authorities['prepared'],CAPS['prepared'])
            refs=[dict(path=k,sha256=h) for k,h in native['bindings'].items() if h==native['physical_report_sha256']]
            require(len(refs)==1,'unique accepted native report authority')
            rref=refs[0];report=reader.read(rref,CAPS['native_report'])
            require(report['schema']==PHYSICAL+'.report' and len(report['rows'])==4,'native own report schema/denominator')
            for root in range(2):
                try:
                    layer=p['layers'][root];ep=layer['task_export']
                    eref=dict(path=ep,sha256=p['sources'][ep])
                    reader.bytes(eref,CAPS['export'])
                    export=cp5.read_pinned(ep,eref['sha256'],CAPS['export'],compressed=True)
                    goals,prov=qualify_root(root,join,p,export,native,report,rref,reader)
                    root_provenance.append(dict(root=root,status='qualified_saved_prefix',provenance=prov))
                    for c in [x for x in cells if x['root']==root]:
                        goal=goals[c['goal_index']]
                        if 'blocked' in goal:block(c,goal['blocked']);continue
                        task=make_task(join['tasks'][root],goal,c,manifest)
                        c.update(status='input_ready_NOT_run_authorized',blocked_reason=None,task=task,
                            goal_evidence=deepcopy(goal),evaluator=dict(native_goal=deepcopy(goal['native_goal']),
                            goal_pixel_sha256=goal['goal_pixel_sha256'],goal_kind=goal['kind'],
                            planner_may_read=False,native_success='unchanged original source predicate'))
                except (Blocked,KeyError) as e:
                    for c in [x for x in cells if x['root']==root]:block(c,'root_input_qualification:'+str(e))
                    root_provenance.append(dict(root=root,status='blocked',reason=str(e)))
    ready=sum(c['status']=='input_ready_NOT_run_authorized' for c in cells)
    out=dict(schema=NAME+'.prepare',status='all16_saved_inputs_resolved_NOT_authorized' if ready==16 else 'blocked_saved_input_qualification',
        arguments=dict(inputs=inputs,inputs_sha256=inputs_sha),authorities=deepcopy(manifest),cells=cells,
        root_provenance=root_provenance,denominator=dict(cells=16,known_roots=2,fresh_roots=0),
        ready=ready,blocked=16-ready,planned_complete_budget=complete_path_budget(),
        planner_allowlist=list(PLANNER_FIELDS),
        evidence=dict(full_upstream_rebuilt_here=False,inherited_MAIN_complete_receipts=join is not None,
            root_PNG_redecoded_here=any(Path(path).suffix=='.png' for path in reader.bindings),
            latent_reencoded_here=False,new_numerical_optimization=False,
            G1_goal_observed_control=0,G0_goal_is_original_frozen_goal=True,
            initial_observation_control=5,numerical_runner_included=False,
            raw_J_or_future_physical_outcomes_used=False),
        new_calls=dict(encoder=0,projector=0,predict=0,AD=0,physics=0),bindings=reader.bindings,**FLAGS)
    require(len(canonical(out))+1<=OUTPUT_CAP,'full16 metadata fits fixed16MiB')
    reader.finish();return out


def write_new(path,value):
    p=absolute(path);require(p.name.startswith(NAME+'_'),'own output stem')
    b=canonical(value)+b'\n';require(len(b)<=OUTPUT_CAP,'bounded complete output')
    require(p.parent.is_dir() and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'existing parent/disk floor')
    with p.open('xb') as f:require(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(p),sha256=sha(p),bytes=len(b))


def verify(path,h):
    old=Reader().read(dict(path=path,sha256=h),OUTPUT_CAP);a=old['arguments']
    require(canonical(old)==canonical(build(a['inputs'],a['inputs_sha256'])),'whole own16 task/source reconstruction')
    return dict(verified=True,ready=old['ready'],blocked=old['blocked'],production_authorized=False)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);ss=q.add_subparsers(dest='cmd',required=True)
    a=ss.add_parser('prepare');a.add_argument('--inputs',required=True);a.add_argument('--inputs-sha',required=True);a.add_argument('--output',required=True)
    a=ss.add_parser('verify');a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
    args=q.parse_args(argv)
    out=write_new(args.output,build(args.inputs,args.inputs_sha)) if args.cmd=='prepare' else verify(args.prepared,args.prepare_sha)
    print(json.dumps(out,sort_keys=True,allow_nan=False),flush=True)


if __name__=='__main__':main()
