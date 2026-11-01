"""Goal-only patch-ablation metric hypothesis in native predicted CLS space.

No learned readout/adapter, endpoint masks, physical-state scoring or admission.
Prepare is RGB/array/source work only. Explicit reviewed --compile-cuda only
encodes frozen goal interventions and rescores a source-bound common bank.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_horizon_residual_budget_20261027_v4 as old
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p3p4_goal_patch_metric_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
PREPARED='results/p3_horizon_residual_budget_20261027_v4_prepared01'
PREPARE_SHA='1374e31190e18fabadc90f8fe8668e3800c863be5083ced8a4fa9bf023a85b98'
SOURCE_RUN='results/p3_horizon_residual_budget_20261027_v4_cuda_main01'
SOURCE_REPORT_SHA='5ce206a88de000141add48e58c4900d8b66e55ae5ee1ec6941bd72110180b57f'
GOALS='results/pusht_goal_object_metric4_v1_20261026_cuda'
SPRITE_CENTERS=((28,28),(196,28),(28,196),(196,196))
require,io,write=old.require,old.io,old.write


def contract():
    return dict(hypothesis_only=True,root_cause_established=False,patch_size=14,image_size=224,
        mask='goal-only exact canonical object color; select any-intersection 14x14 cell, row-major, no endpoint updates',
        agent_removal='complete sprite bbox with two-pixel margin, require wholly white perimeter and no object intersection',
        nuisance='four fixed-corner full sprites, original-clean and original five agent-goal vectors',
        object_interventions='each selected cell replaced by white in source-removed goal; all other pixels unchanged',
        representation='native projector(CLS),192; patch token averaging is NOT predicted and is forbidden as planner score',
        metric='Qnew=Proj_col((I-Proj_nuisance) D_object); two relative-cutoff 1e-6 SVDs with source-scale machine-roundoff floor; no fitted/learned weights',
        goal='unchanged original goal0; nuisance includes original-minus-clean so Qnew anchors agree',
        score='mean((Q@(native_terminal-goal0))**2); Q used consistently by residual/response/selection',
        first_comparison='old_Q vs goal_patch_Q on identical union of sealed uniform/dynamic candidate prefixes',
        ties='cost then horizon then pool index',new_candidates_generated=False,
        before_physics='all metrics/selections frozen; new winners need independent double replay and NEW physical consumer',
        compile_encoder_samples_per_root='6+Kgoal_object_cells; batch1; same number existing-projector forwards; no predicts or AD',
        compile_svds_per_root=2,export_validation_saved_array_svds=8,
        no_total_resource_equivalence=True,rank_zero='refuse, no fallback/re-tuning',
        P1_admission=False,P3_admission=False,P4_admission=False,efficacy_claim=False)


def rgb(value):
    a=np.asarray(value);require(a.shape==(224,224,3) and a.dtype==np.uint8,'official uint8 goal RGB');return a


def catalog(goal):
    """Masks and eligibility derive only from one independently supplied goal."""
    goal=rgb(goal);agent=np.all(goal==[65,105,225],axis=-1);obj=np.all(goal==[119,136,153],axis=-1)
    require(agent.any() and obj.any(),'canonical goal agent/object missing')
    ys,xs=np.nonzero(agent);x0,x1=int(xs.min())-2,int(xs.max())+3;y0,y1=int(ys.min())-2,int(ys.max())+3
    require(0<=x0<x1<=224 and 0<=y0<y1<=224,'complete sprite margin clipped')
    sprite=goal[y0:y1,x0:x1].copy()
    require(all(np.all(edge==255) for edge in (sprite[0],sprite[-1],sprite[:,0],sprite[:,-1])) and not obj[y0:y1,x0:x1].any(),'isolated white-background sprite required; no inferred occlusion fill')
    clean=goal.copy();clean[y0:y1,x0:x1]=255
    require(np.array_equal(clean[obj],goal[obj]),'source object pixels preserved')
    images=[dict(id='original',rgb=goal.copy()),dict(id='clean',rgb=clean)]
    height,width=sprite.shape[:2]
    for cx,cy in SPRITE_CENTERS:
        left,top=cx-width//2,cy-height//2;right,bottom=left+width,top+height
        require(0<=left<right<=224 and 0<=top<bottom<=224 and np.all(clean[top:bottom,left:right]==255),'fixed sprite destination ineligible; no replacement position')
        edited=clean.copy();edited[top:bottom,left:right]=sprite
        images.append(dict(id=f'nuisance_{cx}_{cy}',rgb=edited))
    cells=[]
    for row in range(16):
        for col in range(16):
            bounds=(14*col,14*row,14*(col+1),14*(row+1));left,top,right,bottom=bounds
            if not obj[top:bottom,left:right].any():continue
            edited=clean.copy();edited[top:bottom,left:right]=255
            identity=f'object_cell_{row}_{col}';cells.append(dict(id=identity,row=row,col=col,bounds=list(bounds)))
            images.append(dict(id=identity,rgb=edited))
    require(cells,'nonempty goal object patch support')
    return dict(images=images,cells=cells,agent_bbox=[x0,y0,x1,y1],agent_pixels=int(agent.sum()),object_pixels=int(obj.sum()),
        goal_rgb_sha256=io.sha(goal.tobytes()),mask_sha256=io.sha(np.ascontiguousarray(obj).tobytes()),
        image_ids=[v['id'] for v in images],expected_encoder_samples=len(images))


def basis(directions,journal=lambda e:None,role='array',reference_norm=None):
    a=old.prior.array(directions,np.asarray(directions).shape);require(a.ndim==2 and a.shape[0]==192 and a.shape[1]>0,'native CLS directions')
    journal(dict(kind='svd',role=role,completed=False))
    u,s,_=np.linalg.svd(a,full_matrices=False);journal(dict(kind='svd',role=role,completed=True))
    reference=float(np.linalg.norm(a)) if reference_norm is None else reference_norm
    tolerance=max(s[0]*1e-6,reference*max(a.shape)*np.finfo(float).eps)
    rank=int(np.sum(s>tolerance)) if s[0]>0 else 0
    return u[:,:rank],s


def compile_metric(vectors,legacy_goals,journal=lambda e:None):
    vectors=old.prior.array(vectors,np.asarray(vectors).shape);goals=old.prior.array(legacy_goals,(5,192))
    require(vectors.ndim==2 and vectors.shape[1]==192 and len(vectors)>6,'ordered original/clean/four nuisance/object cells')
    require(np.allclose(vectors[0],goals[0],rtol=1e-5,atol=1e-5),'same pretrained projector CLS goal0')
    nuisance=np.concatenate([goals[1:]-goals[0],vectors[:1]-vectors[1],vectors[2:6]-vectors[1]],axis=0).T
    n,ns=basis(nuisance,journal,'nuisance');complement=np.eye(192)-n@n.T
    object_directions=(vectors[6:]-vectors[1]).T
    task,ts=basis(complement@object_directions,journal,'object',float(np.linalg.norm(object_directions)))
    require(task.shape[1]>0,'no surviving object-patch response: strict refusal, no old-Q fallback')
    q=task@task.T
    require(np.linalg.norm(q@n)<1e-7 and np.allclose(q@q,q,atol=1e-8),'valid nuisance-orthogonal task projector')
    return dict(projection=q.tolist(),task_rank=task.shape[1],nuisance_rank=n.shape[1],nuisance_singular_values=ns.tolist(),
        object_singular_values=ts.tolist(),svd_calls=2,physical_labels_used=False,learned_parameters=0,
        goal_anchor_max_difference=float(np.max(np.abs(q@(vectors[0]-vectors[1])))),root_cause_established=False)


def native_cost(terminal,goal,projection):
    z=old.prior.array(terminal,(192,));g=old.prior.array(goal,(192,));q=old.prior.array(projection,(192,192))
    return float(np.mean((q@(z-g))**2))


def compile_action_response(packet,metric,jacobian,nominal,horizon):
    """Executable array bridge for P3 residual and native SRA DLS compilation.

    Uses supplied source-bound native action derivatives, never patch tokens or
    state. It does NOT certify a fresh nonlinear candidate or execute one.
    """
    from experiments.spectral_reachability.pusht_full_packet_compiler_20261027 import bounded_ridge
    q=old.prior.array(metric['projection'],(192,192));j=old.prior.array(jacobian,(192,10*horizon))
    n=old.prior.array(nominal,(192,));g=old.prior.array(packet['goal'],(192,))
    signal=old.residual_signal(j,n,g,q,horizon)
    solve=bounded_ridge(q@j,q@(g-n),damping=.002,cap=float(np.sqrt(5*horizon)*.75))
    return dict(signal=signal,proposal_logits=solve['params'],raw_proposal=np.tanh(solve['params']).reshape(5*horizon,2).tolist(),
        svd_calls=2,new_predict_calls=0,new_jacobian_materializations=0,
        scope='supplied derivative array bridge; nonlinear proposal evaluation/replay still required')


def select_common_bank(packet,metric,candidates):
    require(candidates,'nonempty full common source bank')
    rows=[];seen=set()
    for c in candidates:
        key=(c['horizon'],c['index']);require(key not in seen and c['horizon'] in (2,3),'duplicate/invalid source candidate');seen.add(key)
        terminal=old.prior.array(c['context']['latent_history'],(3,192))[-1]
        raw=old.prior.array(c['actions'],(5*c['horizon'],2));require(np.all(np.abs(raw)<=1),'bounded source actions')
        rows.append(dict(horizon=c['horizon'],index=c['index'],actions=raw.tolist(),context=deepcopy(c['context']),
            old_Q_cost=native_cost(terminal,packet['goal'],packet['projection']),
            goal_patch_Q_cost=native_cost(terminal,packet['goal'],metric['projection'])))
    winners={}
    for arm in ('old_Q','goal_patch_Q'):
        winners[arm]=deepcopy(min(rows,key=lambda r:(r[arm+'_cost'],r['horizon'],r['index'])))
    return dict(candidates=rows,winners=winners,common_candidates=len(rows),cost_evaluations=2*len(rows),
        key_comparisons=2*(len(rows)-1),new_predict_calls=0,
        physical_ready=False,independent_selected_double_replay_required=True)


def prepare(root,output):
    from PIL import Image
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive NEW preparation')
    body,identity=old.validate_prepared(root/PREPARED,PREPARE_SHA)
    previous,_=old.prior.validate_prepared(body['prior_prepared'],body['prior_prepare_sha256'])
    compilation_path=root/old.prior.old.COMPILATION
    compilation=io.decode(compilation_path.read_bytes());source=io.index(compilation['rows'],('root_sha256',))
    ref_path=root/'results/p3_horizon_residual_physical_20261027_v4_source_prepared01/protocol.json'
    refs=io.decode(ref_path.read_bytes())['references']
    bindings={str(root/PREPARED/'prepare.json'):PREPARE_SHA,str(root/SOURCE_RUN/'report.json'):SOURCE_REPORT_SHA,
        str(compilation_path):old.prior.old.COMPILATION_SHA,str(ref_path):'bf2194c23a028d9d5ffed4854888378c96c3102bc7ac479e355c0e62d23c024f'}
    goals=[]
    for i,p in enumerate(body['packets']):
        path=root/GOALS/f'root_{i}.original_goal.png'
        with Image.open(path) as im:image=np.asarray(im)
        plan=catalog(image);require(plan['goal_rgb_sha256']==refs[p['root_sha256']]['goal_rgb_sha256'],'original independent goal RGB reference')
        legacy=source[p['root_sha256'],]['compiled']['goal_vectors'];require(np.array_equal(legacy[0],p['goal']),'same original goal0')
        goals.append(dict(root_index=i,root_sha256=p['root_sha256'],goal_path=str(path),catalog={k:v for k,v in plan.items() if k!='images'},legacy_goal_vectors=legacy))
        bindings[str(path)]=io.sha(path.read_bytes())
    for path,h in bindings.items():require(io.sha(Path(path).read_bytes())==h,'bound bytes')
    sources=old.prior.physical.source_closure(root,[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md')])
    protocol=dict(schema=NAME+'.protocol',contract=contract(),planner_prepared=str(root/PREPARED),planner_prepare_sha256=PREPARE_SHA,
        planner_protocol_id=identity,planner_output=str(root/SOURCE_RUN),bindings=bindings,packets=body['packets'],model_bindings=body['model_bindings'],goals=goals,
        expected_encoder_samples=sum(g['catalog']['expected_encoder_samples'] for g in goals),expected_metric_svds=8,
        sources_sha256={str(p):io.sha(p.read_bytes()) for p in sources},P1_admission=False,P3_admission=False,P4_admission=False)
    output.mkdir(exist_ok=False);write(output/'protocol.json',protocol)
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical({'protocol_sha256':io.sha((output/'protocol.json').read_bytes())}),output_dir=output/f'source_seal_{i//40}') for i in range(0,len(sources),40)]
    result=dict(status='prepared_hypothesis_not_executed',protocol_sha256=io.sha((output/'protocol.json').read_bytes()),source_seals=seals,
        expected_encoder_samples=protocol['expected_encoder_samples'],new_neural_calls=0,new_physical_steps=0,P1_admission=False,P3_admission=False,P4_admission=False)
    write(output/'prepare.json',result);return result


def validate_prepared(prepared,expected_sha):
    prepared=Path(prepared).absolute();require(io.sha((prepared/'prepare.json').read_bytes())==expected_sha,'prepare SHA')
    receipt=io.decode((prepared/'prepare.json').read_bytes());require(io.sha((prepared/'protocol.json').read_bytes())==receipt['protocol_sha256'],'protocol SHA')
    p=io.decode((prepared/'protocol.json').read_bytes());require(p['schema']==NAME+'.protocol' and p['contract']==contract(),'frozen goal metric protocol')
    for path,h in {**p['bindings'],**p['sources_sha256']}.items():require(io.sha(Path(path).read_bytes())==h,'source/input drift: '+path)
    body,identity=old.validate_prepared(p['planner_prepared'],p['planner_prepare_sha256'])
    require(identity==p['planner_protocol_id'] and p['packets']==body['packets'] and p['model_bindings']==body['model_bindings'],'original root/model/context/goal/Q')
    for seal in receipt['source_seals']:verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    return p


def compile_cuda(prepared,expected_sha,output):
    """Reviewed MAIN-only encoder compilation. No simulator or predict calls."""
    protocol=validate_prepared(prepared,expected_sha);output=Path(output).absolute();require(not output.exists(),'exclusive NEW CUDA compilation output')
    # Frozen exporter validates all old candidates; never loads physical outcomes.
    old.selected_actions(protocol['planner_output'])
    source=io.decode((Path(protocol['planner_output'])/'report.json').read_bytes())
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from PIL import Image
    import torch
    for path,h in protocol['model_bindings'].items():require(checked.digest(path)==h,'original model bytes')
    policy=Path(next(p for p in protocol['model_bindings'] if Path(p).name=='weights.pt')).parent
    output.mkdir(exist_ok=False);write(output/'execution_protocol.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected_sha))
    rows=[];entries=[];projector_entries=[];operations=[];failure=None;model=None;before=after=None;host=None;forbidden=[]
    with (output/'compile.events.jsonl').open('x') as stream:
        def journal(event):stream.write(io.canonical(event)+'\n');stream.flush();os.fsync(stream.fileno())
        try:
            host=require_cuda_allocation('cuda');device=torch.device('cuda:0');model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False)
            before=checked.tensor_state_hash(model);transform=p0.image_transform();original_forward=model.encoder.forward
            def forbid(*a,**kw):forbidden.append('predict');journal(dict(kind='forbidden_predict'));raise RuntimeError('goal compiler forbids predict')
            model.predict=forbid
            def counted(*args,**kwargs):
                tensor=args[0];require(tensor.device.type=='cuda' and tuple(tensor.shape)==(1,3,224,224),'single CUDA goal image')
                require(len(entries)<protocol['expected_encoder_samples'],'goal encoder cap before dispatch')
                event=dict(kind='encoder',index=len(entries),completed=False);entries.append(event);journal(deepcopy(event))
                try:result=original_forward(*args,**kwargs);event['completed']=True;return result
                finally:journal(deepcopy(event))
            model.encoder.forward=counted
            original_projector=model.projector.forward
            def counted_projector(*args,**kwargs):
                require(args[0].device.type=='cuda' and tuple(args[0].shape)==(1,192),'single native CLS projector input')
                require(len(projector_entries)<protocol['expected_encoder_samples'],'existing projector cap before dispatch')
                event=dict(kind='existing_projector',index=len(projector_entries),completed=False);projector_entries.append(event);journal(deepcopy(event))
                try:result=original_projector(*args,**kwargs);event['completed']=True;return result
                finally:journal(deepcopy(event))
            model.projector.forward=counted_projector
            for goal,p in zip(protocol['goals'],protocol['packets'],strict=True):
                with Image.open(goal['goal_path']) as im:planned=catalog(np.asarray(im))
                require(physical_equal_catalog(planned,goal['catalog']),'goal-only frozen interventions changed')
                vectors=[]
                for image in planned['images']:
                    with torch.no_grad():z=model.encode({'pixels':p0._image_tensor(image['rgb'],transform=transform,device=device)})['emb'][0,0]
                    require(z.shape==(192,) and z.device.type=='cuda' and bool(torch.isfinite(z).all()),'native finite192 CLS')
                    vectors.append(z.cpu().tolist())
                journal(dict(kind='metric_svd_pair',root_sha256=p['root_sha256'],completed=False))
                def operation(event):
                    operations.append(deepcopy(event));journal(dict(root_sha256=p['root_sha256'],**event))
                metric=compile_metric(vectors,goal['legacy_goal_vectors'],operation);journal(dict(kind='metric_svd_pair',root_sha256=p['root_sha256'],completed=True))
                bank={}
                for row in source['rows']:
                    if row['root_sha256']!=p['root_sha256']:continue
                    for c in row['search']['candidates']:
                        key=(c['horizon'],c['index'])
                        if key in bank:require(io.canonical(bank[key])==io.canonical(c),'shared candidate witness differs across scheduler arms')
                        bank[key]=c
                selection=select_common_bank(p,metric,[bank[k] for k in sorted(bank)])
                row=dict(root_sha256=p['root_sha256'],root_index=goal['root_index'],metric=metric,goal_vectors=vectors,
                    catalog=goal['catalog'],selection=selection,actual_encoder_samples=len(vectors),physical_ready=False)
                rows.append(row);write(output/f'root_{goal["root_index"]}.json',row)
            torch.cuda.synchronize(device);after=checked.tensor_state_hash(model)
        except (Exception,KeyboardInterrupt) as exc:failure=dict(type=type(exc).__name__,message=str(exc))
    source_ok=True
    try:validate_prepared(prepared,expected_sha)
    except Exception as exc:source_ok=False;failure=dict(type='SourceDrift',message=str(exc))
    complete=failure is None and source_ok and not forbidden and before is not None and before==after and len(rows)==4 and len(entries)==len(projector_entries)==protocol['expected_encoder_samples'] and all(e['completed'] for e in entries+projector_entries)
    result=dict(schema=NAME+'.report',status='model_only_metric_compiled' if complete else 'refused_or_incomplete',rows=rows,failure=failure,host=host,
        actual_encoder_calls=len(entries),actual_encoder_samples=len(entries),completed_encoder_calls=sum(e['completed'] for e in entries),
        actual_existing_projector_calls=len(projector_entries),completed_existing_projector_calls=sum(e['completed'] for e in projector_entries),
        actual_neural_module_entries=len(entries)+len(projector_entries),
        attempted_metric_svds=sum(not e['completed'] for e in operations),completed_metric_svds=sum(e['completed'] for e in operations),
        forbidden_predict_attempts=len(forbidden),compile_journal_sha256=io.sha((output/'compile.events.jsonl').read_bytes()),
        export_verification_saved_array_svds=8,new_predict_calls=0,new_jacobians=0,new_vjp_directions=0,
        physical_ready=False,root_cause_established=False,P1_admission=False,P3_admission=False,P4_admission=False,
        preparation_sha256=expected_sha,efficacy_claim=False,model_unchanged=before is not None and before==after)
    write(output/'report.json',result)
    if not complete:raise RuntimeError('strict refusal/incomplete: retain partials; no metric fallback or automatic retry')
    return result


def physical_equal_catalog(actual,expected):return io.canonical({k:v for k,v in actual.items() if k!='images'})==io.canonical(expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','verify','compile-cuda'):action.add_argument('--'+name,action='store_true')
    parser.add_argument('--project-root',type=Path,default=PROJECT);parser.add_argument('--output',type=Path)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256');args=parser.parse_args()
    if args.prepare:
        require(args.output is not None,'new prepare output');result=prepare(args.project_root,args.output)
    else:
        require(args.prepared is not None and args.prepare_sha256 is not None,'reviewed preparation SHA')
        if args.verify:validate_prepared(args.prepared,args.prepare_sha256);result=dict(status='verified')
        else:
            require(args.output is not None,'new compile output');result=compile_cuda(args.prepared,args.prepare_sha256,args.output)
    print(io.canonical({k:v for k,v in result.items() if k not in ('rows','source_seals')}))


if __name__=='__main__':main()
