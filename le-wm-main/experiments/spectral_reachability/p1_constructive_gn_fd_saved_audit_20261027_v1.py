"""Independent saved-only constructive zero-J FD audit; never import producer.

External actual report SHA required. No torch/model/physics/producer execution.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_line_search_gn_fd_saved_audit_20261027_v1 as arithmetic
from experiments.spectral_reachability import p1_constructive_blind_planners_saved_audit_20261027_v1 as upstream

NAME='p1_constructive_gn_fd_saved_audit_20261027_v1'
PRODUCER='p1_constructive_gn_fd_cuda_20261027_v1'
PRODUCER_SHA='f6147161d839952f2671592b275bf382908a8c79b579ff8fabee435fae8b098a'
PLANNER_REPORT_SHA='4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d'
PLANNER_PREPARE_SHA='30b58155a59f73fdc8cc132ff069e645ed6c2bbff9434f474b1f7b9c412f7eb4'
ROOT=Path(__file__).resolve().parents[2]
EPS=(1e-3,3e-3,1e-2)
CAP,FLOOR=2*1024**2,200*1024**2
require,array=upstream.need,upstream.arr
decode,exact,digest=arithmetic.decode,arithmetic.exact,arithmetic.digest
directions=arithmetic.directions


def canonical(v): return upstream.io.canonical(v)
def hash_value(v):
    require(type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v),'full external SHA256 required')


def decode_artifact(raw,path):
    return decode(arithmetic.gunzip(raw) if str(path).endswith('.gz') else raw)


def requests(points):
    require(len(points)==2 and [(p['point_order'],p['root_index'],p['iteration']) for p in points]==[(0,0,1),(1,1,1)] and
        len({p['task_id'] for p in points})==2, 'two distinct first-zero points')
    rows=[]
    for p in points:
        require(p['root_sha256'] is None and not array(p['at_logits'],(2,10)).any(),'zero and opaque task identity, not native root')
        common={k:p[k] for k in ('point_order','root_index','root_sha256','task_id','iteration')}
        rows.append(dict(common,order=len(rows),mode='nominal',direction=None,epsilon=None,sign=0,
            logits=p['at_logits'],commands=np.zeros((10,2)).tolist()))
        for d in directions():
            for eps in EPS:
                for sign in (-1,1):
                    # Preserve the specified add-to-base operation, including
                    # IEEE signed-zero serialization in untouched coordinates.
                    x=(np.asarray(p['at_logits'],float)+sign*eps*np.asarray(d['vector']).reshape(2,10)).astype(np.float32)
                    rows.append(dict(common,order=len(rows),mode='central_side',direction=d['id'],epsilon=eps,sign=sign,
                        logits=x.astype(float).tolist(),commands=np.tanh(x).reshape(10,2).astype(float).tolist()))
    return rows


def consume(events,points,tasks):
    cursor=module_index=0; completed=[]
    counts={r:dict(attempted=0,completed=0) for r in ('predict','action_encoder')}
    def take():
        nonlocal cursor
        require(cursor<len(events),'truncated journal'); e=events[cursor]; cursor+=1; return e
    for req in requests(points):
        exact(take(),dict(kind='evaluation_attempt',request=req),'independent fixed request')
        previous=tasks[req['root_index']]['root_context']; hashes=[]
        for packet in (0,1):
            for resource in ('action_encoder','predict'):
                e=dict(index=module_index,resource=resource,samples=1,completed=False,evaluation_order=req['order'],packet=packet)
                exact(take(),dict(kind='module_attempt',entry=e),'ordered nonrefundable attempt')
                counts[resource]['attempted']+=1
                exact(take(),dict(kind='module_completed',entry=dict(e,completed=True)),'unique completed return')
                counts[resource]['completed']+=1; module_index+=1
            e=take()
            require(set(e)=={'kind','evaluation_order','packet','context_sha256','terminal'} and
                e['kind']=='native_packet' and e['evaluation_order']==req['order'] and e['packet']==packet,'fullH2 packet identity')
            previous=arithmetic.reconstruct_context(tasks[req['root_index']],previous,req['commands'][5*packet:5*packet+5],e['terminal'])
            require(digest(previous)==e['context_sha256'],'independent normalized action/latent history context hash')
            hashes.append(digest(previous))
        row=dict(req,terminal=previous['latent_history'][-1],context_sha256=hashes)
        exact(take(),dict(kind='evaluation_complete',evaluation=row),'complete raw result')
        if req['mode']=='nominal':
            p=points[req['point_order']]
            exact(row['terminal'],p['at_terminal'],'exact saved AD terminal anchor')
            exact(hashes,p['saved_context_sha256'],'both exact saved AD context anchors')
        completed.append(row)
    require(cursor==len(events)==456 and module_index==152 and len(completed)==38,'all events, no extra work/filtered branches')
    exact(counts,{k:dict(attempted=76,completed=76) for k in counts},'raw76/76 census')
    return completed,counts


def recompute(points,matrices,rows):
    wanted=requests(points)
    require(len(rows)==38,'all38 results')
    for row,req in zip(rows,wanted,strict=True): exact({k:row[k] for k in req},req,'ordered nonlinear probe inputs')
    comparisons=[]; stability=[]; cursor=0
    for point in points:
        cursor+=1; j=array(matrices[point['point_order']],(192,20))
        for d in directions():
            v=np.asarray(d['vector']); group=[]
            for eps in EPS:
                minus,plus=rows[cursor:cursor+2]; cursor+=2
                m,p=array(minus['terminal'],(192,)),array(plus['terminal'],(192,))
                fd=(p-m)/(2*eps)
                effective=(np.asarray(plus['logits']).reshape(20)-np.asarray(minus['logits']).reshape(20))/(2*eps)
                ref=j@v; effective_ref=j@effective
                exact(point['saved_Jv'][d['id']],ref.tolist(),'independent saved Jv')
                group.append(dict(point_order=point['point_order'],root_index=point['root_index'],iteration=1,
                    direction=d['id'],epsilon=eps,fd=fd.tolist(),saved_Jv=ref.tolist(),effective_float32_Jv=effective_ref.tolist(),
                    effective_direction=effective.tolist(),effective_direction_error_l2=float(np.linalg.norm(effective-v)),
                    midpoint_drift_l2=float(np.linalg.norm((p+m)/2-np.asarray(point['at_terminal']))),
                    requested_direction_metrics=arithmetic.error_metrics(fd,ref),effective_direction_metrics=arithmetic.error_metrics(fd,effective_ref)))
            comparisons.extend(group)
            stability.append(dict(point_order=point['point_order'],direction=d['id'],adjacent_epsilon_fd_l2=[
                float(np.linalg.norm(np.asarray(a['fd'])-b['fd'])) for a,b in zip(group,group[1:])]))
    require(cursor==38 and len(comparisons)==18,'all18 arithmetic comparisons')
    return dict(schema=PRODUCER+'.analysis',comparisons=comparisons,epsilon_stability=stability,
        nominal_anchors=2,scientific_pass_threshold=None,derivative_verdict='not_assigned',P1_admission=False)


def original_points(p,report,load):
    """Rebuild from authenticated raw planner, not FD producer helpers."""
    old=decode(load(p['input_report'],PLANNER_REPORT_SHA))
    require(p['input_report_sha256']==PLANNER_REPORT_SHA and p['input_prepare_sha256']==PLANNER_PREPARE_SHA and
        old['prepare_sha256']==PLANNER_PREPARE_SHA and old['schema']==upstream.PLANNER+'.report' and
        old['status']=='model_plans_sealed_not_admitted' and old['errors']==[] and old['commands_released'] is True,'original actual planner')
    op=decode(load(old['prepared'],PLANNER_PREPARE_SHA)); folder=Path(p['input_report']).parent
    require(len(op['slots'])==2 and [s['order'] for s in op['slots']]==[0,1] and
        old['model_hash_before']==old['model_hash_after']==op['expected_model_hash']==p['expected_model_hash']==upstream.io.TENSOR_SHA and
        old['model_unchanged'] is old['source_verified'] is True,'same original model/two roots')
    require(op['bindings_sha256'][str(ROOT/'experiments/spectral_reachability'/(upstream.PLANNER+'.py'))]==upstream.PLANNER_SHA,'original planner source')
    require(op['contract']['numerical']==p['numerical']==dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),'numerical contract')
    load(folder/'events.jsonl.gz',old['artifact_sha256']['events.jsonl.gz'],limit=8*1024**2)
    raw=upstream.load(folder/'events.jsonl.gz',old['artifact_sha256']['events.jsonl.gz'])
    require(raw.endswith(b'\n'),'original complete journal')
    wrapped=[decode(line) for line in raw.splitlines()]; points=[]; matrices={}; tasks={}
    for slot in op['slots']:
        k=slot['order']; export=decode_artifact(load(slot['task_export'],slot['task_sha256']),slot['task_export'])
        task=upstream.export_task(slot['task_export'],slot['task_sha256'])
        require(export['task_id']==slot['task_id'],'original approved task identity')
        group=[e for e in wrapped if e['root_order']==k and e['arm']==upstream.ARMS[0]]
        require(group and all(e['task_id']==slot['task_id'] for e in group),'original opaque GN task join')
        events=[e['event'] for e in group]; label='iter_1_AD'
        ds=[e for e in events if e['kind']=='derivative_complete' and e['evaluation_id']==label]
        require(len(ds)==1,'unique first J'); d=ds[0]['derivative']; j=array(d['raw_terminal_jacobian'],(192,20))
        require(d['task_id']==slot['task_id'] and not array(d['at_logits'],(2,10)).any(),'first J zero')
        path=[e['context'] for e in events if e['kind']=='jacobian_forward_packet' and e['evaluation_id']==label]
        upstream.path_check(task,np.zeros((10,2)),path)
        require(path[-1]['latent_history'][-1]==d['at_terminal'],'original AD terminal')
        rawrows=[e for e in events if e['kind']=='jacobian_row' and e['evaluation_id']==label]
        require([e['row'] for e in rawrows]==list(range(192)) and np.array_equal(j,[e['raw_control_derivative'] for e in rawrows]),'raw zero tanh-chain rows')
        for kind,key in (('joint_logit_jacobian','jacobian'),('jacobian_forward_complete','logits')):
            es=[e for e in events if e['kind']==kind and e['evaluation_id']==label]
            require(len(es)==1 and es[0]['terminal']==d['at_terminal'],'original AD forward/joint identity')
            exact(es[0][key],j.tolist() if key=='jacobian' else d['at_logits'],'original AD content')
        initial=[e['prediction'] for e in events if e['kind']=='evaluation_complete' and e['evaluation_id']=='initial']
        require(len(initial)==3 and all(e['contexts']==path and e['logits']==d['at_logits'] for e in initial),'original initial triple anchors')
        points.append(dict(point_order=k,root_index=k,root_sha256=None,
            root_identity='approved opaque task_id; no invented native root hash',task_id=slot['task_id'],task_path=slot['task_export'],
            task_sha256=slot['task_sha256'],iteration=1,evaluation_id=label,at_logits=d['at_logits'],at_terminal=d['at_terminal'],
            saved_context_sha256=[digest(c) for c in path],logit_J_sha256=digest(j.tolist()),
            saved_Jv={v['id']:(j@np.asarray(v['vector'])).tolist() for v in directions()}))
        tasks[k]=task; matrices[k]=j
    exact(p['points'],points,'independent original two zero J points')
    exact(report['inherited_planner_resources'],old['resources'],'inherited actual resources preserved')
    exact(p['inherited_planner_resources'],old['resources'],'prepare inherited actual resources')
    return points,tasks,matrices


def summarize(comparisons):
    return dict(by_epsilon=[dict(epsilon=e,**arithmetic.summary([c for c in comparisons if c['epsilon']==e])) for e in EPS],
        by_root_and_epsilon=[dict(root_index=k,epsilon=e,**arithmetic.summary([c for c in comparisons if c['root_index']==k and c['epsilon']==e])) for k in (0,1) for e in EPS],
        by_direction_and_epsilon=[dict(direction=d['id'],epsilon=e,**arithmetic.summary([c for c in comparisons if c['direction']==d['id'] and c['epsilon']==e])) for d in directions() for e in EPS])


def audit(report_path,report_sha):
    hash_value(report_sha); bindings={}
    def load(path,h,limit=CAP):
        hash_value(h); f=Path(path).absolute()
        require(f.is_file() and not f.is_symlink() and f.stat().st_size<=limit,'bounded regular artifact')
        raw=f.read_bytes(); require(hashlib.sha256(raw).hexdigest()==h,'external/bound file SHA '+str(f))
        require(str(f) not in bindings or bindings[str(f)]==h,'conflicting binding'); bindings[str(f)]=h
        return raw
    path=Path(report_path).absolute(); r=decode(load(path,report_sha)); folder=path.parent
    require(r['schema']==PRODUCER+'.report' and r['status']=='cuda_fd_completed' and r['errors']==[] and
        r['first_error'] is None and r['forbidden_AD']==[],'only complete FD can be verified')
    require(set(r['artifact_sha256'])=={'binding.json','analysis.json.gz','events.jsonl.gz'} and
        {f.name for f in folder.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'exact artifact census')
    require(sum(f.stat().st_size for f in folder.iterdir())<=r['output_cap_bytes']==CAP and r['floor_bytes']==FLOOR,'whole-output cap and floor declaration')
    artifacts={n:load(folder/n,h) for n,h in r['artifact_sha256'].items()}
    binding=decode(artifacts['binding.json'])
    require(binding==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'report/prepare/binding exact')
    p=decode(load(r['prepared'],r['prepare_sha256']))
    require(p['schema']==PRODUCER+'.prepare' and p['bindings_sha256'][str(ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py'))]==PRODUCER_SHA,'frozen FD source')
    exact(p['directions'],directions(),'fixed directions'); exact(p['epsilons'],list(EPS),'fixed epsilon'); exact(p['signs'],[-1,1],'fixed signs')
    # Only source text/model hashes/approved task exports and original planner
    # artifacts may be followed; never sourcepack, certificate, images or NPZ.
    allowed_results={str(path),str(Path(r['prepared']).absolute()),str(Path(p['input_report']).absolute())}
    old=decode(load(p['input_report'],PLANNER_REPORT_SHA))
    allowed_results.add(str(Path(old['prepared']).absolute()))
    oldp=decode(load(old['prepared'],PLANNER_PREPARE_SHA))
    allowed_results.update(s['task_export'] for s in oldp['slots'])
    for n in old['artifact_sha256']:
        require(Path(n).name==n and n not in ('','.','..'),'flat upstream artifact')
        allowed_results.add(str(Path(p['input_report']).parent/n))
    for name,h in p['bindings_sha256'].items():
        f=Path(name)
        require(name in allowed_results or name in upstream.io.MODELS or
            (f.is_relative_to(ROOT) and 'results' not in f.parts and f.suffix in ('.py','.md')),'forbidden privileged input binding')
        require(upstream.io.sha(f)==h,'source/model/artifact hash '+name); bindings[name]=h
    points,tasks,matrices=original_points(p,r,load)
    require(r['input_report_sha256']==PLANNER_REPORT_SHA,'original report identity')
    require(r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']==upstream.io.TENSOR_SHA and
        r['source_verified'] is r['model_unchanged'] is True,'unchanged original model')
    exact(r['numerical'],p['numerical'],'same numerical policy')
    exact(r['budget'],p['budget'],'budget binding')
    required=dict(points=2,directions=3,epsilons=3,central_comparisons=18,signed_H2_evaluations=36,
        nominal_H2_evaluations=2,total_H2_evaluations=38,predict=76,action_encoder=76,FD_predict=72,nominal_predict=4,
        AD_calls=0,new_jacobians=0,new_vjp_directions=0,encoder=0,projector=0,physics_steps=0,SVD=0,solve=0,selection=0,
        inherited_GN_predict=156,inherited_GN_action_encoder=156,inherited_GN_AD_forward_predict_inside156=16,
        inherited_GN_jacobians=8,inherited_GN_mathematical_vjp_directions=1536,inherited_D2P_predict=132,
        inherited_D2P_jacobians=22,inherited_D2P_mathematical_vjp_directions=4224,equal_total_compute_claim=False)
    for key,v in required.items(): exact(r['budget'][key],v,'budget '+key)
    for key in ('P1_admission','training','new_Q'): require(r[key] is False,'no scope upgrade')
    require(p['P1_admission'] is False and r['scientific_pass_threshold'] is p['scientific_pass_threshold'] is None and
        r['derivative_verdict']==p['derivative_verdict']=='not_assigned','no scientific threshold')
    for key in ('new_AD_calls','new_images','new_physics','independent_sample_increment','saved_validation_AD_SVD_solves'):
        require(type(r[key]) is int and r[key]==0,'no new resource '+key)
    exact(r['saved_validation_calls'],dict(attempted=2,completed=2),'both saved validations')
    raw=arithmetic.gunzip(artifacts['events.jsonl.gz']); require(raw.endswith(b'\n'),'no torn final journal')
    events=[decode(line) for line in raw.splitlines()]; rows,counts=consume(events,points,tasks)
    exact(r['resources'],counts,'report vs raw module counts'); require(r['evaluations_completed']==38,'all evaluations')
    rebuilt=recompute(points,matrices,rows)
    require(r['analysis_sha256']==r['artifact_sha256']['analysis.json.gz'],'analysis binding')
    arithmetic.numeric_equal(decode(arithmetic.gunzip(artifacts['analysis.json.gz'])),rebuilt)
    for name,h in bindings.items(): require(upstream.io.sha(name)==h,'final input drift')
    files=[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    return dict(schema=NAME+'.report',status='saved_fd_arithmetic_verified_not_admitted',input_report=str(path),input_report_sha256=report_sha,
        prepare_sha256=r['prepare_sha256'],input_planner_report_sha256=PLANNER_REPORT_SHA,
        auditor_sha256={str(f):upstream.io.sha(f) for f in files},resources=counts,
        denominators=dict(points=2,nominal_anchors=2,H2_evaluations=38,central_comparisons=18,
            reconstructed_contexts=76,module_attempts=152,module_completions=152,events=len(events)),
        **summarize(rebuilt['comparisons']),epsilon_stability=rebuilt['epsilon_stability'],
        comparisons=[{k:v for k,v in c.items() if k not in ('fd','saved_Jv','effective_float32_Jv','effective_direction')} for c in rebuilt['comparisons']],
        inherited_planner_resources=r['inherited_planner_resources'],new_neural=0,new_physics=0,new_AD=0,
        scientific_pass_threshold=None,derivative_verdict='not_assigned',P1_admission=False,equal_total_compute_claim=False,
        limitations=['Saved ledger/arithmetic consistency, not an independent neural rerun or synchronization proof.',
            'Small fixed two-root zero-point directional FD; no global derivative or task-success guarantee.',
            'No epsilon chosen post hoc; all errors, null ratios and negative evidence retained.',
            'Declared historical floor checked as contract only, not reconstructed historical disk free space.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('report','report-sha','output'): parser.add_argument('--'+key,required=True)
    args=parser.parse_args(); result=audit(args.report,args.report_sha); raw=canonical(result); output=Path(args.output)
    require(len(raw)<=100*1024 and not output.exists() and shutil.disk_usage(output.parent).free>=FLOOR+len(raw),'NEW small report +200MiB floor')
    with output.open('xb') as f:
        require(f.write(raw)==len(raw),'short audit report'); f.flush(); os.fsync(f.fileno())
    print(raw.decode().strip())


if __name__=='__main__': main()
