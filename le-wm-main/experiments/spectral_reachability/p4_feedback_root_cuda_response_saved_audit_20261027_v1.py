"""Independent saved-array response audit. No producer/torch/physics imports.

Sourcepack.verify is used for provenance ONLY. Candidate algebra, history,
selection and the exact nested API journal are independently reconstructed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal

NAME='p4_feedback_root_cuda_response_saved_audit_20261027_v1'
REPO=Path(__file__).resolve().parents[2]
DEFAULT=REPO/'results/p4_feedback_root_cuda_response_20261027_v1_cuda_main01'
RECEIPT_SHA='a7fc0c38e17bf379d663ad639d1e2d9ae170b279f172eeda8c71d3b8f07936a8'
ARMS=('imagined','actual')
META=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10)
COUNTS=dict(action_encoder=13,predict=13,jacobian_api=1,grad_api=192,engine_api=192)
RESERVE=200*1024**2


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'regular source')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=False)


def read(path):
    path=Path(path);require(path.stat().st_size<=16*1024**2,'bounded JSON')
    def unique(items):
        out={}
        for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    value=json.loads(path.read_bytes(),object_pairs_hook=unique);canonical(value);return value


def array(v,shape):
    a=np.asarray(v);require(a.dtype.kind in 'fiu' and a.shape==shape and np.isfinite(a).all(),'finite array '+str(shape));return a.astype(float)


def close(a,b,label,atol=1e-10):
    require(np.asarray(a).shape==np.asarray(b).shape and np.allclose(a,b,rtol=0,atol=atol),label)


def content_sha(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def context(c):
    require(all(c.get(k)==v and type(c.get(k)) is type(v) for k,v in META.items()),'native context metadata')
    states=array(c['latent_history'],(3,192));actions=array(c['action_history'],(2,10))
    require(np.array_equal(states,states.astype(np.float32).astype(float)) and np.array_equal(actions,actions.astype(np.float32).astype(float)),'native float32 array entries')
    return states,actions


def history(before,after,commands,norm):
    s,h=context(before);t,a=context(after);raw=array(commands,(5,2))
    require(np.all(np.abs(raw)<=1),'raw action bounds')
    mean=array(norm['mean'],(2,)).astype(np.float32);std=array(norm['std'],(2,)).astype(np.float32)
    require(np.all(std>0),'positive standard deviations')
    token=((raw.astype(np.float32)-mean)/std).reshape(10).astype(float)
    require(np.array_equal(t[:2],s[1:]) and np.array_equal(a,np.stack((h[-1],token))),'exact causal state/action rolling history')
    require({k:v for k,v in before.items() if k not in ('latent_history','action_history')}==
        {k:v for k,v in after.items() if k not in ('latent_history','action_history')},'all context metadata retained')


def ridge(k,r,damping,radius):
    """Independent normal-equation trust-region solve; no producer/SVD solve."""
    gram=k.T@k;rhs=k.T@r
    def solve(mu):return np.linalg.solve(gram+(damping+mu)*np.eye(10),rhs)
    x=solve(0.)
    if np.linalg.norm(x)<=radius:return x
    lo,hi=0.,1.
    for _ in range(100):
        if np.linalg.norm(solve(hi))<=radius:break
        hi*=2
    else:raise ValueError('ridge radius not bracketed')
    for _ in range(90):
        mid=(lo+hi)/2
        if np.linalg.norm(solve(mid))>radius:lo=mid
        else:hi=mid
    return solve(hi)


def bank_check(j,nominal,goal,q,bank):
    k=q@j;singular=array(bank['singular_values'],(10,));right=array(bank['right_vectors'],(10,10))
    close(singular,np.linalg.svd(k,compute_uv=False),'saved singular values',1e-9)
    close(right@right.T,np.eye(10),'orthogonal right vectors')
    close((k@right.T).T@(k@right.T),np.diag(singular**2),'right-vector response eigensystem',1e-8)
    require(all(v[np.argmax(np.abs(v))]>=0 for v in right),'canonical spectral signs')
    radius=np.sqrt(5)*.75;radii=(np.sqrt(5)*.25,radius)
    expected=[np.zeros(10),ridge(k,q@(goal-nominal),.002,radius)];kinds=['zero','root_dls_progress']
    for r in radii:
        for mode in range(2):
            for sign in (-1,1):
                expected.append(sign*r*right[mode]);kinds.append(f'spectral_mode_{mode}_radius_{r:.8f}_sign_{sign}')
    params=array(bank['params'],(10,10));close(params,np.stack(expected),'independent ten-candidate bank/ridge',2e-9)
    require(bank['kinds']==kinds,'ordered bank roles')
    require(np.all(np.linalg.norm(params,axis=1)<=radius+1e-10),'one total logit norm budget')
    return params


def expected_journal():
    """Independent grammar, including every nested entry/return and its index."""
    events=[];entries=[]
    def call(arm,phase,kind,inside=None):
        entry=dict(index=len(entries),arm=arm,phase=phase,kind=kind,completed=True);entries.append(entry)
        events.append(dict(entry,event='attempt',completed=False))
        if inside is not None:inside()
        events.append(dict(entry,event='return'))
    for arm in ARMS:
        events.append(dict(event='arm_intent',arm=arm))
        for phase in ('nominal','jacobian',*[f'candidate_{i}' for i in range(10)],'replay'):
            events.append(dict(event='phase_intent',arm=arm,phase=phase))
            def forward():
                call(arm,phase,'action_encoder');call(arm,phase,'predict')
            if phase=='jacobian':
                def ad():
                    forward()
                    for _ in range(192):call(arm,phase,'grad_api',lambda:call(arm,phase,'engine_api'))
                call(arm,phase,'jacobian_api',ad)
            else:forward()
    events.append(dict(event='run_finished',complete=True,errors=[]))
    return events,entries


def work_check(receipt,events):
    expected,entries=expected_journal()
    require(events==expected,'complete ordered nested journal; no dropped/reordered/refunded/extra API')
    require(receipt['entries']==entries,'receipt array entry index/order/completion identity')
    counts={arm:{k:dict(attempted=n,completed=n) for k,n in COUNTS.items()} for arm in ARMS}
    require(receipt['counts']==counts,'exact per-arm useful work counts')
    return dict(predict_samples=26,action_encoder_calls=26,jacobian_API_calls=2,
        grad_API_calls=384,engine_API_entries=384,mathematical_scalar_VJP_directions=384,
        processed_context_tokens=78,total_metered_entries=len(entries),journal_events=len(events))


def file_names():
    names={'execution_binding.json','events.jsonl'}
    for arm in ARMS:
        names.update(arm+'.'+suffix+'.json' for suffix in ('intent','nominal','jacobian_forward','jacobian','bank','replay','response'))
        names.update(arm+f'.candidate_{i}'+suffix+'.json' for i in range(10) for suffix in ('','.intent'))
    return names


def arm_check(p,request,row,files):
    arm=request['arm'];prefix=arm+'.'
    require(row==files[prefix+'response.json'] and row['arm']==arm and row['root_sha256']==p['root_sha256'] and
        row['boundary_step']==5 and row['context_sha256']==request['context_sha256']==source_digest(request['context']) and
        row['status']=='response_completed' and row['context_specific_AD'] is row['independent_replay_exact'] is True and
        row['initial_J_reused'] is row['actual_physics'] is False,'same-root/context response identity')
    require(files[prefix+'intent.json']==dict(request=request,root_sha256=p['root_sha256'],active_metric_id=p['active_metric_id']),'own root/metric request intent')
    nominal=files[prefix+'nominal.json'];forward=files[prefix+'jacobian_forward.json']
    history(request['context'],nominal,np.zeros((5,2)),p['normalization']);require(nominal==forward,'fresh AD forward equals nominal')
    j=array(files[prefix+'jacobian.json'],(192,10));require(np.linalg.norm(j)>0 and np.array_equal(j,j.astype(np.float32).astype(float)),'nonzero finite native float32 192x10 saved J')
    q=array(p['q'],(192,192));goal=array(p['goal'],(192,))
    close(q,q.T,'symmetric Q');close(q@q,q,'projector Q');require(np.trace(q)>.5,'nonempty Q')
    bank=files[prefix+'bank.json'];params=bank_check(j,np.asarray(nominal['latent_history'])[-1],goal,q,bank)
    require(len(row['candidates'])==10,'complete ten candidates')
    costs=[]
    for i,c in enumerate(row['candidates']):
        require(c==files[prefix+f'candidate_{i}.json'] and c['index']==i and c['kind']==bank['kinds'][i],'own unique ordered candidate slot')
        require(c['logits']==bank['params'][i],'candidate exact bank logit entry')
        commands=np.tanh(params[i].astype(np.float32)).reshape(5,2).tolist()
        require(c['commands']==commands,'float32 tanh time-major action identity')
        require(files[prefix+f'candidate_{i}.intent.json']==dict(index=i,commands=commands,kind=c['kind']),'pre-execution candidate intent')
        history(request['context'],c['context'],commands,p['normalization'])
        residual=q@(np.asarray(c['context']['latent_history'])[-1]-goal);cost=float(residual@residual/192)
        require(abs(c['cost']-cost)<=1e-12,'independent nonlinear original-Q cost')
        costs.append(cost)
    winner=min(row['candidates'],key=lambda c:(c['cost'],c['index']))
    require(row['winner']==winner and files[prefix+'replay.json']==winner['context'],'stable winner and exact complete-context independent replay')
    return dict(arm=arm,winner_index=winner['index'],winner_Q_cost=costs[winner['index']],candidate_Q_costs=costs,
        context_sha256=request['context_sha256'],J_frobenius=float(np.linalg.norm(j)),
        QJ_frobenius=float(np.linalg.norm(q@j)),J_shape=list(j.shape),J_array_SHA256=content_sha(j.tolist()),
        native_history_verified=True,bank_and_ridge_recomputed=True,AD_derivative_numerically_rederived=False)


def source_digest(value):
    # Frozen protocol content IDs use sorted JSON without a trailing newline.
    return content_sha(value)


def live_interface():
    return dict(status='interface_assessed_not_implemented_not_launchable',owner='MAIN_only_neural_and_physics',
        old_runtime_admitted=False,saved_response_dispatch_allowed=False,
        boundaries=[5,10,15],terminal_evaluation_boundary=20,
        request_fields=['root_source_SHA','root','arm','boundary','previous_execution_SHA',
            'own_actual_RGB3_pixel_SHAs','own_last_two_raw_packets','encoding_receipt_SHA',
            'observed_native192_context','chosen_compile_context_SHA','original_model_and_goalQ_SHA'],
        transitions=['verify actual reset RGB/raw history and execute guarded prefix5',
            'observe own boundary5/10/15; actual uses observed, imagined uses own predicted',
            'fresh context-bound AD + all10 candidates + independent replay',
            'durably seal chosen5commands -> one-shot guarded execution -> own new RGB',
            'after20 only terminal encoder and evaluator; never feed backward into step15'],
        missing=['full root reset/bootstrap/environment identity and standalone source seal',
            'NEW request-scoped compiler/ledger bridge: old response material fixes step5 and two-arm session',
            'six uniquely keyed live round ledgers and global no-refund resource supervisor',
            'real RGB guard, partial-failure journaling, encoder instrumentation and terminal evaluator'],
        prospective_budget=dict(round_compiler_predict=78,prefix_predict_recomputed=2,total_predict=80,
            total_action_encoder=80,Jacobian_API=6,grad_API=1152,engine_API=1152,
            processed_context_tokens=240,encoder_calls=8,projector_calls=8,encoded_frames=20,
            guarded_primitive_steps=40,warmup_primitive_steps=30,warmup_per_arm=15,
            total_public_env_steps=70,root_collection_steps=0,
            reset_calls_cap='MAIN must prospectively bound root collector and branch resets',
            global_walltime_and_VRAM='MAIN must explicitly set; do not copy one-response cap blindly'),
        scientific_admission=False,known_roots_not_independent_confirmation=True)


def audit(folder,receipt_sha,provenance=True):
    folder=Path(folder).resolve();require(sha(folder/'receipt.json')==receipt_sha,'external actual receipt SHA')
    r=read(folder/'receipt.json');binding=read(folder/'execution_binding.json')
    prepared=Path(binding['prepared']);require(sha(prepared/'prepare.json')==binding['prepare_sha256']==r['prepare_sha256'],'receipt->actual prepare SHA')
    pr=read(prepared/'prepare.json');require(pr['status']=='prepared_not_run' and sha(prepared/'protocol.json')==pr['protocol_sha256'],'original prepared protocol SHA')
    p=read(prepared/'protocol.json')
    require(r['schema']=='p4_feedback_root_cuda_response_20261027_v1.receipt' and r['status']=='two_prefix_specific_responses_saved' and
        r['errors']==r['blocked']==[] and r['source_verified'] is r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after'] and bool(r['model_hash_before']),'completed actual response scope')
    require(all(r[k] is p[k] is False for k in ('P1_admission','P4_admission','runtime_admissible','independent_confirmation')) and
        r['known_roots_only'] is p['known_roots_only'] is True and r['training'] is False and
        r['new_encoder_calls']==r['new_projector_calls']==r['new_physical_steps']==0,'no admission/extra acquisition')
    for key in ('root_sha256','source_arm','active_metric_id','model_bindings','limits','sourcepack_prepare_sha256'):
        require(r[key]==p[key],'same original '+key)
    require(binding['sourcepack_prepare_sha256']==p['sourcepack_prepare_sha256'],'execution sourcepack link')
    if provenance:
        verify_seal(output_dir=pr['source_seal']['output_dir'],expected_manifest_sha256=pr['source_seal']['manifest_sha256'])
        args=read(Path(pr['source_seal']['output_dir'])/'args.json')
        require(args==dict(protocol_sha256=pr['protocol_sha256'],limits=p['limits']),'original seal args')
        from experiments.spectral_reachability import p4_feedback_root_sourcepack_20261027_v1 as pack
        origin=pack.verify(p['sourcepack'],p['sourcepack_prepare_sha256'])
        for k in ('root_sha256','source_arm','active_metric_id'):require(p[k]==origin[k],'sourcepack root/metric identity')
        for k,pk in (('requests','response_requests'),('goal','goal'),('q','q'),('normalization','normalization'),('prefix_evidence','prefix_evidence'),('model_bindings','model_bindings')):
            require(p[k]==origin[pk],'sourcepack input binding '+k)
        for path,h in {**p['sources_sha256'],**p['model_bindings']}.items():require(sha(path)==h,'sealed source/model bytes')
        require(r['host']['device']=='cuda:0','recorded actual host CUDA device')
    names=file_names();require(set(r['output_files_sha256'])==names,'exact original 56-file inventory; no path substitution')
    files={}
    for name in sorted(names):
        require(sha(folder/name)==r['output_files_sha256'][name],'own saved file SHA: '+name)
        if name.endswith('.json'):files[name]=read(folder/name)
    require(sha(folder/'events.jsonl')==r['journal_sha256'],'own complete journal hash')
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    work=work_check(r,events)
    require([x['arm'] for x in r['rows']]==list(ARMS) and [x['arm'] for x in p['requests']]==list(ARMS),'two ordered arms, one root')
    arms=[arm_check(p,req,row,files) for req,row in zip(p['requests'],r['rows'],strict=True)]
    limits=r['limits'];require(0<=r['wall_seconds']<=limits['wall_seconds'] and r['total_wall_including_preservation']>=r['wall_seconds'],'recorded walltime cap')
    require(0<=r['peaks']['allocated_bytes']<=limits['allocated_bytes'] and
        r['peaks']['allocated_bytes']<=r['peaks']['reserved_bytes']<=limits['reserved_bytes'],'recorded allocated/reserved caps')
    require(limits['grad_api_per_arm']>=192 and limits['engine_api_per_arm']>=192 and limits['jacobian_api_per_arm']==1 and
        limits['predict_per_arm']==limits['action_encoder_per_arm']==13,'explicit useful-work limits')
    for name,h in r['output_files_sha256'].items():require(sha(folder/name)==h,'output drift during audit')
    require(sha(folder/'receipt.json')==receipt_sha and sha(prepared/'prepare.json')==r['prepare_sha256'],'receipt/prepare drift during audit')
    return dict(schema=NAME+'.report',status='independent_saved_data_audit_passed' if provenance else 'mock_arrays_verified_without_provenance',receipt=str(folder/'receipt.json'),receipt_sha256=receipt_sha,
        original_prepare_sha256=r['prepare_sha256'],sourcepack_prepare_sha256=p['sourcepack_prepare_sha256'],
        provenance_verified=provenance,arms=arms,work=work,recorded_peaks=r['peaks'],recorded_wall_seconds=r['wall_seconds'],
        boundary_scope='one known-root actual saved-prefix at step5, two conditional model-response arms; not live rounds or two batches',
        gaps=['saved J is finite/context-bound; independent derivative recomputation needs new authorized CUDA AD/FD',
            'API/host/source records are evidence, not independent GPU replay or physical execution',
            'winner costs across different input contexts are not physical task improvements'],
        live_driver_interface=live_interface(),new_neural_calls=0,new_physical_steps=0,P1_admission=False,P4_admission=False)


def write_report(folder,value):
    folder=Path(folder).absolute();raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(not folder.exists() and len(raw)<256*1024 and shutil.disk_usage(folder.parent).free>=RESERVE+1024**2,'NEW small report, 200MiB reserve')
    folder.mkdir()
    with (folder/'report.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    parser.add_argument('--receipt-sha',default=RECEIPT_SHA);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    result=audit(a.run_dir,a.receipt_sha)
    result['audit_sources_sha256']={str(p):sha(p) for p in (Path(__file__).resolve(),REPO/'tests'/('test_'+NAME+'.py'),REPO/'innerdocs'/(NAME+'.md'))}
    write_report(a.output,result);print(json.dumps(dict(status=result['status'],work=result['work'],report_sha256=sha(a.output/'report.json'))))


if __name__=='__main__':main()
