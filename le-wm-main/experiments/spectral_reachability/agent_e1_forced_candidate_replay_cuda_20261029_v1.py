"""Six forced E1 raw10 paid replays; MAIN must review before CUDA."""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from experiments.spectral_reachability import agent_e1_forced_candidate_contract_20261029_v1 as contract

host=contract.host
old=host.old
phase=old.phase
ROOT=host.ROOT
NAME='agent_e1_forced_candidate_replay_cuda_20261029_v1'
CONTRACT=ROOT/'results/agent_e1_forced_candidate_contract_20261029_v1_prepare_MAIN01.json'
CONTRACT_SHA='494f806e5bc5d46221b28ce2ad7f288bee4b46d0b3eb3020706f4a07ecf2ed37'
SLOTS=tuple((r,k) for r in (0,1) for k in contract.METHODS)


def forced():
    value=old.parse_canonical(old.read_bound(CONTRACT,CONTRACT_SHA,16*1024*1024))
    old.require(value==contract.build(),'whole fixed contract rebuild')
    old.require([(r['root'],r['method']) for r in value['slots']]==list(SLOTS),'six fixed slots')
    return value


def manifest():
    fixed=forced();_,previous=host.authorities()
    closure=host.source_closure(previous)
    names=[ROOT/'experiments/spectral_reachability'/(NAME+'.py'),ROOT/'tests'/('test_'+NAME+'.py'),
           ROOT/'innerdocs'/(NAME+'.md')]
    for p in names:closure[str(p)]=host.stream_sha(p)
    for p,h in fixed['source_hashes'].items():
        old.require(host.stream_sha(p)==h,'contract source closure drift');closure[p]=h
    return dict(schema=NAME+'.prepare',contract=str(CONTRACT),contract_sha256=CONTRACT_SHA,
                target_actions_sha256=old.sha(old.canonical(fixed['slots'])),closure=closure,
                expected_model_hash=previous['expected_model_hash'],storage=old.storage_bound(),
                slots=[list(x) for x in SLOTS],predict_per_slot=4,predict_total=24,
                GPU_cap=host.GPU_CAP,wall=old.WALL,scientific_admission=False,
                total_cost_equal=False,flops=None,capacity_note='hard ceilings; unseen native values may exceed them')


def verify_manifest(path,expected):
    value=old.parse_canonical(old.read_bound(Path(path),expected,16*1024*1024))
    old.require(value==manifest(),'manifest/closure drift')
    return value


def comparisons(rows):
    old.require(len(rows)==6 and [(r['root'],r['method']) for r in rows]==list(SLOTS),'fixed replay denominator')
    return [dict(root=r['root'],method=r['method'],checkpoints=[{
        key:old.f32_diff(a[key],b[key]) for key in ('latent_history','action_history')}
        for a,b in zip(r['whole'],r['segmented'],strict=True)]) for r in rows]


def execute(store, fixed, tasks, backend_for, guard_for, sync, check, rows):
    """The input barrier must already be durable, all scoring follows replay barrier."""
    replay_rows=[]
    for i,slot in enumerate(fixed['slots']):
        root,method=SLOTS[i];row=rows[i]
        old.require((slot['root'],slot['method'])==(root,method),'slot identity')
        old.require(old.sha(old.canonical(slot['raw10']))==slot['raw_sha256'],'raw sealed bytes')
        row['status']='attempted'
        sink=lambda event,i=i:store.event(dict(kind='replay_event',slot=i,event=event))
        meter=phase.PhaseMeter('replay',sink,sync,check)
        try:
            with guard_for(i,meter):
                replay=phase.replay(tasks[root],slot['raw10'],backend_for(i),meter,sink)
            old.require(meter.counts()==phase.beam.budget(4) and meter.predictor_samples()==dict(attempted=4,completed=4),'four actual packet counts')
            name=f'slot_{i}.replay.z';store.put(name,replay)
            saved=old.reload_packed(store,name)
            old.require(old.canonical(saved)==old.canonical(replay),'replay reload')
            store.event(dict(kind='slot_replay_sealed',slot=i,sha256=old.sha(old.canonical(saved))))
            replay_rows.append(dict(root=root,method=method,raw=saved['raw'],whole=saved['whole'],segmented=saved['segmented']))
            row['status']='sealed'
        finally:
            row['resources']=meter.counts();row['predictor_samples']=meter.predictor_samples()
    store.put('all_replays.z',replay_rows)
    seal=old.sha(old.canonical(replay_rows))
    old.require(old.canonical(old.reload_packed(store,'all_replays.z'))==old.canonical(replay_rows),'all replay reload')
    store.event(dict(kind='all_replays_sealed_before_score',sha256=seal))
    diffs=comparisons(replay_rows);store.put('comparisons.z',diffs)
    store.event(dict(kind='comparisons_sealed',sha256=old.sha(old.canonical(diffs))))
    return diffs


def verify_saved(fixed,tasks,report,payload,events):
    names={'inputs.z','all_replays.z','comparisons.z'}|{f'slot_{i}.replay.z' for i in range(6)}
    old.require(set(payload)==names and payload['inputs.z']==fixed['slots'] and len(report['rows'])==6,'input and payload completeness')
    old.require(events[0]==dict(kind='inputs_sealed_before_model',sha256=old.sha(old.canonical(fixed['slots']))),'input barrier')
    loaded=report['loaded']
    old.require(events[1]==dict(kind='model_loaded',loaded=loaded,model_hash=report['model_hash_before'],cuda=True,
                               tf32=False,autocast=False,math_requested=True),'model load event')
    at=2;result=[]
    for i,(root,method) in enumerate(SLOTS):
        row=report['rows'][i];slot=fixed['slots'][i]
        old.require((row['root'],row['method'],row['status'])==(root,method,'sealed')
                    and row['resources']==phase.beam.budget(4)
                    and row['predictor_samples']==dict(attempted=4,completed=4),'replay hook resources')
        group=[]
        while at<len(events) and events[at].get('kind')=='replay_event' and events[at].get('slot')==i:
            old.require(set(events[at])=={'kind','slot','event'},'event envelope');group.append(events[at]['event']);at+=1
        replay=phase.verify_saved_replay(tasks[root],slot['raw10'],group)
        old.require(old.canonical(replay)==old.canonical(payload[f'slot_{i}.replay.z'])
                    and events[at]==dict(kind='slot_replay_sealed',slot=i,sha256=old.sha(old.canonical(replay))), 'slot replay and seal')
        at+=1;result.append(dict(root=root,method=method,raw=replay['raw'],whole=replay['whole'],segmented=replay['segmented']))
    seal=old.sha(old.canonical(result))
    old.require(old.canonical(payload['all_replays.z'])==old.canonical(result)
                and events[at]==dict(kind='all_replays_sealed_before_score',sha256=seal),'global replay barrier');at+=1
    diffs=comparisons(result)
    old.require(payload['comparisons.z']==diffs and events[at]==dict(kind='comparisons_sealed',sha256=old.sha(old.canonical(diffs)))
                and at+1==len(events),'all comparisons and no extra events')
    return dict(slots=6,predict_per_axis=24,comparisons=diffs,scientific_admission=False)


def run(prepared,expected,output):
    started=time.monotonic();binding=verify_manifest(prepared,expected);fixed=forced()
    authority,previous=host.authorities();tasks,_,_=host.tasks_and_targets(authority,previous)
    store=old.Store(Path(output),started=started)
    rows=[dict(root=r,method=k,status='not_started',resources=None,predictor_samples=None) for r,k in SLOTS]
    errors=[];model=None;before=after=loaded=host_info=gpu=None;torch=None
    try:
        store.put('inputs.z',fixed['slots']);sealed=old.reload_packed(store,'inputs.z')
        old.require(old.canonical(sealed)==old.canonical(fixed['slots']),'inputs reload')
        store.event(dict(kind='inputs_sealed_before_model',sha256=old.sha(old.canonical(sealed))))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch as torch_module
        torch=torch_module
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from hydra.utils import instantiate
        from experiments.checkpoint_compat import remap_legacy_vit_keys
        from experiments.gpu_preflight import require_cuda_allocation
        host_info=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats(device)
        policy=Path(previous['policy']);config=policy/'config.json';weights=policy/'weights.pt'
        cb=old.read_bound(config,previous['bindings_sha256'][str(config)],4*1024*1024)
        wb=old.read_bound(weights,previous['bindings_sha256'][str(weights)],512*1024*1024)
        model,loaded=old.load_bound_buffers(cb,wb,instantiate,lambda stream:torch.load(stream,map_location='cpu',weights_only=True),remap_legacy_vit_keys,device)
        del cb,wb
        old.core.old.enc.frozen(model);old.require(all(p.device.type=='cuda' for p in model.parameters()),'CUDA only')
        before=old.core.old.enc.tensor_hash(model);old.require(before==binding['expected_model_hash'],'loaded weights hash')
        store.event(dict(kind='model_loaded',loaded=loaded,model_hash=before,cuda=True,tf32=False,autocast=False,math_requested=True))
        def check():
            store.check();old.require(not torch.is_autocast_enabled('cuda') and not torch.backends.cuda.matmul.allow_tf32
                and not torch.backends.cudnn.allow_tf32,'precision guard')
            old.require(torch.cuda.max_memory_allocated(device)<=host.GPU_CAP and torch.cuda.max_memory_reserved(device)<=host.GPU_CAP,'GPU cap')
        backends=[phase.beam.NativeEdge(model,device,tasks[r]) for r,_ in SLOTS]
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            execute(store,dict(fixed,slots=sealed),tasks,lambda i:backends[i],lambda i,m:backends[i].guard(m),
                    lambda:torch.cuda.synchronize(device),check,rows)
        check()
    except BaseException as exc:
        errors.append(dict(type=type(exc).__name__,message=str(exc)[:2048]))
        try:store.event(dict(kind='execution_error',error=errors[-1]))
        except BaseException as second:errors.append(dict(type=type(second).__name__,message='journal: '+str(second)[:2048]))
    finally:
        if model is not None:
            try:
                torch.cuda.synchronize(device);old.core.old.enc.frozen(model);after=old.core.old.enc.tensor_hash(model)
                gpu=dict(cap=host.GPU_CAP,peak_allocated=int(torch.cuda.max_memory_allocated(device)),peak_reserved=int(torch.cuda.max_memory_reserved(device)),
                         tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                         autocast=bool(torch.is_autocast_enabled('cuda')),requested_math=True,observed_math=None)
                old.require(before==after==binding['expected_model_hash'],'post weights')
            except BaseException as exc:errors.append(dict(type=type(exc).__name__,message='final model: '+str(exc)[:2048]))
        try:verify_manifest(prepared,expected)
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,message='final closure: '+str(exc)[:2048]))
        complete=not errors and before==after==binding['expected_model_hash'] and all(r['status']=='sealed' for r in rows)
        report=dict(schema=NAME+'.report',status='all_six_sealed' if complete else 'partial_or_failed',rows=rows,errors=errors,
                    prepared=str(Path(prepared)),prepare_sha256=expected,model_hash_before=before,model_hash_after=after,
                    loaded=loaded,host=host_info,gpu=gpu,storage=old.storage_bound(),training=False,
                    new_physics=0,new_encoder=0,new_ad=0,scientific_admission=False)
        terminal=host.stage.finish(store,report)
    timing=old.parse_canonical((store.path/'timing.json').read_bytes())
    old.require(terminal['report']['status']=='all_six_sealed' and timing['wall_within_limit'] is True,'partial/late evidence retained')
    return {k:v for k,v in terminal.items() if k!='report'}


def consume(output,report_sha,completion_sha,timing_sha):
    output=Path(output);old.require(output.is_absolute() and output.is_dir() and not output.is_symlink(),'output directory')
    report=old.parse_canonical(old.read_bound(output/'report.json',report_sha,old.RESERVE))
    completion=old.parse_canonical(old.read_bound(output/'completion.json',completion_sha,old.RESERVE))
    timing=old.parse_canonical(old.read_bound(output/'timing.json',timing_sha,old.RESERVE))
    binding=verify_manifest(report['prepared'],report['prepare_sha256']);fixed=forced()
    authority,previous=host.authorities();tasks,_,_=host.tasks_and_targets(authority,previous)
    result=consume_checked(output,report_sha,completion_sha,timing_sha,report,completion,timing,binding,fixed,previous,tasks)
    verify_manifest(report['prepared'],report['prepare_sha256'])
    return result


def consume_checked(output,report_sha,completion_sha,timing_sha,report,completion,timing,binding,fixed,previous,tasks):
    """Internal saved check: caller authenticates all supplied arguments."""
    keys={'schema','status','rows','errors','prepared','prepare_sha256','model_hash_before','model_hash_after',
          'loaded','host','gpu','storage','training','new_physics','new_encoder','new_ad','scientific_admission',
          'artifacts','journal_events','journal_sha256','array_bytes','metadata_bytes'}
    old.require(set(report)==keys and report['schema']==NAME+'.report' and report['status']=='all_six_sealed' and report['errors']==[]
                and report['scientific_admission'] is False and report['training'] is False
                and report['new_physics']==report['new_encoder']==report['new_ad']==0
                and report['storage']==old.storage_bound() and report['model_hash_before']==report['model_hash_after']==binding['expected_model_hash'],'report/model')
    old.require(completion==dict(schema=old.NAME+'.completion',report_sha256=report_sha,
                timing_boundary='completion file and directory fsync measured by separate timing witness')
                and set(timing)=={'schema','report_sha256','completion_sha256','elapsed_after_completion_fsync','wall_limit','wall_within_limit','timing_boundary'}
                and timing['schema']==old.NAME+'.timing' and timing['report_sha256']==report_sha and timing['completion_sha256']==completion_sha
                and timing['wall_limit']==old.WALL and timing['wall_within_limit'] is True
                and type(timing['elapsed_after_completion_fsync']) in (int,float) and 0<=timing['elapsed_after_completion_fsync']<=old.WALL
                and timing['timing_boundary']=='run entry through completion file and directory fsync; excludes timing witness fsync','completion/timing')
    gpu=report['gpu'];old.require(gpu['cap']==host.GPU_CAP and 0<=gpu['peak_allocated']<=host.GPU_CAP
        and 0<=gpu['peak_reserved']<=host.GPU_CAP and gpu['tf32_matmul'] is gpu['tf32_cudnn'] is gpu['autocast'] is False
        and gpu['requested_math'] is True and gpu['observed_math'] is None,'CUDA precision')
    policy=Path(previous['policy']);loaded=report['loaded']
    old.require(loaded['config_sha256']==previous['bindings_sha256'][str(policy/'config.json')]
                and loaded['weights_sha256']==previous['bindings_sha256'][str(policy/'weights.pt')]
                and loaded['strict_key_coverage'] is True,'loaded config and weights')
    old.require(sum(p.stat().st_size for p in output.iterdir())<=old.CAP,'output cap')
    payload,events=old.read_store(output,report);result=verify_saved(fixed,tasks,report,payload,events)
    return dict(schema=NAME+'.consume',status='six_forced_saved_replays_verified',report_sha256=report_sha,
                completion_sha256=completion_sha,timing_sha256=timing_sha,result=result,scientific_admission=False,
                total_cost_equal=False,flops=None,boundary='saved IO and loader attestation; no neural re-execution')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for name in ('output','prepared','prepare-sha','report-sha','completion-sha','timing-sha','receipt'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.command=='prepare':
        path=Path(a.output);old.require(path.is_absolute(),'absolute output');result=dict(path=str(path),sha256=old.put_new(path.parent,path.name,old.canonical(manifest())))
    elif a.command=='verify':verify_manifest(a.prepared,a.prepare_sha);result=dict(status='prepare_verified')
    elif a.command=='run':result=run(a.prepared,a.prepare_sha,a.output)
    else:
        result=consume(a.output,a.report_sha,a.completion_sha,a.timing_sha)
        path=Path(a.receipt);old.require(path.is_absolute(),'absolute receipt');result=dict(path=str(path),sha256=old.put_new(path.parent,path.name,old.canonical(result)))
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
