"""MAIN-owned real CUDA direct-H2 calibration. Never P1 admission.

Source generation is separate. No simulator, training, witness fabrication or
outcome-driven threshold selection in this driver.
"""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_conditional_direct_screen_20261027_v1 as screen
from experiments.spectral_reachability import p1_conditional_procedural_source_20261027_v1 as source
from experiments.spectral_reachability import p3_joint_prefix_cuda_encoding_20261027_v1 as io

NAME='p1_direct_cuda_calibration_main_20261027_v1'
ROOT=io.REPO
METHODS=('full_H2_GN','full_H2_gradient')
OUTPUT_LIMIT=40*1024**2
require=io.require


def config(method):
    require(method in METHODS,'fixed method')
    return dict(method=method,B1=1,B2=3,initial_logits=[np.zeros((2,10)).tolist()],
                step_scale=1.,damping=.002,extra_line_search_trials=0)


def write(path,value):
    raw=(io.io.canonical(value)+'\n').encode();require(len(raw)<12*1024**2,'bounded JSON')
    with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def material(pack,expected):
    pack=Path(pack).absolute();require(io.sha(pack/'sourcepack.json')==expected,'sourcepack external SHA')
    require(not (pack/'main_cleanup_error.json').exists(),'failed source cleanup is not completed provenance')
    r=io.read(pack/'sourcepack.json');p=r['plan']
    # The first durable journal entry contains the exact frozen exclusions.
    events=source.read_journal(pack/'journal.jsonl')
    require(events and events[0]['kind']=='run_started','actual source journal')
    e=events[0]['exclusions'];checked=source.validate_plan(p,e)
    require(checked['ready'] and p['purpose']=='calibration_exploration' and
            r['status']=='completed_calibration_source_only' and r['first_error'] is None and
            r['secondary_errors']==[] and not r['freshness_certified_globally'],'complete calibration-only source')
    require(io.sha(pack/'journal.jsonl')==r['journal_sha256'],'full source journal SHA')
    require(len(r['slots'])==2 and all(s['status']=='completed' for s in r['slots']),'all two predeclared source slots required')
    bindings={str(pack/'sourcepack.json'):expected,str(pack/'journal.jsonl'):r['journal_sha256']}
    for row in r['slots']:
        path=pack/row['raw_source_file'];require(io.sha(path)==row['raw_source_sha256'],'raw source bytes')
        source.load_raw_source(path,p);bindings[str(path)]=row['raw_source_sha256']
    # Native original checkpoint and normalization, not a transplanted old Q.
    original=io.collect()
    norm=io.read(ROOT/'results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_prepared01/protocol.json')['normalization']
    return dict(schema=NAME+'.protocol',sourcepack=str(pack),sourcepack_sha256=expected,
                source_plan=p,exclusions=e,source_checks=checked,source_resources=r['resources'],
                source_reset_internal_physics_steps=r['reset_internal_physics_steps'],
                bindings=bindings,model_bindings=original['model_bindings'],policy=original['policy'],
                normalization=norm,metric=dict(kind='identity_native192',projection_rank=192,no_fit=True),
                methods=[config(m) for m in METHODS],packet_logit_cap=float(.75*np.sqrt(5)),
                source_history='causal_padding_not_observed_transitions',
                maximum_neural_predicts=120,maximum_action_encoder=120,
                maximum_J=12,mathematical_VJP_directions=2304,
                maximum_new_visual_encoder=8,maximum_new_projector=8,
                output_limit_bytes=OUTPUT_LIMIT,remaining_disk_reserve_bytes=200*1024**2,
                training=False,P1_admission=False,model_task_eligible=False,
                gaps=['Independent witness producer/replay','Independent calibration thresholds and confirmation',
                      'Observed-transition history rather than causal padding','Real physical non-greedy capability and task benefit'])


def prepare(pack,expected,output):
    output=Path(output).absolute();require(not output.exists(),'new prepare')
    p=material(pack,expected)
    files=set(io.source_files()+[Path(__file__).resolve(),Path(screen.__file__).resolve(),Path(source.__file__).resolve(),
        ROOT/'experiments/spectral_reachability/p1_conditional_source_metadata_20261027_v1.py',
        ROOT/'experiments/spectral_reachability/p1_conditional_task_admission_20261027_v1.py',
        ROOT/'tests'/('test_'+NAME+'.py')])
    p['sources_sha256']={str(f):io.sha(f) for f in files}
    output.mkdir();write(output/'protocol.json',p)
    seal=io.create_seal(files=sorted(files),args_json=io.io.canonical(dict(protocol_sha256=io.sha(output/'protocol.json'))),output_dir=output/'source_seal')
    write(output/'prepare.json',dict(protocol_sha256=io.sha(output/'protocol.json'),source_seal=seal))


def verify(folder,expected):
    folder=Path(folder).absolute();require(io.sha(folder/'prepare.json')==expected,'prepare external hash')
    r=io.read(folder/'prepare.json');require(io.sha(folder/'protocol.json')==r['protocol_sha256'],'protocol bytes')
    p=io.read(folder/'protocol.json');s=r['source_seal'];io.verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    for path,h in {**p['bindings'],**p['sources_sha256'],**p['source_plan']['source_closure'],**p['model_bindings']}.items():require(io.sha(path)==h,'input/source/model drift')
    rebuilt=material(p['sourcepack'],p['sourcepack_sha256']);rebuilt['sources_sha256']=p['sources_sha256']
    require(io.same(p,rebuilt),'full input reconstruction')
    return p


def run(folder,expected,output):
    p=verify(folder,expected);output=Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free>200*1024**2+OUTPUT_LIMIT,'complete output budget plus200MiB reserve')
    folder=Path(folder).absolute();write(folder.with_name(folder.name+'.launch.json'),dict(output=str(output),prepare_sha256=expected))
    output.mkdir();errors=[];model=None;before=after=None;host=None;tasks=[];direct=[];components=[];mapping={};source_ok=False
    with (output/'image.events.jsonl').open('x') as stream:
        def image_event(e):
            stream.write(io.io.canonical(e)+'\n');stream.flush();os.fsync(stream.fileno())
        try:
            import torch
            from torch.nn.attention import SDPBackend,sdpa_kernel
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.backends.cuda.matmul.allow_tf32=False
            model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
            io.frozen(model);before=io.tensor_hash(model)
            encode=screen.native_image_encoder(model,device,p0.image_transform(),image_event)
            r=io.read(Path(p['sourcepack'])/'sourcepack.json')
            model_sha=next(h for f,h in p['model_bindings'].items() if Path(f).name=='weights.pt')
            model_config=next(h for f,h in p['model_bindings'].items() if Path(f).name=='config.json')
            for row in r['slots']:
                path=Path(p['sourcepack'])/row['raw_source_file'];raw=source.load_raw_source(path,p['source_plan'])
                template=dict(environment='swm/PushT-v1',task_contract_sha256=io.digest(p),
                    projection=np.eye(192).tolist(),model_sha256=model_sha,model_config_sha256=model_config,
                    metric_source_sha256=io.digest(p['metric']),goal_source_sha256=screen.core.sha(np.ascontiguousarray(raw['goal_rgb']).tobytes()),
                    representation='native_projected_cls',latent_dim=192,horizon=2,packet_steps=5,control_dim=2,
                    action_mean=p['normalization']['mean'],action_std=p['normalization']['std'],packet_logit_cap=p['packet_logit_cap'],feedback='open_loop_model_only')
                encoded=screen.encode_source(path,p['source_plan'],p['exclusions'],row['raw_source_sha256'],template,encode)
                write(output/f'task_{row["slot"]["candidate_id"]}.json',encoded)
                tasks.append(encoded);mapping[encoded['root_sha256']]=encoded['task_id']
            scan=source.to_admission_scan(p['source_plan'],p['exclusions'],r,Path(p['sourcepack']),mapping)
            write(output/'source_scan.json',scan)
            with sdpa_kernel(SDPBackend.MATH):
                for i,t in enumerate(tasks):
                    for c in p['methods']:
                        sink=screen.DiskJournal(output/f'root_{i}_{c["method"]}',byte_limit=8*1024**2)
                        result=screen.run_direct(t['task'],c,screen.NativeCuda(model,device,t['task']),sink)
                        components.append(result)
                        write(sink.directory/'report.json',result);direct.append(dict(root_sha256=t['root_sha256'],method=c['method'],report_path=str(sink.directory/'report.json'),report_sha256=io.sha(sink.directory/'report.json'),status=result['status']))
                        require(result['status']=='direct_component_completed_not_admitted' and result['error'] is None,'direct component failed; inspect durable arrays')
            torch.cuda.synchronize(device)
        except BaseException as exc:errors.append(io.error('execution',exc))
        finally:
            if model is not None:
                try:io.frozen(model);after=io.tensor_hash(model)
                except BaseException as exc:errors.append(io.error('final_model',exc))
            try:verify(folder,expected);source_ok=True
            except BaseException as exc:errors.append(io.error('final_sources',exc))
            counts={}
            for resource in ('predict','action_encoder','jacobian','vjp_directions','svd','linear_solve','proposal','selection'):
                es=[e for c in components for e in c['extended_journal'] if e['resource']==resource]
                counts[resource]=dict(attempted=sum(e['samples'] for e in es),completed=sum(e['samples'] for e in es if e['completed']))
            image_counts={}
            try:
                events=[json.loads(line) for line in (output/'image.events.jsonl').read_bytes().splitlines()]
                for kind in ('encoder','projector'):
                    image_counts[kind]=dict(attempted=sum(e.get('samples',0) for e in events if e['kind']=='native_image_attempted' and e['resource']==kind),
                        completed=sum(e['counts'][kind] for e in events if e['kind']=='native_image_completed'))
            except BaseException as exc:errors.append(io.error('image_journal',exc))
            budget_ok=(all(counts[k]==dict(attempted=n,completed=n) for k,n in
                dict(predict=120,action_encoder=120,jacobian=12,vjp_directions=2304,svd=6,linear_solve=6,proposal=12,selection=12).items())
                and image_counts=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')})
            complete=not errors and budget_ok and len(direct)==4 and len(tasks)==2 and source_ok and before is not None and before==after
            report=dict(status='direct_calibration_completed' if complete else 'partial_or_failed',host=host,errors=errors,
                actual_direct_resources=counts,actual_image_resources=image_counts,exact_budget_verified=budget_ok,
                direct=direct,encoded_tasks=len(tasks),prepare_sha256=expected,model_hash_before=before,model_hash_after=after,
                model_unchanged=before is not None and before==after,source_verified=source_ok,source_checks=p['source_checks'],
                source_resources=p['source_resources'],source_reset_internal_physics_steps=p['source_reset_internal_physics_steps'],
                source_history=p['source_history'],new_physical_steps=0,training=False,P1_admission=False,model_task_eligible=False,gaps=p['gaps'])
            write(output/'report.json',report)
    require(complete,'real direct calibration incomplete; no automatic retries')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--sourcepack',type=Path);parser.add_argument('--sourcepack-sha');parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha');parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.sourcepack,a.sourcepack_sha,a.output);print(io.sha(a.output/'prepare.json'))
    elif a.command=='verify':print(verify(a.prepared,a.prepare_sha)['schema'])
    else:print(io.io.canonical(run(a.prepared,a.prepare_sha,a.output)))


if __name__=='__main__':main()
