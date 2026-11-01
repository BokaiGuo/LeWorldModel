"""MAIN-only observed-history direct calibration; prepare/audit are NumPy-only.

Never resets an environment, invokes a padding loader, trains, or admits P1.
All original screen events are retained losslessly, not sampled/compressed away.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_observed_history_source_20261027_v1 as observed
from experiments.spectral_reachability import p1_conditional_direct_screen_20261027_v1 as screen
from experiments.spectral_reachability import p3_joint_prefix_cuda_encoding_20261027_v1 as io

NAME='p1_observed_direct_cuda_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
FLOOR=200*1024**2
CAP=56*1024**2
METHODS=('full_H2_GN','full_H2_gradient')
TOTAL=dict(predict=280,action_encoder=280,jacobian=32,vjp_directions=6144,
           svd=16,linear_solve=16,proposal=32,selection=32)
require=observed.require
digest=observed.source.digest
sha=io.sha
read=io.read


def read_lines(path):
    return [io.io.decode(line) for line in Path(path).read_bytes().splitlines()]


def config(method):
    require(method in METHODS,'fixed genuine H2 method')
    return dict(method=method,B1=4,B2=8,initial_logits=[np.zeros((2,10)).tolist()],
                step_scale=1.,damping=.002,extra_line_search_trials=0)


def safe_child(directory,name):
    require(isinstance(name,str) and Path(name).name==name and name not in ('','.','..'),'flat source file name')
    path=Path(directory)/name
    io.regular(path)
    return path


def angle_diagnostic(state,goal):
    s=observed.source.vector(state);g=observed.source.vector(goal)
    position=float(np.linalg.norm(g[:4]-s[:4]));delta=abs(float(g[4]-s[4]))
    native=min(delta,2*np.pi-delta);wrapped=abs(float((g[4]-s[4]+np.pi)%(2*np.pi)-np.pi))
    n=bool(position<20 and native<np.pi/9);w=bool(position<20 and wrapped<np.pi/9)
    return dict(position_distance=position,abs_angle_delta=delta,native_angle=float(native),wrapped_angle=wrapped,
                native_terminal=n,wrapped_terminal=w,discrepancy=n!=w)


def audit_source(report,prepared,events,directory):
    """Full successful source-only gate, no placeholder tasks or old loaders.

    Original source bindings are hashed as bytes. The NEW observed NPZ loader
    is used only for observed arrays. Legacy parent NPZs are directly inspected
    for ancestry/goal equality, never passed to an encoding/padding bridge.
    """
    require(report['schema']==observed.NAME+'.sourcepack' and report['prepare']==prepared,'same observed protocol')
    previous=None
    for index,event in enumerate(events):
        value=dict(event);claimed=value.pop('event_sha256')
        require(value['index']==index and value['previous_sha256']==previous and digest(value)==claimed,'source journal complete hash chain')
        previous=claimed
    require(report['status']=='observed_calibration_source_completed' and report['first_error'] is None and
            report['secondary_errors']==[] and report['durable_journal_complete'] is True,'whole source actual terminal required')
    require(report['purpose']=='calibration_exploration' and report['global_freshness']=='unknown' and
            report['P1_admission'] is False and report['neural_calls']==0 and
            report['downstream_future_labels_read'] is False,'calibration-only acquisition')
    require(prepared['schema']==observed.NAME+'.prepare' and prepared['counts']==observed.COUNTS and
            prepared['observed_public_step_indices']==[0,5,10] and prepared['replacement'] is False and
            np.array_equal(prepared['raw_commands'],np.zeros((10,2))) and prepared['normalization']==observed.normalization(),
            'fixed acquisition schedule and bound public normalization')
    for path,h in prepared['bindings'].items():require(sha(path)==h,'original acquisition binding drift: '+path)
    require(events and events[0]['kind']=='run_started' and events[0]['prepare']==prepared and
            events[-1]['kind']=='run_finished' and events[-1]['status']==report['status'] and
            events[-1]['counts']==report['counts'] and events[-1]['first_error'] is None,'complete acquisition terminal journal')
    require([e['receipt'] for e in events if e['kind']=='root_finished']==report['rows'],'all ordered source rows in durable journal')
    require(len(report['rows'])==len(prepared['selections'])==2,'both original roots required, no replacement')
    counts={k:dict(attempted=0,completed=0) for k in observed.COUNTS};stack=[];completed=[]
    for event in events:
        if event['kind']=='operation_started':
            resource=event['resource'];require(resource in counts,'known actual source resource')
            counts[resource]['attempted']+=1
            stack.append((event,counts['physics_step']['completed']))
        elif event['kind']=='operation_completed':
            require(bool(stack),'source completion without dispatch')
            start,physics_at_start=stack.pop();resource=event['resource']
            require(all(event[k]==v for k,v in start.items() if k not in ('kind','index','event_sha256','previous_sha256')),
                    'source nested operation coordinate/input mismatch')
            counts[resource]['completed']+=1;completed.append(event)
            if resource in ('reset','public_step'):
                require(counts['physics_step']['completed']-physics_at_start==(2 if resource=='reset' else 10),
                        'actual source internal physics count per external dispatch')
    require(not stack and counts==report['counts']=={k:dict(attempted=n,completed=n) for k,n in observed.COUNTS.items()},'exact acquisition resources')
    require(report['internal_physics']['actual_hook_counts']==counts['physics_step'] and
            report['internal_physics']['completed_resets_imply']==8 and
            report['internal_physics']['completed_public_steps_imply']==200,'actual and implied source physics')
    origin=read(prepared['inputs']['report']);origin_dir=Path(prepared['inputs']['report']).parent
    packets=[];roots=[];angles=[];seen=set()
    for row,selection,parent in zip(report['rows'],prepared['selections'],origin['slots']):
        cid=selection['candidate_id']
        require(row['candidate_id']==cid==parent['slot']['candidate_id'] and row['status']=='completed' and row['completed_public_steps']==10,
                'fixed complete source row')
        path=safe_child(directory,row['file']);require(path.name==f'source_{cid}.npz' and sha(path)==row['file_sha256'],'observed source SHA')
        packet=observed.load_observed(path,prepared);meta=packet['metadata']
        require(row['metadata']==meta and meta['root_sha256'] not in seen,'source metadata and unique roots')
        seen.add(meta['root_sha256']);packets.append(packet)
        require(np.array_equal(row['last_state'],packet['state']),'final observed source state')
        parent_path=safe_child(origin_dir,parent['raw_source_file']);require(sha(parent_path)==parent['raw_source_sha256'],'bound parent SHA')
        with np.load(parent_path,allow_pickle=False) as z:
            parent_state=z['state'].copy();parent_goal=z['goal'].copy();parent_rgb=z['rgb_history'].copy()
            parent_goal_rgb=z['goal_rgb'].copy();parent_raw=z['raw_action_history'].copy()
        require(selection['parent_root_sha256']==parent['metadata']['root_sha256'] and
                np.array_equal(packet['goal'],parent_goal) and np.array_equal(packet['goal_rgb'],parent_goal_rgb) and
                np.array_equal(packet['observed_states'][0],parent_state) and np.array_equal(packet['rgb_history'][0],parent_rgb[0]),
                'same reconstructed original state and actual goal frame')
        row_events=[e for e in completed if e['candidate_id']==cid]
        steps=[e for e in row_events if e['resource']=='public_step']
        resets=[e for e in row_events if e['resource']=='reset']
        renders=[e for e in row_events if e['resource']=='explicit_render']
        require(len(steps)==10 and len(resets)==2 and len(renders)==3,'fixed per-root acquisition calls')
        require(resets[0]['seed']==row['random_reset_seed']==selection['selected_attempt']['random_reset_seed'] and
                resets[1]['seed']==row['placement_reset_seed']==selection['selected_attempt']['placement_reset_seed'],'original reset seeds')
        require(np.array_equal(resets[1]['receipt']['state'],parent_state) and np.array_equal(resets[0]['receipt']['goal'],packet['goal']) and
                np.array_equal(resets[1]['receipt']['goal'],packet['goal']),'original reset receipt and goal')
        states={0:packet['observed_states'][0]}
        for step,event in enumerate(steps,1):
            receipt=event['receipt'];require(event['public_step']==step and np.array_equal(event['raw_action'],np.zeros(2)), 'executed fixed raw-zero chronology')
            require(receipt['terminated'] is False and receipt['truncated'] is False and np.array_equal(receipt['goal'],packet['goal']), 'no terminal/truncated or goal drift')
            states[step]=np.asarray(receipt['state']);diagnostic=angle_diagnostic(states[step],packet['goal'])
            require(not diagnostic['native_terminal'],'saved native terminal conflicts with successful source')
            angles.append(dict(candidate_id=cid,public_step=step,**diagnostic))
        initial_angle=angle_diagnostic(states[0],packet['goal']);require(not initial_angle['native_terminal'],'initial native terminal')
        angles.append(dict(candidate_id=cid,public_step=0,**initial_angle))
        require(len(row['checkpoints'])==3 and [x['public_step'] for x in row['checkpoints']]==[0,5,10],'all checkpoints required')
        checkpoint_events=[e['receipt'] for e in events if e['kind']=='checkpoint_write_completed' and e['candidate_id']==cid]
        require(checkpoint_events==row['checkpoints'],'durable checkpoint receipt equality')
        for k,checkpoint in enumerate(row['checkpoints']):
            cp=safe_child(directory,checkpoint['file']);require(cp.name==f'checkpoint_{cid}_{[0,5,10][k]}.npz' and sha(cp)==checkpoint['sha256'],'checkpoint SHA')
            with np.load(cp,allow_pickle=False) as z:
                require(set(z.files)==set('rgb state goal goal_rgb snapshot_json public_step'.split()),'checkpoint schema')
                snap=json.loads(str(z['snapshot_json']))
                require(int(z['public_step'])==[0,5,10][k] and np.array_equal(z['state'],states[[0,5,10][k]]) and
                        np.array_equal(z['state'],packet['observed_states'][k]) and np.array_equal(z['state'],snap['state7']), 'checkpoint/state/step history binding')
                require(np.isfinite(snap['block_velocity']).all() and np.isfinite(snap['block_angular_velocity']),'finite hidden provenance')
                require(np.array_equal(z['rgb'],packet['rgb_history'][k]) and np.array_equal(z['goal'],packet['goal']) and np.array_equal(z['goal_rgb'],packet['goal_rgb']),'checkpoint RGB/goal exact')
            require(observed.image_sha(packet['rgb_history'][k])==checkpoint['rgb_sha256']==renders[k]['receipt']['rgb_sha256'] and
                    renders[k]['before']==renders[k]['receipt']['state'] and np.array_equal(renders[k]['before']['state7'],packet['observed_states'][k]),'render/state preservation')
        for step in (0,10):
            state=states[step];domain=read(prepared['inputs']['request'])['plan']['domain']
            low,high=domain['canvas_xy'];position=float(np.linalg.norm(state[2:4]-packet['goal'][2:4]));angle=angle_diagnostic(state,packet['goal'])['wrapped_angle']
            receipt=row['domain_checks'][0 if step==0 else 1]
            checks=dict(canvas=bool(((state[:4]>=low)&(state[:4]<=high)).all()),contact=bool(domain['contact_gap_range'][0]<=receipt['contact_gap']<=domain['contact_gap_range'][1]),
                        nonterminal=bool(position>domain['nonterminal_position_tolerance'] or angle>domain['nonterminal_angle_tolerance']))
            require(receipt['checks']==checks and receipt['accepted'] is True and all(checks.values()),'frozen source-domain receipt')
        roots.append(dict(candidate_id=cid,root_sha256=meta['root_sha256'],parent_root_sha256=meta['parent_root_sha256'],
            root_equals_parent=meta['root_sha256']==meta['parent_root_sha256'],
            rgb_at_0_5_10_identical=all(np.array_equal(packet['rgb_history'][0],x) for x in packet['rgb_history']),
            state_at_0_5_10_identical=all(np.array_equal(packet['observed_states'][0],x) for x in packet['observed_states']),
            numerical_arrays_identical_to_parent_padding=bool(np.array_equal(packet['rgb_history'],parent_rgb) and
                np.array_equal(packet['state'],parent_state) and np.array_equal(packet['raw_action_history'],parent_raw))))
    return dict(packets=packets,summary=dict(validated_rows=2,actual_acquisition_counts=counts,roots=roots,
        novel_root_count=sum(not x['root_equals_parent'] for x in roots),independent_sample_increment=0,
        independent_sample_note='same two predeclared parent source families; acquisition does NOT add independent samples',
        native_wrapped_discrepancies=sum(x['discrepancy'] for x in angles),angle_diagnostics=angles,
        history='actually_observed_even_when_stationary',global_freshness='unknown',P1_admission=False))


def material(pack,expected_sha,source_prepare,source_prepare_sha):
    pack=Path(pack).resolve();source_prepare=Path(source_prepare).resolve()
    require(sha(pack/'sourcepack.json')==expected_sha and sha(source_prepare)==source_prepare_sha,'external actual-source/prepare SHA')
    r=read(pack/'sourcepack.json');p=read(source_prepare)
    require(sha(pack/'journal.jsonl')==r['journal_sha256'],'full observed journal SHA')
    events=observed.source.read_journal(pack/'journal.jsonl')
    data=audit_source(r,p,events,pack)
    bindings={str(pack/'sourcepack.json'):expected_sha,str(source_prepare):source_prepare_sha,str(pack/'journal.jsonl'):r['journal_sha256']}
    for row in r['rows']:
        bindings[str(safe_child(pack,row['file']))]=row['file_sha256']
        for checkpoint in row['checkpoints']:bindings[str(safe_child(pack,checkpoint['file']))]=checkpoint['sha256']
    return dict(**data,bindings=bindings,source_prepare=p)


def model_metadata():
    policy=Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
    bindings={str(policy/'config.json'):'2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09',
              str(policy/'weights.pt'):'48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'}
    for path,h in bindings.items():require(sha(path)==h,'original checkpoint SHA')
    return dict(policy=str(policy),bindings=bindings)


def template(material,contract_sha,model_sha,model_config_sha):
    n=material['source_prepare']['normalization']
    return dict(environment='swm/PushT-v1',task_contract_sha256=contract_sha,projection=np.eye(192).tolist(),
        model_sha256=model_sha,model_config_sha256=model_config_sha,metric_source_sha256=digest(dict(kind='identity_native192',no_fit=True)),
        representation='native_projected_cls',latent_dim=192,horizon=2,packet_steps=5,control_dim=2,
        action_mean=n['mean'],action_std=n['std'],packet_logit_cap=float(.75*np.sqrt(5)),feedback='open_loop_model_only')


def encode_observed(packet,base,encode_rgb):
    """RGB-only callback; no legacy source loader, no simulated state inputs."""
    norm=packet['metadata']['normalization'];t=deepcopy(base)
    require(t['action_mean']==norm['mean'] and t['action_std']==norm['std'],'same source action normalization')
    t['goal_source_sha256']=observed.image_sha(packet['goal_rgb'])
    raw=packet['raw_action_history'];tokens=observed.normalized(raw,norm)
    require(np.array_equal(tokens,packet['normalized_action_history']),'observed between-state tokens')
    bootstrap=dict(t,goal=[0.]*192,root_raw_action_history=raw.tolist(),root_context=dict(
        layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.zeros((3,192)).tolist(),action_history=tokens.tolist()))
    screen.core.validate_task(bootstrap)  # Reject privileged/unknown fields BEFORE encoding.
    values=[screen.core.array(encode_rgb(rgb.copy()),(192,)) for rgb in [*packet['rgb_history'],packet['goal_rgb']]]
    task=dict(bootstrap,goal=values[-1].tolist(),root_context=dict(bootstrap['root_context'],latent_history=[x.tolist() for x in values[:3]]))
    screen.core.validate_task(task)
    return dict(task=task,task_id=digest(task),root_sha256=packet['metadata']['root_sha256'],
        parent_root_sha256=packet['metadata']['parent_root_sha256'],root_equals_parent=packet['metadata']['root_sha256']==packet['metadata']['parent_root_sha256'],
        actual_observed_public_steps=[0,5,10],history_semantics='observed_not_padding_even_when_numerically_identical',
        global_freshness='unknown',independent_sample_increment=0,P1_admission=False)


def used_bytes(directory):
    return sum(x.stat().st_size for x in Path(directory).rglob('*') if x.is_file())


def write(root,path,value):
    raw=(observed.source.canonical(value)+'\n').encode()
    require(used_bytes(root)+len(raw)<=CAP,'56MiB complete output cap')
    require(shutil.disk_usage(root).free>=FLOOR+len(raw),'200MiB write floor')
    with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


class SharedJournal:
    """Original screen events, lossless JSONL; shared 56MiB cap and 200MiB floor."""
    def __init__(self,root,directory):
        self.root=Path(root);self.directory=Path(directory);self.directory.mkdir(exist_ok=False)
        self.path=self.directory/'direct_journal.jsonl'
        with self.path.open('xb') as f:f.flush();os.fsync(f.fileno())
    def __call__(self,event):
        raw=(observed.source.canonical(event)+'\n').encode()
        require(used_bytes(self.root)+len(raw)<=CAP,'56MiB complete output cap')
        require(shutil.disk_usage(self.root).free>=FLOOR+len(raw),'200MiB journal floor')
        with self.path.open('ab') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def build(pack,source_sha,source_prepare,source_prepare_sha):
    data=material(pack,source_sha,source_prepare,source_prepare_sha);model=model_metadata()
    paths=set(io.source_files()+[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
        Path(observed.__file__).resolve(),Path(screen.__file__).resolve(),Path(screen.core.__file__).resolve()])
    return dict(schema=NAME+'.prepare',inputs=dict(pack=str(Path(pack).resolve()),source_sha=source_sha,
        source_prepare=str(Path(source_prepare).resolve()),source_prepare_sha=source_prepare_sha),
        source_bindings=data['bindings'],source_audit=data['summary'],source_sha256={str(p):sha(p) for p in sorted(paths)},
        model=model,normalization=data['source_prepare']['normalization'],methods=[config(m) for m in METHODS],
        direct_resources=TOTAL,image_resources=dict(encoder=8,projector=8),output_cap_bytes=CAP,disk_floor_bytes=FLOOR,
        metric=dict(kind='identity_native192',no_fit=True),training=False,new_physical_steps=0,
        source_acquisition_is_previously_paid=True,global_freshness='unknown',P1_admission=False,
        model_task_eligible=False,independent_sample_increment=0,
        gaps=['No independently validated legal two-stage witness/replay','No calibration-frozen confirmation threshold',
              'Eight optimization iterations do NOT certify plateau/stationarity','No physical task/mechanism efficacy evidence'])


def prepare(output,**inputs):
    p=build(**inputs);path=Path(output)
    raw=(observed.source.canonical(p)+'\n').encode()
    require(len(raw)<256*1024,'small hash-only prepare; no source closure copies')
    require(shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'prepare-only 200MiB floor')
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return p


def verify(prepared,expected_sha):
    require(sha(prepared)==expected_sha,'external direct prepare SHA');p=read(prepared)
    require(p['schema']==NAME+'.prepare','own prepare schema')
    require(p==build(**p['inputs']),'full small protocol/source reconstruction')
    return p


def native_transform():
    # Same native image_transform recipe from the SHA-bound original public
    # source, without importing its simulator/old planner module.
    import torch
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms
    return transforms.Compose([transforms.ToDtype(torch.float32,scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),transforms.Resize(size=224)])


def run(prepared,expected_sha,output):
    p=verify(prepared,expected_sha);output=Path(output)
    require(not output.exists(),'exclusive NEW output')
    require(shutil.disk_usage(output.parent).free>=FLOOR+CAP,'256MiB required: 56MiB cap + 200MiB floor; remain prepare-only, never delete assets')
    output.mkdir(exist_ok=False)
    errors=[];model=None;before=after=None;host=None;tasks=[];components=[];direct=[];image_events=[];source_ok=False
    def fail(phase,exc):errors.append(dict(phase=phase,type=type(exc).__name__,message=str(exc)))
    image_path=output/'image.events.jsonl'
    with image_path.open('xb') as stream:stream.flush();os.fsync(stream.fileno())
    def image_event(event):
        image_events.append(deepcopy(event));raw=(observed.source.canonical(event)+'\n').encode()
        require(used_bytes(output)+len(raw)<=CAP,'56MiB complete output cap')
        require(shutil.disk_usage(output).free>=FLOOR+len(raw),'200MiB image journal floor')
        with image_path.open('ab') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:
        data=material(p['inputs']['pack'],p['inputs']['source_sha'],p['inputs']['source_prepare'],p['inputs']['source_prepare_sha'])
        write(output,output/'source_audit.json',data['summary'])
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0');torch.backends.cuda.matmul.allow_tf32=False
        model=load_compatible_pretrained(p['model']['policy'],device=device).eval().requires_grad_(False)
        io.frozen(model);before=io.tensor_hash(model)
        encoder=screen.native_image_encoder(model,device,native_transform(),image_event)
        weights=next(h for path,h in p['model']['bindings'].items() if Path(path).name=='weights.pt')
        model_config=next(h for path,h in p['model']['bindings'].items() if Path(path).name=='config.json')
        base=template(data,digest(p),weights,model_config)
        for index,packet in enumerate(data['packets']):
            encoded=encode_observed(packet,base,encoder);tasks.append(encoded)
            write(output,output/f'task_{index}.json',encoded)
        with sdpa_kernel(SDPBackend.MATH):
            for index,encoded in enumerate(tasks):
                for c in p['methods']:
                    sink=SharedJournal(output,output/f'root_{index}_{c["method"]}')
                    result=screen.run_direct(encoded['task'],c,screen.NativeCuda(model,device,encoded['task']),sink)
                    components.append(result)
                    write(output,sink.directory/'report.json',result)
                    direct.append(dict(root_sha256=encoded['root_sha256'],method=c['method'],status=result['status'],
                        report_path=str((sink.directory/'report.json').resolve()),report_sha256=sha(sink.directory/'report.json'),
                        journal_sha256=sha(sink.path)))
                    require(result['status']=='direct_component_completed_not_admitted' and result['error'] is None and not result['secondary_errors'],'direct component failure; no retries')
        torch.cuda.synchronize(device)
    except BaseException as exc:fail('execution',exc)
    finally:
        if model is not None:
            try:io.frozen(model);after=io.tensor_hash(model)
            except BaseException as exc:fail('final_model',exc)
        try:verify(prepared,expected_sha);source_ok=True
        except BaseException as exc:fail('final_source',exc)
        counts={}
        for resource in TOTAL:
            events=[e for c in components for e in c['extended_journal'] if e['resource']==resource]
            counts[resource]=dict(attempted=sum(e['samples'] for e in events),completed=sum(e['samples'] for e in events if e['completed']))
        image_counts={}
        for kind in ('encoder','projector'):
            image_counts[kind]=dict(attempted=sum(e.get('samples',0) for e in image_events if e['kind']=='native_image_attempted' and e['resource']==kind),
                completed=sum(e['counts'][kind] for e in image_events if e['kind']=='native_image_completed'))
        budget_ok=(counts=={k:dict(attempted=n,completed=n) for k,n in TOTAL.items()} and
            image_counts=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')})
        complete=not errors and budget_ok and len(tasks)==2 and len(components)==4 and source_ok and before is not None and before==after
        report=dict(schema=NAME+'.report',status='observed_direct_calibration_completed' if complete else 'partial_or_failed',
            errors=errors,host=host,prepare_sha256=expected_sha,source_verified=source_ok,source_audit=p['source_audit'],
            actual_source_acquisition_resources=p['source_audit']['actual_acquisition_counts'],source_acquisition_previously_paid=True,
            new_physical_steps=0,training=False,actual_direct_resources=counts,actual_image_resources=image_counts,
            exact_budget_verified=budget_ok,direct=direct,encoded_tasks=len(tasks),model_hash_before=before,
            model_hash_after=after,model_unchanged=before is not None and before==after,independent_sample_increment=0,
            global_freshness='unknown',P1_admission=False,model_task_eligible=False,gaps=p['gaps'])
        try:write(output,output/'report.json',report)
        except BaseException as exc:
            fail('final_report',exc);report['status']='partial_or_failed'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('audit-source','prepare','verify','run'))
    parser.add_argument('--sourcepack',type=Path);parser.add_argument('--source-sha');parser.add_argument('--source-prepare',type=Path)
    parser.add_argument('--source-prepare-sha');parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    inputs=dict(pack=args.sourcepack,source_sha=args.source_sha,source_prepare=args.source_prepare,source_prepare_sha=args.source_prepare_sha)
    if args.command=='audit-source':
        inputs['expected_sha']=inputs.pop('source_sha');value=material(**inputs)['summary']
        if args.output:
            raw=(observed.source.canonical(value)+'\n').encode();require(shutil.disk_usage(args.output.parent).free>=FLOOR+len(raw),'audit write floor')
            with args.output.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    elif args.command=='prepare':
        value=prepare(args.output,**inputs)
        value=dict(prepared=str(args.output.resolve()),sha256=sha(args.output),source_audit=value['source_audit'],status='prepare_only_no_cuda')
    elif args.command=='verify':value=dict(schema=verify(args.prepared,args.prepare_sha)['schema'],status='verified_no_cuda')
    else:value=run(args.prepared,args.prepare_sha,args.output)
    print(observed.source.canonical(value))
    if args.command=='run' and value['status']!='observed_direct_calibration_completed':raise SystemExit(2)


if __name__=='__main__':main()
