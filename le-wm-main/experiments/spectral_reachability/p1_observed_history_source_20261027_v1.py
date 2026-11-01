"""MAIN-only fixed observed-history producer. Import/prepare never runs physics.

Independent schema; legacy causal-padding source contracts are NOT relabelled.
No neural imports, task selection, replacement, or threshold calibration.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_conditional_procedural_source_20261027_v1 as source
from experiments.spectral_reachability import p1_source_reset_goal_audit_20261027_v1 as reset_audit

NAME='p1_observed_history_source_20261027_v1'
FLOOR=200*1024**2
CAP=8*1024**2
AUTH='MAIN_EXPLICIT_REBUILD_AND_20_PUBLIC_ZERO_STEPS'
COUNTS=dict(reset=4,public_step=20,explicit_render=6,set_state=8,goal_render=4,physics_step=208)
require=source.require


def image_sha(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def normalization():
    path=source.PROJECT/'experiments/run_pusht_counterfactual.py'
    require(source.file_sha(path)=='0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34','public normalizer source drift')
    constants={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'):
                require(isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='np.asarray','literal public normalizer')
                constants[name]=np.asarray(ast.literal_eval(node.value.args[0]),np.float32).tolist()
    mean=constants['PUBLIC_ACTION_MEAN'];std=constants['PUBLIC_ACTION_STD']
    require(np.asarray(mean).shape==np.asarray(std).shape==(2,) and np.isfinite(mean).all() and np.isfinite(std).all() and (np.asarray(std)>0).all(),'valid normalization')
    return dict(mean=mean,std=std,source=str(path),source_sha256=source.file_sha(path),training_provenance_verified=False)


def normalized(raw, norm):
    a=np.asarray(raw,np.float32)
    require(a.ndim==2 and a.shape[1]==2 and len(a)%5==0,'whole five-step packets')
    return (a.reshape(-1,10)-np.tile(np.asarray(norm['mean'],np.float32),5))/np.tile(np.asarray(norm['std'],np.float32),5)


def build(request,report,request_sha256,report_sha256,reset_audit_report,reset_audit_sha256):
    req,saved,_,_=reset_audit.material(request,report,request_sha256,report_sha256)
    audit_path=Path(reset_audit_report).resolve()
    require(source.file_sha(audit_path)==reset_audit_sha256,'reset audit SHA drift')
    audit=json.loads(audit_path.read_text())
    require(audit['status']=='diagnostic_completed' and audit['all_reproduction_checks'] is True and
            audit['first_error'] is None and audit['secondary_errors']==[],'actual reset-goal audit must have passed')
    require(audit['prepare']['request_sha256']==request_sha256 and audit['prepare']['report_sha256']==report_sha256 and
            Path(audit['prepare']['request']).resolve()==Path(request).resolve() and
            Path(audit['prepare']['report']).resolve()==Path(report).resolve(),'same audit origin')
    norm=normalization();parent=Path(report).resolve().parent
    bindings={str(Path(request).resolve()):request_sha256,str(Path(report).resolve()):report_sha256,
              str(audit_path):reset_audit_sha256,str(parent/'journal.jsonl'):saved['journal_sha256'],
              **req['plan']['source_closure'],norm['source']:norm['source_sha256']}
    for row in saved['slots']:bindings[str(parent/row['raw_source_file'])]=row['raw_source_sha256']
    for path in (Path(__file__).resolve(),Path(reset_audit.__file__).resolve(),Path(source.__file__).resolve(),
                 source.PROJECT/'tests'/('test_'+NAME+'.py')):bindings[str(path)]=source.file_sha(path)
    return dict(schema=NAME+'.prepare',inputs=dict(request=str(Path(request).resolve()),report=str(Path(report).resolve()),
                request_sha256=request_sha256,report_sha256=report_sha256,reset_audit_report=str(audit_path),reset_audit_sha256=reset_audit_sha256),
                bindings=bindings,normalization=norm,rgb_shape=req['plan']['rgb_shape'],counts=COUNTS,artifact_cap_bytes=CAP,minimum_free_bytes=FLOOR,
                observed_public_step_indices=[0,5,10],raw_commands=np.zeros((10,2),np.float32).tolist(),
                purpose='calibration_exploration',global_freshness='unknown',P1_admission=False,
                selections=[dict(candidate_id=row['slot']['candidate_id'],source=deepcopy(row['metadata']['source']),
                            selected_attempt=deepcopy(row['metadata']['selected_attempt']),parent_root_sha256=row['metadata']['root_sha256']) for row in saved['slots']],
                final_domain='original_predeclared_initial_domain_plus_native_nonterminal',replacement=False)


def verify(prepared):
    require(prepared['schema']==NAME+'.prepare','independent observed-history schema')
    for path,h in prepared['bindings'].items():require(source.file_sha(path)==h,'bound input/source drift: '+path)
    require(prepared==build(**prepared['inputs']),'prepared protocol drift')
    i=prepared['inputs']
    return reset_audit.material(i['request'],i['report'],i['request_sha256'],i['report_sha256'])


def bounded_write(directory,path,raw):
    directory=Path(directory)
    require(sum(p.stat().st_size for p in directory.iterdir() if p.is_file())+len(raw)<=CAP,'8MiB artifact cap')
    require(shutil.disk_usage(directory).free>=FLOOR+len(raw),'200MiB write floor')
    with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def prepare(output,**inputs):
    value=build(**inputs);path=Path(output)
    require(shutil.disk_usage(path.parent).free>=FLOOR+CAP,'200MiB plus 8MiB reserve')
    raw=(source.canonical(value)+'\n').encode()
    require(len(raw)<CAP,'prepare cap')
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return value


class Rejected(ValueError):
    """Fixed root failed bootstrap safety/domain: retain it; never replace."""


def execute(prepared,output,factory,*,authorization):
    require(authorization==AUTH,'explicit main physics authorization')
    req,saved,operations,packets=verify(prepared);plan=req['plan'];output=Path(output)
    require(shutil.disk_usage(output.parent).free>=FLOOR+CAP,'200MiB plus 8MiB reserve')
    output.mkdir(exist_ok=False)
    journal=source.DurableJournal(output/'journal.jsonl',CAP)
    result=dict(schema=NAME+'.sourcepack',prepare=deepcopy(prepared),status='running',
        rows=[dict(candidate_id=s['slot']['candidate_id'],status='not_started',file=None) for s in saved['slots']],
        counts={k:dict(attempted=0,completed=0) for k in COUNTS},first_error=None,secondary_errors=[],
        durable_journal_complete=False,purpose='calibration_exploration',global_freshness='unknown',
        P1_admission=False,neural_calls=0,downstream_future_labels_read=False,
        bootstrap_terminal_flags_read=True,history_semantics='observed_rgb_at_public_steps_0_5_10_v1')
    env=body=None;originals={};space_hooks=[];phase='setup';cid=None;row=None

    def retain(exc,where):
        error=dict(phase=where,type=type(exc).__name__,message=str(exc))
        if result['first_error'] is None:result['first_error']=error
        else:result['secondary_errors'].append(error)
        result['status']='failed'

    def emit(kind,**data):journal.append(kind,candidate_id=cid,phase=phase,**data)

    def count(kind,fn,summary=lambda v:None,**data):
        c=result['counts'][kind];require(c['attempted']<COUNTS[kind],'predispatch operation cap '+kind)
        emit('operation_started',resource=kind,**data);c['attempted']+=1
        value=fn();c['completed']+=1
        emit('operation_completed',resource=kind,receipt=summary(value),**data)
        return value

    def install_space():
        space=body.space
        if any(s is space for s,_,_ in space_hooks):return
        owned='step' in space.__dict__;original=space.step
        def physics(*args,**kwargs):return count('physics_step',lambda:original(*args,**kwargs))
        space_hooks.append((space,owned,original));space.step=physics

    def setter(value):
        install_space()
        return count('set_state',lambda:originals['_set_state'][1](value),lambda v:reset_audit.snapshot(body),requested=source.vector(value).tolist())

    def render():
        resource='goal_render' if phase.endswith('_reset') else 'explicit_render'
        before=reset_audit.snapshot(body)
        def render_summary(value):
            after=reset_audit.snapshot(body);require(before==after,'render changed state/hidden velocities')
            return dict(rgb_sha256=image_sha(source.rgb(value,plan)),state=after)
        return count(resource,originals['render'][1],render_summary,before=before)

    def safety(state,goal,full_domain=False):
        if not np.isfinite(state).all():raise Rejected('nonfinite observed source')
        terminal,_=body.eval_state(goal,state)
        if terminal:raise Rejected('native terminal observed source')
        low,high=plan['domain']['canvas_xy']
        if not ((state[:4]>=low)&(state[:4]<=high)).all():raise Rejected('observed source outside canvas')
        if full_domain:
            domain=source.initial_domain(env,state,goal,plan)
            row.setdefault('domain_checks',[]).append(domain);emit('domain_check',receipt=domain)
            if not domain['accepted']:raise Rejected('observed source violates frozen source domain')

    def archive(frames,indices,states,goal,goal_rgb,complete):
        if not frames:return
        # Failed roots must not relabel an unexecuted suffix as observations.
        raw=np.zeros((row.get('completed_public_steps',0),2),np.float32)
        state=source.vector(row['last_state']);root=source.root_hash(state,goal)
        meta=dict(schema=NAME+('.raw_observed_source' if complete else '.partial_observed_source'),
            prepare_id=source.digest(prepared),candidate_id=cid,parent_root_sha256=packets[cid]['metadata']['root_sha256'],
            parent_source=deepcopy(packets[cid]['metadata']['source']),selected_attempt=deepcopy(packets[cid]['metadata']['selected_attempt']),
            root_sha256=root,split='calibration',global_freshness='unknown',actual_observed_frames=len(frames),
            observed_public_step_indices=indices,completed_public_steps=row.get('completed_public_steps',0),
            history_semantics=result['history_semantics'],normalization=deepcopy(prepared['normalization']),
            downstream_future_labels_read=False,P1_admission=False)
        arrays=dict(metadata_json=np.asarray(source.canonical(meta)),state=state,goal=goal,rgb_history=np.stack(frames),
            goal_rgb=goal_rgb,raw_action_history=raw,normalized_action_history=normalized(raw[:len(raw)//5*5],prepared['normalization']),
            observed_states=np.asarray(states),observed_public_step_indices=np.asarray(indices,np.int64))
        bound=sum(a.nbytes for a in arrays.values())+65536
        require(sum(p.stat().st_size for p in output.iterdir() if p.is_file())+bound<=CAP,'8MiB array cap')
        require(shutil.disk_usage(output).free>=FLOOR+bound,'200MiB array floor')
        path=output/f'{"source" if complete else "partial"}_{cid}.npz'
        emit('source_write_started',file=path.name,metadata=meta)
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        row.update(file=path.name,file_sha256=source.file_sha(path),metadata=meta)
        if complete:load_observed(path,prepared)
        emit('source_write_completed',file=path.name,sha256=row['file_sha256'])

    def checkpoint(step,state,goal,goal_rgb):
        frame=source.rgb(env.render(),plan).copy()
        snap=reset_audit.snapshot(body)
        require(np.array_equal(state,snap['state7']),'render checkpoint observation drift')
        arrays=dict(rgb=frame,state=state,goal=goal,goal_rgb=goal_rgb,
                    snapshot_json=np.asarray(source.canonical(snap)),public_step=np.asarray(step,np.int64))
        bound=sum(a.nbytes for a in arrays.values())+65536
        require(sum(p.stat().st_size for p in output.iterdir() if p.is_file())+bound<=CAP,'8MiB checkpoint cap')
        require(shutil.disk_usage(output).free>=FLOOR+bound,'200MiB checkpoint floor')
        path=output/f'checkpoint_{cid}_{step}.npz'
        emit('checkpoint_write_started',file=path.name,public_step=step,snapshot=snap)
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        receipt=dict(file=path.name,sha256=source.file_sha(path),public_step=step,rgb_sha256=image_sha(frame))
        row.setdefault('checkpoints',[]).append(receipt);emit('checkpoint_write_completed',receipt=receipt)
        return frame

    try:
        emit('run_started',prepare=prepared)
        env=factory();body=env.unwrapped
        path=inspect.getsourcefile(type(body))
        require(path is not None and plan['source_closure'].get(str(Path(path).resolve()))==plan['environment_sha256'],'actual environment source binding')
        require(body.dt==.01 and body.control_hz==10,'sealed PushT step semantics: ten physics substeps')
        for name,hook in (('_set_state',setter),('render',render)):
            originals[name]=(name in body.__dict__,getattr(body,name));setattr(body,name,hook)
        for row,origin in zip(result['rows'],saved['slots']):
            cid=row['candidate_id'];slot=origin['slot'];packet=packets[cid];m=packet['metadata']
            i=m['source']['proposal_index'];j=m['selected_attempt']['attempt_index']
            seed=slot['reset_seed_span'][0]+i*plan['reset_attempts_per_proposal']+j
            row.update(status='running',completed_public_steps=0,random_reset_seed=seed,placement_reset_seed=plan['placement_reset_seed'])
            frames=[];indices=[];states=[];goal=goal_rgb=None
            try:
                phase='random_reset'
                initial=count('reset',lambda:env.reset(seed=seed),lambda v:source.reset_summary(v,plan),seed=seed)
                require(source.reset_summary(initial,plan)==operations[cid,i,j,'random_reset'],'original random reset mismatch')
                goal=source.vector(initial[1]['goal_state'])
                requested=source.sample_state(goal,list(body.block.shapes),slot,i,j,plan)
                row['placement_requested_state']=requested.tolist();emit('placement_request',state=requested.tolist(),goal=goal.tolist())
                phase='placement_reset'
                placed=count('reset',lambda:env.reset(seed=plan['placement_reset_seed'],options=dict(state=requested.copy(),goal_state=goal.copy())),lambda v:source.reset_summary(v,plan),seed=plan['placement_reset_seed'])
                require(source.reset_summary(placed,plan)==operations[cid,i,j,'placement_reset'],'original placement reset mismatch')
                state=source.vector(placed[0]['state']);goal_rgb=source.rgb(placed[1]['goal'],plan).copy()
                require(np.array_equal(state,packet['state']) and np.array_equal(goal,packet['goal']) and np.array_equal(goal_rgb,packet['goal_rgb']),'original selected root/actual goal RGB mismatch')
                row['last_state']=state.tolist();phase='observing'
                frame=checkpoint(0,state,goal,goal_rgb)
                require(np.array_equal(frame,packet['rgb_history'][0]),'original selected RGB mismatch')
                frames.append(frame);indices.append(0);states.append(state.copy())
                safety(state,goal,True)
                for step in range(1,11):
                    action=np.zeros(2,np.float32)
                    def step_receipt(v):
                        obs,_,terminated,truncated,info=v
                        return dict(state=source.vector(obs['state']).tolist(),terminated=bool(terminated),truncated=bool(truncated),goal=source.vector(info['goal_state']).tolist())
                    value=count('public_step',lambda:env.step(action.copy()),step_receipt,public_step=step,raw_action=action.tolist())
                    obs,_,terminated,truncated,info=value
                    state=source.vector(obs['state']);row['last_state']=state.tolist();row['completed_public_steps']=step
                    require(np.array_equal(state,body._get_obs()) and np.array_equal(info['goal_state'],goal) and np.array_equal(info['goal'],goal_rgb),'step observation/goal drift')
                    if step in (5,10):
                        frames.append(checkpoint(step,state,goal,goal_rgb));indices.append(step);states.append(state.copy())
                    if terminated or truncated:raise Rejected('public step terminal/truncated; no continuation or replacement')
                    safety(state,goal,step==10)
                archive(frames,indices,states,goal,goal_rgb,True)
                row['status']='completed'
            except Rejected as exc:
                retain(exc,'root_rejected');row.update(status='failed',failure_reason=str(exc))
                archive(frames,indices,states,goal,goal_rgb,False)
            except Exception as exc:
                retain(exc,'execution');row['status']='failed'
                try:archive(frames,indices,states,goal,goal_rgb,False)
                except Exception as secondary:retain(secondary,'partial_archive')
                raise
            finally:
                emit('root_finished',receipt=deepcopy(row))
        if all(r['status']=='completed' for r in result['rows']):
            require(all(result['counts'][k]==dict(attempted=n,completed=n) for k,n in COUNTS.items()),'exact completed resource census')
            result['status']='observed_calibration_source_completed'
    except Exception as exc:
        if result['first_error'] is None:retain(exc,'execution')
        elif result['first_error']['message']!=str(exc):retain(exc,'execution_secondary')
    finally:
        for space,owned,original in reversed(space_hooks):
            try:
                if owned:space.step=original
                else:delattr(space,'step')
            except Exception as exc:retain(exc,'restore_space')
        if body is not None:
            for name,(owned,original) in originals.items():
                try:
                    if owned:setattr(body,name,original)
                    else:delattr(body,name)
                except Exception as exc:retain(exc,'restore_'+name)
        if env is not None:
            try:env.close()
            except Exception as exc:retain(exc,'close')
        for path,h in prepared['bindings'].items():
            try:require(source.file_sha(path)==h,'final bound source/input drift: '+path)
            except Exception as exc:retain(exc,'final_binding')
        result['internal_physics']=dict(actual_hook_counts=deepcopy(result['counts']['physics_step']),
            completed_resets_imply=2*result['counts']['reset']['completed'],
            completed_public_steps_imply=10*result['counts']['public_step']['completed'],
            note='hook counts include partial failed reset/step work, not just completed-call inference')
        try:emit('run_finished',status=result['status'],counts=result['counts'],first_error=result['first_error']);result['durable_journal_complete']=True
        except Exception as exc:retain(exc,'final_journal')
        try:journal.close()
        except Exception as exc:retain(exc,'journal_close');result['durable_journal_complete']=False
        try:result['journal_sha256']=source.file_sha(output/'journal.jsonl')
        except Exception as exc:retain(exc,'journal_hash')
        try:bounded_write(output,output/'sourcepack.json',(source.canonical(result)+'\n').encode())
        except Exception as exc:retain(exc,'report_write')
    return result


def load_observed(path,prepared):
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files)==set('metadata_json state goal rgb_history goal_rgb raw_action_history normalized_action_history observed_states observed_public_step_indices'.split()),'independent observed array schema')
        data={k:archive[k].copy() for k in archive.files}
    meta=json.loads(str(data.pop('metadata_json')));data['metadata']=meta
    source.fields(meta,'schema prepare_id candidate_id parent_root_sha256 parent_source selected_attempt root_sha256 split global_freshness actual_observed_frames observed_public_step_indices completed_public_steps history_semantics normalization downstream_future_labels_read P1_admission','observed source metadata')
    require(meta['schema']==NAME+'.raw_observed_source' and meta['prepare_id']==source.digest(prepared),'bound complete observed source, not partial/legacy padding')
    selected=next((s for s in prepared['selections'] if s['candidate_id']==meta['candidate_id']),None)
    require(selected is not None and meta['parent_root_sha256']==selected['parent_root_sha256'] and
            meta['parent_source']==selected['source'] and meta['selected_attempt']==selected['selected_attempt'],'frozen parent source/attempt')
    require(meta['normalization']==prepared['normalization'] and meta['downstream_future_labels_read'] is False and
            meta['history_semantics']=='observed_rgb_at_public_steps_0_5_10_v1','frozen normalizer and observed semantics')
    require(meta['actual_observed_frames']==3 and meta['observed_public_step_indices']==[0,5,10] and meta['completed_public_steps']==10,'actual observed chronology')
    require(np.array_equal(data['observed_public_step_indices'],[0,5,10]) and data['observed_states'].shape==(3,7) and
            np.isfinite(data['observed_states']).all(),'three finite observed states')
    require(np.array_equal(data['state'],data['observed_states'][-1]) and meta['root_sha256']==source.root_hash(data['state'],data['goal']),'root identity is final observed state plus original goal')
    require(data['rgb_history'].dtype==np.uint8 and data['rgb_history'].shape==(3,*prepared['rgb_shape']),'actual RGB3')
    require(data['goal_rgb'].dtype==np.uint8 and data['goal_rgb'].shape==data['rgb_history'].shape[1:],'same-shape goal RGB')
    raw=data['raw_action_history']
    require(raw.shape==(10,2) and raw.dtype==np.float32 and not np.count_nonzero(raw),'ten executed raw zero actions')
    require(data['normalized_action_history'].dtype==np.float32 and np.array_equal(data['normalized_action_history'],normalized(raw,prepared['normalization'])),'two between-state normalized packets, not zeros by assumption')
    require(meta['split']=='calibration' and meta['global_freshness']=='unknown' and meta['P1_admission'] is False,'no freshness/admission promotion')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',type=Path);group.add_argument('--run-main',type=Path)
    for name in ('request','report','reset-audit-report'):parser.add_argument('--'+name,type=Path)
    for name in ('request-sha256','report-sha256','reset-audit-sha256','prepare-sha256'):parser.add_argument('--'+name)
    parser.add_argument('--output',type=Path);parser.add_argument('--authorize-physics',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        value=prepare(args.prepare,**{k:getattr(args,k) for k in ('request','report','request_sha256','report_sha256','reset_audit_report','reset_audit_sha256')})
        print(source.canonical(dict(prepared=str(args.prepare.resolve()),sha256=source.file_sha(args.prepare),counts=value['counts'])))
    else:
        require(args.authorize_physics and args.output is not None,'main explicit physics authorization/output')
        require(source.file_sha(args.run_main)==args.prepare_sha256,'external prepare file SHA')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401 -- main-only registration
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224)
        value=execute(json.loads(args.run_main.read_text()),args.output,factory,authorization=AUTH)
        print(source.canonical(value))
        if value['status']!='observed_calibration_source_completed':raise SystemExit(2)


if __name__=='__main__':main()
