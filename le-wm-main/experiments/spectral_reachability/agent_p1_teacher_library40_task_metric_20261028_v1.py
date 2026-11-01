"""Fixed40 saved latent/task metrics; returned wrapper success is not object coverage."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import itertools
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_teacher_library40_rank_error_20261028_v1 as rank
from experiments.spectral_reachability import agent_p1_teacher_cross_library_physical_20261028_v1 as physics

NAME='agent_p1_teacher_library40_task_metric_20261028_v1'
ROOT,rgb,teacher=rank.ROOT,rank.rgb,rank.rgb.teacher
require,sha,digest,canonical=rank.require,rank.sha,rank.digest,rank.canonical
RANK=ROOT/'results/agent_p1_teacher_library40_rank_error_20261028_v1_analysis_MAIN01.json'
RANK_SHA='e23e60588fa9766b690e605022e089656c4c80ad185853773e23a942b5e0228f'
METRICS=dict(task_success=-1,native_distance=1,all_xy_error=1,agent_xy_error=1,object_xy_error=1,
             object_wrapped_angle_error=1,agent_velocity_error=1,native_coverage=-1,object_only_coverage=-1)


def vector(x,n):
    a=np.asarray(x,np.float64);require(a.shape==(n,) and np.isfinite(a).all(),'complete finite original vector');return a


def task_metrics(state,goal,block_pose,outline,returned_success,distance,reward):
    s,g,b,o=vector(state,7),vector(goal,7),vector(block_pose,3),vector(outline,3)
    require(np.array_equal(b[:2],s[2:4]) and b[2]%(2*np.pi)==s[4],'full returned unwrapped block pose vs wrapped state')
    d=s-g;raw_angle=abs(float(d[4]));native_angle=min(raw_angle,2*np.pi-raw_angle)
    allxy=float(np.linalg.norm(d[:4]));nd=float(np.linalg.norm(d))
    success=bool(allxy<20 and native_angle<np.pi/9)
    require(type(returned_success) is bool and success==returned_success and nd==distance and reward==-nd,
            'original wrapper eval_state/reward, no object-only substitute or repaired angle formula')
    angle=float(b[2]-g[4]);wrapped=float(np.arctan2(np.sin(angle),np.cos(angle)))
    oa=float(b[2]-o[2]);ow=float(np.arctan2(np.sin(oa),np.cos(oa)))
    return dict(task_success=success,native_distance=nd,all_xy_error=allxy,
        agent_xy_error=float(np.linalg.norm(d[:2])),object_xy_error=float(np.linalg.norm(d[2:4])),
        object_wrapped_angle_error=abs(wrapped),agent_velocity_error=float(np.linalg.norm(d[5:])),
        native_angle_term=native_angle,native_angle_term_negative=native_angle<0,
        native_coverage=None,object_only_coverage=None,
        coverage_status='unknown_not_returned_no_coverage_formula_in_bound_wrapper_step',
        returned_object_pose=b.tolist(),task_goal_object_pose=g[2:5].tolist(),
        task_object_pose_error=dict(xy_delta=(b[:2]-g[2:4]).tolist(),xy_l2=float(np.linalg.norm(b[:2]-g[2:4])),
            raw_unwrapped_angle_delta=angle,wrapped_angle_delta=wrapped,absolute_wrapped_angle=abs(wrapped)),
        renderer_outline_pose=o.tolist(),outline_is_taskgoal=False,
        object_vs_outline_diagnostic=dict(xy_delta=(b[:2]-o[:2]).tolist(),xy_l2=float(np.linalg.norm(b[:2]-o[:2])),
            wrapped_angle_delta=ow,absolute_wrapped_angle=abs(ow)),
        wrapper_success_contract='norm(agentXY+objectXY delta)<20 AND min(abs(angle delta),2pi-abs(angle delta))<pi/9; velocity excluded',
        object_only_coverage_is_wrapper_success=False)


def classify(a,b):
    if a is None or b is None:return dict(available=False,classification='unknown',reversal=None,delta_a=a,delta_b=b)
    sa,sb=rank.sign(a),rank.sign(b)
    label='both_tie' if sa==sb==0 else 'left_tie_only' if sa==0 else 'right_tie_only' if sb==0 else 'concordant' if sa==sb else 'discordant'
    return dict(available=True,classification=label,reversal=sa*sb<0,delta_a=a,delta_b=b,sign_a=sa,sign_b=sb)


def compare(a,b,kind):
    rank.matching(a,b,kind)
    require(a['task_id']==b['task_id'] and a['physical_goal']==b['physical_goal'] and a['outline_pose']==b['outline_pose'],
            'strict same task/native goal/renderer contract; no cross condition/root matching')
    dp=a['predicted_Q']-b['predicted_Q']
    da=None if a['actual_Q'] is None or b['actual_Q'] is None else a['actual_Q']-b['actual_Q']
    physical={}
    for k,orientation in METRICS.items():
        va,vb=a['physical_metrics'][k],b['physical_metrics'][k]
        raw=None if va is None or vb is None else float(va)-float(vb)
        oriented=None if raw is None else orientation*raw
        physical[k]=dict(raw_delta=raw,lower_is_better_oriented_delta=oriented,orientation=orientation,
            actual_latent_to_physical=classify(da,oriented))
    return dict(i=a['index'],j=b['index'],kind=kind,predicted_to_actual_latent=classify(dp,da),
        actual_latent_to_physical_by_metric=physical,causal_interpretation=False,C2_failure_inferred=False)


def counts(items):
    return dict(fixed=len(items),available=sum(x['available'] for x in items),unknown=sum(not x['available'] for x in items),
        discordant=sum(x['reversal'] is True for x in items),ties=sum(x.get('classification') in ('both_tie','left_tie_only','right_tie_only') for x in items))


def assemble(rows):
    require(len(rows)==40 and [r['index'] for r in rows]==list(range(40)),'full40 original order')
    # Same frozen roster/rank completeness rule; no new choice of matching strata.
    rank.assemble(deepcopy(rows))
    groups=[];pairs=[]
    for anchor in range(8):
        members=rows[anchor*5:anchor*5+5]
        ps=[compare(a,b,'within_role_library') for a,b in itertools.combinations(members,2)]
        groups.append(dict(root_slot=members[0]['root_slot'],condition=members[0]['condition'],role=members[0]['role'],pairs=ps));pairs+=ps
    zs=[compare(rows[t*10+w],rows[t*10+5+w],'same_word_ZS') for t in range(4) for w in range(5)]
    def summary(ps):
        return dict(predicted_to_actual_latent=counts([p['predicted_to_actual_latent'] for p in ps]),
            actual_latent_to_physical_by_metric={k:counts([p['actual_latent_to_physical_by_metric'][k]['actual_latent_to_physical'] for p in ps]) for k in METRICS})
    return dict(rows=rows,groups=groups,within_role_summary=summary(pairs),same_word_ZS=zs,same_word_ZS_summary=summary(zs),
                fixed_slots=40,available_actual_Q=39,missing_actual_Q=[12],returned_physical_metric_slots=40)


def build():
    a=rgb.read(RANK,RANK_SHA);require(a==rank.build(),'entire original rank/native arrays strict rebuild')
    path=ROOT/rgb.INPUTS['physical'][0];h=rgb.INPUTS['physical'][1]
    r=rgb.read(path,h);receipt_path=ROOT/rgb.INPUTS['physical_receipt'][0]
    receipt=rgb.read(receipt_path,rgb.INPUTS['physical_receipt'][1])
    require(receipt==physics.consume(path.parent,h),'original full partial40 physical consumer, not only hashes')
    bindings={**a['bindings'],str(RANK):RANK_SHA,str(path):h,str(receipt_path):rgb.INPUTS['physical_receipt'][1]}
    prep_path=r['execution_binding']['prepared'];prep_h=r['execution_binding']['prepare_sha256'];prep=rgb.read(prep_path,prep_h)
    bindings.update(prep['bindings']);bindings[prep_path]=prep_h
    env=ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
    require(str(env) in bindings and sha(env)==bindings[str(env)],'original bound wrapper implementation')
    for name,h in r['artifact_sha256'].items():bindings[str(path.parent/name)]=h
    events=physics.read_events(path.parent/r['journal']['file'],r['journal_tail'])
    rows=[];array_receipts={}
    for latent,physical in zip(a['rows'],r['rows'],strict=True):
        require(all(latent[k]==physical[k] for k in rgb.IDENTITY),'exact same40 task/role/word/plan/job joins')
        returns=[e for e in events if e['kind']=='raw_return' and e['job_digest']==latent['job_digest']]
        require(len(returns)==20 and [(e['phase'],e['step']) for e in returns[-5:]]==[('suffix',i) for i in range(1,6)],'full actual20 returns even partialRGB12')
        e=returns[-1];last=physical['last_return'];require(all(last[k]==e[k] for k in ('phase','step','state','native_success','native_distance','reward','terminated','truncated')) and
            e['state']==physical['last_state'] and not e['terminated'] and not e['truncated'],'actual final returned state, never earlier checkpoint')
        info=e['info'];ip=teacher.flat(path.parent,info['file']);require(sha(ip)==info['sha256'],'actual native info NPZ SHA')
        tree=info['tree']
        require('coverage' not in tree and 'final_coverage' not in tree,'unexpected coverage schema requires explicit authority, no silent discard')
        with np.load(ip,allow_pickle=False) as z:
            def array(key,n):
                meta=tree[key];x=z[meta['array']]
                require(x.dtype.str==meta['dtype'] and list(x.shape)==meta['shape'] and x.shape==(n,) and np.isfinite(x).all(),'native full info array '+key)
                array_receipts[str(ip)+':'+meta['array']]=dict(file_sha256=info['sha256'],array_sha256=hashlib.sha256(x.tobytes()).hexdigest(),dtype=x.dtype.str,shape=list(x.shape))
                return x.copy()
            goal=array('goal_state',7);outline=array('goal_pose',3);block=array('block_pose',3)
            pos=array('pos_agent',2);vel=array('vel_agent',2)
        state=vector(e['state'],7)
        require(np.array_equal(goal,physical['goal']) and np.array_equal(outline,physical['goal_pose']) and
            np.array_equal(pos,state[:2]) and np.array_equal(vel,state[5:]),'actual returned native goal/outline/agent state bound')
        require(latent['goal_binding']==dict(state=goal.tolist(),pose=outline.tolist(),pixel_sha256=physical['goal_pixel_sha256']), 'original rendered latent goal vs physical task contract')
        metrics=task_metrics(e['state'],goal,block,outline,e['native_success'],e['native_distance'],e['reward'])
        item=deepcopy(latent);item.update(physical_goal=goal.tolist(),outline_pose=outline.tolist(),physical_metrics=metrics,
            physical_source_status=physical['status'],actual_return_absolute_step=20,actual_return_event_sha256=e['event_sha256'],
            native_info_receipt=deepcopy(info),returned_state=state.tolist(),returned_reward=e['reward'],
            latent_missing_does_not_imply_physical_state_missing=True)
        require(metrics['task_success']==latent['native_success'],'native task success unchanged')
        rows.append(item)
    own=[Path(__file__).resolve(),ROOT/f'tests/test_{NAME}.py',ROOT/f'innerdocs/{NAME}.md']
    for f in own:bindings[str(f)]=sha(f)
    rgb.verify_bindings(bindings)
    return dict(schema=NAME+'.analysis',status='fixed40_saved_task_metric_alignment_not_admitted',bindings=bindings,
        source_files={str(f):bindings[str(f)] for f in own},native_info_array_receipts=array_receipts,
        coverage_authority=dict(wrapper_source=str(env),wrapper_source_sha256=bindings[str(env)],returned_coverage_fields=False,
            recomputed_object_coverage=False,reason='bound wrapper eval_state uses whole agent+object XY and angle, native step/info do not return coverage; no substitute geometry'),
        known_roots=2,independent_roots=0,P1_admission=False,C2_failure_inferred=False,physical_benefit_claim=False,
        winner_selection=False,goal_or_Q_changed=False,new_neural_calls=0,new_physics_calls=0,**assemble(rows))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'));ap.add_argument('--output');ap.add_argument('--analysis');ap.add_argument('--analysis-sha');args=ap.parse_args()
    a=build()
    if args.command=='verify':
        require(rgb.read(args.analysis,args.analysis_sha,16*1024**2)==a,'whole saved metric report rebuild');out=dict(verified=True,analysis_sha256=args.analysis_sha)
    elif args.output:
        p=Path(args.output).absolute();require(p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix=='.json','exclusive own new result')
        raw=canonical(a);require(len(raw)<16*1024**2,'bounded own report')
        with p.open('xb') as f:require(f.write(raw)==len(raw),'whole saved report write');f.flush();os.fsync(f.fileno())
        rgb.sync_dir(p.parent);out=dict(output=str(p),analysis_sha256=sha(p),within_role_summary=a['within_role_summary'])
    else:out=dict(status=a['status'],within_role_summary=a['within_role_summary'])
    print(canonical(out).decode())


if __name__=='__main__':main()
