"""Saved-only six-slot predicted tracking/real-task-goal distances; no native claim."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import shutil

import numpy as np
from experiments.spectral_reachability import agent_p2_six_target_controller_cuda_20261029_v1 as runtime

TAG='agent_p2_six_target_prediction_comparison_20261029_v1'
ROOT=runtime.ROOT
need,sha,read,encode=runtime.need,runtime.sha,runtime.read,runtime.encode
CP=(0,5,10,15)
CAP,FLOOR=16*1024**2,200*1024**2
ACTUAL={
 'manifest':('agent_p2_six_target_controller_cuda_20261029_v1_prepare_MAIN01.json','cccfadc20911dc1be008a047c57bca7400272f6602deb435fb0341a0c8bbc8f9'),
 'report':('agent_p2_six_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json','b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934'),
 'receipt':('agent_p2_six_target_controller_cuda_20261029_v1_consume_MAIN01.json','592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103')}
PINS={Path(runtime.__file__):'075a5c09b8c57051b3e7bc3eb3acac7b824f0c1a983151ec3f8ee10c1c813a5c',
 ROOT/'tests/test_agent_p2_six_target_controller_cuda_20261029_v1.py':'03067ecd140728f5a727948a326637d85b8603fa3284c218a23c9560f8ae0e4e',
 ROOT/'innerdocs/agent_p2_six_target_controller_cuda_20261029_v1.md':'cc59f2b8ef9d36329cad143506aececc9a41831f6bb0f1205c8a7203417fdf23'}


def actual_refs():return {k:dict(path=str(ROOT/'results'/p),sha256=h) for k,(p,h) in ACTUAL.items()}


def closure():
    out=runtime.closure()
    for p,h in PINS.items():need(sha(p)==h,'frozen six runner triple');runtime.merge(out,{str(p):h})
    for folder,name in (('experiments/spectral_reachability',TAG+'.py'),('tests','test_'+TAG+'.py'),('innerdocs',TAG+'.md')):
        p=ROOT/folder/name;runtime.merge(out,{str(p):sha(p)})
    return out


def mse64(z,g):
    need(type(z) is np.ndarray and type(g) is np.ndarray and z.shape==g.shape==(192,) and
         z.dtype==g.dtype==np.float32 and np.isfinite(z).all() and np.isfinite(g).all(),'original finite f32 latent192')
    delta=np.subtract(z.astype(np.float64),g.astype(np.float64),dtype=np.float64)
    squares=np.multiply(delta,delta,dtype=np.float64)
    result=math.fsum(float(x) for x in squares)/192
    need(math.isfinite(result),'finite independent f64 identity192 meanMSE');return result


def exact(a,b,message):need(runtime.delivery.same(a,b),message)


def check_complete(m,r,c,refs):
    need(m['schema']==runtime.TAG+'.prepare' and r['schema']==runtime.TAG+'.report' and
         c['schema']==runtime.TAG+'.receipt','exact six schemas')
    need(m['protocol']==r['protocol']==runtime.protocol() and r['completed'] is True and r['errors']==[], 'complete six runtime, never score partial')
    need(r['manifest_sha256']==c['manifest_sha256']==refs['manifest']['sha256'] and
         Path(r['manifest']).resolve()==Path(refs['manifest']['path']).resolve() and c['report_sha256']==refs['report']['sha256'],'exact external report/manifest/receipt join')
    need(c['status']=='all6_new_target_controller_saved_verified' and c['denominator']==dict(total=6,returned=6,missing=0) and
         c['predict_API']==72 and c['predictor_samples']==324 and len(c['local_checks'])==6,'complete actual six receipt denominator')
    need(r['model_before']==r['model_after']==m['expected_state'] and r['sources_after']==m['sources'],'original model/history/source guards')
    need(len(m['slots'])==len(r['rows'])==6 and all(x['status']=='returned' for x in r['rows']),'missing slot has no invented score')
    for i,(slot,row,check) in enumerate(zip(m['slots'],r['rows'],c['local_checks'],strict=True)):
        identity=(i,i//3,'off',runtime.ARMS[i%3])
        need((slot['slot'],slot['root'],slot['condition'],slot['arm'])==identity and
             (row['slot'],row['root'],row['condition'],row['arm'])==identity,'fixed six slot identities')
        need(check==dict(slot=i,root=i//3,arm=runtime.ARMS[i%3],local_check=row['local_check']),'complete per-slot receipt exact join')


def resolve_goals(m,ap,source_report,get_source):
    """Only original sealed task goal, never a prior/program target substitution."""
    need(ap['schema']==runtime.arbitrary.NAME+'.prepare' and len(ap['slots'])==2 and
         ap['parent_delivery']==m['authorities']['delivery'] and
         ap['inherited_expected_state_sha256']==runtime.arbitrary.digest(m['expected_state']),'original task goal coordinate/model authority')
    values=runtime.select_payloads(m);goals=[]
    for root,s in enumerate(ap['slots']):
        slot=m['slots'][3*root]
        need((s['root'],s['condition'])==(root,'off') and all(s[k]==slot[k] for k in ('task_id','task_sha256','blind_sha256')),'original known-root source identity')
        need(s['goal_pointer']==f'/rows/{root}/pixels/3/paths/hi/output','sealed real goal image pointer')
        original=source_report['rows'][root]
        need(original['root']==root and original['status']=='returned' and len(original['pixels'])==4 and
             original['pixels'][3]['image']==3,'original actual goal source index')
        names=s['source_arrays'];need(names==dict(z=original['routes']['hi_whole']['z'],a=original['routes']['hi_whole']['a'],
            goal=original['pixels'][3]['paths']['hi']['output']),'exact original history/goal array pointers')
        z=runtime.arbitrary.f32(s['z'],(1,3,192));a=runtime.arbitrary.f32(s['a'],(1,2,10));goal=runtime.arbitrary.f32(s['goal'],(192,))
        exact(z,get_source(names['z']),'source full z history');exact(a,get_source(names['a']),'source full between-action history')
        actual=get_source(names['goal']);need(actual.shape==(1,192) and actual.dtype==np.float32,'original encoded goal array')
        exact(goal,actual[0],'original task goal bytes, not tracking target')
        need(s['input_sha256']==dict(z=runtime.core.digest(z.tobytes()),a=runtime.core.digest(a.tobytes()),goal=runtime.core.digest(goal.tobytes())),'complete original context/goal SHA')
        task=s['task'];need(task['context']==s['context'] and task['goal']==s['goal'] and task['root_slot']==root and task['condition']=='off','original task goal/context recipe')
        ctx=task['context'];need(ctx['layout']=='states_with_between_state_actions_v1' and ctx['context_length']==3 and
            ctx['latent_dim']==192 and ctx['action_dim']==10,'original history coordinate contract')
        exact(np.asarray(ctx['latent_history'],np.float32),z[0],'complete causal task z');exact(np.asarray(ctx['action_history'],np.float32),a[0],'complete causal task a')
        for i in range(3*root,3*root+3):
            exact(values[i]['z'],z,'all arms exact same full original history');exact(values[i]['a'],a,'all arms exact same original actions')
        goals.append(dict(root=root,value=goal,source=dict(prepare_pointer=f'/slots/{root}/goal',goal_pointer=s['goal_pointer'],
            source_arrays=names,input_sha256=s['input_sha256'],task_id=s['task_id'],task_sha256=s['task_sha256'],blind_sha256=s['blind_sha256'],
            task_goal_report=s['task']['coordinate_refs']['task_goal'],goal_sha256=runtime.core.digest(goal.tobytes()))))
    return goals


def checkpoint_vectors(tr,arrays,values):
    need(tr['completed'] is True and tr['errors']==[] and len(tr['replays'])==2 and
         tr['selected_mean_replay_exact'] is True,'whole/segmented prerequisite')
    w,s=tr['replays'];need((w['phase'],s['phase'])==('mean_whole','mean_segmented') and w['returned'] and s['returned'],'both paid actual mean replays')
    for k,v in values.items():exact(arrays[tr['inputs'][k]],v,'exact actual input history/target')
    mean=arrays[tr['selected']['mean']];need(mean.shape==(1,3,10) and mean.dtype==np.float32,'actual mean all fifteen grouped actions')
    for replay in (w,s):exact(arrays[replay['future']],mean[:,None],'same actual mean, not elite/witness')
    pw,ps=(arrays[x['rollout']['prediction']] for x in (w,s))
    need(pw.shape==ps.shape==(1,1,3,192) and pw.dtype==ps.dtype==np.float32,'all three predicted checkpoints')
    exact(pw,ps,'whole/segment all prediction bytes')
    for replay in (w,s):
        checkpoints=replay['rollout']['checkpoints'];need(len(checkpoints)==3,'complete checkpoint denominator')
        for j,c in enumerate(checkpoints):
            zz,aa=(arrays[c[k]] for k in ('z','a'))
            need(zz.shape==(1,4+j,192) and aa.shape==(1,3+j,10),'full retained causal histories')
            expected_z=np.concatenate((values['z'],pw[0,:,:j+1,:]),axis=1)
            expected_a=np.concatenate((values['a'],mean[:,:j+1]),axis=1)
            exact(zz,expected_z,'every full z checkpoint');exact(aa,expected_a,'every full a checkpoint')
        for k in ('z','a'):exact(arrays[replay['rollout']['final'][k]],arrays[checkpoints[-1][k]],'entire final context')
    exact(arrays[w['cost']],arrays[s['cost']],'both actual replay scores exact')
    return [(values['z'][0,-1].copy(),dict(kind='observed_initial_context',array=tr['inputs']['z'],index=[0,2])),
        *[(pw[0,0,j].copy(),dict(kind='actual_mean_predicted',array=w['rollout']['prediction'],segmented_array=s['rollout']['prediction'],index=[0,0,j])) for j in range(3)]]


def panel(m,r,goals,get_trace):
    values=runtime.select_payloads(m);rows=[]
    need(len(goals)==2 and [g['root'] for g in goals]==[0,1],'two fixed original goals')
    for i in range(6):
        need(r['rows'][i]['status']=='returned','missing/partial no scores')
        tr,arrays=get_trace(i);vectors=checkpoint_vectors(tr,arrays,values[i]);goal=goals[i//3];target=values[i]['target'][0]
        tracking0=mse64(vectors[0][0],target);goal0=mse64(vectors[0][0],goal['value'])
        source=m['slots'][i]['source']
        source_time=(source['row']['origin']['physical_time'] if i%3!=2 else source['plan']['source_time'])
        for cp,(z,pointer) in zip(CP,vectors,strict=True):
            qt,qg=mse64(z,target),mse64(z,goal['value'])
            rows.append(dict(slot=i,root=i//3,arm=runtime.ARMS[i%3],condition='off',cp_public_step=cp,
                state_kind='observed_context' if cp==0 else 'model_prediction_not_native',latent_sha256=runtime.core.digest(z.tobytes()),latent_source=pointer,
                tracking_target_sha256=runtime.core.digest(target.tobytes()),original_task_goal_sha256=goal['source']['goal_sha256'],
                tracking_target_mse64=qt,task_goal_mse64=qg,tracking_reduction_from_cp0=tracking0-qt,task_goal_reduction_from_cp0=goal0-qg,
                source_time=source_time,controller_future_raw_actions=False,native_metrics=None,scientific_admission=False))
    pairs=[]
    for root in (0,1):
        for left,right in itertools.combinations(range(3*root,3*root+3),2):
            for j,cp in enumerate(CP):
                a,b=rows[4*left+j],rows[4*right+j]
                need(a['original_task_goal_sha256']==b['original_task_goal_sha256'],'common task goal only for pair comparison')
                pairs.append(dict(root=root,left_slot=left,right_slot=right,cp_public_step=cp,
                    right_minus_left_task_goal_mse64=b['task_goal_mse64']-a['task_goal_mse64'],
                    cross_target_tracking_loss_difference=None,prediction_only=True))
    return dict(denominator=dict(roots=2,slots=6,checkpoints_per_slot=4,total_rows=24,pairs=24),rows=rows,same_task_goal_pairs=pairs)


def build(refs):
    need(set(refs)=={'manifest','report','receipt'} and all(set(v)=={'path','sha256'} for v in refs.values()),'explicit actual three authority refs')
    need(refs==actual_refs(),'fixed actual MAIN01 complete authority only')
    m,r,c=(read(refs[k]['path'],refs[k]['sha256']) for k in ('manifest','report','receipt'))
    check_complete(m,r,c,refs)
    need(m==runtime.build(m['upstream_authentication']),'whole original six manifest reconstructed, no new authentication writer')
    folder=Path(refs['report']['path']).parent
    completion_ref=dict(path=str(folder/'completion.json'),sha256=c['completion_sha256'])
    done=read(completion_ref['path'],completion_ref['sha256'])
    need(done['schema']==runtime.TAG+'.completion' and done['complete'] is True and done['report_sha256']==refs['report']['sha256'] and
        r['whole_wall_seconds_before_report']<=done['whole_wall_through_report_fsync_hash']<=runtime.WALL and
        done['boundary']=='Store start before verify through report fsync/hash; completion acknowledgement excluded','actual inclusive complete job')
    runtime.check_authentication(c['saved_consume_authentication'],m['full_receipts']['arbitrary'])
    rebuilt=runtime.analyze_saved(m,r,folder)
    rebuilt.update(report_sha256=refs['report']['sha256'],manifest_sha256=refs['manifest']['sha256'],completion_sha256=completion_ref['sha256'],
        saved_consume_authentication=c['saved_consume_authentication'],upstream_arbitrary_receipt=m['authorities']['arbitrary_receipt'])
    need(rebuilt==c,'entire six strict receipt reconstructed, not just status')
    sources=dict(m['sources']);runtime.merge(sources,closure());runtime.merge(sources,{v['path']:v['sha256'] for v in refs.values()})
    runtime.merge(sources,{completion_ref['path']:completion_ref['sha256']})
    apref=m['authorities']['arbitrary_prepare'];ap=read(apref['path'],apref['sha256'])
    need(m['sources'].get(apref['path'])==apref['sha256'],'sealed arbitrary prepare goal source')
    goal_refs=[s['task']['coordinate_refs']['task_goal'] for s in ap['slots']]
    need(len(goal_refs)==2 and goal_refs[0]==goal_refs[1] and m['sources'].get(goal_refs[0]['path'])==goal_refs[0]['sha256'],'both roots exact same original goal authority report')
    pr=read(goal_refs[0]['path'],goal_refs[0]['sha256']);pf=Path(goal_refs[0]['path']).parent
    def get_source(name):
        need(Path(name).name==name,'flat original source array name');desc=pr['arrays'][name];p=pf/name
        need(m['sources'].get(str(p))==desc['file_sha256'] and ap['bindings'].get(str(p))==desc['file_sha256'],'original goal/history NPY authority')
        return runtime.tracker.parity.load_array(pf,name,desc)
    goals=resolve_goals(m,ap,pr,get_source)
    def get_trace(i):
        row=r['rows'][i];td=row['trace'];runtime.merge(sources,{str(folder/td['file']):td['file_sha256']})
        tr=read(folder/td['file'],td['file_sha256']);arrays={}
        for key,d in row['arrays'].items():
            arrays[key]=runtime.load_array(folder,d);runtime.merge(sources,{str(folder/d['file']):d['file_sha256']})
        return tr,arrays
    out=panel(m,r,goals,get_trace)
    need(all(sha(p)==h for p,h in sources.items()),'all input/source/model/array bytes unchanged')
    return dict(schema=TAG+'.comparison',authorities=refs,completion=completion_ref,full_strict_receipt=c,goal_prepare=apref,
        goal_sources=[g['source'] for g in goals],sources=sources,**out,
        arithmetic='f32 inputs -> f64 subtract/square -> math.fsum/192; identity, no fit',
        resource_accounting27=c['accounting27'],evidence_scope='saved model predictions only; imported strict authenticates source, not independent gradients',
        known_roots_only=True,fresh=False,prior_physical_time=None,equal_generation_budget=False,time_matched=False,
        native_comparison=False,raw_deployment_authorized=False,scientific_admission=False,
        new_calls=dict(encoder=0,predict=0,AD=0,physics=0),winner=None)


def output_path(p):
    p=Path(p);need(p.parent.resolve()==ROOT/'results' and p.name.startswith(TAG+'_') and not p.is_symlink(),'exclusive own results child');return p


def write(output,value):
    p=output_path(output);b=encode(value);need(len(b)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'16MiB/floor200MiB')
    runtime.tracker.durable(p,b);return dict(path=str(p.resolve()),sha256=sha(p))


def verify(path,h):
    v=read(path,h);need(v==build(v['authorities']),'entire saved comparison reconstructed');return dict(verified=True,new_neural=0,physics=0)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','verify'))
    for key in ('manifest','report','receipt','summary','output'):p.add_argument('--'+key,type=Path)
    for key in ('manifest','report','receipt','summary'):p.add_argument('--'+key+'-sha256')
    a=p.parse_args(argv)
    if a.command=='verify':out=verify(a.summary,a.summary_sha256)
    else:
        refs={k:dict(path=str(getattr(a,k).resolve()),sha256=getattr(a,k+'_sha256')) for k in ('manifest','report','receipt')}
        out=write(a.output,build(refs))
    print(encode(out).decode())


if __name__=='__main__':main()
