"""Fixed native task metrics for all four saved words, not C2 admission.

No simulator/model execution. Existing observations are retrospective calibration;
this newly frozen metric specification cannot preregister already seen outcomes.
"""
from __future__ import annotations
import argparse
import importlib.util
import math
from pathlib import Path

TAG='agent_p5_task_benefit_prerequisite_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
OWNER='agent_p5_contact_word_response_analysis_20261028_v1'
OWNER_SHA='0fd96aa3487242ee87e947c5172286c7a49bdf0ce70a9838010a72439dca480f'
ANALYSIS_SHA='f38b62550c347b66e5533885694a53db22ae1b58798d18de3dc11f6f1c3edf2c'
ENV='external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py'
ENV_SHA='6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723'
WORDS=('xy','yx','x_minus_x','zero_zero')
PAIRS=(('xy','yx'),('x_minus_x','zero_zero'))


def require(ok,message):
    if not ok: raise ValueError(message)


def owner():
    # Import only a hash-bound saved analyzer; never an environment or loader.
    import hashlib
    path=ROOT/'experiments/spectral_reachability'/(OWNER+'.py')
    require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==OWNER_SHA,'frozen saved analyzer SHA')
    spec=importlib.util.spec_from_file_location(TAG+'_owner',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol():
    return dict(words=list(WORDS),denominator=4,source_roots=1,prefix_frame=10,suffix_frames=list(range(11,21)),
        native_goal=[256.,256.,math.pi/4],native_success_threshold=.95,
        primary='terminal native coverage(frame20) minus own prefix coverage(frame10); higher is better',
        secondary=['terminal native coverage','mean native coverage across all10 suffix frames',
            'mean coverage minus own prefix coverage','terminal native clipped reward'],
        paired_contrasts=[list(p) for p in PAIRS],sign='left minus right, no ranking or word selection',
        reward_definition='clip(native goal intersection area / goal area / .95,0,1)',
        coverage_definition='native goal intersection area / goal area',
        native_done=False,task_success=None,new_success_threshold=None,
        measurement='native scalar coverage, not RGB similarity, feature MSE, contact or pose displacement',
        timing='specified after existing outcomes; retrospective calibration only, not prospective evidence',
        inference='one constructed OOD root with paired actions, no independent four-root inference',
        resource_policy='retain each resource axis separately; no equal total compute/FLOPs claim',
        C2_established=False,scientific_admission=False)


def quantity(q):
    for k in ('coverage','native_reward','native_success_threshold'):
        require(type(q[k]) in (float,int) and math.isfinite(q[k]),'finite native task metric')
    c=q['coverage']
    require(0<=c<=1 and q['native_success_threshold']==.95,'bound native coverage/threshold')
    require(q['native_reward']==min(max(c/.95,0.),1.),'native reward formula')
    require(q['native_done'] is False and q['task_success'] is None,'native done not task success')
    goal=q['native_goal_pose']
    require(goal==dict(dtype='<f8',shape=[3],values=[256.,256.,math.pi/4]),'original native goal identity')
    return float(c)


def metrics(analysis):
    require(analysis['schema']==OWNER+'.report' and analysis['status']=='all4_saved_native_word_responses_analyzed',
            'complete saved response analysis')
    require(analysis['scientific_admission'] is False and analysis['task_success'] is None,'no inherited task admission')
    baselines=analysis['baselines']; rows=analysis['suffix_frame_rows']
    require([r['word'] for r in baselines]==list(WORDS),'all4 ordered baseline denominator')
    require([(r['word'],r['frame']) for r in rows]==[(w,f) for w in WORDS for f in range(11,21)],
            'all40 ordered suffix denominator')
    base=[quantity(r['quantities']) for r in baselines]
    require(all(r['frame']==10 for r in baselines) and len(set(base))==1,'identical prefix native task baseline')
    by_word={}; frame_rows=[]
    for i,w in enumerate(WORDS):
        branch=rows[i*10:(i+1)*10]
        cs=[quantity(r['quantities']) for r in branch]
        for r,c in zip(branch,cs):
            require(r['suffix_step']==r['frame']-10,'exact native suffix action horizon')
            frame_rows.append(dict(word=w,frame=r['frame'],coverage=c,coverage_gain_from_prefix=c-base[i],
                native_reward=r['quantities']['native_reward'],native_done=False,task_success=None))
        by_word[w]=dict(word=w,actual_suffix_steps=10,prefix_coverage=base[i],terminal_coverage=cs[-1],
            terminal_coverage_gain=cs[-1]-base[i],mean_suffix_coverage=math.fsum(cs)/10,
            mean_suffix_coverage_gain=math.fsum(c-base[i] for c in cs)/10,
            terminal_native_reward=branch[-1]['quantities']['native_reward'],task_success=None)
    contrasts=[]
    for left,right in PAIRS:
        contrasts.append(dict(left_word=left,right_word=right,**{k:by_word[left][k]-by_word[right][k]
            for k in ('terminal_coverage','terminal_coverage_gain','mean_suffix_coverage','mean_suffix_coverage_gain',
                      'terminal_native_reward')}))
    return dict(word_rows=list(by_word.values()),frame_rows=frame_rows,paired_contrasts=contrasts,
        selected_word=None,C2_established=False,task_success=None)


def recompute(a,saved):
    inputs=saved['inputs']
    def unique(suffix):
        matches=[p for p in inputs if p.endswith('/'+suffix)]
        require(len(matches)==1,'unique saved authority '+suffix)
        return Path(matches[0]),inputs[matches[0]]
    mp,mh=unique(a.PHYSICAL+'_prepare_MAIN01.json')
    rp,rh=unique(a.PHYSICAL+'_physics_MAIN01/report.json')
    cp,ch=unique('consume_contact_words_MAIN_20261028_v1.py')
    sp,sh=unique(a.PHYSICAL+'_consume_MAIN01.json')
    require(a.analyze(mp,mh,rp,rh,cp,ch,sp,sh)==saved,'full strict response recomputation')


def prepare(path,expected):
    a=owner()
    require(expected==ANALYSIS_SHA,'actual MAIN verified response SHA')
    saved=a.read_json(path,expected)
    require(a.sha(ROOT/ENV)==ENV_SHA,'bound native task implementation')
    recompute(a,saved)  # full strict physics+numeric saved consumers, never run()
    values=metrics(saved)
    pins={str(Path(path).resolve()):expected,str(ROOT/ENV):ENV_SHA,
        str(ROOT/'experiments/spectral_reachability'/(OWNER+'.py')):OWNER_SHA}
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md')):
        pins[str(p)]=a.sha(p)
    require(all(a.sha(p)==h for p,h in pins.items()),'final authority drift')
    return dict(schema=TAG+'.report',status='saved_native_task_metric_calibration_ready_C2_blocked',
        protocol=protocol(),inputs=pins,metrics=values,model_only_parallel=saved['model_only_parallel'],
        physical_meter=saved['physical_meter'],operation_meter=saved['operation_meter'],
        wall_seconds_physical=saved['wall_seconds_physical'],
        engineering_ready=True,prospective_trial_ready=False,C2_established=False,scientific_admission=False,
        neural_executed=False,physics_executed=False,
        hard_missing=[
            'MAIN-sealed prospective goal-conditioned decision rule using public context+goal only, before new trial outcomes',
            'Legal frozen native goal-scoring interface with checkpoint/preprocessing/action normalization binding; pair feature MSE is not a goal cost',
            'Predeclared comparator and deterministic tie rule, same four-word action set/horizon and separate resource caps',
            'New MAIN-approved evaluation declaration before execution; existing observed single OOD root cannot become fresh or preregistered retroactively'],
        release='descriptive native task metrics only; no planner winner, no autonomous execution permission')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify'))
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--analysis-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--output-sha256')
    args=p.parse_args(argv)
    a=owner()
    if args.command=='prepare': require(not args.output.exists(),'exclusive new output')
    else: require(args.output_sha256,'actual output SHA required')
    result=prepare(args.analysis,args.analysis_sha256)
    if args.command=='prepare': a.write_new(args.output,result)
    else: require(a.read_json(args.output,args.output_sha256)==result,'deterministic task metric recomputation')
    print(a.canonical(dict(status=result['status'],sha256=a.sha(args.output),C2_established=False)).decode())
    return 0


if __name__=='__main__': raise SystemExit(main())
