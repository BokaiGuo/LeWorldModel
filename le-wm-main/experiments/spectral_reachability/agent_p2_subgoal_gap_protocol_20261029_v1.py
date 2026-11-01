"""P2/E1 fixed known-root subgoal-gap preparation and explicitly RAM-only seams.

No model, simulator, foreign loader or full saved strict is called. This is not
another three-arm runtime: reuse the existing E1 scorer/runner after MAIN review.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_subgoal_gap_contract_20261028_v1 as scoring

NAME = 'agent_p2_subgoal_gap_protocol_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
ARMS = scoring.ARMS
CP = (5, 10, 15)
PHASES = ('context_prepare','proposal','controller','AD_primal','whole_replay','segmented_replay')
MODES = ('predict_API_calls_matched','predictor_forward_samples_matched')
ASSETS = ('arbitrary_optimizer','prior_pairing','common_controller','native_rebuild','tracking_criterion')
COMMON = ('model','encoder','preprocess','normalizer','context','q','raw_domain','task_goal')
CAP, FLOOR = 16*1024**2, 200*1024**2
PROPOSAL = Path('/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt')
CATALOG = {
    'compiled_report':'results/agent_p2_three_program_20261028_v1_cuda_MAIN01/report.json',
    'compiled_receipt':'results/agent_p2_three_program_20261028_v1_consume_MAIN01.json',
    'prior_report':'results/agent_hilewm_low_cem_20261029_v1_cuda_MAIN01/report.json',
    'prior_receipt':'results/agent_hilewm_low_cem_20261029_v1_consume_MAIN01.json',
    'prior_rgb_receipt':'results/agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json',
    'safe_weights_export':'results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json',
}
READ_SOURCES = (
    'experiments/spectral_reachability/agent_subgoal_gap_runner_20261028_v1.py',
    'experiments/spectral_reachability/agent_subgoal_gap_assets_20261028_v1.py',
    'experiments/spectral_reachability/agent_p2_three_program_20261028_v1.py',
    'experiments/spectral_reachability/agent_hilewm_low_cem_20261029_v1.py',
    'innerdocs/agent_hilewm_rgb_high_low_20261028_v1.md',
    'innerdocs/agent_hilewm_normalizer_asset_route_20261029_v2.md',
)
require = scoring.require


def raw(x): return scoring.encode(x)+b'\n'
def digest(x): return hashlib.sha256(raw(x)).hexdigest()


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def file_bytes(path):
    p=Path(path)
    require(p.is_absolute() and p.suffix in ('.json','.py','.md','.txt','.h')
        and p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,
        'absolute regular bounded metadata/source only; no weights or archive loading')
    b=p.read_bytes();require(len(b)<=CAP,'actual metadata cap');return b


def ref(path):
    p=Path(path);b=file_bytes(p)
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def check_ref(reference):
    scoring.keys(reference,('path','sha256'));scoring.sha(reference['sha256'])
    b=file_bytes(reference['path'])
    require(hashlib.sha256(b).hexdigest()==reference['sha256'],'external metadata SHA mismatch')
    return b


def bound(reference): return scoring.parse(check_ref(reference))


def inventory(paths=None):
    """Finite named metadata only, no directory scan and no full consume.

    paths injection is a RAM-test seam, not a way to certify other experiments.
    Returned status fields are saved claims, never new authentication tokens.
    """
    paths = {k:ROOT/v for k,v in CATALOG.items()} if paths is None else paths
    require(set(paths)==set(CATALOG),'exact bounded catalog')
    files={};docs={}
    for key,path in paths.items():
        p=Path(path)
        if not p.exists():
            files[key]=dict(path=str(p),sha256=None,bytes=None,status='missing');continue
        r=ref(p);obj=scoring.parse(file_bytes(p));docs[key]=obj
        files[key]=dict(**r,status='metadata_read_NOT_full_strict',schema=obj.get('schema'),saved_status=obj.get('status'))
    compiled=[]
    if 'compiled_receipt' in docs:
        u=docs['compiled_receipt'];plans=u['plans']
        expected=[(r,a) for r in (0,1) for a in ('joint_H3_no_composition','three_program_imagined_recompile')]
        require(u['schema']=='agent_p2_three_program_20261028_v1.consume' and len(plans)==4
            and [(p['root_slot'],p['arm']) for p in plans]==expected,'actual full four-plan compiled metadata ABI')
        if 'compiled_report' in docs:
            require(u['planner_report_sha256']==files['compiled_report']['sha256'],'compiled report/receipt actual byte join')
        for i in (1,3):
            p=plans[i]
            require(p['raw_steps']==15 and p['program_count']==3 and p['compiled_nodes']==[0,5,10]
                and p['condition']=='off' and len(p['edges'])==3,'existing full15 three-program, not joint or H2')
            compiled.append(dict(root_slot=p['root_slot'],receipt_pointer=f'/plans/{i}',task_id=p['task_id'],
                task_file=p['task_file'],task_sha256=p['task_sha256'],model_hash=p['model_hash'],
                root_context_sha256=digest(p['root_context']),goal_sha256=digest(p['goal']),
                normalization_sha256=digest(p['normalization']),selected_plan_sha256=p['selection_digest'],
                checkpoint_context_pointers=[f'/plans/{i}/edges/{j}/output_context' for j in range(3)],
                evidence_scope='existing own selected program, NOT a new equal-budget proposal'))
        require(compiled[0]['model_hash']==compiled[1]['model_hash'],'common existing P2 frozen model')
    if 'prior_receipt' in docs:
        u=docs['prior_receipt']
        require(u['schema']=='agent_hilewm_low_cem_20261029_v1.consume'
            and u['physical_release'] is False,'actual normalized Hi low receipt ABI, no raw release')
        if 'prior_report' in docs:require(u['report_sha256']==files['prior_report']['sha256'],'Hi report/receipt actual byte join')
    return dict(files=files,compiled=compiled,
        availability=dict(arbitrary='no authenticated direct-latent optimizer on this common ABI supplied',
            learned='Hi saved normalized high-FIRST/low integration exists if catalog present; common-P2 pairing and raw inverse NOT certified',
            compiled='saved full15 witnessed programs exist if catalog present; cannot reuse as new equal-budget generator work'),
        full_strict_rebuilt=False,checkpoint_loaded=False,new_neural_calls=0,new_physics_calls=0)


def budget_check(b):
    scoring.keys(b,('mode','cap','matched_model_sha256','matched_callable_sha256','batch_limit','scope','failed_attempts_refunded'))
    require(b['mode'] in MODES and type(b['cap']) is int and b['cap']>0
        and type(b['batch_limit']) is int and b['batch_limit']>0,'external positive cap/mode/batch, no outcome-derived default')
    for k in ('matched_model_sha256','matched_callable_sha256'):scoring.sha(b[k])
    require(b['scope']==list(PHASES) and b['failed_attempts_refunded'] is False,
        'all prepare/proposal/controller/AD-primal/replays/failed attempts charged')
    return b


def contract(budget=None):
    if budget is not None:budget_check(budget)
    return dict(schema=NAME+'.contract',arms=list(ARMS),roots=[0,1],condition='off',
        known_roots=2,fresh_roots=0,independent_confirmation=False,horizon=15,checkpoints=list(CP),latent_dim=192,
        trial_denominator=2,branch_denominator=6,checkpoint_denominator=18,
        arbitrary='direct latent optimization; no random-prior/final-goal/action-endpoint substitution',
        learned='actual frozen learned prior on authenticated common coordinates; Hi/SAGE method identities distinct',
        compiled='finite-horizon action program with full causal-context/whole+segmented replay, targets=predicted postconditions',
        common='same root/task/full causal observation/controller/model/encoder/Q/normalizer/raw domain and feedback times',
        controller='same controller receives ONLY root/public context and delivered targets; no privileged compiled raw-action warmstart',
        deployment='fresh original reset/recipe/warm10; same cp5/10/15 RGB observation; no saved-poststate reset',
        budget=deepcopy(budget),budget_claim='same declared matched-boundary cap; NOT equal actual use; NOT all-model total compute',
        inherited_generation='old plans are interface evidence only; fresh paid generators required for prospective comparison',
        high_prior_cost='separate actual high-model API/sample/token axes, never relabelled common low predictor or free overhead',
        resources=['predict_API','predictor_forward_samples','encoder','projector','action_encoder','pred_proj',
            'AD_primal','scalar_backward','Jacobian','mathematical_VJP','SVD','physics','render','wall_seconds',
            'CPU_capture_seconds','high_prior_API','high_prior_samples','high_prior_context_tokens'],
        total_FLOPs=None,engine_backward_calls=None,equal_FLOPs=False,equal_total_compute=False,
        P1_admission=False,P2_admission=False,scientific_admission=False,dispatch_allowed=False)


def declaration_check(d):
    scoring.keys(d,('schema','route','budget','assets'))
    require(d['schema']==NAME+'.declaration' and d['route']=='current_P2_frozen192','explicit current P2 route; no cross-model latent cast')
    scoring.keys(d['assets'],ASSETS)
    if d['budget'] is not None:budget_check(d['budget'])
    return d


def asset_check(role, obj, model):
    """Metadata predicates only. A claimed certificate is never a run release."""
    require(obj['schema']==NAME+'.'+role and obj['model_hash']==model,'common actual P2 model in each asset')
    require(type(obj['bindings']) is dict and obj['bindings'],'explicit real dependency refs')
    for r in obj['bindings'].values(): check_ref(r)
    require(all(k in obj['bindings'] for k in COMMON),'complete common coordinate references, not model hash alone')
    if role=='arbitrary_optimizer':
        require(obj['method']=='direct_latent_optimization' and obj['parameter_shape']==[3,192]
            and obj['uses_action_rollout_as_target'] is False,'arbitrary is optimized latent, not a renamed action waypoint')
    elif role=='prior_pairing':
        require(obj['method'] in ('HiLeWM_hierarchical_adaptation','SAGE_generator_adaptation')
            and obj['fallback'] is False and obj['raw_inverse_available'] is True,
            'actual learned method/no fallback and raw inverse authority required for physical comparison')
        for k in ('target_delivery','prior_checkpoint_inventory'):
            require(k in obj['bindings'],'missing actual prior pairing field: '+k)
    elif role=='common_controller':
        require(obj['checkpoint_steps']==list(CP) and obj['same_callable_all_arms'] is True
            and obj['privileged_compiled_warmstart'] is False and 'predictor_callable' in obj['bindings'],
            'same full15 controller without witness-action privilege and exact predictor source')
    elif role=='native_rebuild':
        require(obj['roots']==[0,1] and obj['warmup']==10 and obj['reset_saved_poststate'] is False,
            'literal original recipe, not approximate saved state reset')
    else:
        t=obj['latent_l2_tolerance']
        require(type(t) in (int,float) and math.isfinite(t) and t>0 and obj['chosen_before_outcomes'] is True,
            'external positive preregistered tracking tolerance; never fit a threshold here')


def prepare(declaration=None,declaration_sha=None):
    require((declaration is None)==(declaration_sha is None),'declaration path/SHA together')
    inv=inventory();bindings={}
    for r in inv['files'].values():
        if r['sha256'] is not None:bindings[r['path']]=r['sha256']
    d=dict(schema=NAME+'.declaration',route='current_P2_frozen192',budget=None,assets=dict.fromkeys(ASSETS))
    if declaration is not None:
        d=declaration_check(bound(dict(path=str(declaration),sha256=declaration_sha)))
        bindings[str(declaration)]=declaration_sha
    blockers=[]
    if len(inv['compiled'])!=2:blockers.append('existing_compiled_source_metadata_missing')
    model=inv['compiled'][0]['model_hash'] if inv['compiled'] else None
    if d['budget'] is None:blockers.append('MAIN_predeclared_predict_mode_cap_scope_missing')
    elif model is not None:require(d['budget']['matched_model_sha256']==model,'budget bound to same P2 model, not Hi high model')
    common_refs=None
    for role,r in d['assets'].items():
        if r is None:blockers.append('missing_'+role);continue
        require(model is not None,'asset needs actual source model identity')
        obj=bound(r);asset_check(role,obj,model);bindings[r['path']]=r['sha256']
        current={k:obj['bindings'][k] for k in COMMON}
        if common_refs is None:common_refs=current
        else:require(current==common_refs,'all proposal/controller/native/criterion coordinate refs exactly equal')
        if role=='common_controller' and d['budget'] is not None:
            require(obj['bindings']['predictor_callable']['sha256']==d['budget']['matched_callable_sha256'],
                'budget exact matched predictor callable source, not unrelated model metadata')
        for dep in obj['bindings'].values():
            require(dep['path'] not in bindings or bindings[dep['path']]==dep['sha256'],'nonconflicting full dependency closure')
            bindings[dep['path']]=dep['sha256']
    for f in [*own_files(),Path(scoring.__file__).resolve(),PROPOSAL,*[ROOT/p for p in READ_SOURCES]]:
        r=ref(f);bindings[r['path']]=r['sha256']
    for path,h in bindings.items():require(ref(path)['sha256']==h,'final preparation source/metadata closure unchanged')
    return dict(schema=NAME+'.prepare',arguments=dict(declaration=declaration,declaration_sha=declaration_sha),
        status='blocked_missing_assets' if blockers else 'ABI_prepared_MAIN_full_strict_and_runtime_review_pending',
        inventory=inv,contract=contract(d['budget']),declaration=d,blockers=blockers,bindings=bindings,
        slots=[dict(root_slot=r,arm=a,proposal=None,observed=None,native_success=None,resources=None,status='not_dispatched') for r in (0,1) for a in ARMS],
        scientific_limitations=['known roots only','P1 remains false','no paired learned-prior delivery proved here'],
        full_strict_rebuilt=False,production_ready=False,dispatch_allowed=False)


class FixtureMeter:
    """Only RAM tests; a fake counter can NEVER authenticate real execution."""
    def __init__(self,budget):
        self.b=deepcopy(budget_check(budget));self.events=[]
        self.attempted=dict(predict_API=0,predictor_forward_samples=0)
        self.completed=deepcopy(self.attempted)
    def before_dispatch(self,phase,samples,model_sha,callable_sha):
        require(phase in PHASES and type(samples) is int and 1<=samples<=self.b['batch_limit'],'declared phase and batch samples')
        require(model_sha==self.b['matched_model_sha256'] and callable_sha==self.b['matched_callable_sha256'],
            'one exact predictor boundary; prior high cannot masquerade as common low')
        charge=dict(predict_API=1,predictor_forward_samples=samples)
        axis='predict_API' if self.b['mode']==MODES[0] else 'predictor_forward_samples'
        require(self.attempted[axis]+charge[axis]<=self.b['cap'],'predispatch cap, no refund for failure')
        n=len(self.events);self.events.append(dict(index=n,phase=phase,samples=samples,status='pending'))
        for k,v in charge.items():self.attempted[k]+=v
        return n
    def after_return(self,index,*,failed=False):
        require(type(index) is int and 0<=index<len(self.events) and type(failed) is bool,'literal event index and failure flag')
        e=self.events[index];require(e['status']=='pending','exact one terminal return, no duplicate')
        e['status']='failed' if failed else 'returned'
        if not failed:
            self.completed['predict_API']+=1;self.completed['predictor_forward_samples']+=e['samples']
    def report(self):
        return dict(fixture_only=True,real_meter_authenticated=False,attempted=deepcopy(self.attempted),
            completed=deepcopy(self.completed),events=deepcopy(self.events),total_FLOPs=None,
            other_resources={k:None for k in contract()['resources'] if k not in self.attempted})


def fixture_measure(rows,tolerance=None):
    """Reuse existing E1 deviations; fixture rows cannot be scientific receipts."""
    require(len(rows)==6,'fixed all6 fixture denominator')
    if tolerance is not None:require(type(tolerance) in (int,float) and math.isfinite(tolerance) and tolerance>0,'positive fixture tolerance')
    output=[]
    for row,(root,arm) in zip(rows,[(r,a) for r in (0,1) for a in ARMS],strict=True):
        require(row['fixture_only'] is True and (row['root_slot'],row['arm'])==(root,arm),'literal synthetic fixed6 identity')
        if row['target'] is None:
            require(row['observed'] is row['predicted'] is row['native_success'] is None,'missing prior/proposal cannot become a zero outcome')
            actual=pred=realizable=None
        else:
            require(len(row['target'])==len(row['observed'])==len(row['predicted'])==3,'complete cp5/10/15 fixture slots')
            actual=scoring.deviations(row['target'],row['observed'],192)
            pred=scoring.deviations(row['target'],row['predicted'],192)
            realizable=[None if e is None or tolerance is None else e['l2']<=tolerance for e in actual]
        require(row['native_success'] is None or type(row['native_success']) is bool,'native task outcome is separate, unknown stays null')
        output.append(dict(root_slot=root,arm=arm,actual_tracking=actual,predicted_tracking=pred,
            realizable=realizable,native_success=row['native_success']))
    return dict(fixture_only=True,scientific_admission=False,new_neural_calls=0,new_physics_calls=0,
        branch_denominator=6,checkpoint_denominator=18,rows=output,
        realization_ratio=None if tolerance is None else [dict(step=step,denominator=6,
            known=sum(r['realizable'] is not None and r['realizable'][j] is not None for r in output),
            realized=sum(r['realizable'] is not None and r['realizable'][j] is True for r in output)) for j,step in enumerate(CP)])


def write_new(path,value):
    p=Path(path);b=raw(value)
    require(p.is_absolute() and p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink()
        and len(b)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive own bounded metadata')
    with p.open('xb') as f:require(f.write(b)==len(b),'full metadata write');f.flush();os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return ref(p)


def verify(path,expected):
    saved=bound(dict(path=str(path),sha256=expected))
    require(saved['schema']==NAME+'.prepare' and raw(saved)==raw(prepare(**saved['arguments'])),'full saved ABI preparation rebuild')
    require(ref(path)['sha256']==expected,'prepare unchanged')
    return dict(verified=True,production_ready=False,dispatch_allowed=False,status=saved['status'],blockers=saved['blockers'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--declaration');a.add_argument('--declaration-sha');a.add_argument('--output')
    a=sub.add_parser('verify');a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
    sub.add_parser('inventory')
    a=p.parse_args(argv)
    if a.command=='inventory':result=inventory()
    elif a.command=='prepare':
        result=prepare(a.declaration,a.declaration_sha)
        if a.output:result=write_new(a.output,result)
    else:result=verify(a.prepared,a.prepare_sha)
    print(raw(result).decode())


if __name__=='__main__':main()
