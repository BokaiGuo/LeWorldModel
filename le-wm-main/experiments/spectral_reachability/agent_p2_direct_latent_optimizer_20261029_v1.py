"""Fixed exploratory direct-latent optimization and common tracking RAM seam.

No dynamics rollout, action endpoint, learned prior, model import or deployment.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

NAME='agent_p2_direct_latent_optimizer_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL='agent_p2_subgoal_gap_protocol_20261029_v1'
PROTOCOL_SHA='274cbd6f3a8539c866fd62b3ab4d0fff44e37915407a8d596adc8c8ba722a6cd'
ARMS=('arbitrary_latent_waypoint','existing_learned_prior','dynamics_compiled_witness')
CP=(5,10,15)
DIM,STEPS,ETA=192,64,16.
COMMON=('model','encoder','preprocess','normalizer','context','q','raw_domain','task_goal')
PHASES=('context_prepare','proposal','controller','AD_primal','whole_replay','segmented_replay')
CAP=16*1024**2


def need(ok,message):
    if not ok:raise ValueError(message)


def keys(v,names):need(type(v) is dict and set(v)==set(names),'exact fields: '+','.join(names))
def raw(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(raw(v)).hexdigest()
def hash_value(v):need(type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v),'literal SHA256')


def vector(v,n=DIM):
    need(type(v) is list and len(v)==n and all(type(x) in (int,float) and math.isfinite(x) for x in v),
        'complete finite vector, no bool/fill/cast')
    return [float(x) for x in v]


def future(v):
    need(type(v) is list and len(v)==3,'three future checkpoints, not context history')
    return [vector(x) for x in v]


def common_refs(refs):
    keys(refs,COMMON)
    for r in refs.values():keys(r,('path','sha256'));hash_value(r['sha256']);need(type(r['path']) is str,'literal source path')


def task_check(t):
    keys(t,('schema','root_slot','condition','context','goal','coordinate_refs'))
    need(t['schema']==NAME+'.task' and type(t['root_slot']) is int and t['root_slot'] in (0,1) and t['condition']=='off',
        'fixed two known off roots, not fresh')
    c=t['context'];keys(c,('layout','context_length','latent_dim','action_dim','latent_history','action_history'))
    need(type(c['layout']) is str and c['layout'] and type(c['context_length']) is int and c['context_length']==3 and
        type(c['latent_dim']) is int and c['latent_dim']==192 and type(c['action_dim']) is int and c['action_dim']==10,
        'original full3 latent192/two token10 history ABI')
    future(c['latent_history']);need(type(c['action_history']) is list and len(c['action_history'])==2,'actual action2 history')
    for row in c['action_history']:vector(row,10)
    vector(t['goal']);common_refs(t['coordinate_refs']);return t


def contract():
    return dict(schema=NAME+'.contract',method='direct_latent_optimization',parameter_shape=[3,DIM],
        route='current_P2_frozen192',root_slots=[0,1],condition='off',checkpoints=list(CP),
        objective='mean((Z3-goal)^2) + sum(k=1..3, mean((Zk-Z[k-1])^2)), Z0=current_actual_last_latent',
        initialization='repeat current actual last latent three times',updates=STEPS,eta=ETA,
        arithmetic='Python binary64, unbounded latent coordinates, no projection/cast/tanh/action rollout',
        terminal_Q='original identity192 mean squared distance to unchanged goal',smoothness_weight=1.,
        selection='final iterate64 only; retain full0..64 history, no winner/best/outcome selection',
        scalar_loss_evaluations=65,analytic_gradient_evaluations=64,parameter_updates=64,
        neural_forward_calls=0,neural_AD_calls=0,physics_calls=0,model_training_steps=0,
        uses_action_rollout_as_target=False,uses_learned_prior=False,learned_prior_fallback=False,
        original_protocol_exact_loss_specified=False,exploratory_objective_fixed_here=True,
        real_controller_authenticated=False,production_ready=False,dispatch_allowed=False,
        P1_admission=False,P2_admission=False,E1_admission=False,total_FLOPs='unknown')


def objective(z,anchor,goal,gradient=True):
    """Own explicit scalar/gradient; no model-Q/dynamics callback accepted."""
    z=future(z);anchor=vector(anchor);goal=vector(goal)
    need(type(gradient) is bool,'literal gradient flag')
    terminal=sum((x-y)**2 for x,y in zip(z[2],goal,strict=True))/DIM
    prev=anchor;edges=[];smooth=0.
    for row in z:
        d=[x-y for x,y in zip(row,prev,strict=True)];edges.append(d)
        smooth+=sum(x*x for x in d)/DIM;prev=row
    total=terminal+smooth
    need(math.isfinite(total),'finite objective; no clipping/repair')
    g=None
    if gradient:
        g=[[2.*(edges[k][i]-(edges[k+1][i] if k<2 else 0.)+
            (z[2][i]-goal[i] if k==2 else 0.))/DIM for i in range(DIM)] for k in range(3)]
        future(g)
    return dict(loss=total,terminal_goal_MSE=terminal,smoothness_MSE=smooth),g


def optimize(task):
    """Actual fixed64 direct parameter updates; only vector arithmetic in RAM."""
    task_check(task);anchor=vector(task['context']['latent_history'][-1]);goal=vector(task['goal'])
    z=[anchor.copy() for _ in CP];history=[];previous=None
    for iteration in range(STEPS+1):
        score,g=objective(z,anchor,goal,gradient=iteration<STEPS)
        if previous is not None:need(score['loss']<=previous,'fixed-step numerical descent failed; no retry/retune')
        history.append(dict(iteration=iteration,targets=deepcopy(z),score=score,gradient=g))
        previous=score['loss']
        if g is not None:z=[[x-ETA*dx for x,dx in zip(row,grad,strict=True)] for row,grad in zip(z,g,strict=True)]
    return dict(schema=NAME+'.numerical_result',status='RAM_direct_latent_optimized_NOT_deployed',contract=contract(),
        task_sha256=digest(task),context_sha256=digest(task['context']),goal_sha256=digest(task['goal']),
        coordinate_refs=deepcopy(task['coordinate_refs']),root_slot=task['root_slot'],history=history,
        delivered_targets=deepcopy(z),delivered_targets_sha256=digest(z),checkpoint_steps=list(CP),
        origin='direct_latent_parameters_not_action_endpoint',realizability_ratio=None,native_success=None,
        training=False,production_ready=False,dispatch_allowed=False)


def optimizer_asset(model_hash,bindings):
    """Pauli-compatible ABI descriptor; never a pairing/freshness certificate."""
    hash_value(model_hash);common_refs(bindings);bound=deepcopy(bindings)
    path=Path(__file__).resolve()
    bound['optimizer_implementation']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    bound['parent_protocol']=dict(path=str(ROOT/'experiments/spectral_reachability'/(PROTOCOL+'.py')),sha256=PROTOCOL_SHA)
    return dict(schema=PROTOCOL+'.arbitrary_optimizer',model_hash=model_hash,bindings=bound,
        method='direct_latent_optimization',parameter_shape=[3,192],uses_action_rollout_as_target=False,
        algorithm=contract(),implementation_source=str(Path(__file__).resolve()),
        full_strict_rebuilt=False,production_ready=False)


def budget_check(b):
    keys(b,('mode','cap','matched_model_sha256','matched_callable_sha256','batch_limit','scope','failed_attempts_refunded'))
    need(b['mode'] in ('predict_API_calls_matched','predictor_forward_samples_matched') and
        type(b['cap']) is int and b['cap']>0 and type(b['batch_limit']) is int and b['batch_limit']>0,'external positive budget, no default')
    hash_value(b['matched_model_sha256']);hash_value(b['matched_callable_sha256'])
    need(b['scope']==list(PHASES) and b['failed_attempts_refunded'] is False,'same full scope and nonrefund')


def tracking_request(task,arm,targets,controller):
    """One identical public target-tracking ABI; NO witness actions/state/labels."""
    task_check(task);need(arm in ARMS,'exact three original arm identities')
    if targets is None:return dict(status='blocked_missing_delivered_targets',arm=arm,request=None,dispatch_allowed=False)
    targets=future(targets)
    if controller is None:return dict(status='blocked_missing_common_controller',arm=arm,request=None,dispatch_allowed=False)
    keys(controller,('source','predictor_source','model_hash','coordinate_refs','budget','same_callable_all_arms','privileged_compiled_warmstart'))
    keys(controller['source'],('path','sha256'));hash_value(controller['source']['sha256']);hash_value(controller['model_hash'])
    keys(controller['predictor_source'],('path','sha256'));hash_value(controller['predictor_source']['sha256'])
    common_refs(controller['coordinate_refs']);budget_check(controller['budget'])
    need(controller['same_callable_all_arms'] is True and controller['privileged_compiled_warmstart'] is False,
        'no compiled witness privileged warmstart')
    need(raw(controller['coordinate_refs'])==raw(task['coordinate_refs']) and
        controller['budget']['matched_model_sha256']==controller['model_hash'] and
        controller['budget']['matched_callable_sha256']==controller['predictor_source']['sha256'],
        'same common coordinate/model and exact matched predictor boundary')
    return dict(schema=NAME+'.tracking_request',status='RAM_interface_only_NOT_runtime_authorized',arm=arm,
        root_slot=task['root_slot'],condition='off',public_context=deepcopy(task['context']),goal=deepcopy(task['goal']),
        delivered_targets=targets,delivered_targets_sha256=digest(targets),checkpoint_steps=list(CP),
        coordinate_refs=deepcopy(task['coordinate_refs']),controller=deepcopy(controller),
        privileged_witness_actions=None,production_ready=False,dispatch_allowed=False)


def tracking_metrics(request,predicted,observed):
    need(request['schema']==NAME+'.tracking_request' and request['checkpoint_steps']==list(CP) and
        digest(request['delivered_targets'])==request['delivered_targets_sha256'],'exact delivered target identity')
    need(type(predicted) is list and type(observed) is list and len(predicted)==len(observed)==3,'all three checkpoints, missing is null')
    rows=[]
    for cp,target,p,o in zip(CP,future(request['delivered_targets']),predicted,observed,strict=True):
        def error(value):
            if value is None:return None
            value=vector(value);sq=sum((x-y)**2 for x,y in zip(value,target,strict=True))
            need(math.isfinite(sq),'finite tracking deviation');return dict(L2=math.sqrt(sq),MSE=sq/DIM)
        rows.append(dict(step=cp,predicted_to_delivered=error(p),actual_to_delivered=error(o)))
    return dict(rows=rows,checkpoint_denominator=3,realizability_ratio=None,native_success=None,
        unknown_criterion=True,latent_tracking_is_not_native_success=True)


def fixture_track(request,callback):
    """Only explicitly RAM-marked injected callbacks, no dynamic module loading."""
    need(request.get('schema')==NAME+'.tracking_request' and getattr(callback,'kind',None)=='RAM_FIXTURE',
        'RAM-only callback, no CPU neural or real tracker dispatch')
    need(getattr(callback,'source_sha256',None)==request['controller']['source']['sha256'],'same injected fixture callable identity')
    response=callback(deepcopy(request));keys(response,('predicted','observed','resources'))
    metrics=tracking_metrics(request,response['predicted'],response['observed'])
    r=response['resources'];keys(r,('attempted','completed'))
    for phase in ('attempted','completed'):
        keys(r[phase],('predict_API','predictor_forward_samples'))
        need(all(type(x) is int and x>=0 for x in r[phase].values()),'literal fixture resource counters')
    for axis in r['attempted']:need(r['completed'][axis]<=r['attempted'][axis],'no failed refunds/invented completion')
    b=request['controller']['budget'];axis='predict_API' if b['mode']=='predict_API_calls_matched' else 'predictor_forward_samples'
    for counts in r.values():
        need(counts['predict_API']<=counts['predictor_forward_samples']<=b['batch_limit']*counts['predict_API'],
            'API and forward samples distinct, literal bounded batch accounting')
    need(r['attempted'][axis]<=b['cap'],'declared matched-boundary cap')
    return dict(status='fixture_tracking_only',metrics=metrics,resources=deepcopy(r),
        fixture_only=True,real_meter_authenticated=False,other_resources='unknown',total_FLOPs='unknown',
        production_ready=False,dispatch_allowed=False)


def read_ref(ref):
    keys(ref,('path','sha256'));hash_value(ref['sha256']);p=Path(ref['path'])
    need(p.is_absolute() and p.resolve()==p and p.suffix in ('.json','.py','.md','.txt','.h') and
        p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,'metadata/source only, never weights/NPY/archive')
    body=p.read_bytes();need(hashlib.sha256(body).hexdigest()==ref['sha256'],'exact source/input SHA');return body


def sealed_numerical(task_path,task_sha):
    """MAIN may run only numerical arithmetic; no production/controller claim."""
    def unique(pairs):
        out={}
        for k,v in pairs:need(k not in out,'duplicate JSON key');out[k]=v
        return out
    task_ref=dict(path=str(Path(task_path).absolute()),sha256=task_sha)
    task=json.loads(read_ref(task_ref),object_pairs_hook=unique);task_check(task)
    refs=[task_ref,*task['coordinate_refs'].values(),dict(path=str(ROOT/'experiments/spectral_reachability'/(PROTOCOL+'.py')),sha256=PROTOCOL_SHA)]
    for path in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        refs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for r in refs:read_ref(r)
    result=optimize(task)
    for r in refs:read_ref(r)
    result['bindings']=refs;result['input_asset_AUTHENTICATION']='SHA_only_not_encoder_pairing_or_full_strict'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('contract','numerical'))
    p.add_argument('--task');p.add_argument('--task-sha');a=p.parse_args()
    result=contract() if a.command=='contract' else sealed_numerical(a.task,a.task_sha)
    print(raw(result).decode())  # stdout only; no workspace/result writes


if __name__=='__main__':main()
