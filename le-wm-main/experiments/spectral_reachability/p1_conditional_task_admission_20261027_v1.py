"""Array-only conditional-task prerequisites, NEVER physical P1 admission.

No model/optimizer/simulator execution or physical-future file discovery.
Numerical margins have no defaults: independently bound authority is required.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

NAME='p1_conditional_task_admission_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
RESOURCES=('predict','action_encoder','jacobian','svd')
SPLITS=('calibration','confirmation')
CURRENT_PREPARED='results/p3_goal_patch_consistent_compiler_20261027_v2_prepared'
CURRENT_PREPARE_SHA='8b514ccaf7ce1f6404cfb04ad77f57d8bd0cc568e48b4d2881e220340de47fc9'
CURRENT_REPORT='results/p3_goal_patch_consistent_compiler_20261027_v2_cuda_main_v1/report.json'
CURRENT_REPORT_SHA='676f245a471aee56486bcd617a7a5c6b10a0d2ee4da340395acbf795a3b7f397'
CURRENT_ORIGIN='results/pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json'
CURRENT_ORIGIN_SHA='99cbb6b6dc4653eb629427353f6f6d05022c219333f3f7362cf4ca43a3076bc2'


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(raw):return hashlib.sha256(raw).hexdigest()


def fields(value,expected,label):
    # Compare keys BEFORE inspecting values; unknown future-label fields never
    # enter the model-screen payload or its digest.
    require(isinstance(value,dict) and set(value)==set(expected.split()),'exact '+label+' fields required')


def integer(value,minimum=0):
    require(type(value) is int and value>=minimum,'literal bounded integer required')
    return value


def hash_value(value):
    require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'SHA256 identity required')
    return value


def bindings(value):
    require(isinstance(value,dict) and bool(value),'nonempty source bindings required')
    for key,h in value.items():require(isinstance(key,str) and bool(key),'source name required');hash_value(h)


def array(value,shape):
    a=np.asarray(value)
    require(a.dtype.kind in 'fi' and a.shape==shape and np.isfinite(a).all(),'finite aligned real array required')
    return a.astype(float)


def context(value,dim=192):
    fields(value,'layout context_length latent_dim action_dim latent_history action_history','context')
    for key,want in (('layout','states_with_between_state_actions_v1'),('context_length',3),('latent_dim',dim),('action_dim',10)):
        require(type(value[key]) is type(want) and value[key]==want,'strict full-context layout '+key)
    array(value['latent_history'],(3,dim));array(value['action_history'],(2,10))


def validate_task(t):
    fields(t,'environment task_contract_sha256 root_context root_raw_action_history goal projection model_sha256 model_config_sha256 metric_source_sha256 goal_source_sha256 representation latent_dim horizon packet_steps control_dim action_mean action_std packet_logit_cap feedback','task')
    require(isinstance(t['environment'],str) and bool(t['environment']),'environment required')
    require(t['representation']=='native_projected_cls' and t['feedback']=='open_loop_model_only','fixed model representation/feedback')
    require(type(t['latent_dim']) is int and t['latent_dim']==192,'native projected CLS192 required')
    for key,want in (('horizon',2),('packet_steps',5),('control_dim',2)):
        require(type(t[key]) is int and t[key]==want,'full H2 / two time-major5x2 packets required')
    for key in ('task_contract_sha256','model_sha256','model_config_sha256','metric_source_sha256','goal_source_sha256'):hash_value(t[key])
    context(t['root_context']);array(t['goal'],(192,))
    q=array(t['projection'],(192,192))
    require(np.allclose(q,q.T,atol=1e-8,rtol=0) and np.allclose(q@q,q,atol=1e-8,rtol=0) and np.trace(q)>.5,'nonzero orthogonal Q required')
    array(t['action_mean'],(2,));std=array(t['action_std'],(2,))
    require((std>0).all() and type(t['packet_logit_cap']) in (int,float) and math.isfinite(t['packet_logit_cap']) and t['packet_logit_cap']>0,'positive action scale and cap')
    raw=array(t['root_raw_action_history'],(10,2))
    require(np.array_equal(t['root_context']['action_history'],normalized_tokens(t,raw)),
        'strict normalized root raw-action history binding required')


def task_id(t):
    validate_task(t)
    return digest(t)


def normalized_tokens(t,commands):
    a=np.asarray(commands)
    require(a.dtype.kind in 'fi' and a.ndim==2 and a.shape[1]==2 and len(a) in (5,10) and np.isfinite(a).all() and (np.abs(a)<=1).all(),'bounded time-major raw packets')
    raw=a.astype(np.float32).reshape(-1,10)
    mean=np.tile(np.asarray(t['action_mean'],np.float32),5);std=np.tile(np.asarray(t['action_std'],np.float32),5)
    return (raw-mean)/std


def model_cost(t,z):
    r=np.asarray(t['projection'])@(array(z,(192,))-np.asarray(t['goal']))
    return float(np.mean(r*r))


def check_prediction(t,p):
    fields(p,'task_id logits commands contexts whole_replay segmented_replay cost','model prediction')
    require(p['task_id']==task_id(t),'same frozen Q/goal/model/context/action contract required')
    logits=array(p['logits'],(2,10));raw=array(p['commands'],(10,2))
    require((np.linalg.norm(logits,axis=1)<=t['packet_logit_cap']+1e-12).all(),'per-packet logit ball violated')
    require(np.array_equal(raw,np.tanh(logits.astype(np.float32)).reshape(10,2)),'exact float32 tanh logits/raw binding')
    require(isinstance(p['contexts'],list) and len(p['contexts'])==2,'complete H2 states, not a suffix-only witness')
    tokens=normalized_tokens(t,raw);previous=t['root_context']
    for k,c in enumerate(p['contexts']):
        context(c)
        require(c['latent_history'][:2]==previous['latent_history'][1:],'full H2 latent chronology')
        expected=np.concatenate((np.asarray(previous['action_history']),tokens[k:k+1]))[-2:]
        require(np.array_equal(c['action_history'],expected),'exact normalized full-H2 history')
        previous=c
    require(p['whole_replay']==previous and p['segmented_replay']==p['contexts'],'independent whole/segmented saved model replay required')
    cost=model_cost(t,previous['latent_history'][-1])
    require(type(p['cost']) in (int,float) and math.isfinite(p['cost']) and p['cost']==cost,'same-Q full192 nonlinear terminal cost')
    return cost


def check_derivative(t,d,incumbent):
    # "raw" = unprojected terminal CLS; columns differentiate LOGITS (including
    # tanh+normalization chain), not raw environment controls. No double chain.
    fields(d,'task_id at_logits at_terminal raw_terminal_jacobian objective_gradient','direct fullH2 derivative')
    require(d['task_id']==task_id(t) and d['at_logits']==incumbent['logits'] and
        d['at_terminal']==incumbent['contexts'][-1]['latent_history'][-1],'derivative must be conditioned on this exact direct incumbent')
    j=array(d['raw_terminal_jacobian'],(192,20))
    g=array(d['objective_gradient'],(20,));q=np.asarray(t['projection'])
    residual=q@(np.asarray(d['at_terminal'])-np.asarray(t['goal']))
    expected=2*(q@j).T@residual/192
    # Numerical receipt consistency only, NOT a plateau/stationarity threshold.
    require(np.allclose(g,expected,rtol=1e-8,atol=1e-10),'same-Q fullH2 objective gradient receipt')


def identity(row):
    fields(row,'candidate_id root_sha256 task_id source split','source identity')
    integer(row['candidate_id']);source_family(row['source'])
    for key in ('root_sha256','task_id'):hash_value(row[key])
    require(row['split'] in SPLITS,'predeclared calibration/confirmation split')


def source_family(s):
    """Dataset rows share episodes; procedural proposals share seed families.

    One admitted root per family, even within a split. Generator/recipe changes
    cannot relabel a previously used environment+seed as fresh.
    """
    require(isinstance(s,dict),'domain-specific source metadata required')
    if s.get('kind')=='dataset_episode':
        fields(s,'kind dataset_sha256 episode start_row','dataset source')
        hash_value(s['dataset_sha256']);integer(s['episode']);integer(s['start_row'])
        return s['kind'],s['dataset_sha256'],s['episode']
    fields(s,'kind generator_sha256 environment_sha256 reset_recipe_sha256 root_seed reset_seed_span proposal_index','procedural source')
    require(s['kind']=='procedural_seed','explicit source domain required')
    for key in ('generator_sha256','environment_sha256','reset_recipe_sha256'):hash_value(s[key])
    integer(s['root_seed']);integer(s['proposal_index'])
    seed_span(s['reset_seed_span'])
    return s['kind'],s['environment_sha256'],s['root_seed']


def seed_span(span):
    require(isinstance(span,list) and len(span)==2,'half-open random-reset seed span required')
    require(integer(span[1])>integer(span[0]),'finite nonempty random-reset seed span')


def sources_overlap(a,b):
    if source_family(a)==source_family(b):return True
    if a['kind']==b['kind']=='procedural_seed' and a['environment_sha256']==b['environment_sha256']:
        return max(a['reset_seed_span'][0],b['reset_seed_span'][0])<min(a['reset_seed_span'][1],b['reset_seed_span'][1])
    return False


def procedural_scan(plan,plan_sha256,receipts):
    """Validate a saved finite seed/reset census; never generate/reset here.

    Each seed slot returns at most ONE root. The source-bound generator must
    enforce the per-seed proposal limit internally and archive its rejection
    details. These are receipts, not independent simulator attestation.
    """
    fields(plan,'schema generator_sha256 environment_sha256 reset_recipe_sha256 proposal_limit_per_seed physical_future_labels_read frozen_before_generation slots','procedural scan plan')
    require(plan['schema']==NAME+'.procedural_scan' and plan['physical_future_labels_read'] is False and
        plan['frozen_before_generation'] is True and digest(plan)==plan_sha256,'externally sealed pre-generation finite scan required')
    for key in ('generator_sha256','environment_sha256','reset_recipe_sha256'):hash_value(plan[key])
    limit=integer(plan['proposal_limit_per_seed'],1)
    require(isinstance(plan['slots'],list) and bool(plan['slots']) and isinstance(receipts,list) and
        len(receipts)==len(plan['slots']),'every finite seed slot needs a receipt, including failures')
    seeds=set();ids=set();identities=[];failed=0;spans=[]
    for slot,receipt in zip(plan['slots'],receipts):
        fields(slot,'candidate_id root_seed reset_seed_span split','procedural seed slot')
        integer(slot['candidate_id']);integer(slot['root_seed'])
        span=slot['reset_seed_span'];seed_span(span)
        require(not any(max(span[0],old[0])<min(span[1],old[1]) for old in spans),'random-reset spans must be disjoint across ALL slots')
        spans.append(span)
        require(slot['split'] in SPLITS and slot['root_seed'] not in seeds and slot['candidate_id'] not in ids,
            'unique finite seed families and candidate IDs')
        seeds.add(slot['root_seed']);ids.add(slot['candidate_id'])
        fields(receipt,'identity status error','procedural metadata receipt')
        r=receipt['identity'];fields(r,'candidate_id root_sha256 task_id source split','metadata identity')
        require(r['candidate_id']==slot['candidate_id'] and r['split']==slot['split'],'fixed seed-slot order/split')
        source_family(r['source']);s=r['source']
        require(s['kind']=='procedural_seed' and s['root_seed']==slot['root_seed'] and s['proposal_index']<limit and
            s['reset_seed_span']==slot['reset_seed_span'] and
            all(s[k]==plan[k] for k in ('generator_sha256','environment_sha256','reset_recipe_sha256')),
            'finite proposal bound and exact seed/generator/reset recipe binding')
        if receipt['status']=='completed':
            identity(r);require(receipt['error'] is None,'successful metadata receipt cannot conceal failure')
            identities.append(deepcopy(r))
        else:
            require(receipt['status'] in ('failed','metadata_rejected') and isinstance(receipt['error'],str) and
                bool(receipt['error']) and r['root_sha256'] is None and r['task_id'] is None,'explicit failed/rejected metadata slot required')
            failed+=1
    return dict(identities=identities,failed_slots=failed,slot_count=len(receipts),
        finite_maximum_proposals=len(receipts)*limit,receipts=deepcopy(receipts),no_replacement=True)


def source_pool(p):
    require(isinstance(p['scan'],list) and bool(p['scan']),'fixed nonempty finite scan required')
    fields(p['pool_counts'],'calibration confirmation','pool counts')
    for n in p['pool_counts'].values():integer(n,1)
    require(len({r['candidate_id'] for r in p['scan']})==len(p['scan']),'unique ordered scan candidate IDs')
    bindings(p['exclusion_sources'])
    excluded_roots=set(p['excluded_roots'])
    require(len(excluded_roots)==len(p['excluded_roots']),'unique historical root exclusions')
    for h in excluded_roots:hash_value(h)
    excluded_sources=set()
    for s in p['excluded_sources']:
        family=source_family(s)
        require(family not in excluded_sources,'unique source-family exclusions');excluded_sources.add(family)
    accepted=[];rejections=[];seen_roots=set();seen_sources=[];counts={s:0 for s in SPLITS}
    kinds=set()
    for r in p['scan']:
        identity(r);reason=None;family=source_family(r['source']);kinds.add(family[0])
        if r['root_sha256'] in excluded_roots:reason='historical_root'
        elif any(sources_overlap(r['source'],s) for s in p['excluded_sources']):reason='historical_source_family'
        elif r['root_sha256'] in seen_roots:reason='pool_root_duplicate'
        elif any(sources_overlap(r['source'],s) for s in seen_sources):reason='pool_source_family_duplicate'
        elif counts[r['split']]>=p['pool_counts'][r['split']]:reason='beyond_frozen_pool_quota'
        if reason:rejections.append(dict(identity=deepcopy(r),reason=reason));continue
        accepted.append(deepcopy(r));seen_roots.add(r['root_sha256']);seen_sources.append(r['source']);counts[r['split']]+=1
    require(len(kinds)==1,'single declared source domain per scan')
    return dict(accepted=accepted,rejections=rejections,counts=counts,
        complete=counts==p['pool_counts'],scan_count=len(p['scan']),no_replacement_after_model_failure=True)


def validate_policy(p):
    fields(p,'schema environment task_contract_sha256 direct scan metadata_scan pool_counts required_confirmation excluded_roots excluded_sources exclusion_sources budgets source_bindings physical_future_labels_read','policy')
    require(p['schema']==NAME+'.policy' and p['physical_future_labels_read'] is False,'own outcome-free policy')
    require(isinstance(p['environment'],str) and bool(p['environment']),'environment required')
    hash_value(p['task_contract_sha256']);bindings(p['source_bindings'])
    d=p['direct'];fields(d,'method implementation_sha256 configuration_sha256 B1 B2 initial_logits incumbent_rule','direct optimizer policy')
    require(d['method'] in ('full_H2_GN','full_H2_gradient'),'frozen joint-H2 gradient/GN implementation required')
    hash_value(d['implementation_sha256']);hash_value(d['configuration_sha256'])
    require(integer(d['B2'],1)>integer(d['B1'],1) and d['incumbent_rule']=='strict_improvement','continuous B1<B2 incumbent policy')
    require(isinstance(d['initial_logits'],list) and bool(d['initial_logits']),'predeclared restarts/initializations')
    for a in d['initial_logits']:array(a,(2,10))
    fields(p['budgets'],'direct producer witness','phase budgets')
    for b in p['budgets'].values():
        fields(b,'predict action_encoder jacobian svd','resource budget')
        for n in b.values():integer(n)
    require(p['budgets']['witness']['predict']==6 and p['budgets']['witness']['action_encoder']==6,
        'H2 prediction2 + whole2 + segmented2 witness forwards')
    iterations=d['B2']*len(d['initial_logits']);minimum=6*(d['B2']+1)*len(d['initial_logits'])+2*iterations
    require(p['budgets']['direct']['predict']==minimum and p['budgets']['direct']['action_encoder']==minimum and
        p['budgets']['direct']['jacobian']==iterations and p['budgets']['direct']['svd']==(iterations if d['method']=='full_H2_GN' else 0),
        'budget must pay each fullH2 ADforward/API and each candidate prediction/replay')
    require(p['budgets']['witness']['jacobian']==p['budgets']['witness']['svd']==0,
        'witness verification only; producer search work needs a separate frozen budget')
    require(p['budgets']['producer']['predict']>=2 and
        p['budgets']['producer']['action_encoder']==p['budgets']['producer']['predict'],
        'explicit fullH2 witness-producer work, inherited or fresh, cannot be omitted')
    require('witness_producer' in p['source_bindings'],'separate source-bound witness producer required')
    pool=source_pool(p)
    if p['scan'][0]['source']['kind']=='procedural_seed':
        fields(p['metadata_scan'],'plan plan_sha256 receipts','procedural metadata manifest')
        census=procedural_scan(**p['metadata_scan'])
        require(census['identities']==p['scan'],'model scan must equal finite procedural metadata census')
        require(p['source_bindings'].get('metadata_scan_plan')==p['metadata_scan']['plan_sha256'],
            'external seed-plan seal pinned in policy')
    else:require(p['metadata_scan'] is None,'dataset uses episode metadata, not procedural plan')
    integer(p['required_confirmation'],1)
    require(p['required_confirmation']<=p['pool_counts']['confirmation'],'finite predeclared confirmation requirement')
    return pool


def freeze_contract(policy=None,authority=None,expected_authority_sha256=None):
    if policy is None:
        return dict(schema=NAME+'.contract',status='unknown',ready=False,policy=None,authority=None,
            unknown_reasons=['missing_frozen_policy','missing_threshold_authority'],contract_id=None)
    pool=validate_policy(policy);unknown=[]
    if not pool['complete']:unknown.append('insufficient_fixed_metadata_pool')
    if authority is None:unknown.append('missing_threshold_authority')
    else:
        fields(authority,'schema environment policy_id task_contract_sha256 values calibration_roots calibration_sources source_bindings frozen_before_confirmation derivation','threshold authority')
        require(authority['schema']==NAME+'.threshold_authority' and expected_authority_sha256 is not None and
            digest(authority)==expected_authority_sha256,'external threshold-authority SHA binding required')
        require(authority['environment']==policy['environment'] and authority['policy_id']==digest(policy) and
            authority['task_contract_sha256']==policy['task_contract_sha256'],'authority is not transferable between environments/tasks/policies')
        require(authority['frozen_before_confirmation'] is True,'thresholds must freeze before confirmation')
        bindings(authority['source_bindings']);fields(authority['values'],'epsilon_plateau epsilon_escape tau','numeric thresholds')
        for v in authority['values'].values():require(type(v) in (float,int) and math.isfinite(v) and v>=0,'explicit finite nonnegative thresholds required')
        calibration=[r for r in pool['accepted'] if r['split']=='calibration']
        require(authority['calibration_roots']==[r['root_sha256'] for r in calibration] and
            authority['calibration_sources']==[r['source'] for r in calibration],
            'threshold calibration must exclude all confirmation roots/source families')
        if authority['derivation'] is not None:
            derivation=authority['derivation'];fields(derivation,'recipe recipe_sha256 rows','calibration derivation')
            rebuilt=calibrate_authority(policy,derivation['recipe'],derivation['rows'],derivation['recipe_sha256'])
            require(authority==rebuilt,'calibration-derived thresholds/arrays changed')
    body=dict(schema=NAME+'.contract',status='unknown' if unknown else 'ready',ready=not unknown,
        policy=deepcopy(policy),authority=deepcopy(authority),authority_sha256=expected_authority_sha256,unknown_reasons=unknown)
    body['contract_id']=digest(body)
    return body


def validate_contract(c):
    require(c==freeze_contract(c['policy'],c['authority'],c.get('authority_sha256')),'frozen contract bytes/identity changed')


def calibrate_authority(policy,recipe,rows,expected_recipe_sha256):
    """Deterministic model-only calibration, not a physical success threshold.

    Recipe and code are frozen BEFORE calibration; authority is independently
    sealed BEFORE confirmation. No scientific quantile/offset defaults.
    External chronology and source completeness remain runner seal obligations.
    """
    fields(recipe,'schema environment task_contract_sha256 method minimum_rows quantiles offsets source_bindings frozen_before_calibration','calibration recipe')
    require(recipe['schema']==NAME+'.calibration_recipe' and recipe['method']=='empirical_quantiles_v1' and
        recipe['frozen_before_calibration'] is True,'predeclared derivation method required')
    require(digest(recipe)==expected_recipe_sha256==policy['source_bindings'].get('calibration_recipe'),
        'calibration recipe must be externally sealed and pinned in policy')
    require(recipe['environment']==policy['environment'] and recipe['task_contract_sha256']==policy['task_contract_sha256'],
        'calibration recipe task/domain binding')
    bindings(recipe['source_bindings']);integer(recipe['minimum_rows'],1)
    for key in ('quantiles','offsets'):fields(recipe[key],'epsilon_plateau epsilon_escape tau','derivation parameters')
    for v in recipe['quantiles'].values():require(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1,'explicit quantile in [0,1]')
    for v in recipe['offsets'].values():require(type(v) in (int,float) and math.isfinite(v),'finite explicit offset')
    c=freeze_contract(policy);pool=source_pool(policy)
    calibration=[r for r in pool['accepted'] if r['split']=='calibration']
    require(len(rows)==len(calibration)>=recipe['minimum_rows'] and
        [r['identity'] for r in rows]==calibration,'all and ONLY ordered independent calibration rows required')
    evidence=[]
    for r in rows:
        require(r['status']=='completed','failed calibration cannot be dropped or replaced')
        evidence.append(check_task(c,r))
    columns=dict(epsilon_plateau='direct_improvement',epsilon_escape='escape_improvement',tau='witness_cost')
    values={key:max(0.,float(np.quantile([e[column] for e in evidence],recipe['quantiles'][key],method='linear'))+
        recipe['offsets'][key]) for key,column in columns.items()}
    return dict(schema=NAME+'.threshold_authority',environment=policy['environment'],policy_id=digest(policy),
        task_contract_sha256=policy['task_contract_sha256'],values=values,
        calibration_roots=[r['root_sha256'] for r in calibration],calibration_sources=[deepcopy(r['source']) for r in calibration],
        source_bindings=deepcopy(recipe['source_bindings']),frozen_before_confirmation=True,
        derivation=dict(recipe=deepcopy(recipe),recipe_sha256=expected_recipe_sha256,rows=deepcopy(rows)))


def ledger(p,journal,complete):
    require(isinstance(journal,list),'complete attempt/completion ledger required')
    counts={phase:{kind:dict(attempted=0,completed=0) for kind in RESOURCES} for phase in ('direct','producer','witness')}
    groups={}
    for i,e in enumerate(journal):
        fields(e,'index phase evaluation_id resource samples completed','resource event')
        require(type(e['index']) is int and e['index']==i and e['phase'] in counts and e['resource'] in RESOURCES and type(e['completed']) is bool,'ordered scoped attempted-work ledger')
        n=integer(e['samples'],1);require(isinstance(e['evaluation_id'],str),'logical evaluation identity')
        v=counts[e['phase']][e['resource']];v['attempted']+=n;v['completed']+=n if e['completed'] else 0
        group=groups.setdefault((e['phase'],e['evaluation_id']),{k:0 for k in RESOURCES});group[e['resource']]+=n
    for phase,counts0 in counts.items():
        for kind,v in counts0.items():
            cap=p['budgets'][phase][kind]
            require(v['attempted']<=cap,'nonrefundable resource cap exceeded')
            if complete:require(v['attempted']==v['completed']==cap,'complete phase budget/failed work mismatch')
    if complete:
        evaluation_ids={('witness','witness')}
        for r in range(len(p['direct']['initial_logits'])):
            for i in range(p['direct']['B2']+1):evaluation_ids.add(('direct',f'run_{r}_iter_{i}'))
        for key in evaluation_ids:
            require(groups.get(key)==dict(predict=6,action_encoder=6,jacobian=0,svd=0),'each saved complete-H2 model evaluation must pay predict/replays')
        allowed=evaluation_ids|{('producer','producer')}|{('direct',f'opt_{r}_iter_{i}') for r in range(len(p['direct']['initial_logits'])) for i in range(1,p['direct']['B2']+1)}
        require(set(groups)==allowed,'all optimizer and model evaluation groups required')
        require(groups['producer','producer']==p['budgets']['producer'],'all paid witness-producer work required')
        for key in allowed-evaluation_ids-{('producer','producer')}:
            require(groups[key]==dict(predict=2,action_encoder=2,jacobian=1,svd=1 if p['direct']['method']=='full_H2_GN' else 0),
                'joint-H2 derivative/optimizer work must bind each budget step')
    return counts


def check_task(c,row):
    validate_contract(c)
    base=dict(P1_admission=False,physical_mechanism_established=False,physical_efficacy_measured=False,
        mathematical_stationarity_established=False,physical_future_labels_read=False,model_task_eligible=False)
    if c['policy'] is None:return dict(base,status='unknown',unknown_reasons=c['unknown_reasons'])
    p=c['policy'];complete=row.get('status')=='completed'
    fields(row,('identity status task direct_runs witness journal source_bindings physical_future_labels_read' if complete else
        'identity status journal error physical_future_labels_read'),'model screening row')
    require(row['physical_future_labels_read'] is False,'physical futures forbidden')
    pool=source_pool(p);require(row['identity'] in pool['accepted'],'only frozen source pool may be evaluated; no replacement')
    work=ledger(p,row['journal'],complete)
    if not complete:
        require(row['status'] in ('failed','not_started') and isinstance(row['error'],str),'explicit failure/blocked receipt')
        return dict(base,status='unknown',unknown_reasons=['incomplete_model_evidence'],resources=work)
    t=row['task'];validate_task(t)
    require(t['environment']==p['environment'] and t['task_contract_sha256']==p['task_contract_sha256'] and
        task_id(t)==row['identity']['task_id'],'same task/Q/goal/full-H2 contract throughout frozen scan')
    require(row['source_bindings']==p['source_bindings'],'runner/source binding drift')
    d=p['direct'];require(len(row['direct_runs'])==len(d['initial_logits']),'all predeclared restarts required')
    checkpoint={b:[] for b in (d['B1'],d['B2'])}
    for r,run in enumerate(row['direct_runs']):
        require(isinstance(run,list) and len(run)==d['B2']+1,'continuous full direct budget trajectory required')
        previous=None
        for i,step in enumerate(run):
            fields(step,'iteration incumbent proposal accepted derivative','direct budget step')
            require(type(step['iteration']) is int and step['iteration']==i and type(step['accepted']) is bool,'no missing/restarted/jumped budget steps')
            loss=check_prediction(t,step['incumbent'])
            if i==0:
                require(step['proposal'] is None and step['derivative'] is None and step['accepted'] is False and
                    step['incumbent']['logits']==d['initial_logits'][r],'predeclared full-H2 initialization')
            else:
                check_derivative(t,step['derivative'],previous)
                proposed=check_prediction(t,step['proposal']);old=check_prediction(t,previous)
                accept=proposed<old
                require(step['accepted'] is accept and step['incumbent']==(step['proposal'] if accept else previous),
                    'direct trajectory must retain exact incumbent across B1/B2; no reinitialization')
            if i in checkpoint:checkpoint[i].append(loss)
            previous=step['incumbent']
    early=min(checkpoint[d['B1']]);late=min(checkpoint[d['B2']]);witness=check_prediction(t,row['witness'])
    require(0<=early-late,'continued direct budget cannot lose its best incumbent')
    result=dict(base,status='unknown' if not c['ready'] else 'model_screened',unknown_reasons=c['unknown_reasons'],
        direct_B1_cost=early,direct_B2_cost=late,witness_cost=witness,
        direct_improvement=early-late,escape_improvement=late-witness,resources=work,
        finite_budget_plateau=None,budget_escape=None,target_reached=None,direct_above_target=None)
    if c['ready']:
        v=c['authority']['values']
        result.update(finite_budget_plateau=early-late<=v['epsilon_plateau'],budget_escape=late-witness>v['epsilon_escape'],
            target_reached=witness<=v['tau'],direct_above_target=late>v['tau'])
        result['model_task_eligible']=all(result[k] for k in ('finite_budget_plateau','budget_escape','target_reached','direct_above_target'))
    return result


def select_cohort(c,rows):
    validate_contract(c);require(c['policy'] is not None,'frozen policy required for cohort census')
    pool=source_pool(c['policy']);identities=[r['identity'] for r in rows]
    require(len({r['root_sha256'] for r in identities})==len(rows) and
        {canonical(r) for r in identities}=={canonical(r) for r in pool['accepted']},'complete fixed pool census; missing/extra roots cannot be silently replaced')
    indexed={r['identity']['root_sha256']:r for r in rows};evaluated=[]
    resources={state:{k:0 for k in RESOURCES} for state in ('attempted','completed')}
    for identity0 in pool['accepted']:
        evidence=check_task(c,indexed[identity0['root_sha256']]);evaluated.append(dict(identity=identity0,evidence=evidence))
        for phase in evidence['resources'].values():
            for k,work in phase.items():
                for state in resources:resources[state][k]+=work[state]
    incomplete=any(indexed[r['root_sha256']]['status']!='completed' for r in pool['accepted'])
    eligible=[r['identity']['root_sha256'] for r in evaluated if r['identity']['split']=='confirmation' and r['evidence']['model_task_eligible']]
    n=c['policy']['required_confirmation'];enough=c['ready'] and pool['complete'] and not incomplete and len(eligible)>=n
    status=('incomplete_model_pool' if incomplete else 'unknown' if not c['ready'] else
        'model_conditioned_cohort_only' if enough else 'insufficient_model_eligible_roots')
    return dict(schema=NAME+'.selection',contract_id=c['contract_id'],status=status,
        selected_roots=eligible[:n] if enough else [],eligible_confirmation_count=len(eligible),
        rows=evaluated,metadata_pool=pool,resources=resources,P1_admission=False,
        physical_future_labels_read=False,physical_mechanism_established=False,physical_efficacy_measured=False)


def audit_current(project=PROJECT):
    """Explicit MODEL-ONLY file allowlist. No physical file or future glob.

    Saved consistency + actual gaps, not a new model/physical task screen.
    """
    project=Path(project).resolve();directory=project/CURRENT_PREPARED;files={}
    source=Path(__file__).resolve();files[str(source)]=sha(source.read_bytes())
    def read(path):
        raw=path.read_bytes();files[str(path)]=sha(raw);return json.loads(raw)
    receipt=read(directory/'prepare.json');require(files[str(directory/'prepare.json')]==CURRENT_PREPARE_SHA,'fixed v2 prepare SHA')
    envelope=read(directory/'protocol.json');p=envelope['protocol']
    require(files[str(directory/'protocol.json')]==receipt['protocol_sha256'] and digest(p)==envelope['protocol_id']==receipt['protocol_id'],'source protocol file/content identity')
    report=read(project/CURRENT_REPORT)
    require(files[str(project/CURRENT_REPORT)]==CURRENT_REPORT_SHA,'fixed saved model report SHA')
    origin=read(project/CURRENT_ORIGIN)
    require(files[str(project/CURRENT_ORIGIN)]==CURRENT_ORIGIN_SHA,'fixed model-only origin report SHA')
    require([r['root_sha256'] for r in origin['rows']]==p['roots'] and
        origin['protocol']['cohort']=='calibration','same historical calibration roots')
    provenance=dict(kind='procedural_seed',source_episode='not_applicable_procedural_roots',
        root_seed=origin['protocol']['root_seed'],root_count=origin['protocol']['root_count'],
        oversample_factor=origin['protocol']['root_oversample_factor'],proposal_count=16,
        per_root_proposal_index='not_recorded',per_root_reset_recipe='not_recorded',
        sampling='ordered unique nonterminal boundary-contact proposals; NOT plateau/target conditioned',
        freshness='historical calibration seed family; exclude entire seed family from new calibration/confirmation',
        seed_warning='Optimizer seed+root_index is not the root generator/reset seed.')
    require(report['schema']=='p3_goal_patch_consistent_compiler_20261027_v2.report' and report['protocol_id']==envelope['protocol_id'] and
        report['status']=='cuda_completed_h2_prerequisite' and report['failure'] is None and report['failures']==[] and
        report['model_unchanged'] is True and report['new_physical_steps']==0 and report['physical_quality_measured'] is False,'complete model-only current compiler report')
    require(report['resources']['attempted_predict_samples']==report['resources']['completed_predict_samples']==528 and
        report['resources']['completed_jacobians']==88 and report['resources']['completed_svds']==96,'current saved resource census')
    indexed={(r['root_sha256'],r['arm']):r for r in report['rows']}
    require(len(indexed)==len(report['rows'])==8 and len(set(p['roots']))==4 and len(p['packets'])==4 and
        set(indexed)=={(r,a) for r in p['roots'] for a in ('old_Q','goal_patch_Q')},'four roots/eight arms, no order join')
    arms=[];roots=[]
    for i,packet in enumerate(p['packets']):
        root=packet['root_sha256'];require(root==p['roots'][i],'frozen root identity')
        roots.append(dict(root_sha256=root,root_index=i,source_episode='not_applicable_procedural_roots',
            source_kind='procedural_seed',root_seed=provenance['root_seed'],proposal_index=None,
            gradient_plateau='unverified',two_stage_target_reachability='unverified',fresh_conditional_cohort=False,
            missing_screening_fields=dict(
                direct_runs='All declared initializations, continuous iterations0..B2 (B1<B2), full20D derivative at each incumbent, legal proposal/incumbent, same-Q fullH2 contexts/replays/costs.',
                witness='Legal full2x10 logits/raw10x2; both normalized-history contexts; whole/segmented model replay; nonlinear same-Q cost <= independently calibrated frozen tau.',
                journal='Attempted/completed direct+producer+witness resource events, including all failures; current66/11/12 compilation ledger is NOT the new direct-screening budget.',
                authority='Independent outcome-free calibration recipe/rows/source hashes, derived epsilon_plateau/epsilon_escape/tau, freeze before confirmation; no user numeric choice required.',
                identity='Historical procedural root; per-proposal mapping absent. Dataset episode is not applicable. These four cannot become fresh confirmation.',
                policy='Finite fresh seed/reset scan, historical root+seed-family exclusion manifest, calibration/confirmation split, no replacement after model failures.')))
        for arm in ('old_Q','goal_patch_Q'):
            row=indexed[root,arm];r=row['compilation'];q=np.asarray(packet['projections'][arm]);goal=np.asarray(packet['goal'])
            require(row['q_sha256']==r['q_sha256']==digest(q.tolist()) and row['active_metric_id']==packet['active_metric_ids'][arm] and
                row['input_sha256']==packet['input_sha256'] and row['source_input_sha256']==packet['source_input_sha256'],'same source/active task binding')
            costs=[]
            for k,candidate in enumerate(r['candidates']):
                raw=array(candidate['commands'],(10,2));context(candidate['context'])
                prefix=array(candidate['prefix_params'],(10,));suffix=array(candidate['suffix_params'],(10,))
                require(np.linalg.norm(prefix)<=p['contract']['cap_per_packet']+1e-12 and np.linalg.norm(suffix)<=p['contract']['cap_per_packet']+1e-12,'legal saved packet logits')
                require(np.array_equal(raw,np.tanh(np.asarray([prefix,suffix],np.float32)).reshape(10,2)),'saved fullH2 actions')
                tokens=normalized_tokens(dict(action_mean=[-.0087,.0068],action_std=[.2019,.2002]),raw)
                require(np.array_equal(candidate['context']['action_history'],tokens),'saved exact normalized H2 history')
                z=array(candidate['context']['latent_history'],(3,192))
                require(np.array_equal(z[0],packet['root_context']['latent_history'][-1]) and
                    np.array_equal(z[1],candidate['predicted_prefix']) and np.array_equal(z[2],candidate['predicted_terminal']),'saved H2 chronology')
                stage=float(np.mean((q@(z[1]-goal))**2));terminal=float(np.mean((q@(z[2]-goal))**2))
                require(candidate['index']==k and candidate['q_sha256']==row['q_sha256'] and candidate['completed'] is True and
                    candidate['independent_model_replay_exact'] is True and
                    stage==candidate['stage_cost'] and terminal==candidate['nonlinear_terminal_cost'] and
                    candidate['objective']==stage+candidate['own_reach_value'],'saved same-Q scores and complete candidate identity')
                costs.append(candidate)
            require(len(costs)==10,'all ten candidates retained')
            greedy=min(costs,key=lambda c:(c['stage_cost'],c['index']));winner=min(costs,key=lambda c:(c['objective'],c['index']))
            require(winner==r['winner'] and r['whole_replay_context']==r['segmented_replay_context']==winner['context'],'saved winner and whole/segmented context')
            arms.append(dict(root_sha256=root,root_index=i,arm=arm,candidates=10,
                same_Q_full_H2_saved_structure=True,greedy_index=greedy['index'],winner_index=winner['index'],
                greedy_tradeoff_indices=[c['index'] for c in costs if c['stage_cost']>greedy['stage_cost'] and c['own_reach_value']<greedy['own_reach_value']],
                missing=['threshold_authority','continuous_same_Q_full_H2_direct_trajectory','target_tau_witness',
                    'fixed_fresh_scan_and_domain_source_exclusions','calibration_confirmation_split']))
    return dict(schema=NAME+'.current_gap_audit',status='unknown_preconditions',bindings=files,
        provenance=provenance,roots=roots,arms=arms,candidate_count=80,gradient_plateau='unverified',two_stage_target_reachability='unverified',
        fresh_scan_interface=dict(
            before_metadata='Freeze finite ordered seed/recipe slots and domain-only acceptance predicate; exclude historical seed202753010 and all other known-used seed families/root hashes. No dynamic resampling.',
            metadata_only='Runner may reset/render initial conditions and inspect current-state contact/nonterminal validity, never step actions to inspect physical futures. Record every slot/proposal/failure and source closure.',
            source_fields='kind=procedural_seed, generator_sha256, environment_sha256, reset_recipe_sha256, root_seed, reset_seed_span=[low,high), proposal_index; bind state+goal root SHA and causal RGB/action history. Distinct parent seeds may NOT overlap their underlying random-reset spans.',
            scan_api='procedural_scan(plan, plan_sha256, receipts): finite pre-generation slots, bounded proposals, every failed/rejected slot retained; bundle pinned as policy.metadata_scan.',
            one_root_per_seed_family=True,
            split='Disjoint seed families for independent calibration and confirmation, fixed before model screening. Freeze metadata policy before calibration; freeze authority before confirmation.',
            arrays='Runner exports task+direct_runs+witness+journal to check_task/select_cohort; all predictions/J use same task_id (Q/goal/fullH2/action/model/history).',
            calibration='calibrate_authority(policy, recipe, calibration_rows, externally_sealed_recipe_sha256), then freeze_contract(policy, authority, externally_sealed_authority_sha256).',
            incomplete='Insufficient finite pool or eligible confirmation roots => empty cohort; retain collisions and failures, never append seeds to pass.'),
        fresh_conditional_cohort=False,model_task_eligible=False,P1_admission=False,
        physical_future_labels_read=False,new_neural_calls=0,new_physical_steps=0,
        scope='Saved model arrays only; no full source/model seal revalidation and no neural replay. Legal forecasts are not authorized target attainment or real mechanism/efficacy.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audit-current',action='store_true');group.add_argument('--screen',type=Path)
    group.add_argument('--calibrate',type=Path);group.add_argument('--freeze',type=Path)
    parser.add_argument('--project-root',type=Path,default=PROJECT);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.audit_current:result=audit_current(args.project_root)
    elif args.calibrate:
        payload=json.loads(args.calibrate.read_text());fields(payload,'policy recipe rows recipe_sha256','calibration input')
        result=calibrate_authority(payload['policy'],payload['recipe'],payload['rows'],payload['recipe_sha256'])
    elif args.freeze:
        payload=json.loads(args.freeze.read_text());fields(payload,'policy authority authority_sha256','freeze input')
        result=freeze_contract(payload['policy'],payload['authority'],payload['authority_sha256'])
    else:
        payload=json.loads(args.screen.read_text());fields(payload,'contract rows','model screening input')
        result=select_cohort(payload['contract'],payload['rows'])
    if args.output:
        with args.output.open('x') as stream:stream.write(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(canonical(result))
    if args.screen and not result['selected_roots']:raise SystemExit(2)


if __name__=='__main__':main()
