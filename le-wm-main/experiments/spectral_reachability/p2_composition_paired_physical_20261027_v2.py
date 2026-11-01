"""Independent eight-rootarm physical consumer; no planning or admission.

Prepare/verify use saved evidence only, never instantiate an environment or
load a neural model. Main explicitly runs guarded physics; optional HOST CUDA
encoding happens only after all physical/partial receipts have been persisted.
The paired planner/exporter and all historical sources remain unchanged.
"""
import argparse
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from experiments.spectral_reachability import p2_composition_paired_20261027_v2 as sampling
from experiments.spectral_reachability import p2_full20_physical_20261027_v2 as helpers
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context, official_checked

TAG = 'p2_composition_paired_physical_20261027_v2'
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNING = PROJECT/'results/p2_composition_paired_20261027_v2_prepared01'
EXPECTED_PLANNER_SHA256 = '8d3f924c2bc38730e3d3eeedb24149cfb3a420857d6f8132bc95eeb01fe87379'
EXPECTED_PROTOCOL_SHA256 = 'adea073583427f78ee5d832c7f647c1c77c66291d3db83c02b7776bd6beedfd9'
sha, write = sampling.full20.sha, durable.write
BOUNDARY = dict(P1_admission=False,P2_admission=False,physical_efficacy_verified=False,
    fair_total_budget_comparison=False,root_selection_changed=False,extra_search_or_training=False,
    no_neural_predict=True,calibration_only=True,
    scope='eight selected known-root H2 paths; retain all guard/termination/partial failures, no retry or reselection')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(rows, key):
    require(isinstance(rows,list) and len(rows)==4,'all four roots required')
    result = {r[key]:r for r in rows}
    require(len(result)==4,'duplicate stable root identity')
    return result


def verify_archived_exporter(prepared):
    source = str(Path(sampling.__file__).resolve())
    found = []
    for seal in prepared['source_seals']:
        directory = Path(seal['output_dir'])
        verify_seal(output_dir=directory,expected_manifest_sha256=seal['manifest_sha256'])
        manifest = json.loads((directory/'manifest.json').read_text())
        for entry in manifest['files']:
            if entry['source']==source:
                archive = directory/entry['archive']
                require(entry['sha256']==sha(archive)==sha(source)==EXPECTED_PLANNER_SHA256,
                    'original paired source required; no retrospective exporter/reseal')
                require('\ndef selected_actions(output):' in archive.read_text(),'original sealed exporter missing')
                found.append(dict(source=source,archive=str(archive),sha256=entry['sha256'],
                    manifest_sha256=seal['manifest_sha256']))
    require(len(found)==1,'unique original source-bound exporter required')
    return found[0]


def plans_from_saved(report, protocol, prepared, compilation, verified_actions):
    """After sealed exporter verification, bind roots/goals/witnesses and budgets.

    No efficacy labels, winner changes, SVD, neural work or physical access here.
    """
    require(report['schema']==sampling.TAG+'.report' and report['status']=='calibration_completed'
        and report['protocol_sha256']==digest(protocol) and report['source_seals']==prepared['source_seals']
        and bool(report['gates']) and all(v is True for v in report['gates'].values())
        and report['gates']==sampling.calibration_gates(report['rows'],True,True),
        'complete source-bound paired calibration required')
    require(protocol['schedule']==sampling.SCHEDULE and protocol['boundary']==sampling.BOUNDARY,
        'frozen paired schedule/boundary required')
    require(all(report[k]==v for k,v in sampling.BOUNDARY.items()),'paired claim boundary changed')
    ready = indexed(protocol['context']['cases'],'root_sha256')
    rows = indexed(report['rows'],'root_sha256')
    source = indexed(compilation['rows'],'root_sha256')
    require(rows.keys()==ready.keys()==source.keys(),'root identity census mismatch')
    require({r['root_index'] for r in ready.values()}==set(range(4)),'four unique root indices required')
    require(set(verified_actions)=={(key,arm) for key in ready for arm in sampling.ARMS},
        'eight exported rootarm actions required')
    plans = []
    for key,case in sorted(ready.items(),key=lambda pair:pair[1]['root_index']):
        row,root,c = rows[key],case['root'],source[key]['compiled']
        initial = c['input_context']
        expected = dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=initial['emb'][0],action_history=initial['actions'][0][1:])
        require(root['root_id']==key and root['context']==expected and row['root_index']==case['root_index'],
            'complete original root/context binding required')
        require(case['metric_sha256']==c['scoring']['metric']['sha256']
            and case['goal_vector_sha256']==digest(c['goal_vectors'][0])
            and case['projection_sha256']==digest(c['scoring']['metric']['projection']),
            'original goal0/Q binding mismatch')
        require(np.asarray(c['goal_vectors'][0]).shape==(192,)
            and np.isfinite(c['goal_vectors'][0]).all()
            and np.asarray(c['scoring']['metric']['projection']).shape==(192,192)
            and np.isfinite(c['scoring']['metric']['projection']).all(),'finite full original goal/Q required')
        results = row['results']
        require(row['complete'] is True and row['error'] is None
            and len(results)==2 and [r['arm'] for r in results]==list(sampling.ARMS)
            and sampling.accounting_ok(row['actual_model_entries'],row['actual_prediction_budget'],results),
            'complete real two-arm dispatch accounting required')
        receipt = sampling.resource_receipt(row['actual_model_entries'],results,row['operations'])
        require(receipt==row['resource_receipt']
            and sampling.operation_counts(row['operations'])==row['completed_operations']
            and row['completed_operations']['jacobian_calls']==41
            and row['completed_operations']['scalar_output_vjp_directions']==7872,
            'actual shared/exclusive/Jacobian/VJP accounting mismatch')
        require(receipt['unique']['predict_calls']==row['predict_calls']==252
            and receipt['unique']['action_encoder_calls']==row['action_encoder_calls']==252
            and receipt['unique']['predict_context_tokens']==row['predict_context_tokens']==756
            and receipt['unique']['encoder_context_tokens']==row['encoder_context_tokens']==756,
            'actual predict/encoder/context work mismatch')
        budget = row['actual_prediction_budget']
        require(budget['binding']==dict(protocol_id=prepared['source_seals'][0]['manifest_sha256'],
            task_id=sampling.TAG,root_id=key,frozen_model_id=sampling.historical.audit.CHECKPOINT_SHA)
            and budget['method']==sampling.TAG and budget['predict_sample_limit']==252
            and budget['attempted_predict_samples']==252 and budget['blocked_calls']==0
            and all(p['completed'] is True and p['failed_predict_calls']==0 for p in budget['phases']),
            'prediction budget root/model/method/limit binding mismatch')
        by_arm = {r['arm']:r for r in results}
        # Fixed counterbalancing, unrelated to model/physical outcomes.
        order = sampling.ARMS if case['root_index']%2==0 else sampling.ARMS[::-1]
        for arm in order:
            result = by_arm[arm]
            require(result['fixture_only'] is False and result['complete'] is True
                and result['replay_error'] is None and result['selection_evaluations']==40
                and len(result['attempts'])==40 and all(a['completed'] is True for a in result['attempts']),
                'all forty real candidate slots required')
            best = min(result['attempts'],key=lambda a:sampling.full20._cost(a['cost']))
            winner = result['winner']
            require(winner is not None and winner['candidate_id']==best['candidate_id']
                and winner['cost']==best['cost'] and winner['witness']==best['witness']
                and winner['independent_full_replay_exact'] is True,'stable replayed winner required')
            witness = winner['witness']
            n = 1 if arm==sampling.ARMS[0] else 2
            require(len(witness['edges'])==n,'exact joint/composition witness topology required')
            edges = witness['edges']
            contexts = [e['input_context'] for e in edges]+[edges[-1]['output_context']]
            chunks = [e['actions'] for e in edges]
            require([len(chunk) for chunk in chunks]==([10] if n==1 else [5,5]),'exact ten-step edge lengths required')
            provenance = witness['receipt']['provenance']
            expected_prov = sampling.sra.provenance_for(protocol,prepared,case)
            require(all(provenance[k]==v for k,v in expected_prov.items() if k!='candidate_bank_sha256'),
                'winner root/metric/model/source provenance mismatch')
            rebound = sampling.saved_witness(root,provenance,case['action_domain'],contexts,chunks,best['candidate_id'])
            raw = np.asarray(verified_actions[key,arm])
            require(raw.dtype.kind in 'fi' and raw.shape==(10,2) and np.isfinite(raw).all() and np.all(np.abs(raw)<=1)
                and raw.tolist()==best['actions']==witness['receipt']['actions']
                and witness==rebound,'exact exported actions/witness required; no casting/clipping')
            require(abs(sampling.full20.metric_for(c)(witness['receipt']['terminal_context'])-winner['cost'])<=1e-12,
                'selected original-Q terminal cost mismatch')
            plan = dict(root_index=case['root_index'],root_sha256=key,method=arm,root=deepcopy(root),
                commands=raw.tolist(),action_domain=deepcopy(case['action_domain']),witness=deepcopy(witness),
                receipt_sha256=witness['receipt']['receipt_sha256'],selected_candidate_id=winner['candidate_id'],
                planned_horizon_tokens=2,planned_commands=10,corrected_search_horizon=2,
                goal_vector_sha256=case['goal_vector_sha256'],projection_sha256=case['projection_sha256'],
                planning_accounting=dict(shared=deepcopy(receipt['shared']),exclusive=deepcopy(receipt['exclusive'][arm]),
                    standalone_attributed=deepcopy(receipt['standalone_attributed'][arm]),
                    paired_unique=deepcopy(receipt['unique']),completed_operations=deepcopy(row['completed_operations']),
                    warning='shared counted once; never sum standalone attributions'))
            plan['selection_sha256'] = digest(plan)
            plans.append(plan)
    expected_totals = dict(actual_unique_predict_calls=1008,actual_unique_action_encoder_calls=1008,
        actual_selection_evaluations=320,completed_jacobian_calls=164,completed_scalar_output_vjp_directions=31488)
    require(all(report[k]==v for k,v in expected_totals.items()),'four-root inherited cost totals mismatch')
    require(report['completed_svd_calls']==sum(r['completed_operations']['svd_calls'] for r in rows.values()),
        'actual SVD total mismatch')
    require(len(plans)==len({(p['root_sha256'],p['method']) for p in plans})==8,'eight unique physical plans required')
    return plans


def inputs(planning):
    planning = Path(planning).resolve()
    require((planning/'report.json').is_file(),'paired CUDA report missing; wait for main, no placeholder plans')
    path,protocol,prepared = sampling.load_prepared(planning)
    require(digest(protocol)==EXPECTED_PROTOCOL_SHA256,'original pre-outcome paired protocol required')
    proof = verify_archived_exporter(prepared)
    reader = sampling.historical.audit.EvidenceReader()
    reader.bindings(protocol['bindings'])
    report = reader.load(path/'report.json')
    exported = sampling.selected_actions(path)
    source = reader.load(protocol['context']['compilation'],sampling.historical.audit.COMPILATION_SHA)
    plans = plans_from_saved(report,protocol,prepared,source,exported['actions'])
    require(exported['report_sha256']==reader.hashes[str(path/'report.json')],'report changed during export')
    for filename in ('protocol.json','prepared.json'):
        reader.load(path/filename)
    for case in protocol['context']['cases']:
        reader.load(path/f'root_{case["root_index"]}_bank.json')
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    reports = [reader.load(p) for p in protocol['bindings'] if Path(p).name=='report.json']
    interventions = [r for r in reports if 'bank_path' in r.get('protocol',{})]
    require(len(interventions)==1,'unique official physical root provenance required')
    parent = interventions[0]['protocol']
    bank_path = Path(parent['bank_path']).resolve()
    bank = reader.load(bank_path,parent['bank_sha256'])
    upstream_path = Path(bank['upstream_path']).resolve()
    reader.load(upstream_path,parent['upstream_sha256'])
    upstream = base.physical._load_preflight(upstream_path)
    refs = indexed(upstream['rows'],'root_sha256')
    require(set(refs)=={p['root_sha256'] for p in plans},'physical reference census mismatch')
    for plan in plans:
        ref = refs[plan['root_sha256']]
        require(plan['root']['root_rgb_sha256']==ref['history_rgb_sha256']
            and plan['root']['goal_rgb_sha256']==ref['goal_rgb_sha256'],'physical root/goal RGB binding mismatch')
    _,_,config = checked.validate_inputs(protocol['context']['compilation'],protocol['context']['policy'])
    reader.unchanged()
    return dict(base=base,checked=checked,source=source,config=config,bindings=deepcopy(reader.hashes),
        plans=plans,upstream=upstream,refs=refs,planning_protocol=protocol,exporter_proof=proof,
        verification_work=deepcopy(exported['verification_work']),
        planning_totals={k:report[k] for k in ('actual_unique_predict_calls','actual_unique_action_encoder_calls',
            'actual_selection_evaluations','completed_jacobian_calls','completed_svd_calls','completed_scalar_output_vjp_directions')})


def environment_creator(base):
    entry = base.gym.spec('swm/PushT-v1').entry_point
    if callable(entry):
        return entry
    module,name = entry.split(':')
    return getattr(importlib.import_module(module),name)


def source_files(context):
    creator = environment_creator(context['base'])
    environment_sources = sorted({str(Path(inspect.getfile(obj)).resolve()) for obj in
        (creator,*[getattr(creator,n) for n in ('eval_state','reset','render','step')])})
    files = context['checked'].sources_before_allocate(context['source'],context['config'])
    files += [Path(m.__file__) for n,m in tuple(sys.modules.items())
        if n.startswith('experiments.') and getattr(m,'__file__','').endswith('.py')]
    files += [Path(p) for p in environment_sources]+[Path(__file__),PROJECT/'tests'/f'test_{TAG}.py',PROJECT/'innerdocs'/f'{TAG}.md']
    return sorted({p.resolve() for p in files}),environment_sources


def prepare(planning, output):
    output = Path(output).resolve()
    require(TAG in output.name,'new tagged physical preparation directory required')
    if output.exists():
        raise FileExistsError(output)
    context = inputs(planning)
    files,environment_sources = source_files(context)
    bindings = deepcopy(context['bindings'])
    for p in environment_sources:
        bindings[p] = sha(p)
    protocol = dict(schema=TAG+'.protocol',planning=str(Path(planning).resolve()),bindings=bindings,
        plans=context['plans'],source_paths=list(map(str,files)),environment_source_paths=environment_sources,
        sealed_exporter=context['exporter_proof'],planning_totals=context['planning_totals'],
        exporter_verification_work=context['verification_work'],claim_boundary=deepcopy(BOUNDARY),
        physical_repetitions=1,planned_commands_per_branch=10,total_planned_commands=80,
        branches=8,branch_order='root index ascending; ARMS even, reversed ARMS odd; no label ordering',
        physical_before_optional_encoding=True,
        root_goal_validation='finite seven-state and goal + exact full root fingerprint for all roots before selected actions; then per-branch RGB/context recheck',
        cost_scope='selected-path attempts/returns only; root collection/reset/warmup excluded, not zero',
        no_predict_calls=0)
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',protocol)
    seals = [create_seal(files=files[i:i+40],args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
        output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
    write(output/'prepared.json',dict(schema=TAG+'.prepared',protocol_sha256=digest(protocol),source_seals=seals,
        status='physical_consumer_prepared_not_executed',new_neural_calls=0,new_physical_steps=0))
    load_prepared(output)
    return protocol


def load_prepared(output):
    output = Path(output).resolve()
    protocol = json.loads((output/'protocol.json').read_text())
    prepared = json.loads((output/'prepared.json').read_text())
    require(protocol['schema']==TAG+'.protocol' and prepared['schema']==TAG+'.prepared'
        and digest(protocol)==prepared['protocol_sha256'] and protocol['claim_boundary']==BOUNDARY
        and protocol['branches']==8 and protocol['total_planned_commands']==80,
        'physical prepared protocol binding mismatch')
    paths = []
    for seal in prepared['source_seals']:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        directory = Path(seal['output_dir'])
        require(json.loads((directory/'args.json').read_text())==dict(protocol_sha256=digest(protocol)),
            'physical source/protocol seal binding mismatch')
        paths += [e['source'] for e in json.loads((directory/'manifest.json').read_text())['files']]
    required = {str(Path(m.__file__).resolve()) for m in (sampling,helpers,durable)}
    required |= {str(Path(__file__).resolve()),str(PROJECT/'tests'/f'test_{TAG}.py'),str(PROJECT/'innerdocs'/f'{TAG}.md')}
    require(sorted(paths)==protocol['source_paths'] and required<=set(paths),'physical source closure missing')
    for p,h in protocol['bindings'].items():
        require(sha(p)==h,'physical bound input/source changed: '+p)
    context = inputs(protocol['planning'])
    require(context['plans']==protocol['plans'] and context['planning_totals']==protocol['planning_totals']
        and context['exporter_proof']==protocol['sealed_exporter']
        and context['verification_work']==protocol['exporter_verification_work'],
        'physical plans/inherited costs differ from source-bound exporter')
    require(all(protocol['bindings'].get(p)==h for p,h in context['bindings'].items()),'physical input binding closure missing')
    return output,protocol,prepared,context


def validate_plan(plan):
    require(plan['method'] in sampling.ARMS and plan['planned_horizon_tokens']==2
        and plan['planned_commands']==10 and plan['corrected_search_horizon']==2,'exact paired H2 arm required')
    require(digest({k:v for k,v in plan.items() if k!='selection_sha256'})==plan['selection_sha256'],
        'physical selection digest mismatch')
    raw = np.asarray(plan['commands'])
    require(raw.dtype.kind in 'fi' and raw.shape==(10,2) and np.isfinite(raw).all() and np.all(np.abs(raw)<=1),
        'exact bounded ten commands required')
    require(plan['witness']['receipt']['actions']==raw.tolist()
        and plan['witness']['receipt']['receipt_sha256']==plan['receipt_sha256'],
        'physical commands differ from sealed witness')


def physical_branch(*,plan,output,reset,env,state_of,evaluate):
    validate_plan(plan)
    return helpers.physical_branch(plan=plan,output=output,reset=reset,env=env,state_of=state_of,evaluate=evaluate)


def validate_roots(roots, refs, fingerprint):
    """Validate ALL state/goal bindings before any selected-path dispatch."""
    require(isinstance(roots,list) and len(roots)==4,'four actual physical roots required')
    result = {}
    for root in roots:
        require(set(root)=={'state','goal'},'complete physical state/goal root required')
        for field in ('state','goal'):
            v = np.asarray(root[field])
            require(v.dtype.kind in 'fi' and v.shape==(7,) and np.isfinite(v).all(),
                'finite seven-state and goal required upfront')
        key = fingerprint(root)
        require(key in refs and key not in result,'physical root/goal fingerprint mismatch or duplicate')
        result[key] = deepcopy(root)
    require(set(result)==set(refs),'physical root census differs; no replacement')
    return result


def validate_reset_history(history, normalization, expected):
    """Original warmup yields three raw constant [2] commands, not [10] tokens."""
    raw = np.asarray(history)
    require(raw.shape==(3,2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all()
        and np.all(np.abs(raw)<=1),'full physical raw warmup history required')
    mean,std = (np.asarray(normalization[k],dtype=np.float32) for k in ('mean','std'))
    require(mean.shape==std.shape==(2,) and np.isfinite([mean,std]).all() and np.all(std>0),
        'original per-command normalization required')
    packet = np.repeat(raw.astype(np.float32)[:,None,:],5,axis=1).reshape(3,10)
    tokens = (packet-np.tile(mean,5))/np.tile(std,5)
    require(np.array_equal(tokens[1:],np.asarray(expected)),'physical normalized between-state actions differ')
    return dict(raw_history=raw.tolist(),normalized_between_history=tokens[1:].tolist())


def optional_encodings(rows, output, setup):
    require((output/'physical_report.json').is_file(),'persist all physical branches before encoding')
    complete = [r for r in rows if r['full_horizon_completed'] and r['completed_commands']==10]
    if complete:
        return helpers.optional_encodings(rows,output,setup)
    # No complete endpoint: do not allocate CUDA just to write eight abstentions.
    receipts = []
    for row in rows:
        receipt = dict(root_sha256=row['root_sha256'],method=row['method'],repetition=0,
            status='blocked_fixed_horizon',attempted_encode_samples=0,
            failure='no complete ten-step endpoint; retain actual partial separately')
        write(output/f'root_{row["root_index"]}_{row["method"]}_replay0.encoding.json',receipt)
        receipts.append(receipt)
    return receipts


def physical_gates(plans, rows, output):
    by_key = {(p['root_sha256'],p['method']):p for p in plans}
    keys = [(r['root_sha256'],r['method']) for r in rows]
    return dict(all_eight_retained=len(rows)==len(set(keys))==8 and set(keys)==set(by_key),
        all_fixed_H2=all(r['full_horizon_completed'] and r['completed_commands']==10 for r in rows),
        trace_matches_selected=all(len(r['trace'])==r['completed_commands'] and
            [e['command'] for e in r['trace']]==by_key[(r['root_sha256'],r['method'])]['commands'][:r['completed_commands']]
            for r in rows),
        no_branch_errors=all(not r.get('failures') and r['status']=='completed' for r in rows),
        physical_persisted=(output/'physical_report.json').is_file())


def run(prepared, output, encode_terminals=False):
    _,protocol,sealed,context = load_prepared(prepared)
    plans = context['plans']
    for plan in plans:
        validate_plan(plan)
    base,checked = context['base'],context['checked']
    output = Path(output).resolve()
    require(TAG in output.name,'new tagged physical execution directory required')
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',dict(schema=TAG+'.execution',prepared=str(Path(prepared).resolve()),
        prepared_protocol_sha256=sealed['protocol_sha256'],plans=plans,encode_terminals=encode_terminals))
    rows,failures,encodings = [],[],[]
    env = model = before = host = None
    try:
        env = base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        require(str(Path(inspect.getfile(type(env.unwrapped))).resolve()) in protocol['environment_source_paths'],
            'environment class source not prepared')
        require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist())==(65,105,225),
            'unsupported renderer color')
        roots = base.collect_nonterminal_boundary_contact_roots(env,count=4,
            seed=int(context['upstream']['protocol']['root_seed']),oversample_factor=4)
        roots = validate_roots(roots,context['refs'],base.root_fingerprint)
        # Validate all evaluator goal/state contracts before the first selected command.
        for root in roots.values():
            official_checked(env,root['state'],root['goal'])
        for plan in plans:
            root = roots[plan['root_sha256']]
            goal = np.asarray(root['goal']).copy()
            def reset():
                require(base.root_fingerprint(root)==plan['root_sha256'] and np.array_equal(root['goal'],goal),
                    'physical root/goal changed before branch')
                frames,history,goal_rgb = base.physical._reset_and_warmup(env,root,history_len=3)
                identity = verify_rgb_context(frames,goal_rgb,context['refs'][plan['root_sha256']])
                history_identity = validate_reset_history(history,context['planning_protocol']['context']['normalization'],
                    plan['root']['context']['action_history'])
                identity.update(history_identity,state=base.p0.packets.state_of(env).tolist(),goal=goal.tolist())
                return identity
            def evaluate(state):
                return dict(official=official_checked(env,state,goal),pose=base.physical._pose_label(state,goal))
            rows.append(physical_branch(plan=plan,output=output,reset=reset,env=env,
                state_of=base.p0.packets.state_of,evaluate=evaluate))
    except (Exception,KeyboardInterrupt) as exc:
        failures.append(durable.failure(exc,'physical_setup_or_branch'))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failures.append(durable.failure(exc,'env_close'))
    helpers.materialize_missing(plans,rows,output)
    write(output/'physical_report.json',dict(schema=TAG+'.physical_before_encoding',rows=rows,
        failures=deepcopy(failures),planning_totals=protocol['planning_totals'],new_predict_calls=0,**deepcopy(BOUNDARY)))
    if encode_terminals:
        def setup():
            nonlocal model,before,host
            import torch
            from PIL import Image
            host = base.require_cuda_allocation('cuda')
            device = torch.device('cuda:0')
            model = base.load_compatible_pretrained(Path(context['planning_protocol']['context']['policy']),device=device).eval().requires_grad_(False)
            before = checked.tensor_state_hash(model)
            def forbid(*a,**kw):
                raise RuntimeError('physical endpoint encoding forbids all neural predict')
            model.predict = forbid
            transform = base.p0.image_transform()
            def encode(path):
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert('RGB')).copy()
                with torch.no_grad():
                    vector = model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                require(vector.device.type=='cuda' and torch.isfinite(vector).all(),'finite HOST CUDA encoding required')
                torch.cuda.synchronize(device)
                return vector.cpu().tolist()
            return encode
        encodings = optional_encodings(rows,output,setup)
    gates = physical_gates(plans,rows,output)
    gates['encodings_complete'] = not encode_terminals or (len(encodings)==8
        and {(e['root_sha256'],e['method']) for e in encodings}=={(p['root_sha256'],p['method']) for p in plans}
        and all(e['status']=='completed' for e in encodings))
    try:
        load_prepared(prepared)
        gates['sources_and_inputs_unchanged'] = True
    except Exception as exc:
        gates['sources_and_inputs_unchanged'] = False
        failures.append(durable.failure(exc,'final_source_check'))
    if model is not None:
        gates.update(model_unchanged=before==checked.tensor_state_hash(model),
            frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()))
    passed = all(gates.values()) and not failures
    write(output/'report.json',dict(schema=TAG+'.report',status='calibration_recorded' if passed else 'integrity_fail',
        prepared_protocol_sha256=sealed['protocol_sha256'],source_seals=sealed['source_seals'],
        rows=rows,encodings=encodings,failures=failures,gates=gates,host=host,new_predict_calls=0,
        selected_path_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        selected_path_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        attempted_terminal_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        planning_totals=protocol['planning_totals'],exporter_verification_work_per_load=protocol['exporter_verification_work'],
        runtime_source_revalidation_loads=2,
        extra_source_verification_scope='each load replays pure-array exporter; no neural predicts; not part of inherited planning budget',
        actual_guard_endpoints_may_differ=True,**deepcopy(BOUNDARY)))
    if not passed:
        raise RuntimeError('physical incomplete; retain all eight branches, no retry/reselection')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--planning',type=Path,default=DEFAULT_PLANNING)
    prep.add_argument('--output',type=Path,required=True)
    sub.add_parser('verify').add_argument('--output',type=Path,required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--prepared',type=Path,required=True)
    execute.add_argument('--output',type=Path,required=True)
    execute.add_argument('--encode-terminals',action='store_true')
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    if command=='prepare':
        prepare(**args)
    elif command=='verify':
        load_prepared(**args)
    else:
        run(**args)
    print(json.dumps(dict(status=command+'_completed',P2_admission=False)))


if __name__=='__main__':
    main()
