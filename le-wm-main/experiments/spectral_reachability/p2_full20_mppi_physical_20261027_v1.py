"""Independent consumer of four sealed MPPI-inspired H2 witnesses.

Prepare/validation are model-free and never instantiate a simulator. Run is
explicit main-thread physics, optional terminal encoding is HOST CUDA only.
No planning, predictions, resampling, old whitelist edits or scientific admission.
"""
import argparse
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from experiments.spectral_reachability import p2_full20_mppi_20261027_v1 as sampling
from experiments.spectral_reachability import p2_full20_physical_20261027_v2 as helpers
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_composition_witness_v2 import compose, digest
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context, official_checked

TAG = 'p2_full20_mppi_physical_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNING = PROJECT/'results/p2_full20_mppi_20261027_v1_prepared01'
sha = sampling.full20.sha
write = durable.write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(rows, key):
    require(isinstance(rows,list) and len(rows) == 4, 'all four roots required')
    result = {r[key]:r for r in rows}
    require(len(result) == 4, 'duplicate stable root identity')
    return result


def verify_kernel_trace(arm):
    """Reconstruct the saved proposal schedule with saved costs, NOT model work."""
    events = arm['attempts']
    require(len(events) == 40, 'all forty candidates required')
    offset = 0
    def saved_costs(batch):
        nonlocal offset
        values = []
        for proposal in batch:
            event = events[offset]
            require(event['candidate_id'] == f'{sampling.ARM}_{offset:02d}'
                and event['completed'] is True and event['proposal_dimensions'] == 20
                and event['packet_shape'] == [2,5,2], 'invalid frozen candidate identity/dimensions')
            raw = sampling.full20.raw_actions(proposal)
            require(raw == event['actions'] == sampling.full20.raw_actions(event['commands']),
                    'saved proposal differs from deterministic MPPI kernel trace')
            values.append(sampling.full20._cost(event['cost']))
            offset += 1
        return np.asarray(values)
    selected, receipt = sampling.kernels.mppi(saved_costs,initial=np.zeros((2,5,2)),
        lower=-1.,upper=1.,samples=12,iterations=3,temperature=1.,noise_scale=.5,seed=arm['seed'])
    require(offset == 40 and receipt == arm['kernel_receipt'], 'kernel receipt reconstruction mismatch')
    best = min(events,key=lambda e:e['cost'])
    require(arm['selected_candidate_id'] == best['candidate_id']
            and arm['selected_model_cost'] == best['cost']
            and sampling.full20.raw_actions(selected) == best['actions'], 'saved best-ever selection mismatch')
    return best['actions']


def plans_from_saved(report, protocol, prepared, compilation):
    require(report['schema'] == sampling.TAG+'.report' and report['status'] == 'calibration_completed'
        and report['protocol_sha256'] == digest(protocol) and report['source_seals'] == prepared['source_seals']
        and bool(report['gates']) and all(v is True for v in report['gates'].values()),
        'complete protocol/source-bound MPPI calibration required')
    require(protocol['schedule'] == sampling.SCHEDULE and protocol['arms'] == [sampling.ARM], 'fixed MPPI schedule required')
    source = indexed(compilation['rows'],'root_sha256')
    rows = indexed(report['rows'],'root_sha256')
    ready = indexed(protocol['prepared'],'root_sha256')
    require(source.keys() == rows.keys() == ready.keys(), 'root identity census mismatch')
    plans = []
    actual_samples = 0
    for key, reference in sorted(ready.items(), key=lambda item:item[1]['root_index']):
        row, root = rows[key], reference['root']
        compiled = source[key]['compiled']
        initial = compiled['input_context']
        expected_context = dict(layout='states_with_between_state_actions_v1',context_length=3,
            latent_dim=192,action_dim=10,latent_history=initial['emb'][0],action_history=initial['actions'][0][1:])
        require(root['root_id'] == key and root['context'] == expected_context and row['root'] == root
            and row['action_domain'] == reference['action_domain'] and row['root_index'] == reference['root_index'],
            'full root/goal/predictor context/domain mismatch')
        require(len(row['rows']) == 1 and row['rows'][0]['arm'] == sampling.ARM, 'one unique MPPI arm required')
        arm = row['rows'][0]
        require(arm['fixture_only'] is False and arm['prediction_budget_matched'] is True
            and arm['kernel_error'] is None and arm['replay_error'] is None
            and arm['planning_predict_samples'] == 80 and arm['replay_predict_samples'] == 4
            and arm['seed'] == reference['seed'], 'actual complete80+4 MPPI evidence required')
        require(sampling.full20.entries_match(arm['actual_model_entries'],arm['ledger'],sampling.ARM)
            and len(arm['ledger']) == 84 and [e['phase'] for e in arm['ledger']] == ['planning']*80+['replay']*4,
            'actual entry/dispatch ledger mismatch')
        actual = arm['actual_prediction_budget']
        provenance = sampling.roots_provenance(protocol,prepared,reference)
        require(actual['binding'] == dict(protocol_id=provenance['source_sha256'],task_id=sampling.TAG,
            root_id=key,frozen_model_id=provenance['checkpoint_sha256'])
            and actual['method'] == sampling.ARM and actual['predict_sample_limit'] == 84
            and actual['attempted_predict_samples'] == 84 and actual['blocked_calls'] == 0
            and [(p['phase'],p['predict_calls'],p['predict_samples']) for p in actual['phases']] ==
                [('planning',80,80),('evaluation',4,4)]
            and all(p['completed'] is True and p['failed_predict_calls'] == 0 for p in actual['phases'])
            and arm['action_encoder_work'] == dict(calls=84,samples=84,processed_context_tokens=252),
            'actual84 budget/binding or encoder work mismatch')
        actions = verify_kernel_trace(arm)
        saved = arm['witness']
        require(saved['independent_full_replay_exact'] is True, 'independent saved replay required')
        receipt, edges = saved['receipt'], saved['edges']
        require(len(edges) == 1 and receipt['root'] == root and receipt['provenance'] == provenance
            and receipt['action_domain'] == reference['action_domain'] and receipt['actions'] == actions,
            'selected receipt/root/actions/provenance mismatch')
        rebuilt = compose(root=root,provenance=provenance,action_domain=reference['action_domain'],
            edges=edges,replay=lambda c,a:deepcopy(receipt['terminal_context']),max_edges=1,max_actions=10)
        require(rebuilt == receipt and len(actions) == 10, 'exact H2 witness digest mismatch')
        require(abs(sampling.full20.metric_for(compiled)(receipt['terminal_context'])-arm['selected_model_cost']) <= 1e-12,
                'saved goal0/Q objective mismatch')
        plan = dict(root_index=reference['root_index'],root_sha256=key,method=sampling.ARM,
            commands=deepcopy(actions),planned_horizon_tokens=2,planned_commands=10,corrected_search_horizon=2,
            root=deepcopy(root),action_domain=deepcopy(reference['action_domain']),
            selected_candidate_id=arm['selected_candidate_id'],receipt_sha256=receipt['receipt_sha256'])
        plan['selection_sha256'] = digest(plan)
        plans.append(plan)
        actual_samples += sum(e['samples'] for e in arm['actual_model_entries'] if e['kind'] == 'predict')
    require(actual_samples == report['actual_predict_samples'] == 336, 'four-root actual336 census mismatch')
    return plans


def inputs(planning):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    path,protocol,prepared,source = sampling.load_prepared(planning)
    require((path/'report.json').is_file(), 'MPPI CUDA report missing; no physical plans may be prepared')
    report = json.loads((path/'report.json').read_text())
    plans = plans_from_saved(report,protocol,prepared,source)
    bindings = deepcopy(protocol['bindings'])
    for filename in ('protocol.json','prepared.json','report.json'):
        bindings[str(path/filename)] = sha(path/filename)
    reports = [json.loads(Path(p).read_text()) for p in protocol['bindings'] if Path(p).name == 'report.json']
    interventions = [r for r in reports if 'bank_path' in r.get('protocol',{})]
    require(len(interventions) == 1, 'unique physical root provenance required')
    upstream_protocol = interventions[0]['protocol']
    bank_path = Path(upstream_protocol['bank_path']).resolve()
    require(sha(bank_path) == upstream_protocol['bank_sha256'], 'physical bank SHA mismatch')
    bank = json.loads(bank_path.read_text())
    upstream_path = Path(bank['upstream_path']).resolve()
    require(sha(upstream_path) == upstream_protocol['upstream_sha256'], 'physical RGB provenance SHA mismatch')
    upstream = base.physical._load_preflight(upstream_path)
    refs = indexed(upstream['rows'],'root_sha256')
    require(set(refs) == {p['root_sha256'] for p in plans}, 'physical reference census mismatch')
    for plan in plans:
        reference = refs[plan['root_sha256']]
        require(plan['root']['root_rgb_sha256'] == reference['history_rgb_sha256']
            and plan['root']['goal_rgb_sha256'] == reference['goal_rgb_sha256'], 'physical root/goal RGB mismatch')
    for p in (bank_path,upstream_path): bindings[str(p)] = sha(p)
    _,_,config = checked.validate_inputs(protocol['compilation'],protocol['policy'])
    return base,checked,source,config,bindings,plans,upstream,refs,protocol


def environment_creator(base):
    """Resolve installed class source without gym.make/reset/step/render."""
    entry = base.gym.spec('swm/PushT-v1').entry_point
    if callable(entry): return entry
    module, name = entry.split(':')
    return getattr(importlib.import_module(module),name)


def prepare(planning,output):
    output = Path(output).resolve()
    require(TAG in output.name, 'exclusive tagged physical prepare directory required')
    if output.exists(): raise FileExistsError(output)
    base,checked,source,config,bindings,plans,upstream,refs,protocol = inputs(planning)
    creator = environment_creator(base)
    files = checked.sources_before_allocate(source,config)
    files += [Path(module.__file__) for name,module in tuple(sys.modules.items())
        if name.startswith('experiments.') and getattr(module,'__file__','').endswith('.py')]
    environment_sources = sorted({str(Path(inspect.getfile(obj)).resolve()) for obj in
        (creator,*[getattr(creator,name) for name in ('eval_state','reset','render','step')])})
    files += [Path(p) for p in environment_sources]
    files += [Path(__file__),PROJECT/'tests'/f'test_{TAG}.py',PROJECT/'docs'/f'{TAG}.md']
    files = sorted({p.resolve() for p in files})
    for path in environment_sources: bindings[path] = sha(path)
    value = dict(schema=TAG+'.protocol',planning=str(Path(planning).resolve()),bindings=bindings,plans=plans,
        environment_source_paths=environment_sources,physical_repetitions=1,planned_commands_per_branch=10,
        total_planned_commands=40,actual_planning_predict_samples_verified=336,
        physical_before_optional_encoding=True,root_selection_changed=False,extra_search_or_training=False,
        P1_admission=False,P2_admission=False,official_MPPI=False,claim_boundary=sampling.CLAIM,
        fair_total_budget_comparison=False,scope='four known calibration roots, MPPI-inspired selected paths only')
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',value)
    seals = [create_seal(files=files[i:i+40],args_json=json.dumps(dict(protocol_sha256=digest(value))),
                        output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
    write(output/'prepared.json',dict(schema=TAG+'.prepared',protocol_sha256=digest(value),source_seals=seals,
        status='physical_consumer_prepared_not_executed',new_neural_calls=0,new_physical_steps=0))
    load_prepared(output)
    print(json.dumps(dict(status='physical_consumer_prepared_not_executed',output=str(output),
        protocol_sha256=digest(value),branches=4,planned_commands=40,actual_planning_samples_verified=336)))


def load_prepared(output):
    output = Path(output).resolve()
    protocol = json.loads((output/'protocol.json').read_text())
    prepared = json.loads((output/'prepared.json').read_text())
    require(protocol['schema'] == TAG+'.protocol' and digest(protocol) == prepared['protocol_sha256'],
            'physical prepared protocol digest mismatch')
    for seal in prepared['source_seals']:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    for path,expected in protocol['bindings'].items():
        require(sha(path) == expected, 'physical bound input/source changed: '+path)
    context = inputs(protocol['planning'])
    require(context[5] == protocol['plans'], 'saved physical plans differ from MPPI witnesses')
    return output,protocol,prepared,context


def physical_branch(*,plan,output,reset,env,state_of,evaluate):
    require(plan['method'] == sampling.ARM, 'MPPI-inspired arm only')
    require(digest({k:v for k,v in plan.items() if k != 'selection_sha256'}) == plan['selection_sha256'],
            'physical plan selection digest mismatch')
    return helpers.physical_branch(plan=plan,output=output,reset=reset,env=env,state_of=state_of,evaluate=evaluate)


def run(prepared,output,encode_terminals=False):
    _,protocol,sealed,context = load_prepared(prepared)
    base,checked,source,config,bindings,plans,upstream,refs,planning_protocol = context
    output = Path(output).resolve()
    require(TAG in output.name, 'new tagged physical execution output required')
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',dict(prepared=str(Path(prepared).resolve()),
        prepared_protocol_sha256=sealed['protocol_sha256'],plans=plans,encode_terminals=encode_terminals))
    rows,failures,encodings = [],[],[]
    env = model = before = host = None
    try:
        env = base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        require(str(Path(inspect.getfile(type(env.unwrapped))).resolve()) in protocol['environment_source_paths'],
                'physical environment class source not prepared')
        require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) == (65,105,225),
                'unsupported renderer color')
        roots = base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        require(len(roots) == 4, 'all four physical roots required')
        roots = {base.root_fingerprint(root):root for root in roots}
        require(set(roots) == set(refs), 'physical roots differ; no replacement')
        for plan in plans:
            root = roots[plan['root_sha256']]
            def reset():
                require(base.root_fingerprint(root) == plan['root_sha256'], 'physical root changed')
                frames,history,goal = base.physical._reset_and_warmup(env,root,history_len=3)
                identity = verify_rgb_context(frames,goal,refs[plan['root_sha256']])
                identity.update(history=np.asarray(history).tolist(),state=base.p0.packets.state_of(env).tolist(),
                                goal=np.asarray(root['goal']).tolist())
                return identity
            def evaluate(state):
                return dict(official=official_checked(env,state,root['goal']),pose=base.physical._pose_label(state,root['goal']))
            rows.append(physical_branch(plan=plan,output=output,reset=reset,env=env,
                state_of=base.p0.packets.state_of,evaluate=evaluate))
    except (Exception,KeyboardInterrupt) as exc:
        failures.append(durable.failure(exc,'physical_setup_or_branch'))
    finally:
        if env is not None:
            try: env.close()
            except Exception as exc: failures.append(durable.failure(exc,'env_close'))
    helpers.materialize_missing(plans,rows,output)
    write(output/'physical_report.json',dict(schema=TAG+'.physical_before_encoding',rows=rows,
        failures=deepcopy(failures),new_predict_calls=0,P1_admission=False,P2_admission=False))
    if encode_terminals:
        def setup():
            nonlocal model,before,host
            import torch
            from PIL import Image
            host = base.require_cuda_allocation('cuda')
            device = torch.device('cuda:0')
            model = base.load_compatible_pretrained(Path(planning_protocol['policy']),device=device).eval().requires_grad_(False)
            before = checked.tensor_state_hash(model)
            def forbid(*a,**kw): raise RuntimeError('terminal encoding forbids predictions')
            model.predict = forbid
            transform = base.p0.image_transform()
            def encode(path):
                with Image.open(path) as image: rgb = np.asarray(image.convert('RGB')).copy()
                with torch.no_grad():
                    vector = model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                require(vector.device.type == 'cuda' and torch.isfinite(vector).all(), 'finite HOST CUDA vector required')
                torch.cuda.synchronize(device)
                return vector.cpu().tolist()
            return encode
        encodings = helpers.optional_encodings(rows,output,setup)
    gates = dict(all_four_retained=len(rows) == 4 and len({r['root_sha256'] for r in rows}) == 4,
        all_fixed_H2=all(r['full_horizon_completed'] and r['completed_commands'] == 10 for r in rows),
        trace_matches_selected=all(len(r['trace']) == r['completed_commands'] and
            [e['command'] for e in r['trace']] == next(p['commands'] for p in plans if p['root_sha256'] == r['root_sha256'])[:r['completed_commands']] for r in rows),
        no_branch_errors=all(not r.get('failures') for r in rows),
        physical_persisted=(output/'physical_report.json').is_file(),
        encodings_complete=not encode_terminals or len(encodings) == 4 and all(e['status'] == 'completed' for e in encodings))
    try:
        load_prepared(prepared)
        gates['sources_and_inputs_unchanged'] = True
    except Exception as exc:
        gates['sources_and_inputs_unchanged'] = False
        failures.append(durable.failure(exc,'final_source_check'))
    if model is not None:
        gates.update(model_unchanged=before == checked.tensor_state_hash(model),
            frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()))
    passed = all(gates.values()) and not failures
    write(output/'report.json',dict(schema=TAG+'.report',status='calibration_recorded' if passed else 'integrity_fail',
        prepared_protocol_sha256=sealed['protocol_sha256'],rows=rows,encodings=encodings,failures=failures,gates=gates,
        source_seals=sealed['source_seals'],host=host,new_predict_calls=0,
        selected_path_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        selected_path_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        attempted_terminal_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        P1_admission=False,P2_admission=False,physical_efficacy_verified=False,official_MPPI=False,
        fair_total_budget_comparison=False,actual_guard_endpoints_may_differ=True))
    if not passed: raise RuntimeError('MPPI physical incomplete; all branches retained; no retries/reselection')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--planning',type=Path,default=DEFAULT_PLANNING)
    prep.add_argument('--output',type=Path,required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--prepared',type=Path,required=True)
    execute.add_argument('--output',type=Path,required=True)
    execute.add_argument('--encode-terminals',action='store_true')
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    prepare(**args) if command == 'prepare' else run(**args)


if __name__ == '__main__':
    main()
