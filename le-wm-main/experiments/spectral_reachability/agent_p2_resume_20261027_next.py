"""Saved-only P2 frontier/contract integration and RAM reservation state machine.

No neural or physics dispatch API. Common supported-axis caps != equal FLOPs.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
from pathlib import Path

from experiments.spectral_reachability import agent_p2_resume_20261027_witness_queue as prior
from experiments.spectral_reachability import p2_composition_predict_frontier_20261027_v1 as frontier
from experiments.spectral_reachability import p2_full_resource_matching_20261027_v1 as cost
from experiments.spectral_reachability import p2_supported_resource_guard_20261027_v1 as guard

NAME = 'agent_p2_resume_20261027_next'
ROOT = prior.ROOT
ARMS = cost.ARMS
COUNTS = dict(zip(ARMS, (80, 40)))
ENV = ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
ENV_SHA = 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'
DEFAULT_WITNESS = ROOT/'results/agent_p2_resume_20261027_prepared_v1.json'
WITNESS_SHA = 'e7b63325cb6e2995a86b5430256f289156d904bb7bece5ffea339c222675d839'
require = cost.require
digest = cost.digest
BLOCKERS = [
    'independent_P1_MAIN_authority_missing', 'freshroot_and_complete_exclusions_missing',
    'prospective_blind_task_goal7_and_renderer_restore_authority_missing',
    'full_cost_instrumentation_and_durable_runner_guard_missing',
    'MAIN_prospective_protocol_and_HOST_CUDA_physics_review_missing',
]


def native_contract(path=ENV):
    """Inspect sealed installed source as TEXT only; never import the simulator."""
    require(prior.file_sha(path) == ENV_SHA, 'native PushT source drift')
    text = Path(path).read_text(); tree = ast.parse(text)
    methods = {n.name: ast.get_source_segment(text, n) for cls in tree.body
               if isinstance(cls, ast.ClassDef) for n in cls.body if isinstance(n, ast.FunctionDef)}
    return dict(source=dict(path=str(Path(path).resolve()), sha256=ENV_SHA),
        native_methods={k: methods[k] for k in ('__init__', 'step', 'eval_state')},
        goal=dict(planner='original frozen goal latent/Q; never object-only proxy',
                  task='original goal_state[7] supplied by exact native reset/warmup',
                  physical_goal7=None, renderer_goal_pose='outline only; do not substitute task goal',
                  mutation_allowed=False),
        success=dict(position='norm(goal_state[:4]-state[:4]) < 20',
                     angle='min(abs(goal[4]-state[4]), 2*pi-abs(goal[4]-state[4])) < pi/9',
                     positions_include='agent_xy AND object_xy', velocities_in_success=False,
                     raw_angle_preserved=True, wrapped_angle='diagnostic only; never relabel native success'),
        controller=dict(relative=True, action_scale=100, k_p=100, k_v=20, dt=.01,
                        control_hz=10, internal_substeps_per_public_step=10,
                        native_step_clips=False, raw_domain=[-1, 1],
                        commands='exact float32[10,2]; no extra clip/tanh/normalization',
                        warmup_public_steps=15, selected_public_steps=10,
                        early_success='record native terminated; fixed horizon requires MAIN-reviewed wrapper'),
        restore=dict(require='exact original state, full history, warmup, goal7, renderer and wrapper identity',
                     approximate_restore_allowed=False, currently_certified=False))


def allocation():
    protocol = cost.next_protocol(); cap = protocol['common_per_root_cap']
    recipes = {a: cost.candidate_recipe(a) for a in ARMS}
    shared = cost.forward(4)
    shared.update(jacobian_192x20_calls=1, math_vjp_192x20_directions=192,
                  jacobian_scalar_entries=3840, svd_192x20_calls=2, bounded_solve_20_calls=1)
    replay = cost.forward(4)
    execution = cost.vector(visual_encoder_calls=1, projector_calls=1,
                           warmup_steps=15, selected_steps=10, warmup_resets=1)
    arms = {}
    for arm, recipe in recipes.items():
        got = guard.allocate(cap, recipe['fixed'], recipe['per_useful_candidate_cap'], COUNTS[arm])
        require(got['candidates'] == COUNTS[arm], 'fixed useful candidate denominator')
        require(cost.add(shared, replay, execution) == recipe['fixed'], 'base/fixed allocation closure')
        if arm == ARMS[0]:
            edges = [dict(cost.forward(2), selection_evaluations=1)]
        else:
            prefix = cost.forward(3)
            prefix.update(jacobian_192x10_calls=1, math_vjp_192x10_directions=192,
                          jacobian_scalar_entries=1920, svd_192x10_calls=1, bounded_solve_10_calls=1)
            edges = [prefix, dict(cost.forward(1), selection_evaluations=1)]
        require(cost.add(*edges) == recipe['per_useful_candidate_cap'], 'edge evaluation allocation closure')
        arms[arm] = dict(candidate_count=COUNTS[arm], allocation=got, edge_evaluation_upper=edges,
                         per_candidate_upper=recipe['per_useful_candidate_cap'])
    return dict(common_standalone_cap=cap, graph_base_shared_upper=shared,
        winner_replay_upper=replay, native_execution_and_endpoint_upper=execution, arms=arms,
        standalone_predict_upper=[168,168], paired_unique_predict_upper=332,
        cohort_unique_predict_upper=1328, supported_axes=list(cost.KNOWN), unknown_axes=list(cost.UNKNOWN),
        accounting='shared graph base once in paired unique, fully in BOTH standalone ledgers; never sum standalone as unique',
        edge_scope='candidate generation/evaluation including suffix compile; NOT another charge for traversing saved edges',
        inherited_acquisition_cost=None, root_collection_cost=None,
        zero_radius='suffix SVD/solve caps are upper bounds; actual zero retained, reservation never refunded',
        claim='common_supported_componentwise_caps; normal-path prediction matched only',
        full_resource_equal=False, padding=False, transfer_unused_budget=False)


def files():
    return [Path(__file__).resolve(), ROOT/'tests'/(NAME+'_tests.py'), ROOT/'innerdocs'/(NAME+'.md'),
            Path(prior.__file__), Path(frontier.__file__), Path(cost.__file__), Path(guard.__file__),
            ROOT/'experiments/spectral_reachability/p2_composition_predict_frontier_cuda_20261027_v2.py',
            ROOT.parent/'docs/D2P_FIXED_CAPABILITY_DIRECTION_RESULT_MAIN_20261027.md', ENV]


def build(witness, banks, task_contract):
    """No winners/outcomes read. Keep old320 witnesses distinct from future480 slots."""
    g = witness['program_graph']; roots = g['root_slots']
    require(g['candidate_denominator'] == len(g['candidates']) == 320 and len(roots) == 4,
            'complete original known-root witness denominator')
    require([r['identity']['root_sha256'] for r in roots] == list(frontier.ROOTS), 'fixed saved known-root order')
    require(len(banks) == 4, 'all four saved banks required')
    resources = allocation(); slots = []
    for index, (root, bank) in enumerate(zip(roots, banks, strict=True)):
        ident = root['identity']; expanded = frontier.bound_candidate_bank(bank, ident['root_sha256'], 80)
        require(bank['goal_sha256'] == ident['goal_vector_sha256'] and
                bank['projection_sha256'] == ident['projection_sha256'], 'saved bank goal/projection identity')
        for arm in ARMS:
            old = [r for r in g['candidates'] if r['identity'] == ident and r['arm'] == arm]
            require([r['candidate_index'] for r in old] == list(range(40)), 'original candidate census/order')
            for j, record in enumerate(old):
                commands = [a for edge in record['edges'] for a in edge['actions']]
                if arm == ARMS[0]:
                    require(commands == [a for packet in expanded['commands'][j] for a in packet], 'original40 action seam')
                else:
                    require(commands[:5] == expanded['commands'][j][0], 'conditional original prefix seam')
            prov = old[0]['edges'][0]['provenance']
            candidates = [dict(index=j, candidate_id=f'{arm}_{j:02d}', status='not_dispatched',
                # Conditional suffix is future model-compiled, never pretend the base suffix is it.
                commands=([a for p in expanded['commands'][j] for a in p] if arm == ARMS[0] else None),
                prefix_commands=expanded['commands'][j][0],
                old_saved_receipt_sha256=old[j]['native_receipt_sha256'] if j < 40 else None)
                for j in range(COUNTS[arm])]
            slots.append(dict(root_slot=index, identity=deepcopy(ident), arm=arm, original_saved_provenance=deepcopy(prov),
                future_bank_sha256=expanded['candidate_bank_sha256'], task_id=None, blind_sha256=None,
                candidate_denominator=COUNTS[arm], candidates=candidates, status='not_dispatched'))
    return dict(schema=NAME+'.plan', resources=resources, native_contract=task_contract, slots=slots,
        old_witness_candidate_denominator=320, future_candidate_denominator=480, root_denominator=4, arm_denominator=8,
        known_roots_only=True, candidate_selection_rule=frontier.contract()['selection'],
        algorithm='sealed frontier CUDA v2 joint80/conditional40; this adapter does NOT dispatch or modify its kernel',
        queue_audit=deepcopy(g['queue_audit']),
        blockers=BLOCKERS, execution_authorized=False, engineering_contract_ready=True,
        strict_full_resource_ready=False, P2_admission=False, scientific_pass=False)


def prepare(path=DEFAULT_WITNESS, expected=WITNESS_SHA):
    path = Path(path).resolve(); require(prior.file_sha(path) == expected, 'external witness file SHA')
    w = prior.verify(prior.read(path), expected)
    directory = Path(w['inputs']['composition']['directory'])
    banks = [prior.read(directory/f'root_{i}_bank.json') for i in range(4)]
    bindings = dict(w['bindings']); bindings[str(path)] = expected
    for p in files(): bindings[str(p.resolve())] = prior.file_sha(p)
    for i in range(4):
        p = directory/f'root_{i}_bank.json'; bindings[str(p.resolve())] = prior.file_sha(p)
    plan = build(w, banks, native_contract())
    for name, h in bindings.items(): require(prior.file_sha(name) == h, 'prepare input/source drift')
    return dict(schema=NAME+'.prepare', witness_path=str(path), witness_sha256=expected,
                plan=plan, bindings=bindings, new_neural_calls=0, new_physics_steps=0,
                execution_authorized=False, P2_admission=False)


def verify(value, expected):
    prior.graph.digest(expected)
    require(digest(value) == expected and value['schema'] == NAME+'.prepare', 'external prepare SHA/schema')
    for name, h in value['bindings'].items(): require(prior.file_sha(name) == h, 'input/source drift')
    rebuilt = prepare(value['witness_path'], value['witness_sha256'])
    require(rebuilt == value, 'saved contract/frontier/resource drift')
    return rebuilt


class ArmQueue:
    """RAM engineering state machine, NOT dispatch permission or durable journal.

    Each instance is standalone attribution, so shared base is charged in full.
    Freeze receipt BEFORE dispatch in a future MAIN runner; no such runner here.
    """
    def __init__(self, arm, identity, model_sha256, protocol_sha256):
        require(arm in ARMS, 'fixed arm')
        r = allocation(); self._arm = arm; self._identity = deepcopy(identity)
        self._protocol = protocol_sha256
        self._ledger = guard.BudgetLedger(r['common_standalone_cap'], dict(
            root_sha256=identity['root_sha256'], model_sha256=model_sha256, protocol_sha256=protocol_sha256,
            task_sha256=digest(dict(scope='pending_task_contract_NOT_authority', identity=identity))))
        self._rows = [dict(work_id='base', purpose='candidate_generation', upper=r['graph_base_shared_upper'])]
        self._rows += [dict(work_id=f'candidate_{i:02d}', purpose='candidate_evaluation',
                           upper=r['arms'][arm]['per_candidate_upper']) for i in range(COUNTS[arm])]
        self._rows += [dict(work_id='replay', purpose='winner_replay', upper=r['winner_replay_upper']),
                       dict(work_id='execution', purpose='physical_execution', upper=r['native_execution_and_endpoint_upper'])]
        for row in self._rows: row.update(status='pending', actual=None, error=None, receipt_sha256=None)

    def begin(self):
        require(not any(r['status'] in ('attempted','failed','skipped') for r in self._rows), 'pending terminal or stopped arm')
        row = next((r for r in self._rows if r['status'] == 'pending'), None)
        require(row is not None, 'queue exhausted')
        self._ledger.reserve(row['work_id'], row['upper'], row['purpose'])
        row['status'] = 'attempted'
        return deepcopy(row)

    def finish(self, actual, receipt_sha256, error=None):
        row = next((r for r in self._rows if r['status'] == 'attempted'), None)
        require(row is not None, 'one attempted work required')
        prior.graph.digest(receipt_sha256)
        require(error is None or type(error) is str and bool(error.strip()), 'explicit failure reason')
        self._ledger.finish(row['work_id'], actual)  # Guard checks BEFORE state changes.
        row.update(status='completed' if error is None else 'failed', actual=deepcopy(actual),
                   error=error, receipt_sha256=receipt_sha256)
        if error is not None: self.stop('prior_work_failed: '+error)

    def stop(self, reason):
        require(type(reason) is str and bool(reason.strip()), 'explicit stop reason')
        for row in self._rows:
            if row['status'] == 'pending': row.update(status='skipped', error=reason)
        # A pending attempt retains its reservation and unknown completion.

    def receipt(self):
        ledger = self._ledger.receipt()
        return dict(arm=self._arm, identity=deepcopy(self._identity), protocol_sha256=self._protocol,
            candidate_denominator=COUNTS[self._arm],
            candidates=[deepcopy(r) for r in self._rows if r['work_id'].startswith('candidate_')],
            all_work=deepcopy(self._rows), resource_ledger=ledger,
            completion_count_scope='ledger.completed is a returned-work LOWER BOUND, not missing-attempt completion',
            attempted_completion_total=None if ledger['pending'] else deepcopy(ledger['completed']),
            actual_dispatch_verified=False, execution_authorized=False, P2_admission=False)


def new_queues(plan, protocol_sha256):
    """Instantiate all eight RAM attribution queues before any attempts."""
    prior.graph.digest(protocol_sha256)
    require(len(plan['slots']) == 8, 'fixed eight arm slots')
    return [ArmQueue(s['arm'], s['identity'], s['original_saved_provenance']['checkpoint_sha256'],
                     protocol_sha256) for s in plan['slots']]


def census_queues(plan, queues):
    """Keep unstarted, pending, failed and skipped rows; no successful-only export."""
    require(len(queues) == len(plan['slots']) == 8, 'complete eight arm denominator')
    rows = [q.receipt() for q in queues]
    for slot, row in zip(plan['slots'], rows, strict=True):
        require(slot['arm'] == row['arm'] and slot['identity'] == row['identity'] and
                row['candidate_denominator'] == len(row['candidates']) == COUNTS[slot['arm']],
                'root/arm/full candidate census binding')
    require(len({r['protocol_sha256'] for r in rows}) == 1, 'single protocol required')
    return dict(schema=NAME+'.ram_census', rows=rows, root_denominator=4, arm_denominator=8,
                candidate_denominator=480, actual_dispatch_verified=False, execution_authorized=False,
                P2_admission=False, scope='RAM reservation receipts, NOT scientific or actual runtime evidence')


def consume(value, expected, *, confirmatory=False):
    if confirmatory: raise ValueError('confirmatory blocked: '+', '.join(BLOCKERS))
    p = verify(value, expected); plan = p['plan']
    return dict(schema=NAME+'.consume', prepare_sha256=expected,
        status='saved_frontier_contract_verified_NOT_executed', slots=plan['slots'],
        root_denominator=4, arm_denominator=8, candidate_denominator=480,
        resources=plan['resources'], blockers=BLOCKERS, new_neural_calls=0, new_physics_steps=0,
        execution_authorized=False, P2_admission=False, scientific_pass=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','verify','run-saved','consume'))
    parser.add_argument('--witness', default=str(DEFAULT_WITNESS)); parser.add_argument('--witness-sha', default=WITNESS_SHA)
    parser.add_argument('--prepared'); parser.add_argument('--prepare-sha'); parser.add_argument('--output')
    parser.add_argument('--confirmatory', action='store_true'); a = parser.parse_args(argv)
    if a.confirmatory: raise ValueError('confirmatory blocked: '+', '.join(BLOCKERS))
    if a.command == 'prepare': value = prepare(a.witness, a.witness_sha)
    elif a.command == 'verify': value = verify(prior.read(a.prepared), a.prepare_sha)
    else: value = consume(prior.read(a.prepared), a.prepare_sha)
    if a.output:
        require(Path(a.output).name.startswith(NAME), 'exclusive NEXT output stem required')
        prior.write(a.output, value)
    print(prior.graph.encode(dict(status=value.get('status','saved_prepared_or_verified'),
        sha256=digest(value), execution_authorized=False, P2_admission=False)).decode())
    return 0


if __name__ == '__main__': raise SystemExit(main())
