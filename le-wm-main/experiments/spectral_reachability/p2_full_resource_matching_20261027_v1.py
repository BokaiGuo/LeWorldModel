"""Pure resource arithmetic, source-bound saved receipts, no neural/simulator imports."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

NAME = 'p2_full_resource_matching_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
ARMS = ('joint_h2_no_composition', 'conditional_suffix_composition')
PLANNER = 'results/p2_composition_predict_frontier_cuda_20261027_v1_main01'
PHYSICAL = 'results/p2_composition_predict_frontier_physical_20261027_v1_main01'
PLANNER_SHA = '4023a0e3896a90407188d260a45dd334e0802ba17ecd59ef62ea931603eba173'
PHYSICAL_SHA = '9144c3f25f2880af62d94dd88257918fa508b5ddbea51d98e9ccbc3d3b7532e2'
FORWARD = ('predict_calls','predict_samples','predict_context_tokens',
    'action_encoder_calls','action_encoder_samples','action_encoder_context_tokens')
KNOWN = FORWARD + ('jacobian_192x20_calls','jacobian_192x10_calls',
    'math_vjp_192x20_directions','math_vjp_192x10_directions','jacobian_scalar_entries',
    'svd_192x20_calls','svd_192x10_calls','bounded_solve_20_calls','bounded_solve_10_calls',
    'selection_evaluations','visual_encoder_calls','projector_calls',
    'root_collection_steps','warmup_steps','selected_steps','root_collection_resets','warmup_resets')
UNKNOWN = ('backward_engine_api_calls','cuda_kernel_calls','flops','synchronized_device_time_ns',
    'peak_gpu_bytes','cpu_validation_time_ns','inherited_context_compute_units','physics_internal_substeps')
AXES = KNOWN + UNKNOWN


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',',':'))


def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def vector(**values):
    """Zeros mean observed/declared absence WITHIN the explicit work scope only."""
    out = dict.fromkeys(KNOWN,0) | dict.fromkeys(UNKNOWN,None)
    require(not set(values)-set(AXES), 'unknown resource axis')
    out.update(values); validate(out); return out


def validate(v):
    require(isinstance(v,dict) and set(v) == set(AXES), 'complete versioned resource vector required')
    require(all(x is None or type(x) is int and x >= 0 for x in v.values()), 'nonnegative literal count or unknown; bool forbidden')


def add(*vectors):
    """Add disjoint non-peak work; unknown propagates, peak bytes take max."""
    require(bool(vectors), 'nonempty disjoint vectors')
    for v in vectors: validate(v)
    return {k: None if any(v[k] is None for v in vectors) else
        (max(v[k] for v in vectors) if k == 'peak_gpu_bytes' else sum(v[k] for v in vectors)) for k in AXES}


def sum_disjoint(charges):
    require(charges and len({c['work_id'] for c in charges}) == len(charges), 'shared work identity cannot be charged twice')
    return add(*(c['vector'] for c in charges))


def compare(left, right):
    validate(left); validate(right)
    unknown = [k for k in AXES if left[k] is None or right[k] is None]
    unequal = {k:dict(left=left[k],right=right[k]) for k in AXES
        if k not in unknown and left[k] != right[k]}
    frontier = all(left[k] is not None and left[k] == right[k] for k in FORWARD)
    return dict(predict_frontier_equal=frontier, full_resource_equal=not unknown and not unequal,
        measured_vector_equal=not unequal, differences=unequal, unknown_axes=unknown,
        status='known_vector_mismatch' if unequal else 'unknown_full_cost' if unknown else 'equal_declared_vector',
        P2_admission=False, hardware_cost_equivalence_certified=False)


def validate_claim(left, right, claim):
    require(claim in ('predict_frontier_equal','full_resource_equal'), 'explicit supported comparison label')
    result = compare(left,right)
    require(result[claim], 'unsupported '+claim+': known differences or unknown axes; 168/168 is not total compute')
    return result


def forward(n):
    return vector(predict_calls=n,predict_samples=n,predict_context_tokens=3*n,
        action_encoder_calls=n,action_encoder_samples=n,action_encoder_context_tokens=3*n)


def candidate_recipe(arm):
    """Prospective conservative units. No dispatch and no padding operation.

    Conditional solve/SVD is a positive-radius upper bound, NOT an actual count
    when a future candidate has zero remaining radius.
    """
    require(arm in ARMS, 'supported unmodified algorithm family')
    fixed = add(forward(8), vector(jacobian_192x20_calls=1,math_vjp_192x20_directions=192,
        jacobian_scalar_entries=3840,svd_192x20_calls=2,bounded_solve_20_calls=1,
        visual_encoder_calls=1,projector_calls=1,warmup_steps=15,selected_steps=10,warmup_resets=1))
    unit = forward(2 if arm == ARMS[0] else 4)
    unit['selection_evaluations'] = 1
    if arm == ARMS[1]:
        unit.update(jacobian_192x10_calls=1,math_vjp_192x10_directions=192,
            jacobian_scalar_entries=1920,svd_192x10_calls=1,bounded_solve_10_calls=1)
    return dict(arm=arm, fixed=fixed, per_useful_candidate_cap=unit,
        scope='per-root standalone, includes own endpoint/warmup; cohort root collection shared separately',
        predicted_caps_not_measurements=True)


def allocate(cap, fixed, unit, requested):
    """Max useful candidates under EVERY known component. Never fill unused budget."""
    for v in (cap,fixed,unit): validate(v)
    require(type(requested) is int and requested >= 1, 'finite positive candidate request')
    require(all(v[k] is not None for v in (cap,fixed,unit) for k in KNOWN), 'all measured-axis caps/units required')
    require(any(unit[k] > 0 for k in KNOWN), 'nonzero useful work unit')
    require(all(fixed[k] <= cap[k] for k in KNOWN), 'mandatory work exceeds cap; no partial replay or endpoint trimming')
    n = min([requested]+[(cap[k]-fixed[k])//unit[k] for k in KNOWN if unit[k]])
    used = {k:fixed[k]+n*unit[k] for k in KNOWN}
    return dict(candidates=n, requested=requested, feasible=n >= 1, measured_cap_usage=used,
        unspent={k:cap[k]-used[k] for k in KNOWN}, unknown_full_cost_axes=list(UNKNOWN),
        strict_full_resource_ready=False, no_padding=True)


class BudgetLedger:
    """Pre-dispatch reservations, explicit completion lower bounds, never refunds.

    This is an adapter for a FUTURE runner, not evidence that work executed.
    Reserve primitive work units; partial failures can report partial completions.
    """
    PURPOSES = ('candidate_generation','candidate_evaluation','winner_replay','endpoint_encoding','physical_execution')
    def __init__(self, cap, bindings):
        validate(cap)
        require(all(cap[k] is not None for k in KNOWN), 'measured budget caps required')
        require(set(bindings) == {'root_sha256','task_sha256','model_sha256','protocol_sha256'} and
            all(isinstance(x,str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x) for x in bindings.values()),
            'frozen root/task/model/protocol bindings required')
        self.cap,self.bindings = deepcopy(cap),deepcopy(bindings)
        self.reserved,self.completed,self.events = vector(),vector(),[]
    def reserve(self, work_id, work, purpose):
        validate(work)
        require(purpose in self.PURPOSES and isinstance(work_id,str) and work_id and
            work_id not in {e['work_id'] for e in self.events}, 'useful purpose, unique work identity; padding forbidden')
        require(all(work[k] is not None for k in KNOWN) and any(work[k] for k in KNOWN), 'explicit nonzero primitive work')
        require(all(self.reserved[k]+work[k] <= self.cap[k] for k in KNOWN), 'predispatch resource cap exceeded')
        for k in KNOWN: self.reserved[k] += work[k]
        self.events.append(dict(work_id=work_id,purpose=purpose,reserved=deepcopy(work),completed=None,bindings=deepcopy(self.bindings)))
    def finish(self, work_id, completed):
        validate(completed)
        matches = [e for e in self.events if e['work_id'] == work_id]
        require(len(matches) == 1 and matches[0]['completed'] is None, 'one terminal receipt per reservation')
        e = matches[0]
        require(all(completed[k] is not None and completed[k] <= e['reserved'][k] for k in KNOWN), 'completion cannot exceed reserved work')
        for k in KNOWN: self.completed[k] += completed[k]
        e['completed'] = deepcopy(completed)
        # Deliberately leave reservation charged after failure/partial completion.
    def receipt(self):
        return dict(bindings=deepcopy(self.bindings),reserved=deepcopy(self.reserved),
            completed=deepcopy(self.completed),events=deepcopy(self.events),
            pending=[e['work_id'] for e in self.events if e['completed'] is None],
            scope='pure reservation arithmetic; not proof of actual neural/physics dispatch',
            full_resource_equal=False,P2_admission=False)


def next_protocol():
    recipes = [candidate_recipe(a) for a in ARMS]
    cap = vector(**{k:max(r['fixed'][k]+n*r['per_useful_candidate_cap'][k] for r,n in zip(recipes,(80,40))) for k in KNOWN})
    allocations = {r['arm']:allocate(cap,r['fixed'],r['per_useful_candidate_cap'],n) for r,n in zip(recipes,(80,40))}
    return dict(status='prospective_common_cap_design_not_execution', common_per_root_cap=cap, allocations=allocations,
        strict_equal_vector_feasible_for_positive_candidates=False,
        infeasibility_certificate='joint has zero exclusive 192x10 Jacobians; conditional has n>=1. Equal derivative histogram forces n=0; without padding no positive pair.',
        minimum_change='Freeze NEW root/task/model/domain and useful method configs, common componentwise caps, per-operation attempt/completion journals, common backend/hardware and synchronized timing. Report cap-matched frontier, NOT equal actual vector.',
        full_cost_prerequisites=list(UNKNOWN), P2_admission=False, new_neural_calls=0,new_physical_steps=0)


def count_operations(ops, results):
    require(len(ops)%2 == 0, 'request/terminal operation pairs required')
    shared,conditional = vector(),vector()
    locals_ = {e['index']:e['suffix_compilation'] for r in results if r['arm'] == ARMS[1] for e in r['attempts']}
    for request,terminal in zip(ops[::2],ops[1::2]):
        require(request['kind'] == terminal['kind'] and request.get('index') == terminal.get('index') and
            request['completed'] is False and terminal['completed'] is True, 'complete nonrefundable operation pair')
        kind = request['kind']; v = conditional if kind.startswith('suffix_') else shared
        if kind in ('jacobian','suffix_jacobian'):
            width = 20 if kind == 'jacobian' else 10
            require(request['shape'] == terminal['shape'] == [192,width] and
                request['vectorize'] is True and terminal['vectorize'] is True, 'explicit derivative dimensions/backend contract')
            v[f'jacobian_192x{width}_calls'] += 1
            v[f'math_vjp_192x{width}_directions'] += 192
            v['jacobian_scalar_entries'] += 192*width
        elif kind in ('bounded_ridge_svd','mode_svd'):
            require(request['svd_calls_requested'] == 1, 'root SVD request')
            v['svd_192x20_calls'] += 1
            if kind == 'bounded_ridge_svd': v['bounded_solve_20_calls'] += 1
        else:
            require(kind == 'suffix_solve', 'unknown operation cannot be silently dropped')
            local = locals_[request['index']]
            require(len(local['jacobian']) == 192 and all(len(r) == 10 for r in local['jacobian']), 'saved suffix response dimensions')
            require(request['svd_calls_requested'] == terminal['svd_calls'] == local['svd_calls'] and
                terminal['bounded_solve_calls'] == local['bounded_solve_calls'] and
                type(local['svd_calls']) is int and local['svd_calls'] in (0,1) and
                type(local['bounded_solve_calls']) is int and local['bounded_solve_calls'] == local['svd_calls'], 'actual zero-radius/nonzero-radius solve census')
            v['svd_192x10_calls'] += local['svd_calls']; v['bounded_solve_10_calls'] += local['bounded_solve_calls']
    return shared,conditional


def audit_current(project=PROJECT):
    """Explicit saved resource allowlist; never read physical scores for allocation."""
    project=Path(project).resolve(); planning=project/PLANNER; physical=project/PHYSICAL; bindings={}
    def bound(path, expected=None):
        h=sha(path); require(expected is None or h == expected, 'source bytes changed: '+str(path))
        bindings[str(path)]=h; return read(path)
    report=bound(planning/'report.json',PLANNER_SHA); execution=bound(physical/'report.json',PHYSICAL_SHA)
    require(report['status'] == 'cuda_completed' and report['model_unchanged'] and execution['status'] == 'guarded_calibration_recorded', 'completed actual inputs')
    prepared_path=Path(report['prepared']); protocol=bound(prepared_path/'protocol.json'); prepared=bound(prepared_path/'prepared.json')
    require(digest(protocol) == report['protocol_sha256'] and digest(prepared) == report['prepared_sha256'], 'planner prepare identity')
    require(bound(planning/'execution.json') == {k:report[k] for k in ('prepared','prepared_sha256','protocol_sha256')}, 'actual execution binding')
    for seal in prepared['source_seals']:
        directory=Path(seal['output_dir']); manifest=bound(directory/'manifest.json',seal['manifest_sha256'])
        for f in manifest['files']:
            require(sha(directory/f['archive']) == f['sha256'] and sha(f['source']) == f['sha256'], 'archived and current source binding')
    proots={r['root_sha256']:r for r in report['rows']}; require(len(proots) == 4, 'four unique planning roots')
    roots=[]; shared_all=[]; exclusive_all={a:[] for a in ARMS}
    for root,receipt in proots.items():
        i=receipt['root_index']; row=bound(planning/f'root_{i}.json')
        require(digest(row) == receipt['row_sha256'] and row['root_sha256'] == root and row['complete'] and row['error'] is None, 'stable root row content binding')
        bank=bound(planning/f'root_{i}_bank.json',row['bank_sha256'])
        require(len(bank['base']['jacobian']) == 192 and all(len(r) == 20 for r in bank['base']['jacobian']), 'saved root response dimensions')
        journal_path=planning/f'root_{i}_journal.jsonl';require(sha(journal_path) == row['journal_sha256'], 'actual journal binding');bindings[str(journal_path)]=row['journal_sha256']
        journal=[json.loads(line) for line in journal_path.read_text().splitlines()]
        attempts=[e['entry'] for e in journal if e.get('kind') == 'entry_attempt']
        terminals=[e['entry'] for e in journal if e.get('kind') == 'entry_terminal']
        require(terminals == row['actual_model_entries'] and len(attempts) == len(terminals) == 664 and
            all(a == dict(t,completed=False) for a,t in zip(attempts,terminals)), 'actual attempts/completions, no refund')
        ops=[e for e in journal if e.get('kind') in ('jacobian','suffix_jacobian','bounded_ridge_svd','mode_svd','suffix_solve')]
        require(ops == row['operations'], 'AD/solve operations from actual journal')
        shared,cond=count_operations(ops,row['results']); exclusive={ARMS[0]:vector(),ARMS[1]:cond}
        for e in terminals:
            require(e['completed'] is True and e['samples'] == 1 and e['context_tokens'] == 3 and not e.get('error'), 'completed single-sample module entry')
            scope=e['scope']; v=shared if scope['phase'] == 'compilation' else exclusive[scope['arm']]
            prefix='predict' if e['kind'] == 'predict' else 'action_encoder'
            require(e['kind'] in ('predict','action_encoder'), 'unaccounted neural entry')
            for suffix,n in (('calls',1),('samples',1),('context_tokens',3)): v[prefix+'_'+suffix] += n
        selections={r['arm']:r for r in row['results']}
        require(set(selections) == set(ARMS), 'complete two arm results')
        for a,n in zip(ARMS,(80,40)):
            require(len(selections[a]['attempts']) == selections[a]['selection_evaluations'] == n and
                all(e['completed'] and e['metric_attempted'] for e in selections[a]['attempts']), 'all80/40 useful candidates retained')
            exclusive[a]['selection_evaluations']=n
        roots.append(dict(root_sha256=root,root_index=i,shared_planner=shared,exclusive_planner=exclusive,
            standalone_planner={a:add(shared,exclusive[a]) for a in ARMS},
            comparison=compare(add(shared,exclusive[ARMS[0]]),add(shared,exclusive[ARMS[1]]))))
        shared_all.append(shared)
        for a in ARMS: exclusive_all[a].append(exclusive[a])
    # Actual physical public calls have root/arm binding except collection.
    physical_shared=vector(); physical_arm={a:vector() for a in ARMS}
    budget_path=physical/'physical_budget.events.jsonl'; require(sha(budget_path) == execution['physical_budget_journal_sha256'], 'physical budget bytes');bindings[str(budget_path)]=execution['physical_budget_journal_sha256']
    events=[json.loads(line) for line in budget_path.read_text().splitlines()]
    require(len(events)%2 == 0, 'public call pairs')
    for request,terminal in zip(events[::2],events[1::2]):
        require(request['completed'] is False and terminal['completed'] is True and terminal == dict(request,completed=True), 'complete physical call binding')
        phase=request['phase'];kind=request['kind']; require(kind in ('env_step','env_reset'), 'known actual public call')
        v=physical_shared if phase == 'root_collection' else physical_arm[request['method']]
        axis=phase+('_steps' if kind == 'env_step' else '_resets')
        require(axis in KNOWN, 'unaccounted physical phase');v[axis] += 1
    enc_path=physical/'encoding.events.jsonl'; bindings[str(enc_path)]=sha(enc_path)
    enc_events=[json.loads(line) for line in enc_path.read_text().splitlines()]; active=None; enc_counts={}
    for e in enc_events:
        if e['kind'] == 'endpoint_encode': active=(e['root_sha256'],e['method'])
        elif e['kind'] in ('encoder','projector'):
            require(active is not None and type(e['completed']) is bool and not e.get('error'), 'endpoint-scoped encoder event')
            key=(*active,e['kind'],e['completed']); enc_counts[key]=enc_counts.get(key,0)+1
        elif e['kind'] == 'endpoint_result': active=None
    require(len(execution['encodings']) == 8, 'all physical endpoint receipts')
    for e in execution['encodings']:
        root,a=e['root_sha256'],e['method']
        require(root in proots and a in ARMS and e['status'] == 'completed' and e['attempted_encode_samples'] == 1, 'complete actual endpoint')
        for kind,axis in (('encoder','visual_encoder_calls'),('projector','projector_calls')):
            require(enc_counts[root,a,kind,False] == enc_counts[root,a,kind,True] == 1, 'actual hook attempts/completions')
            physical_arm[a][axis] += 1
    shared=add(*shared_all,physical_shared)
    exclusive={a:add(*exclusive_all[a],physical_arm[a]) for a in ARMS}
    standalone={a:add(shared,exclusive[a]) for a in ARMS}
    unique=add(shared,*exclusive.values());comparison=compare(standalone[ARMS[0]],standalone[ARMS[1]])
    require(unique['predict_calls'] == report['actual_unique_predict_calls'] == 1328 and
        unique['action_encoder_calls'] == report['actual_unique_action_encoder_calls'] == 1328 and
        unique['jacobian_192x20_calls']+unique['jacobian_192x10_calls'] == report['completed_operations']['jacobian_calls'] == 164 and
        unique['selected_steps'] == 80 and unique['warmup_steps'] == 120 and unique['root_collection_resets'] == 32, 'actual global resource totals')
    require(sum(unique[k] for k in ('math_vjp_192x20_directions','math_vjp_192x10_directions')) == report['completed_operations']['scalar_output_vjp_directions'] == 31488 and
        unique['svd_192x20_calls']+unique['svd_192x10_calls'] == report['completed_operations']['svd_calls'] == 168 and
        unique['bounded_solve_20_calls']+unique['bounded_solve_10_calls'] == report['completed_operations']['bounded_solve_calls'] == 164, 'AD/SVD/solve global cross-check')
    return dict(schema=NAME+'.audit',status='actual_resource_vectors_not_full_cost_matched',roots=roots,
        shared_once=shared,exclusive=exclusive,standalone_attributed=standalone,paired_unique=unique,
        comparison=comparison,do_not_sum_standalone_attributions=True,bindings=bindings,
        next_protocol=next_protocol(),new_neural_calls=0,new_physical_steps=0,P2_admission=False,
        scope='Resource metadata only; no scores used for allocation. Mathematical VJP directions are NOT backward API calls or measured FLOPs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit-current',action='store_true');mode.add_argument('--next-protocol',action='store_true')
    parser.add_argument('--project-root',type=Path,default=PROJECT);parser.add_argument('--output',type=Path)
    args=parser.parse_args(); result=audit_current(args.project_root) if args.audit_current else next_protocol()
    raw=json.dumps(result,sort_keys=True,allow_nan=False,indent=2)+'\n'
    if args.output:
        require(shutil.disk_usage(args.output.parent).free-len(raw.encode()) >= 200*1024**2, 'keep200MiB; never delete')
        with args.output.open('x') as stream: stream.write(raw)
    print(raw,end='')


if __name__ == '__main__': main()
