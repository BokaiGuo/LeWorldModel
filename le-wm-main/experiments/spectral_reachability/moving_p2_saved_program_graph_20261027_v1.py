"""Search exact, saved action-witness edges; never infer or execute a transition."""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct

STEM = 'moving_p2_saved_program_graph_20261027_v1'
AXES = ('predict_calls', 'predict_samples', 'action_encoder_calls', 'encoder_calls',
        'jacobian_calls', 'jvp_directions', 'vjp_directions', 'backward_calls',
        'svd_calls', 'solve_calls', 'physics_steps', 'host_wall_ns', 'cuda_ns', 'flops')
POLICY = ('training', 'adapters', 'q_fitting', 'model_updates', 'cpu_neural',
          'oracle_selection', 'outcome_filtering')
ASSETS = ('model', 'encoder', 'normalizer', 'q', 'goal', 'task', 'controller', 'action_domain')


class InvalidGraph(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise InvalidGraph(message)


def keys(obj, names):
    require(type(obj) is dict and set(obj) == set(names), 'exact fields required: ' + ','.join(names))


def label(value):
    require(type(value) is str and bool(value.strip()), 'nonempty string required')


def integer(value, positive=False):
    require(type(value) is int and value >= int(positive), 'nonnegative/positive integer required')


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'finite numeric value required')
    return Fraction(value)  # Exact arithmetic on the saved binary scalar, no tolerance.


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    require(type(value) is str and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), 'lowercase SHA256 required')


def parse(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key: ' + key)
            out[key] = value
        return out

    def constant(value):
        raise InvalidGraph('nonfinite JSON constant: ' + value)

    obj = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    encode(obj)  # Reject overflowed JSON exponents as well.
    return obj


def vector(obj):
    keys(obj, AXES)
    for value in obj.values():
        integer(value)  # Missing/null is never zero.
    return obj


def endpoint(node, contract_sha256):
    """Full saved identity, including predictor history/context and state domain."""
    return dict(node_id=node['node_id'], contract_sha256=contract_sha256,
                state_kind=node['state_kind'],
                **{k + '_sha256': node[k]['sha256'] for k in ('latent', 'state', 'context')})


class Snapshot:
    """Read each explicitly named file once, pin bytes and check every reference."""

    def __init__(self, load):
        self.load = load
        self.data = {}

    def read(self, ref):
        keys(ref, ('file', 'sha256'))
        label(ref['file']); digest(ref['sha256'])
        path = Path(ref['file'])
        require(not path.is_absolute() and '..' not in path.parts, 'contained relative reference required')
        name = ref['file']
        if name not in self.data:
            self.data[name] = self.load(name)
        raw = self.data[name]
        require(type(raw) is bytes and bool(raw) and sha(raw) == ref['sha256'], 'missing/empty/mismatched evidence bytes: ' + name)
        return raw


def validate_graph(graph, load):
    """Strict whole-graph validation: never quietly select around broken edges."""
    g = deepcopy(graph)
    keys(g, ('schema', 'contract', 'base_resources', 'nodes', 'edges', 'start_node', 'target_nodes'))
    require(g['schema'] == STEM + '.graph', 'graph schema mismatch')
    c = g['contract']
    keys(c, ('assets', 'action_dtype', 'action_dim', 'lower', 'upper', 'max_steps',
             'max_l1', 'max_l2_squared', 'min_programs', 'max_programs',
             'resource_caps', 'prohibited', 'objective'))
    snap = Snapshot(load)
    keys(c['assets'], ASSETS)
    for ref in c['assets'].values():
        snap.read(ref)  # Sealed metadata only; no model loader or directory discovery.
    keys(c['prohibited'], POLICY)
    require(all(v is True for v in c['prohibited'].values()), 'frozen/training-free policy required')
    require(c['objective'] == 'sum_saved_nonnegative_cost', 'fixed saved-cost objective required')
    require(c['action_dtype'] in ('float32-le-raw', 'float64-le-raw'), 'raw action dtype required')
    integer(c['action_dim'], True); integer(c['max_steps'], True)
    integer(c['min_programs'], True); integer(c['max_programs'], True)
    require(c['min_programs'] <= c['max_programs'], 'invalid program bounds')
    for bound in ('lower', 'upper'):
        require(type(c[bound]) is list and len(c[bound]) == c['action_dim'], 'action bound dimension mismatch')
        for value in c[bound]:
            number(value)
    require(all(number(lo) <= number(hi) for lo, hi in zip(c['lower'], c['upper'])), 'inverted action bounds')
    require(number(c['max_l1']) >= 0 and number(c['max_l2_squared']) >= 0, 'nonnegative cumulative action caps required')
    vector(c['resource_caps']); vector(g['base_resources'])
    require(all(g['base_resources'][k] <= c['resource_caps'][k] for k in AXES), 'base resources exceed cap')
    contract_sha = sha(encode(c))
    require(type(g['nodes']) is list and bool(g['nodes']), 'saved nodes missing')
    nodes, identities = {}, set()
    for n in g['nodes']:
        keys(n, ('node_id', 'state_kind', 'latent', 'state', 'context'))
        label(n['node_id'])
        require(n['node_id'] not in nodes, 'duplicate node_id')
        require(n['state_kind'] in ('saved_predicted', 'saved_observed'), 'explicit saved state domain required')
        for key in ('latent', 'state', 'context'):
            snap.read(n[key])
        # Alias nodes with exactly the same context/domain are rejected, not merged.
        ident = (n['state_kind'], *(n[k]['sha256'] for k in ('latent', 'state', 'context')))
        require(ident not in identities, 'duplicate endpoint alias')
        identities.add(ident)
        nodes[n['node_id']] = n
    label(g['start_node'])
    require(g['start_node'] in nodes, 'unknown start node')
    require(type(g['target_nodes']) is list and bool(g['target_nodes']), 'explicit saved targets required')
    for target in g['target_nodes']:
        label(target)
        require(target in nodes, 'unknown target node')
    require(len(set(g['target_nodes'])) == len(g['target_nodes']), 'duplicate targets')
    require(type(g['edges']) is list, 'explicit saved edge list required')
    edges, adjacency = {}, {node: [] for node in nodes}
    for e in g['edges']:
        keys(e, ('edge_id', 'program_type', 'parent', 'child', 'contract_sha256',
                 'steps', 'commands', 'resource_delta', 'cost', 'receipt'))
        label(e['edge_id'])
        require(e['edge_id'] not in edges, 'duplicate edge_id')
        require(e['program_type'] in ('progress', 'reconfiguration'), 'saved program type required')
        require(e['contract_sha256'] == contract_sha, 'edge goal/model/action/resource contract mismatch')
        for side in ('parent', 'child'):
            require(type(e[side]) is dict and e[side].get('node_id') in nodes, 'unknown saved endpoint')
            require(e[side] == endpoint(nodes[e[side]['node_id']], contract_sha), 'exact ' + side + ' latent/state/context identity mismatch')
        integer(e['steps'], True)
        vector(e['resource_delta'])
        require(number(e['cost']) >= 0, 'negative saved edge cost')
        raw = snap.read(e['commands'])
        fmt, width = ('f', 4) if c['action_dtype'] == 'float32-le-raw' else ('d', 8)
        require(len(raw) == e['steps'] * c['action_dim'] * width, 'raw action shape/byte length mismatch')
        values = struct.unpack('<' + fmt * (len(raw) // width), raw)
        for value in values:
            number(value)
        commands = [list(values[i:i+c['action_dim']]) for i in range(0, len(values), c['action_dim'])]
        for command in commands:
            require(all(number(lo) <= number(v) <= number(hi) for v, lo, hi in
                        zip(command, c['lower'], c['upper'])), 'saved raw action outside fixed domain')
        receipt = parse(snap.read(e['receipt']))
        keys(receipt, ('schema', 'edge_id', 'parent', 'child', 'contract_sha256', 'commands_sha256',
                       'steps', 'resource_delta', 'cost', 'program_type', 'backend',
                       'status', 'producer_source_sha256', 'invocation'))
        require(receipt['schema'] == STEM + '.saved_transition', 'saved transition receipt schema')
        for k in ('edge_id', 'parent', 'child', 'contract_sha256', 'steps', 'resource_delta', 'cost', 'program_type'):
            # canonical JSON prevents bool/int or int/float equality coercions.
            require(encode(receipt[k]) == encode(e[k]), 'receipt binding mismatch: ' + k)
        require(receipt['commands_sha256'] == e['commands']['sha256'], 'receipt raw action SHA mismatch')
        require(receipt['backend'] == 'HOST_CUDA' and receipt['status'] == 'completed', 'completed saved HOST CUDA generation receipt required')
        digest(receipt['producer_source_sha256']); snap.read(receipt['invocation'])
        edges[e['edge_id']] = dict(saved=e, commands=commands,
                                   l1=sum((abs(Fraction(v)) for v in values), Fraction()),
                                   l2=sum((Fraction(v)**2 for v in values), Fraction()))
        adjacency[e['parent']['node_id']].append(e['edge_id'])
    for outgoing in adjacency.values():
        outgoing.sort()  # Input ordering is not a search policy.
    return g, nodes, edges, adjacency, snap


def rational(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def search(graph, load, *, algorithm, max_expansions, beam_width=None):
    """Bounded simple-path uniform-cost or beam search over saved edges only.

    Uniform-cost retains full resource/action/path labels, not one best cost per
    node. Beam is approximate; exhaustion after pruning is never unreachability.
    """
    result = dict(schema=STEM + '.search', status='invalid_saved_graph', errors=[],
                  graph_sha256=None, path=None, optimality_proven=False,
                  search_scope='fixed_saved_simple_paths_only',
                  search_policy=None, scientific_evidence_generated=False,
                  consumer_host_wall_ns=None,
                  resources_are_saved_declared_not_new_measurements=True,
                  execution_authorized=False, P2_admission=False,
                  execution_blockers=['P1_and_freshroot_not_admitted_by_this_core',
                                      'MAIN_review_and_HOST_CUDA_executor_required'],
                  physical_outcome=None, evidence_authenticity_independently_proven=False,
                  expanded_labels=0, generated_labels=0, beam_pruned_labels=0,
                  rejections=[], rejection_counts={}, saved_input_sha256={})
    try:
        require(algorithm in ('shortest', 'beam'), 'algorithm must be shortest or beam')
        integer(max_expansions, True)
        if algorithm == 'beam':
            integer(beam_width, True)
        else:
            require(beam_width is None, 'beam_width is not a shortest-path parameter')
        result['search_policy'] = dict(algorithm=algorithm, max_expansions=max_expansions,
                                      beam_width=beam_width)
        result['graph_sha256'] = sha(encode(graph))
        g, nodes, edges, adjacency, snap = validate_graph(graph, load)
    except (ValueError, TypeError, KeyError, OSError, OverflowError, struct.error) as error:
        result['errors'].append(str(error))
        return result
    result['saved_input_sha256'] = {name: sha(raw) for name, raw in sorted(snap.data.items())}
    c = g['contract']; targets = set(g['target_nodes'])
    # label = node, visited nodes, edge path, cost, steps, l1, l2, resource vector
    initial = (g['start_node'], (g['start_node'],), (), Fraction(), 0,
               Fraction(), Fraction(), dict(g['base_resources']))
    reasons = Counter()

    def reject(path, edge_id, reason):
        reasons[reason] += 1
        result['rejections'].append(dict(prefix_edge_ids=list(path), edge_id=edge_id, reason=reason))

    def is_goal(item):
        return item[0] in targets and len(item[2]) >= c['min_programs']

    def expand(item):
        node, visited, path, cost, steps, l1, l2, resources = item
        if len(path) >= c['max_programs']:
            reject(path, None, 'max_programs_reached')
            return []
        if not adjacency[node]:
            reject(path, None, 'no_saved_outgoing_edge')
        children = []
        for edge_id in adjacency[node]:
            e = edges[edge_id]; saved = e['saved']; child = saved['child']['node_id']
            if child in visited:
                reject(path, edge_id, 'cycle_not_in_simple_path_scope'); continue
            new_steps, new_l1, new_l2 = steps + saved['steps'], l1 + e['l1'], l2 + e['l2']
            if new_steps > c['max_steps']:
                reject(path, edge_id, 'cumulative_steps_exceeded'); continue
            if new_l1 > number(c['max_l1']) or new_l2 > number(c['max_l2_squared']):
                reject(path, edge_id, 'cumulative_action_cap_exceeded'); continue
            total = {k: resources[k] + saved['resource_delta'][k] for k in AXES}
            exceeded = [k for k in AXES if total[k] > c['resource_caps'][k]]
            if exceeded:
                reject(path, edge_id, 'resource_cap_exceeded:' + ','.join(exceeded)); continue
            children.append((child, visited + (child,), path + (edge_id,),
                             cost + number(saved['cost']), new_steps, new_l1, new_l2, total))
            result['generated_labels'] += 1
        return children

    best = None
    limited = False
    if algorithm == 'shortest':
        heap, serial = [(Fraction(), (), 0, initial)], 0
        while heap:
            if result['expanded_labels'] >= max_expansions:
                limited = True; break
            _, _, _, item = heapq.heappop(heap)
            result['expanded_labels'] += 1
            if is_goal(item):
                best = item
                result['optimality_proven'] = True
                break
            for child in expand(item):
                serial += 1
                heapq.heappush(heap, (child[3], child[2], serial, child))
    else:
        frontier = [initial]
        while frontier:
            next_level = []
            for item in frontier:
                if result['expanded_labels'] >= max_expansions:
                    limited = True; break
                result['expanded_labels'] += 1
                if is_goal(item):
                    if best is None or (item[3], item[2]) < (best[3], best[2]):
                        best = item
                else:
                    next_level.extend(expand(item))
            if limited:
                break
            next_level.sort(key=lambda item: (item[3], item[2]))
            result['beam_pruned_labels'] += max(0, len(next_level) - beam_width)
            frontier = next_level[:beam_width]
    result['rejection_counts'] = dict(sorted(reasons.items()))
    if best is not None:
        _, visited, path, cost, steps, l1, l2, resources = best
        result['path'] = dict(edge_ids=list(path), node_ids=list(visited),
            endpoints=[endpoint(nodes[n], sha(encode(c))) for n in visited],
            saved_edges=[deepcopy(edges[e]['saved']) for e in path],
            commands=[command for e in path for command in deepcopy(edges[e]['commands'])],
            action_dtype=c['action_dtype'], steps=steps, l1=rational(l1), l2_squared=rational(l2),
            cost=rational(cost), resources=resources,
            witness_refs=[deepcopy(edges[e]['saved']['commands']) for e in path],
            receipt_refs=[deepcopy(edges[e]['saved']['receipt']) for e in path])
    if limited:
        result['status'] = 'search_limit'
        result['errors'].append('max_expansions_reached; no optimality or unreachability claim')
    elif best is not None:
        result['status'] = 'saved_path_found'
    elif algorithm == 'beam' and result['beam_pruned_labels']:
        result['status'] = 'beam_exhausted_no_path'
        result['errors'].append('beam pruned saved path labels; no unreachability claim')
    else:
        result['status'] = 'no_feasible_saved_path'
        result['errors'].append('no path within exact saved topology and fixed simple-path constraints')
    return result


def file_loader(root):
    root = Path(root).resolve()

    def load(name):
        path = (root / name).resolve()
        require(path.is_relative_to(root), 'reference symlink escapes input root')
        return path.read_bytes()
    return load


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--algorithm', choices=('shortest', 'beam'), required=True)
    parser.add_argument('--max-expansions', type=int, required=True)
    parser.add_argument('--beam-width', type=int)
    args = parser.parse_args(argv)
    try:
        report = search(parse(args.graph.read_bytes()), file_loader(args.input_root),
                        algorithm=args.algorithm, max_expansions=args.max_expansions,
                        beam_width=args.beam_width)
    except (ValueError, TypeError, OSError) as error:
        report = dict(schema=STEM + '.input_error', status='invalid_saved_graph',
                      errors=[str(error)], execution_authorized=False, P2_admission=False)
    print(encode(report).decode())  # No result files, research reports or executor.
    return 0 if report['status'] == 'saved_path_found' else 2


if __name__ == '__main__':
    raise SystemExit(main())
