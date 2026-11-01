"""Saved native P2 witness-to-queue seam. No neural/physics execution API.

Reuse paired-v2's all-candidate action witnesses, standalone SRA's full exporter,
and the existing prospective queue. Unknown strict graph resources stay unknown.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import struct

from experiments.spectral_reachability import moving_p2_saved_program_graph_20261027_v1 as graph
from experiments.spectral_reachability import moving_p2_composition_prospective_queue_20261027_v1 as queue
from experiments.spectral_reachability import d2p_composition_witness_v2 as witness
from experiments.spectral_reachability import chat_p2_20261027_candidate_census as census
from experiments.spectral_reachability import chat_p2_20261027_producer_bundle as entry_metadata

PREFIX = 'agent_p2_resume_20261027_'
NAME = PREFIX+'witness_queue'
ROOT = Path(__file__).resolve().parents[2]
KINDS = ('composition', 'no_composition_sra')
BLOCKERS = ['independent_P1_MAIN_authority_missing', 'freshroot_and_complete_exclusions_missing',
            'native_blind_packet_and_task_authority_missing',
            'full_resource_caps_and_graph_base_edge_allocation_missing',
            'prospective_goal_success_controller_rules_and_MAIN_spec_missing']


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path):
    path = Path(path)
    graph.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 96*1024**2,
                  'bounded regular saved JSON')
    return graph.parse(path.read_bytes())


def write(path, value):
    path = Path(path).absolute()
    graph.require(path.name.startswith(PREFIX) and path.parent.resolve().is_relative_to(ROOT/'results'),
                  'exclusive prefix under results required')
    data = graph.encode(value)
    graph.require(len(data) <= 32*1024**2, 'bounded saved metadata output')
    with path.open('xb') as stream:
        graph.require(stream.write(data) == len(data), 'short saved write')
        stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/(PREFIX+'test_witness_queue.py'),
            ROOT/'innerdocs'/(NAME+'.md'), Path(graph.__file__), Path(queue.__file__),
            Path(witness.__file__), Path(census.__file__),
            Path(entry_metadata.__file__), Path(entry_metadata.life.__file__),
            ROOT.parent/'docs/parallel_chats_20261027/03_p2_composition.md',
            ROOT.parent/'docs/D2P_MOVING_CAPABILITY_PROBE_MAIN_20261027.md']


def load_native(kind, directory, expected):
    """Use original full exporter, never winner-only report reads as validation."""
    from experiments.spectral_reachability import p2_composition_paired_20261027_v2 as paired
    from experiments.spectral_reachability import p2_no_composition_sra_20261027_v2 as sra
    graph.require(kind in KINDS, 'fixed native producer kind')
    module = paired if kind == 'composition' else sra
    directory = Path(directory).resolve(); graph.digest(expected)
    graph.require(file_sha(directory/'report.json') == expected, 'external native report SHA')
    _, protocol, prepared = module.load_prepared(directory)
    exported = module.selected_actions(directory)  # Saved NumPy arithmetic only.
    graph.require(exported['report_sha256'] == expected, 'native full exporter report identity')
    report = read(directory/'report.json'); bindings = {}
    def bind(path):
        path = Path(path).resolve(); bindings[str(path)] = file_sha(path)
    for name, h in protocol['bindings'].items():
        bind(name); graph.require(bindings[str(Path(name).resolve())] == h, 'original input seal')
    for name in ('protocol.json', 'prepared.json', 'report.json'): bind(directory/name)
    for case in protocol['context']['cases']: bind(directory/f'root_{case["root_index"]}_bank.json')
    for seal in prepared['source_seals']:
        sd = Path(seal['output_dir'])
        bind(sd/'manifest.json'); bind(sd/'args.json')
        for entry in read(sd/'manifest.json')['files']:
            bind(entry['source'])
    graph.require(file_sha(directory/'report.json') == expected, 'native report drift')
    return dict(directory=str(directory), protocol=protocol, prepared=prepared, report=report,
                report_sha256=expected, bindings=bindings, verification_work=exported['verification_work'])


def identity(case):
    root = case['root']; witness._context(root['context'])
    for key in ('root_sha256', 'goal_vector_sha256', 'metric_sha256', 'projection_sha256'):
        graph.digest(case[key])
    graph.require(root['root_id'] == case['root_sha256'], 'native root id mismatch')
    for key in ('root_rgb_sha256', 'goal_rgb_sha256'): graph.digest(root[key])
    return dict(root_sha256=case['root_sha256'], root_rgb_sha256=root['root_rgb_sha256'],
                goal_rgb_sha256=root['goal_rgb_sha256'], goal_vector_sha256=case['goal_vector_sha256'],
                history_sha256=witness.digest(root['context']), metric_sha256=case['metric_sha256'],
                projection_sha256=case['projection_sha256'], action_domain=deepcopy(case['action_domain']))


def candidate_record(case, attempt, arm, index, source_ref, nodes):
    """Project existing transitions, never manufacture intermediate states."""
    graph.require(attempt['candidate_id'] == f'{arm}_{index:02d}' and attempt['index'] == index and
                  attempt['completed'] is True, 'fixed complete native candidate identity')
    w = attempt['witness']; edges = w['edges']; receipt = w['receipt']
    expected_edges = 2 if arm == 'conditional_suffix_composition' else 1
    graph.require(len(edges) == expected_edges and [len(e['actions']) for e in edges] ==
                  ([5, 5] if expected_edges == 2 else [10]), 'original program segmentation')
    def replay(context, actions):
        found = [e for e in edges if graph.encode(e['input_context']) == graph.encode(context) and
                 graph.encode(e['actions']) == graph.encode(actions)]
        graph.require(len(found) == 1, 'unique original action transition')
        return deepcopy(found[0]['output_context'])
    rebound = witness.compose(root=case['root'], provenance=receipt['provenance'],
        action_domain=case['action_domain'], edges=edges, replay=replay,
        max_edges=expected_edges, max_actions=10)
    graph.require(graph.encode(rebound) == graph.encode(receipt) and
                  graph.encode(receipt['actions']) == graph.encode(attempt['actions']),
                  'native receipt/history/goal/model/action witness mismatch')
    if expected_edges == 2:
        graph.require(attempt['suffix_compilation']['input_context_sha256'] ==
                      witness.digest(edges[0]['output_context']), 'suffix compiled at wrong history')
    ident = identity(case)
    def node(context):
        # This is the native MODEL state/context, explicitly not native physics7.
        key = witness.digest(dict(identity=ident, context=context))
        value = dict(node_id=key, state_kind='saved_predicted',
                     state_domain='native_predictor_full_context_not_physical_pose',
                     context_sha256=witness.digest(context), context=deepcopy(context))
        graph.require(key not in nodes or graph.encode(nodes[key]) == graph.encode(value), 'node identity collision')
        nodes[key] = value; return key
    projected = []
    for e in edges:
        values = [v for row in e['actions'] for v in row]
        bits = struct.pack('<'+'f'*len(values), *values)
        decoded = list(struct.unpack('<'+'f'*len(values), bits))
        graph.require(graph.encode(decoded) == graph.encode(values), 'raw actions must be exact float32, no repair')
        projected.append(dict(program_id=e['program_id'], parent=node(e['input_context']),
            child=node(e['output_context']), native_edge_sha256=witness.digest(e),
            actions=deepcopy(e['actions']), commands_float32_le_hex=bits.hex(),
            commands_sha256=graph.sha(bits), steps=len(e['actions']),
            provenance=deepcopy(e['provenance']), action_domain=deepcopy(e['action_domain'])))
    return dict(candidate_index=index, candidate_id=attempt['candidate_id'], arm=arm,
                identity=ident, source_ref=source_ref, native_witness_sha256=witness.digest(w),
                native_receipt_sha256=receipt['receipt_sha256'], edges=projected,
                execution_authorized=False)


def project(native):
    comp, sra = (native[k] for k in KINDS)
    cases = [n['protocol']['context']['cases'] for n in (comp, sra)]
    graph.require(all(len(c) == 4 for c in cases), 'all four known native roots required')
    graph.require(graph.encode(cases[0]) == graph.encode(cases[1]), 'cross-producer full root/history/goal/Q/domain mismatch')
    comp_rows = {r['root_sha256']: r for r in comp['report']['rows']}
    sra_rows = {r['root_sha256']: r for r in sra['report']['rows']}
    roots = [c['root_sha256'] for c in cases[0]]
    graph.require(len(comp['report']['rows']) == len(sra['report']['rows']) == 4 and
                  len(set(roots)) == len(comp_rows) == len(sra_rows) == 4 and
                  set(comp_rows) == set(sra_rows) == set(roots), 'complete duplicate-free root join')
    nodes, records, slots = {}, [], []
    for i, case in enumerate(cases[0]):
        graph.require(type(case['root_index']) is int and case['root_index'] == i, 'fixed original root order')
        row = comp_rows[case['root_sha256']]; standalone = sra_rows[case['root_sha256']]['result']
        graph.require([r['arm'] for r in row['results']] ==
                      ['joint_h2_no_composition', 'conditional_suffix_composition'], 'fixed native arm order')
        graph.require(len(standalone['attempts']) == 40, 'fixed standalone SRA denominator')
        summaries = []
        for result in row['results']:
            summary = census.census(result, arm=result['arm'], expected_count=40)
            graph.require(summary['deployable_engineering_candidate'], 'complete original candidate/replay census')
            summaries.append({**{k: v for k, v in summary.items() if k != 'slots'},
                'slots': [{**{k: v for k, v in s.items() if k != 'native_attempt'},
                           'native_attempt_sha256': None if s['native_attempt'] is None else
                           witness.digest(s['native_attempt'])} for s in summary['slots']]})
            for j, attempt in enumerate(result['attempts']):
                if result['arm'] == 'joint_h2_no_composition':
                    other = standalone['attempts'][j]
                    graph.require(other['candidate_id'] == f'joint_h2_no_composition_sra_{j:02d}' and
                                  other['completed'] is True and
                                  graph.encode(other['actions']) == graph.encode(attempt['actions']),
                                  'paired joint vs standalone SRA raw candidate bank differs')
                ref = dict(report_sha256=comp['report_sha256'], root_sha256=case['root_sha256'],
                           arm=result['arm'], candidate_index=j)
                records.append(candidate_record(case, attempt, result['arm'], j, ref, nodes))
        slots.append(dict(root_index=i, identity=identity(case), censuses=summaries,
                          actual_native_entry_records_sha256=witness.digest(row['actual_model_entries']),
                          saved_paired_entries=entry_metadata.entry_counts(row['actual_model_entries']),
                          saved_standalone_sra_entries=entry_metadata.entry_counts(sra_rows[case['root_sha256']]['actual_model_entries']),
                          saved_paired_resource_receipt=deepcopy(row['resource_receipt'])))
    manifest = queue.build_draft([r for r in roots])
    manifest['contract']['horizon_steps'] = 10
    for row, case in zip(manifest['slots'], cases[0], strict=True):
        row['identity'] = dict(root_sha256=case['root_sha256'], goal_sha256=case['goal_vector_sha256'],
                              history_sha256=witness.digest(case['root']['context']), blind_sha256=None,
                              task_id=None)  # Native schema has neither; NEVER fabricate authority.
    audit = queue.build_queue(manifest, lambda name: (_ for _ in ()).throw(ValueError('unprovided queue asset: '+name)))
    return dict(schema=NAME+'.program_graph', nodes=list(nodes.values()), candidates=records,
                root_slots=slots, root_denominator=4, candidates_per_root_arm=40, candidate_denominator=320,
                edge_denominator=sum(len(r['edges']) for r in records),
                topology='only original native candidate edges; no new bridge, root, waypoint or search',
                strict_resource_graph_ready=False, strict_resource_unknown_axes=list(graph.AXES),
                strict_graph_not_constructed_reason='no full graph-base/edge-increment/caps allocation; no invented resource zeros',
                queue_manifest=manifest, queue_audit=audit,
                confirmatory_blockers=BLOCKERS, execution_authorized=False, P2_admission=False,
                scientific_evidence_generated=False)


def prepare(inputs):
    graph.keys(inputs, KINDS)
    native = {k: load_native(k, inputs[k]['directory'], inputs[k]['report_sha256']) for k in KINDS}
    bindings = {}
    for n in native.values():
        for name, h in n['bindings'].items():
            graph.require(name not in bindings or bindings[name] == h, 'conflicting source/input seal')
            bindings[name] = h
    for p in own_files(): bindings[str(p.resolve())] = file_sha(p)
    value = dict(schema=NAME+'.prepare', inputs=deepcopy(inputs), program_graph=project(native),
                 bindings=bindings, additional_saved_verification_work={k: n['verification_work'] for k, n in native.items()},
                 engineering_ready=True, scientific_pass=False, execution_authorized=False)
    for name, h in bindings.items(): graph.require(file_sha(name) == h, 'prepare final input drift')
    return value


def verify(prepared, expected):
    graph.digest(expected)
    graph.require(graph.sha(graph.encode(prepared)) == expected and prepared['schema'] == NAME+'.prepare',
                  'external canonical prepare SHA/schema')
    for name, h in prepared['bindings'].items(): graph.require(file_sha(name) == h, 'input/source drift')
    rebuilt = prepare(prepared['inputs'])
    graph.require(graph.encode(rebuilt) == graph.encode(prepared), 'full saved witness/queue/census drift')
    return rebuilt


def consume(prepared, expected, *, confirmatory=False):
    # No caller boolean or forged queue pass can authorize actual execution.
    if confirmatory:
        raise ValueError('confirmatory dispatch blocked: '+', '.join(BLOCKERS))
    p = verify(prepared, expected); g = p['program_graph']
    return dict(schema=NAME+'.consume', status='engineering_witness_queue_verified_science_blocked',
                prepare_sha256=expected, root_denominator=4, candidate_denominator=320,
                edge_denominator=g['edge_denominator'],
                candidate_actions=[dict(identity=r['identity'], arm=r['arm'], candidate_index=r['candidate_index'],
                    candidate_id=r['candidate_id'], commands=[a for e in r['edges'] for a in e['actions']],
                    native_receipt_sha256=r['native_receipt_sha256']) for r in g['candidates']],
                queue_status=g['queue_audit']['status'], confirmatory_blockers=BLOCKERS,
                scope='saved_action_witness_export_NOT_dispatch', new_neural_calls=0, new_physics_steps=0,
                execution_authorized=False, P2_admission=False, scientific_pass=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run-saved', 'consume'))
    for k in KINDS:
        parser.add_argument('--'+k.replace('_', '-')+'-output')
        parser.add_argument('--'+k.replace('_', '-')+'-sha')
    parser.add_argument('--prepared'); parser.add_argument('--prepare-sha'); parser.add_argument('--output')
    parser.add_argument('--confirmatory', action='store_true'); a = parser.parse_args(argv)
    if a.confirmatory:
        raise ValueError('P1/freshness and complete resource/MAIN authority missing; no confirmatory run API')
    if a.command == 'prepare':
        inputs = {k: dict(directory=str(Path(getattr(a, k+'_output')).resolve()),
                          report_sha256=getattr(a, k+'_sha')) for k in KINDS}
        value = prepare(inputs); write(a.output, value)
        result = dict(status='engineering_prepared_science_blocked', prepared=str(Path(a.output).absolute()),
                      prepare_sha256=graph.sha(graph.encode(value)))
    else:
        p = read(a.prepared)
        if a.command == 'verify':
            verify(p, a.prepare_sha); result = dict(status='engineering_verified_science_blocked')
        else:
            result = consume(p, a.prepare_sha)
            if a.output: write(a.output, result)
        result = {k: v for k, v in result.items() if k != 'candidate_actions'}
    print(graph.encode(result).decode()); return 0


if __name__ == '__main__': raise SystemExit(main())
