"""Exclusive saved-program lifecycle, reusing existing graph and queue contracts.

No neural dispatch, physics, outcome selection, or scientific authorization.
"""
import argparse
from copy import deepcopy
import importlib
import os
from pathlib import Path
import time

from experiments.spectral_reachability import moving_p2_saved_program_graph_20261027_v1 as graph
from experiments.spectral_reachability import moving_p2_composition_prospective_queue_20261027_v1 as queue

PREFIX = 'chat_p2_20261027_'
PROJECT = Path(__file__).resolve().parents[2]
SOURCES = (Path(__file__).resolve(), Path(graph.__file__).resolve(), Path(queue.__file__).resolve())
PRODUCERS = {
    'composition': 'p2_composition_predict_frontier_cuda_20261027_v2',
    'no_composition_sra': 'p2_no_composition_sra_20261027_v2',
    'strong_sampling': 'p2_full20_sampling_20261027_v2',
}


def write(path, value):
    """Only new prefixed files under authorized subtrees; durable exclusive writes."""
    path = Path(path).absolute()
    graph.require(path.name.startswith(PREFIX), 'exclusive prefix required')
    allowed = [PROJECT / p for p in ('results', 'innerdocs', 'tests', 'experiments/spectral_reachability')]
    allowed.append(PROJECT.parent / 'docs')
    graph.require(any(path.parent.resolve().is_relative_to(p.resolve()) for p in allowed),
                  'output outside authorized subtree')
    with path.open('xb') as stream:
        stream.write(graph.encode(value)); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read(path):
    return graph.parse(Path(path).read_bytes())


def prepare(saved_graph, load, *, algorithm, max_expansions, beam_width=None):
    """Pin complete graph and all referenced bytes, not just the winning path."""
    graph.require(algorithm in ('shortest', 'beam'), 'unknown fixed search rule')
    graph.integer(max_expansions, True)
    if algorithm == 'beam':
        graph.integer(beam_width, True)
    else:
        graph.require(beam_width is None, 'shortest has no beam width')
    _, _, _, _, snapshot = graph.validate_graph(saved_graph, load)
    sources = {}
    for p in SOURCES:
        raw = p.read_bytes()
        sources[str(p)] = dict(sha256=graph.sha(raw), bytes_hex=raw.hex())
    return dict(schema=PREFIX+'prepared.v1', saved_graph=deepcopy(saved_graph),
                # Raw bytes pinned without filesystem aliases or lossy numeric decoding.
                evidence_hex={k: v.hex() for k, v in snapshot.data.items()},
                sources=sources,
                policy=dict(algorithm=algorithm, max_expansions=max_expansions, beam_width=beam_width),
                candidate_edge_ids=[e['edge_id'] for e in saved_graph['edges']],
                P2_admission=False, execution_authorized=False)


def verify(prepared, expected_sha):
    graph.digest(expected_sha)
    graph.require(graph.sha(graph.encode(prepared)) == expected_sha, 'external prepared SHA mismatch')
    graph.keys(prepared, ('schema', 'saved_graph', 'evidence_hex', 'sources', 'policy',
                          'candidate_edge_ids', 'P2_admission', 'execution_authorized'))
    graph.require(prepared['schema'] == PREFIX+'prepared.v1', 'prepared schema')
    graph.require(prepared['P2_admission'] is False and prepared['execution_authorized'] is False,
                  'saved lifecycle cannot authorize execution')
    graph.require(set(prepared['sources']) == set(map(str, SOURCES)), 'source closure mismatch')
    for name, entry in prepared['sources'].items():
        raw = bytes.fromhex(entry['bytes_hex'])
        graph.require(graph.sha(raw) == entry['sha256'] and Path(name).read_bytes() == raw,
                      'source snapshot/current drift: '+name)
    evidence = {k: bytes.fromhex(v) for k, v in prepared['evidence_hex'].items()}
    rebuilt = prepare(prepared['saved_graph'], evidence.__getitem__, **prepared['policy'])
    graph.require(graph.encode(rebuilt) == graph.encode(prepared), 'prepared census or binding drift')
    return evidence


def run(prepared, expected_sha, *, confirmatory=False, manifest=None, load=None):
    """Run only saved search. Confirmation is always blocked; no trusted MAIN input."""
    started = time.monotonic_ns()
    evidence = verify(prepared, expected_sha)
    audit = None
    if manifest is not None:
        graph.require(load is not None, 'queue evidence loader required')
        audit = queue.build_queue(manifest, load, trusted=None)
    report = dict(schema=PREFIX+'run.v1', prepared_sha256=expected_sha,
                  candidate_edge_ids=prepared['candidate_edge_ids'],
                  candidate_edge_denominator=len(prepared['candidate_edge_ids']),
                  confirmatory_requested=confirmatory, queue_audit=audit,
                  execution_authorized=False, P2_admission=False,
                  scientific_evidence_generated=False, neural_calls=0, physics_steps=0,
                  measured_world_model_predict_calls=0, measured_world_model_batch_samples=0,
                  saved_generation_costs_are_not_new_measurements=True,
                  fair_total_budget_comparison=False, search=None,
                  errors=[], status='started', consumer_host_wall_ns=None)
    try:
        if confirmatory:
            report['status'] = 'blocked_confirmatory'
            report['errors'] = ['P1/freshness/MAIN authority and cross-contract identity bridge required; no neural dispatch']
        else:
            report['search'] = graph.search(prepared['saved_graph'], evidence.__getitem__, **prepared['policy'])
            report['status'] = report['search']['status']
    except BaseException as exc:
        report['status'] = 'failed_saved_search'
        report['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        report['consumer_host_wall_ns'] = time.monotonic_ns()-started
    return report


def run_file(prepared_path, expected_sha, output, **kwargs):
    """Start marker prevents retries/refunds. Failure artifact survives exceptions."""
    output = Path(output)
    graph.require(not output.exists(), 'output already exists; no retry/overwrite')
    marker = output.with_name(output.stem+'_started.json')
    write(marker, dict(prepared_sha256=expected_sha, failed_attempts_refunded=False))
    try:
        result = run(read(prepared_path), expected_sha, **kwargs)
    except BaseException as exc:
        write(output, dict(status='failed', error_type=type(exc).__name__, error=str(exc),
                           prepared_sha256=expected_sha, P2_admission=False, execution_authorized=False))
        raise
    write(output, result)
    return result


def consume(prepared, expected_sha, report):
    """Recompute saved result; never trust a winner-only or altered report."""
    graph.require(report.get('schema') == PREFIX+'run.v1', 'completed saved report required')
    graph.require(report['confirmatory_requested'] is False and report['queue_audit'] is None,
                  'consume accepts only unprivileged saved-search reports')
    recomputed = run(prepared, expected_sha)
    for key in recomputed:
        if key != 'consumer_host_wall_ns':
            graph.require(graph.encode(report[key]) == graph.encode(recomputed[key]), 'saved report drift: '+key)
    graph.require(report['status'] == 'saved_path_found', 'no complete saved path to consume')
    return dict(commands=report['search']['path']['commands'],
                path=report['search']['path'], report_sha256=graph.sha(graph.encode(report)),
                export_scope='saved_action_bytes_only_not_execution',
                execution_authorized=False, P2_admission=False)


def consume_producer(kind, output):
    """Use existing full-census exporter, including its source and solve checks.

    Never weaken it to read a report or selected_actions array directly.
    """
    graph.require(kind in PRODUCERS, 'unknown existing producer')
    module = importlib.import_module('experiments.spectral_reachability.'+PRODUCERS[kind])
    if kind == 'strong_sampling':
        directory, protocol, prepared, compilation = module.load_prepared(Path(output))
        report = read(directory/'report.json')
        graph.require(report['source_seals'] == prepared['source_seals'], 'sampling report seal drift')
        return module.selected_actions(report, protocol, compilation)
    return module.selected_actions(Path(output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run', 'consume'))
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--algorithm', choices=('shortest', 'beam'), default='shortest')
    parser.add_argument('--max-expansions', type=int, default=10000)
    parser.add_argument('--beam-width', type=int)
    parser.add_argument('--confirmatory', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        graph.require(args.graph is not None and args.input_root is not None, 'graph and input-root required')
        value = prepare(read(args.graph), graph.file_loader(args.input_root), algorithm=args.algorithm,
                        max_expansions=args.max_expansions, beam_width=args.beam_width)
        write(args.output, value)
        result = dict(status='engineering_prepared', sha256=graph.sha(graph.encode(value)), P2_admission=False)
    elif args.command == 'verify':
        verify(read(args.prepared), args.sha256)
        result = dict(status='engineering_verified', P2_admission=False)
    elif args.command == 'run':
        result = run_file(args.prepared, args.sha256, args.output, confirmatory=args.confirmatory)
    else:
        result = consume(read(args.prepared), args.sha256, read(args.report))
        if args.output:
            write(args.output, result)
    print(graph.encode(result).decode())
    return 2 if result.get('status') == 'blocked_confirmatory' else 0


if __name__ == '__main__':
    raise SystemExit(main())
