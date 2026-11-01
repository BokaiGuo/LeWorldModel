"""Read-only native-producer preparation and saved validation, never CUDA dispatch."""
import argparse
import importlib
from pathlib import Path
import time

from experiments.spectral_reachability import chat_p2_20261027_lifecycle as life

SCHEMA = 'chat_p2_20261027_producer_bundle.v1'
KINDS = tuple(life.PRODUCERS)


def native(kind, path):
    module = importlib.import_module('experiments.spectral_reachability.'+life.PRODUCERS[kind])
    loaded = module.load_prepared(Path(path))
    directory, protocol, prepared = loaded[:3]
    return module, directory, protocol, prepared


def identity_rows(kind, protocol):
    rows = protocol['prepared'] if kind == 'strong_sampling' else protocol['context']['cases']
    out = []
    for i, row in enumerate(rows):
        life.graph.require(type(row['root_index']) is int and row['root_index'] == i,
                           'original complete root order required')
        life.graph.digest(row['root_sha256'])
        life.graph.require(row['root']['root_id'] == row['root_sha256'], 'root id binding')
        out.append({k: row[k] for k in ('root_index', 'root_sha256', 'root', 'action_domain', 'metric_sha256')})
    life.graph.require(out and len({r['root_sha256'] for r in out}) == len(out), 'unique complete roots')
    return out


def declared_budgets(protocols):
    """Read native schedules; these are declarations, NOT measured operations."""
    c = protocols['composition']['contract']
    s = protocols['no_composition_sra']['schedule']
    f = protocols['strong_sampling']['schedule']
    point = c['budget_points'][1]
    life.graph.require(c['horizon'] == s['horizon_tokens'] == f['horizon_tokens'] == 2,
                       'native horizon mismatch')
    life.graph.require(c['raw_commands'] == s['primitive_steps'] == f['primitive_steps'] == 10,
                       'native action length mismatch')
    return dict(scope='native_declared_per_root_only_not_total_budget_match',
                composition_frontier=dict(joint_candidates=point['joint_candidates'],
                    composition_candidates=point['composition_candidates'],
                    standalone_predict_calls=point['standalone_predict']),
                no_composition_sra=dict(candidates=s['candidates'],predict_calls=s['total_predict_calls'],
                    predict_samples=s['total_predict_samples'],jacobian_calls=s['jacobian_calls']),
                strong_sampling=dict(candidates_per_arm=f['candidates'],
                    predict_calls_per_arm=f['planning_predict_calls']+f['replay_predict_calls'],
                    predict_samples_per_arm=f['planning_predict_samples']+f['replay_predict_samples']),
                flops=None, upstream_rgb_cost=None, fair_total_budget_comparison=False)


def prepare(paths):
    life.graph.keys(paths, KINDS)
    protocols = {}; records = {}; shared_rows = None
    closure = {Path(__file__).resolve(), *life.SOURCES}
    for kind in KINDS:
        module, directory, protocol, prepared = native(kind, paths[kind])
        rows = identity_rows(kind, protocol)
        if shared_rows is None:
            shared_rows = rows
        life.graph.require(life.graph.encode(rows) == life.graph.encode(shared_rows),
                           'cross-producer root/history/goal/domain/metric mismatch: '+kind)
        protocols[kind] = protocol
        records[kind] = dict(path=str(directory.resolve()),protocol_sha256=life.graph.sha(life.graph.encode(protocol)),
                            prepared_sha256=life.graph.sha(life.graph.encode(prepared)))
        closure.add(Path(module.__file__).resolve())
        closure.update(Path(p).resolve() for p in protocol.get('source_paths', []))
        for seal in prepared['source_seals']:
            manifest = life.read(Path(seal['output_dir'])/'manifest.json')
            closure.update(Path(e['source']).resolve() for e in manifest['files'])
    sources = {}
    for path in sorted(closure):
        raw = path.read_bytes()
        sources[str(path)] = dict(sha256=life.graph.sha(raw),bytes_hex=raw.hex())
    return dict(schema=SCHEMA,producers=records,identities=shared_rows,
                root_denominator=len(shared_rows),budgets=declared_budgets(protocols),sources=sources,
                known_engineering_roots_only=True,execution_authorized=False,P2_admission=False,
                confirmatory_blockers=['external_P1_and_freshness_authority_missing',
                    'native_producer_to_prospective_queue_identity_bridge_not_authorized',
                    'full_resource_matched_caps_and_measurements_missing'])


def verify(bundle, expected_sha):
    life.graph.digest(expected_sha)
    life.graph.require(life.graph.sha(life.graph.encode(bundle)) == expected_sha, 'external bundle SHA drift')
    life.graph.require(bundle['schema'] == SCHEMA, 'bundle schema')
    for name, entry in bundle['sources'].items():
        raw = bytes.fromhex(entry['bytes_hex'])
        life.graph.require(life.graph.sha(raw) == entry['sha256'] and Path(name).read_bytes() == raw,
                           'bundle source drift: '+name)
    rebuilt = prepare({k:bundle['producers'][k]['path'] for k in KINDS})
    life.graph.require(life.graph.encode(rebuilt) == life.graph.encode(bundle), 'native bundle binding drift')
    return True


def entry_counts(entries):
    """Keep failed actual entries charged; never equate calls/samples/tokens.

    These counters describe only supplied instrumented entry records, not FLOPs
    or attempted dispatches blocked before the underlying model entry.
    """
    totals = {kind:dict(api_calls=0,batch_samples=0,context_tokens=0,failed_api_calls=0)
              for kind in ('predict','action_encoder')}
    for entry in entries:
        life.graph.require(entry['kind'] in totals, 'unknown actual entry kind')
        life.graph.integer(entry['samples'], True)
        life.graph.integer(entry['context_tokens'], True)
        life.graph.require(type(entry['completed']) is bool, 'explicit entry completion required')
        life.graph.require(entry['context_tokens'] >= entry['samples'], 'invalid context sample count')
        total = totals[entry['kind']]
        total['api_calls'] += 1
        total['batch_samples'] += entry['samples']
        total['context_tokens'] += entry['context_tokens']
        total['failed_api_calls'] += int(not entry['completed'])
    return dict(entry_streams=totals,observed_records=len(entries),
                failed_entries_refunded=False,missing_instrumentation_independently_proven=False,
                entry_scope='supplied_actual_entry_records_only')


def run_saved(bundle, expected_sha):
    """All kinds retained even when one is incomplete. Never executes native cuda()."""
    verify(bundle, expected_sha)
    rows = []
    for kind in KINDS:
        start = time.monotonic_ns()
        row = dict(kind=kind,status='not_verified',error=None)
        try:
            result = life.consume_producer(kind,bundle['producers'][kind]['path'])
            metadata = saved_metadata(kind,bundle['producers'][kind]['path'])
            row.update(status='native_full_export_verified',result_type=type(result).__name__,saved_metadata=metadata)
        except Exception as exc:
            row.update(status='rejected_not_resealed',error=dict(type=type(exc).__name__,message=str(exc)))
        row['verification_host_wall_ns'] = time.monotonic_ns()-start
        rows.append(row)
    return dict(schema=SCHEMA+'.saved',bundle_sha256=expected_sha,rows=rows,producer_denominator=len(KINDS),
                root_denominator=bundle['root_denominator'],
                status='all_native_exports_verified' if all(r['error'] is None for r in rows) else 'incomplete_native_exports',
                execution_authorized=False,P2_admission=False,new_neural_dispatch=False,
                new_physics_dispatch=False,fair_total_budget_comparison=False)


def saved_metadata(kind, path):
    """Called only AFTER the native full exporter; no outcome scores projected."""
    directory = Path(path); report = life.read(directory/'report.json'); rows = []
    for row in report['rows']:
        if kind == 'composition':
            row = life.read(directory/f"root_{row['root_index']}.json")
            # Native frontier exporter already checks all root files and journals.
            entries = row['actual_model_entries']
            arms = row['results']
        elif kind == 'no_composition_sra':
            entries = row['actual_model_entries']; arms = [row['result']]
        else:
            arms = row['rows']; entries = [e for arm in arms for e in arm['actual_model_entries']]
        rows.append(dict(root_sha256=row['root_sha256'],actual_entries=entry_counts(entries),
                         candidate_denominators={arm['arm']:len(arm['attempts']) for arm in arms}))
    return dict(rows=rows,saved_records_not_new_neural_measurements=True)


def run_saved_file(bundle_path, expected_sha, output):
    output = Path(output)
    life.graph.require(not output.exists(), 'exclusive new saved report required')
    life.write(output.with_name(output.stem+'_started.json'),dict(bundle_sha256=expected_sha,failed_work_refunded=False))
    try:
        result = run_saved(life.read(bundle_path),expected_sha)
        life.write(output,result)
    except BaseException as primary:
        failure = output.with_name(output.stem+'_failure.json')
        try:
            life.write(failure,dict(error_type=type(primary).__name__,error=str(primary),
                                   bundle_sha256=expected_sha,P2_admission=False,execution_authorized=False))
        except BaseException as secondary:
            primary.add_note('failure persistence also failed: '+type(secondary).__name__)
        raise
    return result


def consume(bundle, expected_sha, report):
    life.graph.require(report['bundle_sha256'] == expected_sha, 'report bundle binding')
    checked = run_saved(bundle, expected_sha)
    def projection(value):
        return dict(value,rows=[{k:v for k,v in r.items() if k != 'verification_host_wall_ns'} for r in value['rows']])
    life.graph.require(life.graph.encode(projection(checked)) == life.graph.encode(projection(report)), 'saved audit drift')
    life.graph.require(checked['status'] == 'all_native_exports_verified', 'complete composition/SRA/sampling exports required')
    return dict(status='engineering_saved_bundle_verified',P2_admission=False,execution_authorized=False,
                report_sha256=life.graph.sha(life.graph.encode(report)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for kind in KINDS:
        parser.add_argument('--'+kind.replace('_','-'),type=Path)
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--output',type=Path)
    a = parser.parse_args(argv)
    if a.command == 'prepare':
        value = prepare({k:getattr(a,k) for k in KINDS})
        life.write(a.output,value)
        result = dict(status='engineering_bundle_prepared',sha256=life.graph.sha(life.graph.encode(value)))
    elif a.command == 'verify':
        verify(life.read(a.bundle),a.sha256); result = dict(status='engineering_bundle_verified')
    elif a.command == 'run':
        result = run_saved_file(a.bundle,a.sha256,a.output)
    else:
        result = consume(life.read(a.bundle),a.sha256,life.read(a.report))
        if a.output:
            life.write(a.output,result)
    print(life.graph.encode(result).decode())
    return 2 if result.get('status') == 'incomplete_native_exports' else 0


if __name__ == '__main__':
    raise SystemExit(main())
