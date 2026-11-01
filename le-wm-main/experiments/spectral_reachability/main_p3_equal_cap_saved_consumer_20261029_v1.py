"""Saved-only equal-cap consumer with the producer's bounded output ABI.

The sealed CUDA producer is unchanged. All original mathematical/resource
validators remain mandatory; only the inherited 64 MiB journal line ABI differs.
"""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p3_equal_cap_parent_20261029_v1 as producer

NAME = 'main_p3_equal_cap_saved_consumer_20261029_v1'
PRODUCER_SHA = 'ea5b8266ed1c0f26c20db91c7cfbb9648a589a3291a5d10480ec0ab3bacc52d1'
require, sha, digest = producer.require, producer.sha, producer.digest
base, core, signal = producer.base, producer.core, producer.signal


def load_events(output, report, *, total_cap, record_cap):
    """Bound BEFORE decoding; preserve duplicate-field and whole hashchain checks."""
    require(type(total_cap) is type(record_cap) is int and 0 < record_cap <= total_cap,
            'positive bounded journal ABI')
    path = base.regular(Path(output)/'events.jsonl', total_cap)
    records, tail, used = [], '0'*64, 0
    with path.open('rb') as stream:
        while True:
            line = stream.readline(record_cap+1)
            if not line:
                break
            used += len(line)
            require(used <= total_cap and len(line) <= record_cap and line.endswith(b'\n'),
                    'bounded complete producer journal record')
            value = json.loads(line, object_pairs_hook=signal.unique_pairs,
                               parse_constant=lambda _: require(False, 'finite JSON'))
            core.keys(value, ('index', 'previous', 'payload', 'sha256'))
            h = value.pop('sha256')
            require(type(value['index']) is int and value['index'] == len(records)
                    and value['previous'] == tail and digest(value) == h, 'complete journal chain')
            records.append(value['payload'])
            tail = h
    require(tail == report['journal_tail'] and path.stat().st_size == used,
            'exact entire journal and final tail')
    return records


def consume(output, report_sha, prepared, prepare_sha):
    require(sha(producer.__file__) == PRODUCER_SHA, 'frozen producer unchanged')
    output = Path(output).absolute()
    require(output.resolve() == output and not output.is_symlink(), 'exact output directory')
    p = producer.verify(prepared, prepare_sha)
    rp = base.regular(output/'report.json', p['profile']['final_reserve_bytes'])
    require(sha(rp) == report_sha, 'external actual report SHA')
    r = signal.read(rp, p['profile']['final_reserve_bytes'])
    core.keys(r, ('schema', 'status', 'prepared', 'prepare_sha256', 'rows', 'errors',
        'model_before', 'model_after', 'classes_before', 'classes_after', 'precision', 'host',
        'sources_verified', 'source_manifest_sha256', 'profile', 'fixed_denominator',
        'wall_seconds', 'CUDA_peak_bytes', 'journal_tail', 'artifact_sha256', *producer.FLAGS))
    require(r['schema'] == producer.NAME+'.report' and r['prepared'] == str(Path(prepared).absolute())
            and r['prepare_sha256'] == prepare_sha, 'original report/prepare binding')
    require(all(r[k] is False for k in producer.FLAGS) and not r['errors']
            and r['sources_verified'] is True and r['source_manifest_sha256'] == digest(p['sources']),
            'no admission and complete source evidence')
    require(r['model_before'] == r['model_after']
            and r['model_before']['original_state_dict'] == p['expected_model_hash'], 'frozen state')
    require(r['classes_before'] == r['classes_after'] and r['classes_before'], 'complete runtime classes')
    for c in r['classes_before']:
        require(p['sources'].get(c['path']) == c['sha256'], 'prebound installed source')
    require(r['host']['device'] == 'cuda:0' and r['host']['allocation_sum'] == 4.
            and r['host']['visible_nvidia_nodes'], 'actual HOST CUDA preflight')
    require(r['precision'] == [base.precision_contract()]*2, 'before/after actual precision')
    require(set(r['artifact_sha256']) == {'binding.json', 'events.jsonl'}
            and {f.name for f in output.iterdir()} == {'binding.json', 'events.jsonl', 'report.json'},
            'all-and-only artifacts')
    for n, h in r['artifact_sha256'].items():
        require(sha(base.regular(output/n, p['profile']['output_bytes'])) == h, 'actual artifact bytes')
    require(signal.read(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=prepare_sha),
            'durable run binding')
    # Same preexisting producer total and reserve caps, NOT a scientific threshold.
    events = load_events(output, r, total_cap=p['profile']['output_bytes'],
                         record_cap=p['profile']['final_reserve_bytes'])
    for phase in ('before', 'after'):
        require([e for e in events if e['kind'] == 'source_'+phase] ==
                [dict(kind='source_'+phase, sha256=digest(p['sources']))], 'all source events')
        require([e for e in events if e['kind'] == 'model_'+phase] ==
                [dict(kind='model_'+phase, hashes=r['model_'+phase], classes=r['classes_'+phase])],
                'all model events')
        require([e['value'] for e in events if e['kind'] == 'precision_'+phase] ==
                [base.precision_contract()], 'all precision events')
    require(0 <= r['wall_seconds'] <= 18000 and len(r['CUDA_peak_bytes']) == 2
            and all(type(v) is int and 0 <= v <= p['profile'][k] for v, k in
                    zip(r['CUDA_peak_bytes'], ('cuda_allocated_bytes', 'cuda_reserved_bytes'))),
            'whole job/memory bounds')
    require(sum(f.stat().st_size for f in output.iterdir()) <= p['profile']['output_bytes'],
            'original whole output cap')
    arms = producer.validate_saved(p, r, events)
    settled = all(x['result'] is not None and not x['result']['errors']
                  and not x['timing']['failed'] for x in r['rows'])
    require(r['status'] == ('four_slots_settled_not_admitted' if settled else 'partial_or_failed'),
            'exact completion semantics')
    require(sha(rp) == report_sha and all(sha(f) == h for f, h in p['sources'].items())
            and all(sha(output/n) == h for n, h in r['artifact_sha256'].items()),
            'all original sources/artifacts unchanged after full validation')
    own = [Path(__file__).resolve(), producer.ROOT/'tests'/('test_'+NAME+'.py'),
           producer.ROOT/'innerdocs'/(NAME+'.md')]
    return dict(schema=NAME+'.consume', status='saved_equal_cap_verified_not_admitted'
                if settled else 'saved_equal_cap_partial_verified_not_admitted',
        prepared=r['prepared'], prepare_sha256=prepare_sha, report_sha256=report_sha,
        fixed_denominator=p['fixed_denominator'], arms=arms, common_budget=p['profile'],
        equal_cap_contract=True, matching_axes=[], total_FLOPs=None,
        inherited_producer_sha256=PRODUCER_SHA, consumer_sources={str(f): sha(f) for f in own},
        compatibility=dict(original_reader_record_cap=64*1024**2,
            record_cap=p['profile']['final_reserve_bytes'], total_cap=p['profile']['output_bytes'],
            original_producer_unchanged=True, original_full_math_resource_timing_validation=True,
            new_neural_calls=0, new_physics_calls=0), **producer.FLAGS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('output', 'report-sha', 'prepared', 'prepare-sha', 'receipt'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    value = consume(a.output, a.report_sha, a.prepared, a.prepare_sha)
    target = Path(a.receipt).absolute()
    require(target.parent == producer.ROOT/'results' and target.name.startswith(NAME+'_')
            and not target.exists() and target.resolve() == target, 'exclusive own receipt')
    data = producer.canonical(value)
    require(len(data) <= producer.profile()['final_reserve_bytes'], 'bounded own receipt')
    with target.open('xb') as stream:
        require(stream.write(data) == len(data), 'whole receipt')
        stream.flush()
        producer.os.fsync(stream.fileno())
    signal.fsync_dir(target.parent)
    print(producer.canonical(dict(path=str(target), sha256=sha(target))).decode())


if __name__ == '__main__':
    main()
