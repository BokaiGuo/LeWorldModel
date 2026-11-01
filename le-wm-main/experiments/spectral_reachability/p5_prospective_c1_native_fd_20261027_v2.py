"""Explicit manifest/receipt binding for the immutable native FD v1 math/core.

No compiler-run path is hardcoded. All prepare/bind operations are non-neural.
Only main may call run-cuda; fixed .01/.005, 40 predictions, zero encodings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import p5_prospective_c1_native_fd_20261027_v1 as base
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p5_prospective_c1_native_fd_20261027_v2'
ROOT = Path(__file__).resolve().parents[2]
compiler = base.compiler
require, file_sha, read_json = base.require, base.file_sha, base.read_json


def merge_bindings(*groups):
    result = {}
    for group in groups:
        for path, sha in group.items():
            require(path not in result or result[path] == sha, 'conflicting frozen binding '+path)
            result[path] = sha
    return result


def new_output(path):
    path = Path(path).resolve()
    require(path.parent == ROOT/'results' and path.name.startswith(TAG+'_') and not path.exists(), 'exclusive new v2 results child required')
    path.mkdir(exist_ok=False)
    return path


def verify(path, expected):
    require(file_sha(path) == expected, 'retained FD manifest FILE SHA differs')
    m = read_json(path)
    require(m['schema'] == TAG and m['policy'] == base.policy(), 'same frozen FD mathematics required')
    require(not compiler.check_bindings(m['bindings']), 'frozen source/loader/input/reference drift')
    s = m['source_seal']
    verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    c = None
    if m['compiler_manifest_path'] is not None:
        require(file_sha(m['compiler_manifest_path']) == m['compiler_manifest_sha256'], 'explicit compiler manifest SHA differs')
        c = read_json(m['compiler_manifest_path'])
        require(c['schema'] == compiler.TAG and c['status'] == 'prepared_for_main_CUDA'
                and not c['blockers'] and c['policy'] == compiler.policy(), 'same ready prospective compiler policy required')
        compiler.validate_bridge(read_json(c['bridge_path']), c['bridge_sha256'], ready=True)
        s = c['source_seal']
        verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    return m, c


def freeze_source(output):
    files = [Path(__file__).resolve(), ROOT/'tests'/('test_'+TAG+'.py'), ROOT/'innerdocs'/(TAG+'.md'),
             Path(base.__file__).resolve(), ROOT/'tests'/('test_'+base.TAG+'.py'),
             ROOT/'innerdocs'/(base.TAG+'.md'), Path(compiler.__file__).resolve(),
             ROOT/'experiments/spectral_reachability/d2p_prerun_source_seal.py']
    bindings = {str(p): file_sha(p) for p in files}
    out = new_output(output)
    seal = create_seal(files=files, args_json=json.dumps(dict(policy=base.policy(), binding='explicit compiler manifest then explicit receipt; no run directory default')),
                       output_dir=out/'source_seal')
    m = dict(schema=TAG, status='source_prepared_awaiting_explicit_compiler_manifest', policy=base.policy(),
        bindings=bindings, source_seal=seal, compiler_manifest_path=None, compiler_manifest_sha256=None,
        reference=None, new_neural_calls=0, new_physics_steps=0, P5_admission=False)
    compiler.bridge.write(out/'manifest_frozen.json', m)
    verify(out/'manifest_frozen.json', file_sha(out/'manifest_frozen.json'))
    return m


def prepare(source_manifest, source_sha, compiler_manifest, compiler_sha, output):
    m, _ = verify(source_manifest, source_sha)
    require(m['status'] == 'source_prepared_awaiting_explicit_compiler_manifest', 'use unbound source freeze')
    require(file_sha(compiler_manifest) == compiler_sha, 'explicit compiler manifest FILE SHA differs')
    c = read_json(compiler_manifest)
    require(c['schema'] == compiler.TAG and c['policy'] == compiler.policy()
            and c['status'] == 'prepared_for_main_CUDA' and not c['blockers'], 'new compiler source prepare must be ready')
    p = read_json(c['bridge_path'])
    compiler.validate_bridge(p, c['bridge_sha256'], ready=True)
    compiler.bridge.compiler_bundle(p, c['bridge_sha256'])  # Same seven allowed assets only.
    bound = dict(m)
    bound.update(status='prepared_awaiting_explicit_actual_JVP_receipt',
        compiler_manifest_path=str(Path(compiler_manifest).resolve()), compiler_manifest_sha256=compiler_sha,
        bridge_sha256=c['bridge_sha256'], compiler_inputs_sha256=p['compiler_inputs_sha256'],
        source_manifest_path=str(Path(source_manifest).resolve()), source_manifest_sha256=source_sha)
    bound['bindings'] = merge_bindings(m['bindings'], c['bindings'],
        {str(Path(source_manifest).resolve()): source_sha, str(Path(compiler_manifest).resolve()): compiler_sha})
    require(not compiler.check_bindings(bound['bindings']), 'new compiler/source/allowed-input drift')
    out = new_output(output)
    compiler.bridge.write(out/'manifest_frozen.json', bound)
    verify(out/'manifest_frozen.json', file_sha(out/'manifest_frozen.json'))
    return bound


def check_receipt_location(path, records):
    path = Path(path).resolve()
    require(path.name == 'compiler_receipt.json' and path.parent.is_relative_to(ROOT/'results'), 'explicit local compiler receipt path required')
    for rec in records.values():
        asset = Path(rec['path']).resolve()
        require(asset.parent == path.parent and asset.suffix == '.npy', 'compiler context/JVP assets must belong to explicit receipt directory')
    return path


def verify_journal(path, receipt):
    require(file_sha(path) == receipt['journal_sha256'], 'actual compiler journal FILE SHA differs')
    previous = None
    terminal = None
    for i, line in enumerate(Path(path).read_text().splitlines()):
        record = compiler.bridge.saved.strict_json(line.encode())
        require(record['sequence'] == i and record['previous_sha256'] == previous, 'compiler journal chronological chain')
        require(compiler.digest({k: v for k, v in record.items() if k != 'entry_sha256'}) == record['entry_sha256'], 'compiler journal entry content hash')
        previous, terminal = record['entry_sha256'], record['event']
    require(terminal is not None and terminal['kind'] == 'terminal' and terminal['status'] == receipt['status'], 'compiler journal complete terminal status required')


def bind_reference(manifest, manifest_sha, receipt_path, receipt_sha, output, diagnostic=False):
    m, c = verify(manifest, manifest_sha)
    require(m['status'] == 'prepared_awaiting_explicit_actual_JVP_receipt' and m['reference'] is None and c is not None, 'explicit new compiler prepare must precede receipt binding')
    require(file_sha(receipt_path) == receipt_sha, 'explicit actual receipt FILE SHA differs')
    r = read_json(receipt_path)
    p = read_json(c['bridge_path'])
    # Exact manifest and bridge digests reject old/new bridge cross-wiring.
    base.reference_arrays(r, dict(path=m['compiler_manifest_path'], manifest=c), p, diagnostic)
    records = {name: r['arrays'][name] for name in base.required_arrays()}
    receipt_path = check_receipt_location(receipt_path, records)
    journal = receipt_path.parent/'journal.jsonl'
    verify_journal(journal, r)
    bound = dict(m)
    bound.update(status='prepared_diagnostic_failed_compiler' if diagnostic else 'prepared_for_main_CUDA_FD',
        reference=dict(receipt_path=str(receipt_path), receipt_sha256=receipt_sha, arrays=records,
                       compiler_status=r['status'], compiler_error=r['error'], diagnostic_failed_compiler=diagnostic))
    bound['bindings'] = merge_bindings(m['bindings'], {str(Path(manifest).resolve()): manifest_sha,
        str(receipt_path): receipt_sha, str(journal): r['journal_sha256']},
        {rec['path']: rec['sha256'] for rec in records.values()})
    require(not compiler.check_bindings(bound['bindings']), 'receipt/source/array bind drift')
    out = new_output(output)
    compiler.bridge.write(out/'manifest_frozen.json', bound)
    verify(out/'manifest_frozen.json', file_sha(out/'manifest_frozen.json'))
    return bound


def run(manifest, manifest_sha, output):
    m, c = verify(manifest, manifest_sha)
    require(c is not None and m['reference'] is not None and m['status'] in ('prepared_for_main_CUDA_FD', 'prepared_diagnostic_failed_compiler'), 'actual new manifest/receipt binding required before CUDA')
    p = read_json(c['bridge_path'])
    receipt = read_json(m['reference']['receipt_path'])
    reference = base.reference_arrays(receipt, dict(path=m['compiler_manifest_path'], manifest=c), p, m['reference']['diagnostic_failed_compiler'])
    out = new_output(output)
    r = dict(schema=TAG+'.receipt', status='FD_failed', error=None, manifest_file_sha256=manifest_sha,
        compiler_manifest_path=m['compiler_manifest_path'], compiler_manifest_sha256=m['compiler_manifest_sha256'],
        compiler_receipt_path=m['reference']['receipt_path'], compiler_receipt_sha256=m['reference']['receipt_sha256'],
        reference_compiler_status=receipt['status'], reference_compiler_error=receipt['error'],
        diagnostic_failed_compiler=m['reference']['diagnostic_failed_compiler'], compiler_inputs_sha256=m['compiler_inputs_sha256'],
        backend='host_cuda_only_SDPA_MATH', training=False, physical_calls=0, used_observed_frame15=False,
        selected_packet_changed=False, P5_admission=False, C2_established=False, partial_work_unknown=True)
    ledger = compiler.Ledger(out, r)
    ledger.event('begin', policy=m['policy'])
    try:
        base.cuda_core(m, c, p, reference, ledger)  # Unchanged sealed forty-predict implementation.
    except Exception as exc:
        r.update(status='FD_failed', error=type(exc).__name__+': '+str(exc), failed_phase=ledger.phase)
    compiler.guarded_final(r, 'source_reference_after', lambda: verify(manifest, manifest_sha))
    compiler.guarded_final(r, 'arrays_after', lambda: require(not compiler.check_bindings(r['artifact_bindings']), 'FD output array drift'))
    r['counts_match'] = r['attempted'] == r['completed'] == base.policy()['completed']
    if r['status'] == 'core_completed_pending_final_checks' and not r['finalization_errors'] and r['counts_match']:
        r.update(status='diagnostic_FD_recorded_from_failed_compiler' if m['reference']['diagnostic_failed_compiler'] else 'native_JVP_FD_comparison_recorded', partial_work_unknown=False)
    else:
        r['status'] = 'FD_failed'
        if r['error'] is None:
            r['error'] = 'final integrity or exact 40-predict count failed'
    ledger.phase = 'terminal'
    ledger.event('terminal', status=r['status'], error=r['error'])
    r['journal_sha256'] = file_sha(out/'journal.jsonl')
    compiler.bridge.write(out/'fd_receipt.json', r)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('freeze-source')
    s.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--compiler-manifest', type=Path, required=True)
    p.add_argument('--compiler-manifest-sha256', required=True)
    b = sub.add_parser('bind-reference')
    b.add_argument('--compiler-receipt', type=Path, required=True)
    b.add_argument('--compiler-receipt-sha256', required=True)
    b.add_argument('--diagnostic-failed-compiler', action='store_true')
    for q in (p, b, sub.add_parser('run-cuda')):
        q.add_argument('--manifest', type=Path, required=True)
        q.add_argument('--manifest-sha256', required=True)
        q.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'freeze-source':
        r = freeze_source(args.output)
    elif args.command == 'prepare':
        r = prepare(args.manifest, args.manifest_sha256, args.compiler_manifest, args.compiler_manifest_sha256, args.output)
    elif args.command == 'bind-reference':
        r = bind_reference(args.manifest, args.manifest_sha256, args.compiler_receipt, args.compiler_receipt_sha256, args.output, args.diagnostic_failed_compiler)
    else:
        r = run(args.manifest, args.manifest_sha256, args.output)
    print(json.dumps(dict(status=r['status'], output=str(args.output), new_neural_calls=r.get('new_neural_calls'), P5_admission=False)))
    if args.command == 'run-cuda' and r['status'] == 'FD_failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
