"""Model-only fixed-prefix action handoff; saved full-strict authority required."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import agent_p1_fixed_prefix_host_cuda_20261029_v1 as host

NAME = 'agent_p1_fixed_prefix_saved_handoff_20261029_v1'
ROOT, F, core, signal = host.ROOT, host.F, host.core, host.signal
require, raw, sha, digest = host.require, host.raw, host.sha, host.digest
HOST_SHA = 'fa11dd40ce0e2c437306101fcd707b5d3ae5c2a131ecbf84529df85bf6cb9e94'
Q_FIELDS = {'index','slot','source_arm','source_iteration','task','prefix_logits','suffix_logits',
            'expected_raw_bits','expected_mid','expected_end','initial_Q','baseline_Q','source_report_sha256'}
RESULT_FIELDS = {'index','slot','source_arm','source_iteration','stage_Q','initial_Q','baseline_Q','iterates',
                 'gradients','selected_iteration','status','P1_admission','physical_release'}
ITER_FIELDS = {'iteration','logits','commands','raw_bits','context','terminal_Q',
               'uphill_from_initial','better_terminal_than_original','witness'}


def contract():
    return dict(schema=NAME+'.contract', slots=4, groups=64, candidates=576, known_roots=2, fresh_roots=0,
        selection='all 64 groups; exactly each source model witness selected_iteration; no cross-group selection',
        witness='stage_Q > initial_Q and terminal_Q < original winner Q; ties terminal_Q then iteration',
        abstention='null; never substitute winner_replay, another candidate or baseline',
        authority='complete host consume rebuilt and byte-identical persisted receipt',
        native_guard='not performed: original recipe and separate prefix-specific cp5 guard required',
        physical_outcomes_read=False, physical_release=False, P1_admission=False, freshness=False)


def select(prepared, receipt, report_sha):
    """Pure model-only join. Caller must independently perform full strict first."""
    core.hash_value(report_sha)
    require(prepared['schema'] == host.NAME+'.prepare' and raw(prepared['contract']) == raw(host.contract()), 'actual host prepare')
    require(receipt['schema'] == host.NAME+'.consume' and receipt['status'] == 'verified_model_only'
            and receipt['complete'] is True and receipt['report_sha256'] == report_sha
            and receipt['group_denominator'] == 64 and receipt['candidate_denominator'] == 576
            and all(receipt[k] is False for k in ('P1_admission','freshness','physical_release'))
            and receipt['actual_new_compute'] == {k:F.caps(1216,512) for k in ('attempted','completed')}, 'complete full strict receipt')
    groups, source_rows, audits = prepared['groups'], receipt['rows'], receipt['audits']
    require(len(groups) == len(source_rows) == len(audits) == 64, 'full group roster')
    rows = []
    for i, (q, saved, checked) in enumerate(zip(groups, source_rows, audits, strict=True)):
        require(type(q) is dict and set(q) == Q_FIELDS and q['index'] == saved['index'] == i and
                q['slot'] == i//16 and q['source_arm'] == host.method.parent.ARMS[(i%16)//8]
                and q['source_iteration'] == i%8+1 and saved['status'] == 'sealed', 'fixed model identity')
        require(type(checked) is dict and checked['index'] == i and checked['native_forwards'] == 19
                and checked['scalar_reverse'] == 8 and checked['scalar_objective']['checked_scalar_losses'] == 8
                and checked['P1_admission'] is False, 'full native/scalar audit')
        result = saved['result']; require(type(result) is dict and set(result) == RESULT_FIELDS, 'model-only result fields')
        require(all(result[k] == q[k] for k in ('index','slot','source_arm','source_iteration')) and
                result['P1_admission'] is False and result['physical_release'] is False and
                result['initial_Q'] == q['initial_Q'] and result['baseline_Q'] == q['baseline_Q'], 'result provenance')
        task = core.validate_task(q['task'])
        require(F.cost(task['goal'], task['root_context']) == q['initial_Q'] and
                F.cost(task['goal'], q['expected_mid']) == result['stage_Q'], 'same task initial/stage Q')
        prefix = F.chart(core.finite(q['prefix_logits'], (10,)))
        initial_suffix = F.chart(core.finite(q['suffix_logits'], (10,)))
        require(F.bits(np.concatenate((prefix, initial_suffix))) == q['expected_raw_bits'], 'source raw bits')
        iters = result['iterates']
        require(type(iters) is list and len(iters) == 9 and len(result['gradients']) == 8, 'complete nine candidates and gradient ledger')
        eligible = []
        for j, v in enumerate(iters):
            require(type(v) is dict and set(v) == ITER_FIELDS and v['iteration'] == j, 'candidate schema/order')
            command = F.chart(core.finite(v['logits'], (10,)))
            require(F.bits(command) == v['raw_bits'] == F.bits(v['commands']) and
                    F.cost(task['goal'], v['context']) == v['terminal_Q'], 'candidate raw and goal Q')
            uphill = result['stage_Q'] > q['initial_Q']; better = v['terminal_Q'] < q['baseline_Q']
            require(v['uphill_from_initial'] is uphill and v['better_terminal_than_original'] is better
                    and v['witness'] is (uphill and better), 'original strict conjunction')
            if uphill and better: eligible.append(v)
        require(iters[0]['raw_bits'] == F.bits(initial_suffix) and raw(iters[0]['context']) == raw(q['expected_end']), 'original suffix replay')
        winner = min(eligible, key=lambda v:(v['terminal_Q'],v['iteration'])) if eligible else None
        selected = None if winner is None else winner['iteration']
        require(result['selected_iteration'] == selected and result['status'] == ('no_witness' if winner is None else 'model_witness'), 'fixed model selection')
        commands = None if winner is None else np.concatenate((prefix, np.asarray(winner['commands'], np.float32)))
        rows.append(dict(index=i, slot=q['slot'], source_arm=q['source_arm'], source_iteration=q['source_iteration'],
            status='abstain' if winner is None else 'model_witness', selected_iteration=selected,
            commands=None if commands is None else commands.tolist(), raw_float32_bits=None if commands is None else F.bits(commands),
            prefix_raw_float32_bits=F.bits(prefix), task=deepcopy(q['task']), source_group_sha256=digest(q),
            initial_Q=q['initial_Q'], stage_Q=result['stage_Q'], original_winner_Q=q['baseline_Q'],
            selected_terminal_Q=None if winner is None else winner['terminal_Q'],
            physical_release=False, P1_admission=False))
    return rows


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def build(host_output, host_report_sha, host_prepared, host_prepare_sha, host_receipt, host_receipt_sha):
    require(sha(host.__file__) == HOST_SHA, 'sealed host source')
    for path, h in ((host_prepared,host_prepare_sha),(host_receipt,host_receipt_sha)):
        core.hash_value(h); require(Path(path).is_absolute() and Path(path).resolve() == Path(path) and sha(path) == h, 'actual artifact path/SHA')
    output = Path(host_output); core.hash_value(host_report_sha)
    require(output.is_absolute() and output.resolve() == output and sha(output/'report.json') == host_report_sha, 'actual report path/SHA')
    report = signal.read(output/'report.json', 256*1024**2)
    require(report['prepared'] == host_prepared and report['prepare_sha256'] == host_prepare_sha, 'host prepared binding')
    receipt = host.consume(str(output), host_report_sha)
    require(Path(host_receipt).read_bytes() == raw(receipt) and sha(host_receipt) == host_receipt_sha, 'actual persisted strict receipt bytes')
    prepared = signal.read(host_prepared, 256*1024**2)
    rows = select(prepared, receipt, host_report_sha)
    bindings = deepcopy(prepared['bindings'])
    for p in own_files(): bindings[str(p)] = sha(p)
    for path, h in ((output/'report.json',host_report_sha),(Path(host_prepared),host_prepare_sha),(Path(host_receipt),host_receipt_sha)):
        bindings[str(path)] = h
    for name, h in report['artifact_sha256'].items(): bindings[str(output/name)] = h
    F.check_bindings(bindings)
    args = dict(host_output=host_output,host_report_sha=host_report_sha,host_prepared=host_prepared,
                host_prepare_sha=host_prepare_sha,host_receipt=host_receipt,host_receipt_sha=host_receipt_sha)
    return dict(schema=NAME+'.prepare', arguments=args, contract=contract(), rows=rows,
                group_denominator=64, candidate_denominator=576, bindings=bindings,
                physical_release=False, P1_admission=False, freshness=False)


def verify(path, expected):
    core.hash_value(expected); require(sha(path) == expected, 'handoff SHA')
    p = signal.read(path, 256*1024**2)
    require(raw(p) == raw(build(**p['arguments'])) and sha(path) == expected, 'full authority and handoff rebuild')
    return p


def write(path, value):
    path = Path(path).absolute(); payload = raw(value)
    require(path.parent == ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve() == path
            and not path.exists() and not path.is_symlink() and len(payload) <= 256*1024**2, 'exclusive bounded output')
    require(shutil.disk_usage(path.parent).free >= 200*1024**2+len(payload), 'disk floor')
    with path.open('xb') as f:
        require(f.write(payload) == len(payload), 'full write'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(path.parent); return dict(path=str(path), sha256=sha(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('command', choices=('prepare','verify'))
    for k in ('host-output','host-report-sha','host-prepared','host-prepare-sha','host-receipt','host-receipt-sha','output','prepared','prepare-sha'):
        ap.add_argument('--'+k)
    a = ap.parse_args()
    if a.command == 'prepare':
        value = write(a.output, build(a.host_output,a.host_report_sha,a.host_prepared,a.host_prepare_sha,a.host_receipt,a.host_receipt_sha))
    else:
        verify(a.prepared,a.prepare_sha); value = dict(verified=True, physical_release=False)
    print(raw(value).decode(), flush=True)


if __name__ == '__main__': main()
