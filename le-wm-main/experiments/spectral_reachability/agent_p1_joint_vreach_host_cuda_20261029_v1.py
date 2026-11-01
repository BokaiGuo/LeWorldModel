"""Four-slot/eight-arm candidate host. No execution without explicit authority.

Saved consumption validates recorded transcripts, not hardware attestation.
The existing scientific gates and known-root/freshness limitations remain intact.
"""
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import time

from experiments.spectral_reachability import agent_p1_joint_vreach_authority_prepare_20261029_v1 as authority
from experiments.spectral_reachability import agent_p1_joint_vreach_multistart_20261029_v1 as method
from experiments.spectral_reachability import agent_p1_joint_vreach_native_callback_20261029_v1 as callback
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_audit_20261029_v1 as audit
from experiments.spectral_reachability import agent_p3_parent_r_native_components_20261029_v1 as component
from experiments.spectral_reachability import agent_p1_two_packet_host_cuda_20261029_v1 as parent
from experiments.spectral_reachability import agent_nonlinear_suffix_observed_recompile_20261028_v2 as old

NAME = 'agent_p1_joint_vreach_host_cuda_20261029_v1'
AUTH = 'MAIN_P1_JOINT_VREACH_HOST_CUDA_ONLY'
F, core, signal, transport = parent.F, parent.core, parent.signal, parent.transport
require, sha, raw, digest = F.require, F.sha, F.raw, F.digest
ROOT = F.ROOT
PRECISION = dict(matmul_TF32=False, cudnn_TF32=False, autocast=False, math=True,
                 flash=False, efficient=False, cudnn_sdp=False)


def contract():
    return dict(schema=NAME+'.contract', method=method.contract(), slot_denominator=4,
        arm_denominator=8, candidate_denominator=160, scalar_per_arm=F.caps(98,16),
        signal_per_arm=component.expected('postcontext_H1'), signal_count_per_arm=20,
        output_bytes=2*1024**3, final_reserve_bytes=256*1024**2, disk_floor_bytes=200*1024**2,
        wall_seconds=7200, cuda_allocated_bytes=3*1024**3, cuda_reserved_bytes=4*1024**3,
        limits_scope='candidate engineering caps, not scientific gate or equal FLOPs',
        training=False, physics=False, P1_admission=False, freshness=False, physical_release=False, FLOPs=None)


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def build(authority_prepared, authority_sha):
    core.hash_value(authority_sha)
    path = Path(authority_prepared)
    require(path.is_absolute() and path.resolve() == path, 'absolute authority path')
    a = authority.verify(str(path), authority_sha)
    require(a['production_ready'] is False and a['P1_admission'] is False and a['physical_release'] is False,
            'authority remains non-admitted')
    src = a['authority']
    require(src['whole_producer_strict_reconstructed'] is True and len(src['rows']) == 4, 'strict original authority')
    requests = [deepcopy(row['request']) for row in src['rows']]
    for i, q in enumerate(requests):
        method.original.request(q)
        require(q['slot'] == i, 'fixed source slot order')
    bindings = deepcopy(a['bindings'])
    for p in own_files() + [Path(authority.__file__).resolve(), Path(parent.__file__).resolve(), Path(old.__file__).resolve(), path]:
        authority.add(bindings, p, sha(p))
    F.check_bindings(bindings)
    return dict(schema=NAME+'.prepare', arguments=dict(authority_prepared=str(path), authority_sha=authority_sha),
        contract=contract(), requests=requests, bindings=bindings, policy=src['policy'],
        expected_model_hash=src['expected_model_hash'], P1_admission=False, freshness=False, physical_release=False)


def verify(path, expected):
    core.hash_value(expected)
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p and sha(p) == expected, 'prepare SHA/path')
    value = signal.read(p, 256*1024**2)
    require(value['schema'] == NAME+'.prepare' and raw(value) == raw(build(**value['arguments']))
            and sha(p) == expected, 'full source authority rebuild')
    return value


def write(path, value):
    p = Path(path)
    data = raw(value)
    require(p.is_absolute() and p.resolve() == p and p.parent == ROOT/'results'
            and p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink()
            and len(data) <= 256*1024**2, 'exclusive bounded metadata')
    require(shutil.disk_usage(p.parent).free >= contract()['disk_floor_bytes']+len(data), 'disk floor')
    with p.open('xb') as f:
        require(f.write(data) == len(data), 'complete write'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(p.parent)
    return dict(path=str(p), sha256=sha(p))


class Store(F.Store):
    def __init__(self, path, started):
        self.path = Path(path); self.started = started; self.finalizing = False
        require(self.path.is_absolute() and self.path.parent == ROOT/'results' and self.path.name.startswith(NAME+'_')
                and self.path.resolve() == self.path and not self.path.exists() and not self.path.is_symlink()
                and shutil.disk_usage(self.path.parent).free >= contract()['output_bytes']+contract()['disk_floor_bytes'],
                'exclusive output and disk floor')
        self.path.mkdir(); signal.fsync_dir(self.path.parent)

    def check(self, size=4096):
        require(self.finalizing or time.monotonic()-self.started < contract()['wall_seconds'], 'wall cap')
        used = sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+size <= contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
                and shutil.disk_usage(self.path).free >= contract()['disk_floor_bytes']+size, 'output cap/final reserve/floor')


def roster():
    return [dict(index=i, slot=i//2, arm=method.ARMS[i%2], status='not_started', result=None, resources=None)
            for i in range(8)]


def dispatch(p, store, rows, check, factory):
    require(len(p['requests']) == 4 and len(rows) == 8, 'fixed roster')
    for i, row in enumerate(rows):
        q = p['requests'][i//2]; arm = method.ARMS[i%2]; ns = f'a{i:02}'
        require(q['slot'] == row['slot'] == i//2 and row['arm'] == arm, 'ordered arm')
        row['status'] = 'attempted'
        try:
            journal = transport.NativeJournal(store, ns, check)
        except BaseException as exc:
            row.update(status='failed_before_journal', error=repr(exc))
            store.put(ns+'_result.json', row)
            raise
        ledger = core.Ledger(F.caps(98,16), sink=journal.event)
        try:
            backend = factory(q, journal, ledger)
            row['result'] = method.run_arm(q, arm, backend, journal.event)
            require(ledger.attempted == ledger.completed == F.caps(98,16), 'scalar native resource completion')
            if i%2:
                first = rows[i-1]['result']; second = row['result']
                require(all(raw(first[k]) == raw(second[k]) for k in
                    ('initial_Q','baseline_terminal_Q','baseline_reach_value','baseline_commands')), 'same paid baseline')
            row['status'] = 'sealed'
        except BaseException as exc:
            row.update(status='failed_partial', error=repr(exc), result=None)
            raise
        finally:
            row['resources'] = ledger.receipt()
            journal.close()
            row.update(journal_file=ns+'/events.jsonl', journal_sha256=sha(journal.path), journal_tail=journal.tail)
            store.put(ns+'_result.json', row)


def make_report(p, prepared, expected, store, rows, errors, *, source_ok, model_before=None, model_after=None,
                host=None, precision_before=None, precision_after=None, peaks=None, wall=None):
    files = {str(f.relative_to(store.path)): sha(f) for f in store.path.rglob('*') if f.is_file()}
    good = not errors and source_ok and model_before is not None and model_before == model_after and all(r['status']=='sealed' for r in rows)
    return dict(schema=NAME+'.report', status='complete_model_only' if good else 'partial_failclosed',
        contract=contract(), prepared=prepared, prepare_sha256=expected, rows=rows, errors=errors,
        slot_denominator=4, arm_denominator=8, candidate_denominator=160, source_verified=source_ok,
        source_manifest_sha256=digest(p['bindings']), model_before=model_before, model_after=model_after,
        host=host, precision_before=precision_before, precision_after=precision_after,
        CUDA_peak_bytes=peaks, wall_seconds=wall, artifact_sha256=files,
        P1_admission=False, freshness=False, physical_release=False)


def signal_account(events):
    """Reconstruct observed H1 resource prefixes; not hidden-call attestation."""
    totals = {kind: dict.fromkeys(core.AXES, 0) for kind in ('attempted','completed')}
    active = None; host_closed = True; returns = 0
    for event in events:
        kind = event['kind']
        if kind == 'joint_signal_attempt':
            require(active is None, 'nested signal')
            active = {'label': event['label'], 'events': [], 'attempted': dict.fromkeys(core.AXES, 0), 'stack': []}
        elif kind == 'joint_signal_event':
            require(active is not None and event['label'] == active['label'], 'signal event ordering')
            item = event['payload']; active['events'].append(item)
            if item['channel'] == 'resource':
                op = item['payload']; entry = op['entry']
                require(entry['delta'] == core.delta(entry['resource'],entry['shape'],entry['directions']), 'signal resource delta')
                if op['kind'] == 'resource_attempt':
                    require(entry['status'] == 'attempted', 'signal attempt status')
                    active['stack'].append(entry)
                elif op['kind'] in ('resource_return','resource_failure'):
                    require(active['stack'], 'signal resource missing attempt')
                    before = active['stack'].pop()
                    require(all(before[k] == entry[k] for k in ('index','scope','resource','shape','directions','delta','started_ns'))
                        and entry['status'] == ('returned' if op['kind'] == 'resource_return' else 'failed'), 'signal resource terminal')
                else:
                    require(op['kind'] == 'resource_blocked' and entry['status'] == 'blocked_before_dispatch', 'signal resource event')
                if op['kind'] in ('resource_attempt','resource_return'):
                    key = 'attempted' if op['kind'] == 'resource_attempt' else 'completed'
                    for k in core.AXES:
                        totals[key][k] += entry['delta'][k]
                        if key == 'attempted': active['attempted'][k] += entry['delta'][k]
        elif kind in ('joint_signal_return','joint_signal_failure'):
            require(active is not None and event['label'] == active['label'], 'signal terminal ordering')
            require(not active['stack'], 'signal resource incomplete')
            receipt = event.get('receipt')
            if receipt is not None:
                require(raw(receipt['events']) == raw(active['events']) and receipt['resources'] == active['attempted'], 'signal receipt prefix')
            if kind == 'joint_signal_return':
                require(receipt is not None and receipt['status'] == 'complete' and
                    receipt['resources'] == component.expected('postcontext_H1'), 'signal complete resources')
                host_closed &= receipt['kind'] == 'HOST_CUDA' and receipt['native_closure_verified'] is True
                returns += 1
            active = None
    require(active is None, 'unterminated signal prefix')
    return dict(attempted=totals['attempted'], completed=totals['completed'], returned_signals=returns,
                host_native_closure=host_closed)


def run(prepared, expected, output, authorization):
    require(authorization == AUTH, 'MAIN explicit CUDA authorization')
    started = time.monotonic(); p = verify(prepared, expected); store = Store(output, started); rows = roster()
    errors = []; model = before = after = host = pb = pa = peaks = flags = None; source_ok = False
    try:
        store.put('binding.json', dict(prepared=prepared, prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        def check():
            store.check(); old.precision(torch)
            require(torch.cuda.memory_allocated(device) <= contract()['cuda_allocated_bytes'] and
                    torch.cuda.memory_reserved(device) <= contract()['cuda_reserved_bytes'], 'GPU cap')
        with torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            check(); model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False); check()
            before = signal.model_hash(model)
            require(before['original_state_dict'] == p['expected_model_hash'], 'model identity')
            for module in model.modules():
                require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'], 'model class source binding')
            pb = old.precision(torch)
            def factory(q, journal, ledger):
                scalar = old.Backend(model, device, q['task'], journal, ledger, check)
                return callback.NativeCallbacks.host(q['task'], scalar, model, device, check, journal.event, main_authorized=True)
            dispatch(p, store, rows, check, factory)
            pa = old.precision(torch); torch.cuda.synchronize(device); check()
    except BaseException as exc:
        errors.append(dict(stage='run', error=repr(exc)))
    finally:
        store.finalizing = True
        try:
            if model is not None:
                old.midpoint.bridge.native.frozen(model); after = signal.model_hash(model)
                require(before == after, 'model unchanged')
                peaks = [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)]
            F.check_bindings(p['bindings']); require(sha(prepared) == expected, 'prepare/source closure'); source_ok = True
        except BaseException as exc:
            errors.append(dict(stage='closure', error=repr(exc)))
        wall = time.monotonic()-started
        if wall >= contract()['wall_seconds']: errors.append(dict(stage='wall', error='wall cap'))
        if flags is not None: torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
        report = make_report(p, prepared, expected, store, rows, errors, source_ok=source_ok,
            model_before=before, model_after=after, host=host, precision_before=pb, precision_after=pa, peaks=peaks, wall=wall)
        store.put('report.json', report, final=True)
    return dict(status=report['status'], output=str(store.path), report_sha256=sha(store.path/'report.json'), errors=errors)


def consume(output, expected):
    output = Path(output); core.hash_value(expected)
    require(output.is_absolute() and output.resolve() == output and sha(output/'report.json') == expected, 'report SHA/path')
    r = signal.read(output/'report.json', 256*1024**2); p = verify(r['prepared'], r['prepare_sha256'])
    require(r['schema'] == NAME+'.report' and raw(r['contract']) == raw(contract()) and
        (r['slot_denominator'], r['arm_denominator'], r['candidate_denominator']) == (4,8,160) and len(r['rows']) == 8
        and all(r[k] is False for k in ('P1_admission','freshness','physical_release')), 'fixed denominator and gates')
    require(r['status'] in ('complete_model_only','partial_failclosed') and type(r['errors']) is list
        and (r['status'] != 'partial_failclosed' or bool(r['errors'])), 'disposition')
    require(signal.read(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']), 'binding')
    files = {str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}
    require(files == set(r['artifact_sha256']) | {'report.json'} and
        sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) <= contract()['output_bytes'], 'file census')
    for name, h in r['artifact_sha256'].items():
        path = Path(name)
        require(not path.is_absolute() and all(x not in ('.','..') for x in path.parts)
            and (output/path).resolve() == output/path and sha(output/path) == h, 'artifact SHA/path')
    receipts = []; audits = []; signals = []; stopped = False
    for i, row in enumerate(r['rows']):
        q = p['requests'][i//2]
        require((row['index'],row['slot'],row['arm']) == (i,i//2,method.ARMS[i%2]) and q['slot'] == i//2, 'arm identity')
        if row['resources'] is None:
            require(row['result'] is None, 'no result without ledger')
            if row['status'] == 'failed_before_journal':
                require(not stopped and signal.read(output/f'a{i:02}_result.json') == row, 'durable journal failure')
            else:
                require(row['status'] == 'not_started' and (stopped or bool(r['errors'])), 'unreached arm')
            stopped = True; audits.append(None); continue
        require(not stopped and signal.read(output/f'a{i:02}_result.json',256*1024**2) == row and
            row['journal_file'] == f'a{i:02}/events.jsonl', 'durable ordered arm')
        events = F.load_journal(output,row)
        signals.append(signal_account(events))
        scalar_events = [e for e in events if e['kind'].startswith('resource_') and 'entry' in e]
        transport.audit_resources(row['resources'],scalar_events); receipts.append(row['resources'])
        if row['status'] == 'sealed':
            audits.append(audit.audit(q,row['arm'],row['result'],row['resources'],events))
            if i%2:
                first=r['rows'][i-1]['result']; second=row['result']
                require(all(raw(first[k]) == raw(second[k]) for k in ('initial_Q','baseline_terminal_Q','baseline_reach_value','baseline_commands')), 'baseline mismatch')
        else:
            require(row['status'] == 'failed_partial' and row['result'] is None and type(row.get('error')) is str, 'partial arm')
            stopped = True; audits.append(None)
    counts = {kind:{k:sum(x[kind][k] for x in receipts) for k in core.AXES} for kind in ('attempted','completed')}
    complete = r['status'] == 'complete_model_only' and not r['errors'] and all(x is not None for x in audits)
    require((r['status'] == 'complete_model_only') == complete, 'complete evidence')
    if complete:
        total = {k:8*v for k,v in F.caps(98,16).items()}
        require(counts['attempted'] == counts['completed'] == total, 'full scalar native resources')
        require(len(signals) == 8 and all(s['returned_signals'] == 20 and s['host_native_closure'] for s in signals),
            'all signal receipts require native HOST closure')
        require(r['source_verified'] is True and r['source_manifest_sha256'] == digest(p['bindings']) and
            r['model_before'] == r['model_after'] and r['model_before']['original_state_dict'] == p['expected_model_hash'], 'model/source closure')
        require(r['precision_before'] == r['precision_after'] == PRECISION, 'precision closure')
        require(0 <= old.scalar(r['wall_seconds'],'wall') < contract()['wall_seconds'] and
            type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes']) == 2 and
            all(type(x) is int and 0 <= x <= contract()[k] for x,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)), 'wall/GPU cap')
    F.check_bindings(p['bindings']); require(sha(output/'report.json') == expected, 'unchanged report')
    return dict(schema=NAME+'.consume', status='verified_model_only' if complete else 'partial_failclosed', complete=complete,
        report_sha256=expected, rows=r['rows'], audits=audits, scalar_resources=counts, signal_resources=signals, P1_admission=False,
        freshness=False, physical_release=False, FLOPs=None,
        limitation='saved transcript verification; not hardware attestation, physical result or fresh roots')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','verify','run','consume'))
    for key in ('authority-prepared','authority-sha','prepared','prepare-sha','output','authorization','report-sha','receipt'):
        parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command == 'prepare': value=write(a.output,build(a.authority_prepared,a.authority_sha))
    elif a.command == 'verify': verify(a.prepared,a.prepare_sha); value=dict(verified=True)
    elif a.command == 'run': value=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else: value=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(value).decode(),flush=True)


if __name__ == '__main__': main()
