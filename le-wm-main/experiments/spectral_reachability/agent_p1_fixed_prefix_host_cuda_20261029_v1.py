"""Fixed 64-group host, MAIN authorized CUDA only; saved-only strict consumer."""
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import time

from experiments.spectral_reachability import agent_p1_fixed_prefix_refinement_20261029_v1 as method
from experiments.spectral_reachability import agent_p1_fixed_prefix_saved_audit_20261029_v1 as audit
from experiments.spectral_reachability import agent_p1_two_packet_host_cuda_20261029_v1 as parent
from experiments.spectral_reachability import agent_nonlinear_suffix_observed_recompile_20261028_v2 as old

NAME = 'agent_p1_fixed_prefix_host_cuda_20261029_v1'
AUTH = 'MAIN_P1_FIXED_PREFIX_HOST_CUDA_ONLY'
F, core, signal, transport = parent.F, parent.core, parent.signal, parent.transport
require, sha, raw, digest = F.require, F.sha, F.raw, F.digest
ROOT = F.ROOT
PINS = ((method, 'e01a7ffb4e65c64574a392257476e8ab3edc925f9093052aa18c847d364e8c84'),
        (audit, '98ac53975adea76fc7181b9917693b255f9aac19e46492b8a041b28516aeec53'),
        (parent, 'a5f4333e12363208e5d333de01fb20cff68e13170ed6b76dfb1d25d8e04836ed'))


def contract():
    return dict(schema=NAME+'.contract', method=method.contract(), per_group=F.caps(19, 8),
        new_total=F.caps(1216, 512), inherited=F.caps(288, 128), combined=F.caps(1504, 640),
        fixed_slots=4, group_denominator=64, candidate_denominator=576, stop_on_first_failure=True,
        output_bytes=2*1024**3, final_reserve_bytes=256*1024**2, disk_floor_bytes=200*1024**2,
        wall_seconds=7200, cuda_allocated_bytes=3*1024**3, cuda_reserved_bytes=4*1024**3,
        wall_scope='prepare verification through pre-report sample; final report fsync excluded',
        training=False, physics=False, P1_admission=False, freshness=False, physical_release=False, FLOPs=None)


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]


def build(parent_output, parent_report_sha, parent_prepared, parent_prepare_sha, parent_receipt, parent_receipt_sha):
    for module, h in PINS:
        require(sha(module.__file__) == h, 'frozen upstream source')
    for path, h in ((parent_prepared, parent_prepare_sha), (parent_receipt, parent_receipt_sha)):
        core.hash_value(h)
        require(Path(path).is_absolute() and Path(path).resolve() == Path(path) and sha(path) == h, 'absolute parent artifact and SHA')
    output = Path(parent_output)
    core.hash_value(parent_report_sha)
    require(output.is_absolute() and output.resolve() == output and sha(output/'report.json') == parent_report_sha, 'actual parent report')
    report = signal.read(output/'report.json', 128*1024**2)
    require(report['prepared'] == parent_prepared and report['prepare_sha256'] == parent_prepare_sha, 'exact parent prepare')
    accepted = parent.consume(str(output), parent_report_sha)
    require(Path(parent_receipt).read_bytes() == raw(accepted) and sha(parent_receipt) == parent_receipt_sha,
            'parent full strict equals actual persisted receipt bytes')
    prepared = signal.read(parent_prepared, 128*1024**2)
    groups = method.requests(prepared, report, accepted, parent_report_sha)
    bindings = deepcopy(prepared['bindings'])
    for module, h in PINS:
        bindings[str(Path(module.__file__).resolve())] = h
        stem = module.__name__.rsplit('.', 1)[-1]
        for path in (ROOT/'tests'/('test_'+stem+'.py'), ROOT/'innerdocs'/(stem+'.md')):
            bindings[str(path)] = sha(path)
    for path in own_files(): bindings[str(path)] = sha(path)
    for path, h in ((output/'report.json', parent_report_sha), (Path(parent_prepared), parent_prepare_sha),
                    (Path(parent_receipt), parent_receipt_sha)):
        bindings[str(path)] = h
    for name, h in report['artifact_sha256'].items(): bindings[str(output/name)] = h
    for v in F.installed_class_sources().values():
        require(v['path'] not in bindings or bindings[v['path']] == v['sha256'], 'class source collision')
        bindings[v['path']] = v['sha256']
    F.check_bindings(bindings)
    args = dict(parent_output=parent_output, parent_report_sha=parent_report_sha, parent_prepared=parent_prepared,
                parent_prepare_sha=parent_prepare_sha, parent_receipt=parent_receipt, parent_receipt_sha=parent_receipt_sha)
    return dict(schema=NAME+'.prepare', arguments=args, contract=contract(), groups=groups,
                policy=prepared['authority']['policy'], expected_model_hash=prepared['authority']['expected_model_hash'],
                bindings=bindings, P1_admission=False, freshness=False, physical_release=False)


def verify(path, expected):
    core.hash_value(expected)
    require(sha(path) == expected, 'actual prepare SHA')
    p = signal.read(path, 256*1024**2)
    require(raw(p) == raw(build(**p['arguments'])) and sha(path) == expected, 'rebuild full parent strict and prepare')
    return p


def write(path, value):
    path = Path(path).absolute(); payload = raw(value)
    require(path.parent == ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve() == path
            and not path.exists() and not path.is_symlink() and len(payload) <= 256*1024**2, 'exclusive bounded metadata')
    require(shutil.disk_usage(path.parent).free >= contract()['disk_floor_bytes']+len(payload), 'disk floor')
    with path.open('xb') as f:
        require(f.write(payload) == len(payload), 'full metadata write'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(path.parent)
    return dict(path=str(path), sha256=sha(path))


class Store(F.Store):
    def __init__(self, path, started):
        self.path = Path(path).absolute(); self.started = started; self.finalizing = False
        require(self.path.parent == ROOT/'results' and self.path.name.startswith(NAME+'_') and self.path.resolve() == self.path
                and not self.path.exists() and shutil.disk_usage(self.path.parent).free >= contract()['output_bytes']+contract()['disk_floor_bytes'], 'exclusive output and disk floor')
        self.path.mkdir(); signal.fsync_dir(self.path.parent)
    def check(self, size=4096):
        require(self.finalizing or time.monotonic()-self.started < contract()['wall_seconds'], 'wall watchdog')
        used = sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+size <= contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
                and shutil.disk_usage(self.path).free >= contract()['disk_floor_bytes']+size, 'output cap/reserve/floor')


def roster():
    return [dict(index=i, status='not_started', result=None, resources=None) for i in range(64)]


def dispatch(p, store, rows, check, factory):
    require(len(p['groups']) == len(rows) == 64, 'fixed64')
    for i, (q, row) in enumerate(zip(p['groups'], rows, strict=True)):
        require(q['index'] == row['index'] == i, 'ordered group identity')
        row['status'] = 'attempted'; ns = f'g{i:02}'
        try:
            journal = transport.NativeJournal(store, ns, check)
        except BaseException as exc:
            row.update(status='failed_before_journal', error=repr(exc))
            store.put(ns+'_result.json', row)
            raise
        ledger = core.Ledger(F.caps(19, 8), sink=journal.event)
        try:
            backend = factory(q['task'], journal, ledger)
            with backend.observe(): row['result'] = method.run_group(q, backend, journal.event)
            require(ledger.attempted == ledger.completed == F.caps(19, 8), 'all group resources charged')
            row['status'] = 'sealed'
        finally:
            row['resources'] = ledger.receipt(); journal.close()
            row.update(journal_file=ns+'/events.jsonl', journal_sha256=sha(journal.path), journal_tail=journal.tail)
            if row['status'] != 'sealed': row['status'] = 'failed_partial'
            store.put(ns+'_result.json', row)


def run(prepared, expected, output, authorization):
    require(authorization == AUTH, 'MAIN explicit CUDA authorization')
    started = time.monotonic(); p = verify(prepared, expected); store = Store(output, started); rows = roster()
    errors = []; model = before = after = host = pb = pa = peaks = flags = None; source_ok = False
    try:
        store.put('binding.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=expected))
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
            before = signal.model_hash(model); require(before['original_state_dict'] == p['expected_model_hash'], 'model identity')
            for module in model.modules(): require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'], 'model class binding')
            pb = old.precision(torch)
            dispatch(p, store, rows, check, lambda task, journal, ledger: old.Backend(model, device, task, journal, ledger, check))
            pa = old.precision(torch); torch.cuda.synchronize(device); check()
    except BaseException as exc: errors.append(dict(stage='run', error=repr(exc)))
    finally:
        store.finalizing = True
        try:
            if model is not None:
                old.midpoint.bridge.native.frozen(model); after = signal.model_hash(model); require(before == after, 'model unchanged')
                peaks = [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)]
            F.check_bindings(p['bindings']); require(sha(prepared) == expected, 'prepare/source closure'); source_ok = True
        except BaseException as exc: errors.append(dict(stage='closure', error=repr(exc)))
        wall = time.monotonic()-started
        if wall >= contract()['wall_seconds']: errors.append(dict(stage='wall', error='wall cap'))
        if flags is not None: torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
        complete = not errors and source_ok and before is not None and before == after and all(x['status'] == 'sealed' for x in rows)
        artifacts = {str(f.relative_to(store.path)): sha(f) for f in store.path.rglob('*') if f.is_file()}
        report = dict(schema=NAME+'.report', status='complete_model_only' if complete else 'partial_failclosed', contract=contract(),
            prepared=str(Path(prepared).absolute()), prepare_sha256=expected, rows=rows, errors=errors, fixed_denominator=64,
            candidate_denominator=576, source_verified=source_ok, source_manifest_sha256=digest(p['bindings']),
            model_before=before, model_after=after, host=host, precision_before=pb, precision_after=pa,
            CUDA_peak_bytes=peaks, wall_seconds=wall, artifact_sha256=artifacts,
            P1_admission=False, freshness=False, physical_release=False)
        store.put('report.json', report, final=True)
    return dict(status=report['status'], report_sha256=sha(store.path/'report.json'), output=str(store.path), errors=errors)


def consume(output, expected):
    output = Path(output).absolute(); core.hash_value(expected); require(sha(output/'report.json') == expected, 'report SHA')
    r = signal.read(output/'report.json', 256*1024**2); p = verify(r['prepared'], r['prepare_sha256'])
    require(r['schema'] == NAME+'.report' and raw(r['contract']) == raw(contract()) and len(r['rows']) == r['fixed_denominator'] == 64
            and r['candidate_denominator'] == 576 and all(r[k] is False for k in ('P1_admission','freshness','physical_release')),
            'fixed denominator and gates')
    require(r['status'] in ('complete_model_only', 'partial_failclosed') and type(r['errors']) is list, 'disposition')
    require(r['status'] != 'partial_failclosed' or bool(r['errors']), 'partial must state failure')
    require(signal.read(output/'binding.json') == dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']), 'execution binding')
    files = {str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}
    require(files == set(r['artifact_sha256']) | {'report.json'} and sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) <= contract()['output_bytes'], 'file census')
    for name, h in r['artifact_sha256'].items():
        path = Path(name)
        require(not path.is_absolute() and all(x not in ('.','..') for x in path.parts) and
                (output/path).resolve() == output/path and sha(output/path) == h, 'artifact SHA/path')
    receipts = []; audits = []; stopped = False
    for i, (q, row) in enumerate(zip(p['groups'], r['rows'], strict=True)):
        require(q['index'] == row['index'] == i, 'ordered groups')
        if row['resources'] is None:
            require(row['result'] is None, 'no result without ledger')
            if row['status'] == 'failed_before_journal':
                require(not stopped and type(row.get('error')) is str and signal.read(output/f'g{i:02}_result.json', 32*1024**2) == row, 'durable journal failure')
            else: require(row['status'] == 'not_started', 'unreached group')
            stopped = True; audits.append(None); continue
        require(not stopped and signal.read(output/f'g{i:02}_result.json', 32*1024**2) == row, 'durable ordered group')
        require(row['journal_file'] == f'g{i:02}/events.jsonl', 'group journal namespace')
        events = F.load_journal(output, row); transport.audit_resources(row['resources'], events); receipts.append(row['resources'])
        if row['status'] == 'sealed': audits.append(audit.audit(q, row['result'], row['resources'], events))
        else:
            require(row['status'] == 'failed_partial', 'partial group retained'); stopped = True; audits.append(None)
    counts = {kind: {k: sum(x[kind][k] for x in receipts) for k in core.AXES} for kind in ('attempted','completed')}
    complete = r['status'] == 'complete_model_only' and not r['errors'] and all(x is not None for x in audits)
    require((r['status'] == 'complete_model_only') == complete, 'complete evidence')
    if complete:
        require(counts['attempted'] == counts['completed'] == F.caps(1216,512), 'full paid group resources')
        require(r['source_verified'] is True and r['model_before'] == r['model_after'] and
                r['model_before']['original_state_dict'] == p['expected_model_hash'] and r['source_manifest_sha256'] == digest(p['bindings']), 'model/source closure')
        precision = dict(matmul_TF32=False, cudnn_TF32=False, autocast=False, math=True, flash=False, efficient=False, cudnn_sdp=False)
        require(r['precision_before'] == r['precision_after'] == precision, 'precision closure')
        require(0 <= old.scalar(r['wall_seconds'], 'wall') < contract()['wall_seconds'] and len(r['CUDA_peak_bytes']) == 2 and
                all(type(x) is int and 0 <= x <= contract()[k] for x, k in zip(r['CUDA_peak_bytes'], ('cuda_allocated_bytes','cuda_reserved_bytes'), strict=True)), 'wall/GPU caps')
    F.check_bindings(p['bindings']); require(sha(output/'report.json') == expected, 'unchanged report')
    return dict(schema=NAME+'.consume', status='verified_model_only' if complete else 'partial_failclosed', complete=complete,
                report_sha256=expected, group_denominator=64, candidate_denominator=576, rows=r['rows'], audits=audits,
                actual_new_compute=counts, P1_admission=False, freshness=False, physical_release=False, FLOPs=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','verify','run','consume'))
    for k in ('parent-output','parent-report-sha','parent-prepared','parent-prepare-sha','parent-receipt','parent-receipt-sha',
              'output','prepared','prepare-sha','report-sha','receipt','authorization'): parser.add_argument('--'+k)
    a = parser.parse_args()
    if a.command == 'prepare': v = write(a.output, build(a.parent_output, a.parent_report_sha, a.parent_prepared, a.parent_prepare_sha, a.parent_receipt, a.parent_receipt_sha))
    elif a.command == 'verify': verify(a.prepared, a.prepare_sha); v = dict(verified=True)
    elif a.command == 'run': v = run(a.prepared, a.prepare_sha, a.output, a.authorization)
    else: v = write(a.receipt, consume(a.output, a.report_sha))
    print(raw(v).decode(), flush=True)


if __name__ == '__main__': main()
