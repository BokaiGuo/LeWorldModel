"""Own full30 prepare/HostCUDA/saved strict; no producer globals or receipt impersonation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import inspect
from importlib.machinery import PathFinder
import io
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np
from experiments.spectral_reachability import agent_p2_three_program_20261028_v1 as native
from experiments.spectral_reachability import agent_p2_saved_program_linker_20261029_v1 as linker
from experiments.spectral_reachability import agent_p2_two_program_postcontext_20261029_v1_kernel as K

NAME, ROOT = K.NAME, native.ROOT
AUTH = 'MAIN_TWO_PROGRAM_FULL30_KNOWN2_HOST_CUDA_ONLY'
LINK = dict(path=str(ROOT/'results/agent_p2_saved_program_linker_20261029_v1_MAIN01.json'),
            sha256='82f73b810e5cd34b12907f11d0d063914578263ae4e41163e9581ac3ce3f5567')
PROTOCOL = ROOT/'innerdocs/agent_p2_two_program_postcontext_protocol_20261029_v1.md'
PROTOCOL_SHA = '4176575497bcf9c7395438613af0460233a9d94abe891b2023e821cf8395da99'
CAP, RESERVE, FLOOR, META = 128*1024**2, 2*1024**2, 200*1024**2, 32*1024**2
raw, digest, require, equal, sha = K.raw, K.digest, K.require, K.equal, native.sha
CLASS_MODULES = tuple('torch.nn.modules.'+s for s in
                      ('activation', 'batchnorm', 'container', 'conv', 'dropout', 'linear', 'normalization'))+('transformers.activations',)


def read(path, h, cap=META, gz=False):
    if gz:
        p = Path(path)
        require(p.is_file() and not p.is_symlink() and p.stat().st_size <= cap and sha(p) == h, 'bounded gzip authority')
        decoder = zlib.decompressobj(31)
        data = decoder.decompress(p.read_bytes(), cap+1)
        require(len(data) <= cap and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
                'single complete bounded gzip member, no tail')
        require(sha(p) == h, 'gzip changed during read')
        return linker.decode(data)
    return linker.read_pin(path, h, cap, gz=gz)


def merge(dst, source):
    for path, h in source.items():
        require(path not in dst or dst[path] == h, 'conflicting immutable binding')
        dst[path] = h


def rehash(bindings):
    for path, h in bindings.items():
        p = Path(path)
        require(p.is_absolute() and p.is_file() and not p.is_symlink() and sha(p) == h, 'immutable file '+path)


def installed_sources():
    """Metadata only: no torch/transformers package execution."""
    out = {}
    for name in CLASS_MODULES:
        search = None
        parts = name.split('.')
        for i in range(len(parts)):
            spec = PathFinder.find_spec('.'.join(parts[:i+1]), search)
            require(spec is not None, 'installed source metadata '+name)
            if i+1 < len(parts):
                require(spec.submodule_search_locations is not None, 'installed package metadata')
                search = list(spec.submodule_search_locations)
        path = Path(spec.origin).resolve()
        require(path.is_file() and path.suffix == '.py', 'installed Python source')
        out[str(path)] = sha(path)
    require(len(out) == 8, 'all eight distinct class sources')
    return out


def files():
    return sorted(set(native.files()) | set(linker.own_files()) | {
        Path(__file__).resolve(), Path(K.__file__).resolve(), PROTOCOL,
        ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')})


def material(catalog, original):
    require(catalog['schema'] == linker.NAME+'.catalog' and catalog['arguments']['requests'] is None,
            'original complete default saved catalog')
    equal(catalog['arguments']['authorities'], linker.ACTUAL, 'pinned original authorities')
    equal(catalog['denominator'], dict(candidate_rows=168, candidates_per_root_per_arm=42,
          default_link_attempts=168, known_roots=2, programs=336), 'full source denominator')
    require(len(catalog['catalog']) == 336 and len(catalog['candidates']) == 168, 'no source candidate pruning')
    by_id = {p['id']: p for p in catalog['catalog']}
    require(len(by_id) == 336 and len(original['slots']) == 2, 'unique complete program ids')
    slots = []
    for root, old in enumerate(original['slots']):
        require(type(old['root_slot']) is int and old['root_slot'] == root and old['condition'] == 'off', 'known2 off order')
        K.context(old['packet']['root_context'])
        require(np.array_equal(K.finite(old['q'], (192, 192)), np.eye(192)), 'original identity Q')
        programs = []
        for candidate in (1, 2):
            p = deepcopy(by_id[f'r{root}/joint_H3_no_composition/c{candidate:02}/p0'])
            h = p.pop('program_sha256')
            require(digest(p) == h, 'literal saved capsule digest')
            p['program_sha256'] = h
            require(p['status'] == 'linked_saved_program' and p['missing'] == [] and
                    p['planned_steps'] == p['committed_steps'] == 15 and len(p['contexts']) == 3, 'complete full program')
            equal(p['input_context'], old['packet']['root_context'], 'original root full context')
            s = p['scope']
            require(s['root_slot'] == root and s['condition'] == 'off' and s['task_id'] == old['task_id'] and
                    s['task_sha256'] == old['task_sha256'] and s['model_hash'] == original['expected_model_hash'] and
                    s['goal_sha256'] == digest(old['packet']['goal']) and s['Q_sha256'] == digest(old['q']), 'original task/goal/Q/model')
            equal(s['normalization'], dict(mean=native.kernel.ACTION_MEAN.tolist(), std=native.kernel.ACTION_STD.tolist()),
                  'unchanged native normalizer')
            K.legal(p['logits'], K.C15)
            a = K.chart(p['logits'])
            require(a.tobytes().hex() == p['raw_float32_bits'] and
                    a.tobytes() == np.asarray(p['commands'], '<f4').tobytes(), 'original literal action chart bits')
            for c in p['contexts']:
                K.context(c)
            programs.append(p)
        K.legal(programs[0]['logits']+programs[1]['logits'])
        warm = K.finite(old['warm_raw10'], (10, 2)).astype(np.float32)
        require(np.array_equal(warm, np.asarray(old['warm_raw10'])) and (abs(warm) <= 1).all(), 'literal legal root raw10')
        equal(native.kernel.normalized_history(warm[:5]).tolist()+native.kernel.normalized_history(warm[5:]).tolist(),
              old['packet']['root_context']['action_history'], 'causal warm10 normalization exactly once')
        slots.append(dict(root_slot=root, condition='off', task_id=old['task_id'], task_file=old['task_file'],
                          task_sha256=old['task_sha256'], blind_sha256=old['blind_sha256'],
                          root_context=deepcopy(old['packet']['root_context']), goal=deepcopy(old['packet']['goal']),
                          root_warm_raw10=old['warm_raw10'], programs=programs))
    return slots


def build(link_path=LINK['path'], link_sha=LINK['sha256']):
    require(str(Path(link_path).absolute()) == LINK['path'] and link_sha == LINK['sha256'], 'only pinned MAIN linker')
    c = read(link_path, link_sha)
    equal(c['arguments']['authorities'], linker.ACTUAL, 'inherited MAIN authorities')
    a = c['arguments']['authorities']
    p, r, receipt = [read(a[k]['path'], a[k]['sha256'], linker.CAPS[k]) for k in ('prepared', 'report', 'receipt')]
    require(p['schema'] == native.NAME+'.prepare' and receipt['schema'] == native.NAME+'.consume' and
            receipt['status'] == 'full15_engineering_verified_science_blocked' and receipt['errors'] == [] and
            receipt['planner_report_sha256'] == a['report']['sha256'] and r['prepare_sha256'] == a['prepared']['sha256'] and
            r['model_before'] == r['model_after'] == p['expected_model_hash'] and r['errors'] == [], 'actual inherited complete receipt join')
    bindings = dict(p['bindings_sha256'])
    merge(bindings, c['bindings'])
    merge(bindings, {link_path: link_sha, **{d['path']: d['sha256'] for d in a.values()}})
    merge(bindings, {str(Path(a['report']['path']).parent/n): h for n, h in r['artifact_sha256'].items()})
    require(sha(PROTOCOL) == PROTOCOL_SHA, 'approved protocol unchanged')
    merge(bindings, {str(f.resolve()): sha(f) for f in files()})
    merge(bindings, installed_sources())
    slots = material(c, p)
    rehash(bindings)
    inherited = dict(original_planner=deepcopy(r['actual_resources']), original_encoding=deepcopy(p['contract']['inherited_encoding']),
                     linker_new_model_calls=0, source_authorities=deepcopy(a), charged_again_as_new=False,
                     later_wallcapped_or_physics_not_used_as_input=True)
    out = dict(schema=NAME+'.prepare', linker=dict(path=link_path, sha256=link_sha), contract=K.contract(),
               original_authorities=a, slots=slots, policy=p['policy'], expected_model_hash=p['expected_model_hash'],
               bindings_sha256=bindings, inherited_resources=inherited, full_upstream_rebuilt_here=False,
               inherited_authority='pinned MAIN complete receipt + saved linker; no upstream fullstrict rerun', **K.FLAGS)
    require(len(raw(out)) <= META, 'bounded own prepare')
    return out


def write_new(path, value):
    p = Path(path).absolute()
    data = raw(value)
    require(p.name.startswith(NAME+'_') and p.parent.is_dir() and not p.is_symlink() and len(data) <= META and
            shutil.disk_usage(p.parent).free >= FLOOR+len(data), 'own exclusive bounded metadata')
    with p.open('xb') as f:
        require(f.write(data) == len(data), 'short metadata')
        f.flush()
        os.fsync(f.fileno())
    native.fsync_dir(p.parent)
    return dict(path=str(p), sha256=sha(p))


def verify(path, h):
    p = read(path, h)
    equal(p, build(p['linker']['path'], p['linker']['sha256']), 'entire own prepare reconstruction')
    return p


class Store:
    """Single-LF native hashchain; inclusive clock; incremental physical byte cap."""
    def __init__(self, output, started):
        self.output = Path(output).absolute()
        require(self.output.name.startswith(NAME+'_') and not self.output.exists() and not self.output.is_symlink() and
                shutil.disk_usage(self.output.parent).free >= CAP+FLOOR, 'exclusive output and floor')
        self.output.mkdir()
        self.started, self.slot_started = started, None
        self.index, self.previous, self.used, self.closed = 0, '0'*64, 0, False
        self.metadata_used = 0
        self.comp = zlib.compressobj(6, zlib.DEFLATED, 31)
        self._write('events.jsonl.gz', b'', final=True)

    def check(self):
        now = time.monotonic()
        require(now-self.started <= 1800 and (self.slot_started is None or now-self.slot_started <= 450), 'whole/slot inclusive wall')

    def _write(self, name, data, append=False, final=False, time_guard=True):
        if not final and time_guard:
            self.check()
        require(Path(name).name == name and name not in ('', '.', '..') and not (self.output/name).is_symlink(), 'flat artifact')
        require(self.used+len(data) <= CAP-(0 if final else RESERVE) and
                shutil.disk_usage(self.output).free >= FLOOR+len(data), 'output cap/reserve/floor')
        with (self.output/name).open('ab' if append else 'xb') as f:
            require(f.write(data) == len(data), 'short artifact')
            f.flush()
            os.fsync(f.fileno())
        self.used += len(data)
        native.fsync_dir(self.output)

    def put(self, name, value, final=False):
        data = raw(value)
        if name.endswith('.gz'):
            data = gzip.compress(data, mtime=0)
        elif name.endswith('.json'):
            require(self.metadata_used+len(data) <= 4*1024**2, 'aggregate runtime JSON metadata4MiB')
            self.metadata_used += len(data)
        self._write(name, data, final=final)

    def event(self, payload):
        require(not self.closed, 'no journal reopen')
        self.check()
        e = dict(index=self.index, previous_sha256=self.previous, payload=deepcopy(payload))
        h = digest(e)
        data = raw(dict(e, event_sha256=h))
        require(self.used+len(data)+128 <= CAP-RESERVE and
                shutil.disk_usage(self.output).free >= FLOOR+len(data)+128, 'admit complete journal record before compression')
        out = self.comp.compress(data)+self.comp.flush(zlib.Z_SYNC_FLUSH)
        self._write('events.jsonl.gz', out, append=True, time_guard=False)
        self.index += 1
        self.previous = h
        self.check()

    def close(self):
        if not self.closed:
            self._write('events.jsonl.gz', self.comp.flush(zlib.Z_FINISH), append=True, final=True)
            self.closed = True


def ad_request(index, c, u, tag):
    K.legal(u)
    require(type(index) is int and 0 <= index < 2, 'two new J per joint slot')
    return dict(index=index, tag=tag, input_context_sha256=digest(c), at_logits=np.asarray(u).tolist(),
                raw=K.chart(u).tolist(), coordinates='raw_incumbent', shape=[192, 60], mathematical_VJP=192,
                vectorize=True, completed=False)


class HostBackend(native.HostBackend):
    """Reuse unchanged observed single-packet bridge; own H6 J, caps and scope."""
    def check(self):
        self.t.cuda.synchronize(self.device)
        self.store.check()
        require(self.t.cuda.max_memory_allocated(self.device) <= K.contract()['allocated_cap_bytes'] and
                self.t.cuda.max_memory_reserved(self.device) <= K.contract()['reserved_cap_bytes'], 'actual allocator peak caps')
        n, j = K.COUNTS[self.row['arm']]
        require(len(self.row['forwards']) <= n and len(self.row['AD']) <= j, 'fixed own call caps')

    def rollout(self, c, commands, tag):
        require(len(self.row['forwards'])+len(commands) <= K.COUNTS[self.row['arm']][0], 'no extra forward dispatch')
        return super().rollout(c, commands, tag)

    def jacobian(self, c, u, tag):
        self.check()
        n = len(self.row['AD'])
        require(self.row['arm'] == K.ARMS[1] and len(self.row['forwards'])+6 <= 96, 'own AD admission')
        request = ad_request(n, c, u, tag)
        self.event(dict(kind='AD_intent', request=request))
        e = dict(request=request, completed=False)
        self.row['AD'].append(e)
        t = self.t
        # The only differentiated input is literal cast32 RAW, not logits or quantization.
        leaf = t.tensor(K.chart(u).reshape(60), device=self.device, dtype=t.float32).requires_grad_(True)
        def fn(value):
            state, path = self._rollout(c, K.chart(u), tag+'/forward', value.reshape(6, 5, 2))
            e['contexts'] = path
            return state[0, -1]
        with t.enable_grad():
            j = t.autograd.functional.jacobian(fn, leaf, vectorize=True)
        self.check()
        require(tuple(j.shape) == (192, 60) and j.dtype == t.float32, 'literal full raw192x60 J')
        a = j.detach().cpu().numpy().copy()
        require(np.isfinite(a).all(), 'finite full J')
        stream = io.BytesIO()
        np.save(stream, a, allow_pickle=False)
        name = f'slot{self.row["slot"]}_J{n}.npy'
        self.store._write(name, stream.getvalue())
        d = dict(file=name, sha256=sha(self.store.output/name), shape=[192, 60], dtype='float32')
        e.update(array=d, completed=True)
        self.last_ad_contexts = deepcopy(e['contexts'])
        self.event(dict(kind='AD_return', index=n, array=d, contexts=e['contexts'], completed=True))
        return a.astype(np.float64)


class SavedBackend(native.SavedBackend):
    def rollout(self, c, commands, tag):
        require(self.n+len(commands) <= K.COUNTS[self.row['arm']][0], 'fixed saved forward cap')
        for f in self.row['forwards'][self.n:self.n+len(commands)]:
            require(set(f) == {'index', 'tag', 'request', 'input', 'raw', 'modules', 'completed', 'output'} and
                    len(f['modules']) == 3 and all(set(m) == {'index', 'resource', 'inputs', 'completed', 'output'}
                    for m in f['modules']), 'exact native full forward and module fields')
            K.context(f['input'])
            K.context(f['output'])
        return super().rollout(c, commands, tag)

    def jacobian(self, c, u, tag):
        require(self.ad < len(self.row['AD']), 'missing J')
        e = self.row['AD'][self.ad]
        request = ad_request(self.ad, c, u, tag)
        require(e['completed'] is True and set(e) == {'request', 'completed', 'contexts', 'array'} and
                e['array']['file'] == f'slot{self.row["slot"]}_J{self.ad}.npy', 'complete own full J authority')
        equal(e['request'], request, 'literal H6 AD request')
        self.event(dict(kind='AD_intent', request=request))
        paths = self.rollout(c, K.chart(u), tag+'/forward')
        equal(paths, e['contexts'], 'AD complete primal context path')
        j = self.load(e['array'])
        require(j.dtype == np.float32 and j.shape == (192, 60), 'strict full192x60 no reduced J')
        self.last_ad_contexts = deepcopy(paths)
        self.event(dict(kind='AD_return', index=self.ad, array=e['array'], contexts=paths, completed=True))
        self.ad += 1
        return j.astype(np.float64)

    def complete(self):
        n, j = K.COUNTS[self.row['arm']]
        require(self.n == len(self.row['forwards']) == n and self.ad == len(self.row['AD']) == j, 'all fixed paid calls')


def classes(model, bindings):
    out = {}
    for module in model.modules():
        p = str(Path(inspect.getfile(type(module))).resolve())
        require(p in bindings and sha(p) == bindings[p], 'all actual loaded class sources presealed, no skip')
        out[p] = bindings[p]
    return out


def row_plan(p, slot, row):
    result = row['result']
    plan = dict(schema=NAME+'.model_plan', root_slot=slot['root_slot'], arm=row['arm'],
                task_id=slot['task_id'], task_sha256=slot['task_sha256'], root_context=slot['root_context'],
                root_warm_raw10=slot['root_warm_raw10'], goal=slot['goal'], model_hash=p['expected_model_hash'],
                source_programs=[dict(id=x['id'], sha256=x['program_sha256']) for x in slot['programs']],
                normalization=slot['programs'][0]['scope']['normalization'], new_raw_steps=30,
                program_count=2 if row['arm'] == K.ARMS[0] else 1,
                program_start_controls=[0, 15] if row['arm'] == K.ARMS[0] else [0],
                new_domain=dict(logit_dim=60, logit_cap=K.C30, raw_bounds=[-1, 1]),
                logits=result['logits'], commands=result['commands'], raw_float32_bits=result['raw_float32_bits'],
                contexts=result['contexts'], predicted_Q=result['final_Q'], **K.FLAGS)
    plan['witness_sha256'] = digest(plan)
    return plan


def run_cuda(prepared, h, output, authorization):
    require(authorization == AUTH, 'explicit MAIN HostCUDA authorization before I/O')
    started = time.monotonic()  # BEFORE whole own saved verify, not after it.
    p = verify(prepared, h)
    store = Store(output, started)
    rows, errors = [], []
    model = before = after = host = loader_memory = None
    source_ok = False
    summaries = [dict(slot=2*r+i, root_slot=r, arm=a, status='not_started', file=None,
                      final_Q=None, contexts=None) for r in (0, 1) for i, a in enumerate(K.ARMS)]
    try:
        store.check()  # Expired verify must not import/dispatch the neural loader.
        store.put('binding.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=h))
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model, before = native.loader.loaded_model(p, torch, device, store, h)
        torch.cuda.synchronize(device)
        store.check()
        loader_memory = dict(allocated_peak=torch.cuda.max_memory_allocated(device), reserved_peak=torch.cuda.max_memory_reserved(device))
        require(loader_memory['allocated_peak'] <= K.contract()['allocated_cap_bytes'] and
                loader_memory['reserved_peak'] <= K.contract()['reserved_cap_bytes'], 'loader allocator peaks before reset')
        store.put('classes.json', classes(model, p['bindings_sha256']))
        with sdpa_kernel(SDPBackend.MATH), torch.autocast(device_type='cuda', enabled=False):
            for summary in summaries:
                slot = p['slots'][summary['root_slot']]
                store.slot_started = time.monotonic()
                torch.cuda.reset_peak_memory_stats(device)
                row = dict(slot=summary['slot'], index=summary['root_slot'], arm=summary['arm'],
                           status='attempted', forwards=[], AD=[])
                rows.append(row)
                summary['status'] = 'attempted'
                store.event(dict(kind='slot_started', slot=row['slot'], root_slot=row['index'], arm=row['arm']))
                emit = lambda e: store.event(dict(slot=row['slot'], event=e))
                backend = HostBackend(model, device, torch, store, row, emit)
                with backend.observe():
                    row['result'] = K.execute(slot, row['arm'], backend)
                require((len(row['forwards']), len(row['AD'])) == K.COUNTS[row['arm']], 'exact own natural counts')
                if row['arm'] == K.ARMS[1]:
                    equal(rows[-2]['result']['contexts'], row['result']['initial_contexts'], 'independent same-action six-context parity')
                row['plan'] = row_plan(p, slot, row)
                row['resources'] = K.resources(row['forwards'], row['AD'])
                row['memory'] = dict(allocated_peak=torch.cuda.max_memory_allocated(device),
                                     reserved_peak=torch.cuda.max_memory_reserved(device))
                row['status'] = 'sealed'
                name = f'slot{row["slot"]}.json.gz'
                store.put(name, row)
                store.event(dict(kind='slot_sealed', slot=row['slot'], witness=row['plan']['witness_sha256']))
                store.check()
                summary.update(status='sealed', file=name, final_Q=row['result']['final_Q'],
                               contexts=row['result']['contexts'], wall_seconds=time.monotonic()-store.slot_started,
                               resources=row['resources'])
                store.slot_started = None
    except BaseException as exc:
        errors.append(native.loader.enc.error('execution', exc))
    finally:
        store.slot_started = None
        for s in summaries:
            if s['status'] == 'attempted':
                s['status'] = 'failed_partial'
        if model is not None:
            try:
                native.loader.enc.frozen(model)
                after = native.loader.enc.tensor_hash(model)
            except BaseException as exc:
                errors.append(native.loader.enc.error('final_model', exc))
        try:
            rehash(p['bindings_sha256'])
            store.check()
            source_ok = True
        except BaseException as exc:
            errors.append(native.loader.enc.error('final_source_wall', exc))
        for row in rows:
            if row['status'] != 'sealed':
                try:
                    store.put(f'partial_slot{row["slot"]}.json.gz', row, final=True)
                except BaseException as exc:
                    errors.append(native.loader.enc.error('partial_persistence', exc))
        try:
            store.close()
        except BaseException as exc:
            errors.append(native.loader.enc.error('journal_close', exc))
        good = not errors and source_ok and before == after == p['expected_model_hash'] and all(s['status'] == 'sealed' for s in summaries)
        actual_resources = census(journal(store.output)) if store.closed else None
        elapsed = time.monotonic()-started
        good = good and elapsed <= 1800
        store.put('seal.json', dict(schema=NAME+'.seal', complete=good,
                  through_durable_data_seconds=elapsed,
                  data_artifacts_sha256={f.name: sha(f) for f in store.output.iterdir()}), final=True)
        report = dict(schema=NAME+'.report', status='all4_full30_sealed_NOT_admitted' if good else 'failed_partial',
                      prepared=str(Path(prepared).absolute()), prepare_sha256=h, contract=K.contract(), rows=summaries,
                      errors=errors, first_error=errors[0] if errors else None, model_before=before, model_after=after,
                      source_verified=source_ok, host=host, wall_seconds=time.monotonic()-started,
                      actual_resources=actual_resources, inherited_resources=p['inherited_resources'], loader_memory=loader_memory,
                      artifact_sha256={f.name: sha(f) for f in store.output.iterdir()}, **K.FLAGS)
        store.put('report.json', report, final=True)
        # Report SHA authenticates the seal. If final publication overruns, preserve an
        # explicit non-complete marker; the strict complete inventory rejects it.
        if good:
            try:
                store.check()
            except BaseException as exc:
                good = False
                store.put('terminal_failure.json', dict(reason='final_publication_wall',
                          error=native.loader.enc.error('final_publication', exc)), final=True)
    require(good, 'failed partial output retained; no retry/refund '+str(errors))
    return dict(report=str(store.output/'report.json'), sha256=sha(store.output/'report.json'))


def journal(output):
    path = Path(output)/'events.jsonl.gz'
    require(path.stat().st_size <= CAP, 'bounded event gzip')
    decoder = zlib.decompressobj(31)
    data = decoder.decompress(path.read_bytes(), CAP+1)
    require(len(data) <= CAP and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and
            (not data or data.endswith(b'\n')), 'single bounded complete gzip / single-LF journal')
    out, previous = [], '0'*64
    for index, line in enumerate(data.splitlines()):
        require(bool(line), 'no empty-record filtering')
        e = linker.decode(line)
        require(set(e) == {'index', 'previous_sha256', 'payload', 'event_sha256'} and
                type(e['index']) is int and e['index'] == index and e['previous_sha256'] == previous and
                raw(e) == line+b'\n', 'exact event bytes/order/fields')
        require(digest({k: v for k, v in e.items() if k != 'event_sha256'}) == e['event_sha256'], 'whole native hashchain')
        previous = e['event_sha256']
        out.append(e['payload'])
    return out


def census(events):
    """Durable intents, including pending failures; completed row lists are not attempts."""
    out = dict(forward_attempted=0, forward_returned=0, jacobian_attempted=0, jacobian_returned=0,
               predict_API_attempted=0, predict_API_returned=0,
               modules_attempted={r: 0 for r in ('action_encoder', 'predictor', 'pred_proj')},
               modules_returned={r: 0 for r in ('action_encoder', 'predictor', 'pred_proj')})
    for x in events:
        if 'event' not in x:
            continue
        e = x['event']
        key = {'forward_intent': 'forward_attempted', 'forward_return': 'forward_returned',
               'AD_intent': 'jacobian_attempted', 'AD_return': 'jacobian_returned',
               'predict_API_intent': 'predict_API_attempted', 'predict_API_return': 'predict_API_returned'}.get(e['kind'])
        if key:
            out[key] += 1
        if e['kind'] == 'module_intent':
            out['modules_attempted'][e['resource']] += 1
        elif e['kind'] == 'module_return':
            out['modules_returned'][e['entry']['resource']] += 1
    return out


def size_budget(rows, events):
    """RAM shape census: hex blobs exact width; every JSON float conservatively 32 chars."""
    def bound(v):
        if type(v) is float:
            require(np.isfinite(v) and len(raw(v))-1 <= 32, 'bounded numeric JSON representation')
            return 32
        if type(v) is list:
            return 2+max(0, len(v)-1)+sum(bound(x) for x in v)
        if type(v) is dict:
            return 2+max(0, len(v)-1)+sum(len(raw(k))-1+1+bound(x) for k, x in v.items())
        return len(raw(v))-1
    envelopes = [dict(index=i, previous_sha256='f'*64, payload=e, event_sha256='f'*64) for i, e in enumerate(events)]
    ledger = sum(len(raw(e)) for e in envelopes)
    row_bytes = sum(len(raw(r)) for r in rows)
    worst_ledger = sum(bound(e)+1 for e in envelopes)
    worst_rows = sum(bound(r)+1 for r in rows)
    matrices = 4*(192*60*4+128)  # Four complete float32 NPY files, real v1.0 header128.
    metadata = 4*1024**2  # Enforced by Store.put across loader/classes/binding/report/seal.
    worst = worst_ledger+worst_rows+matrices+metadata
    require(worst < CAP-RESERVE, 'full-shape JSON/NPY envelope exceeds approved output cap')
    return dict(fixture_uncompressed_ledger=ledger, fixture_uncompressed_rows=row_bytes,
                worst_uncompressed_ledger=worst_ledger, worst_uncompressed_rows=worst_rows,
                full_J_NPY_bytes=matrices, enforced_metadata_bytes=metadata,
                conservative_uncompressed_total=worst, physical_output_cap=CAP, reserve=RESERVE)


def consume(output, report_sha):
    output = Path(output).absolute()
    r = read(output/'report.json', report_sha)
    p = verify(r['prepared'], r['prepare_sha256'])
    require(r['schema'] == NAME+'.report' and r['status'] == 'all4_full30_sealed_NOT_admitted' and
            r['errors'] == [] and r['first_error'] is None and r['source_verified'] is True and
            r['model_before'] == r['model_after'] == p['expected_model_hash'], 'own complete engineering authority')
    for key in K.FLAGS:
        require(r[key] is False, 'scientific gates remain false')
    equal(r['contract'], K.contract(), 'immutable approved contract')
    equal(r['inherited_resources'], p['inherited_resources'], 'inherited and new costs remain separate')
    require(type(r['loader_memory']) is dict and set(r['loader_memory']) == {'allocated_peak', 'reserved_peak'} and
            all(type(r['loader_memory'][k]) is int and 0 <= r['loader_memory'][k] <= cap for k, cap in
                (('allocated_peak', K.contract()['allocated_cap_bytes']), ('reserved_peak', K.contract()['reserved_cap_bytes']))),
            'actual loader memory before slot counter reset')
    require(type(r['wall_seconds']) in (int, float) and np.isfinite(r['wall_seconds']) and 0 <= r['wall_seconds'] <= 1800, 'whole wall admission')
    seal = read(output/'seal.json', sha(output/'seal.json'))
    require(seal['schema'] == NAME+'.seal' and seal['complete'] is True and
            type(seal['through_durable_data_seconds']) in (int, float) and
            np.isfinite(seal['through_durable_data_seconds']) and 0 <= seal['through_durable_data_seconds'] <= r['wall_seconds'],
            'durable data clock seal, no overwall complete')
    names = {'binding.json', 'loader.json', 'classes.json', 'events.jsonl.gz', 'seal.json'} | {f'slot{i}.json.gz' for i in range(4)} | {
        f'slot{i}_J{j}.npy' for i in (1, 3) for j in range(2)}
    require(set(r['artifact_sha256']) == names and {f.name for f in output.iterdir()} == names | {'report.json', 'seal.json'} and
            all(f.is_file() and not f.is_symlink() for f in output.iterdir()) and
            sum(f.stat().st_size for f in output.iterdir()) <= CAP, 'exact complete bounded inventory')
    for n, h in r['artifact_sha256'].items():
        require(sha(output/n) == h, 'all actual artifact hashes')
    equal(seal['data_artifacts_sha256'], {n: h for n, h in r['artifact_sha256'].items() if n != 'seal.json'}, 'report-bound seal complete data inventory')
    equal(read(output/'binding.json', r['artifact_sha256']['binding.json']),
          dict(prepared=r['prepared'], prepare_sha256=r['prepare_sha256']), 'actual own prepare binding')
    load = read(output/'loader.json', r['artifact_sha256']['loader.json'])
    native.loader.validate_loader(load, p, r['prepare_sha256'])
    cs = read(output/'classes.json', r['artifact_sha256']['classes.json'])
    require(type(cs) is dict and bool(cs) and all(p['bindings_sha256'].get(f) == h for f, h in cs.items()), 'all reported classes presealed')
    events, cursor = journal(output), 0
    equal(r['actual_resources'], census(events), 'durable attempted/returned census including AD primal')
    actual = r['actual_resources']
    require(actual['forward_attempted'] == actual['forward_returned'] == actual['predict_API_attempted'] == actual['predict_API_returned'] == 228 and
            actual['jacobian_attempted'] == actual['jacobian_returned'] == 4 and
            all(n == 228 for v in (actual['modules_attempted'], actual['modules_returned']) for n in v.values()), 'complete fixed module/AD census')
    def accept(e):
        nonlocal cursor
        require(cursor < len(events), 'missing lifecycle event')
        equal(events[cursor], e, 'full event chronology, unknown events rejected')
        cursor += 1
    for f in load['files']:
        accept(dict(kind='load_intent', path=f['path'], expected_sha256=f['sha256']))
        accept(dict(kind='consumed_buffer_frozen', receipt=f))
    def array(d):
        require(set(d) == {'file', 'sha256', 'shape', 'dtype'} and d['file'] in names and
                r['artifact_sha256'][d['file']] == d['sha256'] and d['shape'] == [192, 60] and d['dtype'] == 'float32', 'full J descriptor')
        data = (output/d['file']).read_bytes()
        stream = io.BytesIO(data)
        require(len(data) <= 1024**2 and np.lib.format.read_magic(stream) == (1, 0), 'bounded exact NPY v1.0')
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        require(shape == (192, 60) and fortran is False and dtype == np.dtype('<f4') and
                len(data)-stream.tell() == 192*60*4, 'authenticate full J header before array allocation')
        stream.seek(0)
        a = np.load(stream, allow_pickle=False)
        require(stream.tell() == len(data) and a.dtype == np.float32 and a.shape == (192, 60) and np.isfinite(a).all(), 'full J NPY bytes/dtype/EOF')
        return a
    require(len(r['rows']) == 4, 'all four fixed slots')
    plans, rebuilt = [], []
    for i, ss in enumerate(r['rows']):
        root, ai = divmod(i, 2)
        arm = K.ARMS[ai]
        require(type(ss['slot']) is int and type(ss['root_slot']) is int and ss['slot'] == i and ss['root_slot'] == root and ss['arm'] == arm and
                ss['status'] == 'sealed' and ss['file'] == f'slot{i}.json.gz' and
                type(ss['wall_seconds']) in (int, float) and np.isfinite(ss['wall_seconds']) and 0 <= ss['wall_seconds'] <= 450, 'fixed slot identity/wall')
        row = read(output/ss['file'], r['artifact_sha256'][ss['file']], CAP, gz=True)
        require(set(row) == {'slot', 'index', 'arm', 'status', 'forwards', 'AD', 'result', 'resources', 'plan', 'memory'} and
                type(row['slot']) is int and type(row['index']) is int and row['slot'] == i and row['index'] == root and
                row['arm'] == arm and row['status'] == 'sealed', 'actual slot native row exact ABI')
        accept(dict(kind='slot_started', slot=i, root_slot=root, arm=arm))
        backend = SavedBackend(row, lambda e: accept(dict(slot=i, event=e)), array)
        result = K.execute(p['slots'][root], arm, backend)
        backend.complete()
        equal(row['result'], result, 'all contexts/evaluations/guards/selection/replays independently reconstructed')
        equal(row['resources'], K.resources(row['forwards'], row['AD']), 'actual complete paid resource census')
        require(set(row['memory']) == {'allocated_peak', 'reserved_peak'} and
                all(type(row['memory'][k]) is int and 0 <= row['memory'][k] <= cap for k, cap in
                    (('allocated_peak', K.contract()['allocated_cap_bytes']), ('reserved_peak', K.contract()['reserved_cap_bytes']))),
                'actual slot allocator peak evidence')
        equal(ss['resources'], row['resources'], 'report resource identity')
        equal(ss['contexts'], result['contexts'], 'report full contexts')
        equal(ss['final_Q'], result['final_Q'], 'report Q')
        plan = row_plan(p, p['slots'][root], row)
        equal(row['plan'], plan, 'new full30 plan exact action and scope authority')
        if ai:
            equal(rebuilt[-1]['contexts'], result['initial_contexts'], 'cross-arm same-action complete context parity')
        rebuilt.append(result)
        accept(dict(kind='slot_sealed', slot=i, witness=plan['witness_sha256']))
        plans.append(plan)
    require(cursor == len(events), 'no extra terminal events')
    rehash(p['bindings_sha256'])
    require(sha(output/'report.json') == report_sha, 'report stable through consume')
    return dict(schema=NAME+'.consume', status='full30_transfer_and_joint_verified_NOT_admitted',
                report_sha256=report_sha, prepare_sha256=r['prepare_sha256'], plans=plans,
                full_upstream_rebuilt_here=False, own_native_saved_reconstructed_here=True,
                actual_new=K.contract()['total'], new_consume_model_calls=0, errors=[], **K.FLAGS)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--linker', default=LINK['path'])
    q.add_argument('--linker-sha', default=LINK['sha256'])
    q.add_argument('--output', required=True)
    for cmd in ('verify', 'run_cuda'):
        q = sub.add_parser(cmd)
        q.add_argument('--prepared', required=True)
        q.add_argument('--prepare-sha', required=True)
        if cmd == 'run_cuda':
            q.add_argument('--output', required=True)
            q.add_argument('--authorization', required=True)
    q = sub.add_parser('consume')
    q.add_argument('--output', required=True)
    q.add_argument('--report-sha', required=True)
    q.add_argument('--receipt-output', required=True)
    a = parser.parse_args()
    if a.cmd == 'prepare':
        result = write_new(a.output, build(a.linker, a.linker_sha))
    elif a.cmd == 'verify':
        result = dict(schema=verify(a.prepared, a.prepare_sha)['schema'], status='verified_no_dispatch')
    elif a.cmd == 'run_cuda':
        result = run_cuda(a.prepared, a.prepare_sha, a.output, a.authorization)
    else:
        result = write_new(a.receipt_output, consume(a.output, a.report_sha))
    print(raw(result).decode(), end='')


if __name__ == '__main__':
    main()
