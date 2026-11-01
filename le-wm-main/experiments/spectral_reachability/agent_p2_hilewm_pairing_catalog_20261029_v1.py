"""MAIN CUDA data-only P2 catalog; independent saved/RAM five-group join.

Cold import is stdlib-only. No forward, normalizer, fitting, foreign Hi objects,
or full upstream consumer. Catalog equality is not computational parity.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import time

TAG = 'agent_p2_hilewm_pairing_catalog_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
OWNER = 'experiments.spectral_reachability.agent_p2_equal144_baseline_cuda_20261028_v1'
PREP = ROOT/'results/agent_p2_three_program_20261028_v1_prepare_MAIN01.json'
PREP_SHA = '86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461'
OLD_LOADER = ROOT/'results/agent_p2_three_program_20261028_v1_cuda_MAIN01/loader.json'
OLD_LOADER_SHA = '2401302225f042755bd9a8e8a06874ebafd367dee5937434b442c739b45b4660'
CONFIG_SHA = '2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09'
WEIGHTS_SHA = '48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'
MODEL_HASH = '4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546'
HI_SHA = '3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3'
IR_SHA = '709b31e28bee8039d577668cbd9613347be1d070a7619e3db3ca61e3674cabbe'
PINS = {
    'experiments/spectral_reachability/agent_p2_equal144_baseline_cuda_20261028_v1.py':
        'f592a485d657c849f6a5036b8e485a0287cb234f00476cf5f9ca262bb5e5a841',
    'experiments/checkpoint_compat.py': '50b01e20864dc929943df7cb3566c271a771e69c06ea0fdbc9b598c59e6e9dff',
    'experiments/spectral_reachability/p1_constructive_blind_encode_cuda_20261027_v1.py':
        'c3b071de1ad0b322e1fff77a716550f42014f0303a6b5fa09271fd4e55609b4a',
}
GROUPS = dict(encoder='encoder', projector='projector', action_encoder='action_encoder',
              low_predictor='predictor', low_pred_proj='pred_proj')
REPLACEMENTS = (
    ('.encoder.layer.', '.layers.'),
    ('.attention.attention.query.', '.attention.q_proj.'),
    ('.attention.attention.key.', '.attention.k_proj.'),
    ('.attention.attention.value.', '.attention.v_proj.'),
    ('.attention.output.dense.', '.attention.o_proj.'),
    ('.intermediate.dense.', '.mlp.fc1.'), ('.output.dense.', '.mlp.fc2.'))
CAP, FLOOR, WALL = 16*1024**2, 200*1024**2, 600
ROW_FIELDS = {'key', 'role', 'persistent', 'is_none', 'dtype', 'shape', 'stride',
              'storage_offset', 'value_bytes', 'value_sha256', 'tensor_aliases', 'storage_aliases'}


def need(ok, msg):
    if not ok: raise ValueError(msg)


def encode(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(path):
    p = Path(path)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size <= CAP, 'bounded regular metadata/source')
    b = p.read_bytes(); need(len(b) <= CAP, 'read cap')
    return hashlib.sha256(b).hexdigest()


def read(path, expected):
    need(re.fullmatch('[0-9a-f]{64}', expected or '') is not None and sha(path) == expected, 'external actual SHA')
    body = Path(path).read_bytes()
    need(len(body) <= CAP and hashlib.sha256(body).hexdigest() == expected, 'exact decoded buffer SHA/cap')
    def pairs(xs):
        d = {}
        for k, v in xs:
            need(k not in d, 'duplicate JSON key'); d[k] = v
        return d
    return json.loads(body, object_pairs_hook=pairs,
                      parse_constant=lambda _: need(False, 'nonfinite JSON'))


def write(path, value):
    p = Path(path); b = encode(value)
    need(len(b) <= CAP and shutil.disk_usage(p.parent).free >= FLOOR+len(b), 'output cap/floor')
    with p.open('xb') as f:
        need(f.write(b) == len(b), 'full exclusive write'); f.flush(); os.fsync(f.fileno())
    return dict(path=str(p), sha256=sha(p))


def remap(key):
    for old, new in REPLACEMENTS: key = key.replace(old, new)
    return key


def sources():
    """Only explicit local Python import closure; no directory recursion/weights."""
    todo = [ROOT/k for k in PINS]; seen = {}
    while todo:
        p = todo.pop()
        if str(p) in seen: continue
        need(len(seen) < 512, 'bounded original import closure')
        seen[str(p)] = sha(p)
        for node in ast.walk(ast.parse(p.read_text())):
            names = []
            if isinstance(node, ast.Import): names = [x.name for x in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]+[node.module+'.'+x.name for x in node.names]
            for name in names:
                child = ROOT.joinpath(*name.split('.')).with_suffix('.py')
                if name.startswith('experiments.') and child.is_file(): todo.append(child)
    for rel, h in PINS.items(): need(seen[str(ROOT/rel)] == h, 'frozen original loader source')
    for p in (Path(__file__).resolve(), ROOT/'tests'/('test_'+TAG+'.py'), ROOT/'innerdocs'/(TAG+'.md')):
        seen[str(p)] = sha(p)
    return seen


def protocol():
    return dict(schema=TAG+'.protocol', CUDA_only=True, data_only=True, forward=0,
        training=False, normalization=False, full_upstream_strict=False, scientific_admission=False,
        groups=GROUPS, row_key_space='P2 runtime remapped names; serialized original-key inventory not exported',
        equality='exact role/dtype/shape/persistence/None/alias/value SHA; layout separately reported',
        config_pairing=False, computational_pairing=False, raw_scaler_pairing=False,
        output_cap_bytes=CAP, disk_floor_bytes=FLOOR, cooperative_wall_seconds=WALL,
        wall_scope='including load, CUDA copies, two full catalogs, hashes, writes; not kernel preemption')


def build():
    p = read(PREP, PREP_SHA); old = read(OLD_LOADER, OLD_LOADER_SHA)
    policy = Path(p['policy'])
    expected = {str(policy/'config.json'): CONFIG_SHA, str(policy/'weights.pt'): WEIGHTS_SHA}
    need(policy == Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm')
         and p['expected_model_hash'] == MODEL_HASH
         and all(p['bindings_sha256'].get(k) == v for k, v in expected.items()), 'original P2 model bindings')
    need(old['prepare_sha256'] == PREP_SHA and old['tensor_hash_before'] == MODEL_HASH
         and old['missing'] == old['unexpected'] == [] and old['frozen_eval'] is True,
         'historical actual loader authority')
    need(old['adapter_source_sha256'] == PINS[OWNER.replace('.', '/')+'.py']
         and old['legacy_loader_source_sha256'] == PINS['experiments/checkpoint_compat.py'], 'original loader/source identity')
    need(len(old['files']) == 2 and {f['path'] for f in old['files']} == set(expected), 'two original model buffers')
    for f in old['files']:
        need(f['same_buffer'] is True and all(f[k] == expected[f['path']] for k in
             ('sha256', 'hash_buffer_sha256', 'deserialize_buffer_sha256')), 'same-buffer exact original load')
    need(sha(policy/'config.json') == CONFIG_SHA, 'small config only during prepare')
    closure = sources()
    for path, h in closure.items():
        if path in p['bindings_sha256']:
            need(h == p['bindings_sha256'][path], 'inherited source identity: '+path)
    return dict(schema=TAG+'.prepare', protocol=protocol(), sources=closure,
        original=dict(prepare=str(PREP), prepare_sha256=PREP_SHA, loader=str(OLD_LOADER), loader_sha256=OLD_LOADER_SHA),
        loader_input={k: p[k] for k in ('policy', 'expected_model_hash', 'bindings_sha256')},
        config=read(policy/'config.json', CONFIG_SHA), weights_read_during_prepare=False)


def prepare(path): return write(path, build())


def verify_prepare(path, h):
    m = read(path, h); need(m == build(), 'exact saved/source prepare reconstruction'); return m


def validate_rows(rows):
    need(type(rows) is list and 0 < len(rows) <= 4096, 'nonempty bounded catalog')
    keys = set()
    for r in rows:
        need(type(r) is dict and set(r) == ROW_FIELDS, 'exact row fields')
        k = r['key']; need(type(k) is str and re.fullmatch(r'[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*', k)
                           and k not in keys, 'unique explicit key'); keys.add(k)
        need(r['role'] in ('parameter', 'buffer') and type(r['persistent']) is bool
             and type(r['is_none']) is bool and (r['role'] != 'parameter' or r['persistent']), 'role/persistence')
        for a in ('tensor_aliases', 'storage_aliases'):
            need(type(r[a]) is list and all(type(x) is str for x in r[a]) and r[a] == sorted(set(r[a])), 'canonical alias list')
        if r['is_none']:
            need(all(r[f] is None for f in ('dtype', 'shape', 'stride', 'storage_offset', 'value_bytes', 'value_sha256'))
                 and r['tensor_aliases'] == r['storage_aliases'] == [], 'explicit None not zero')
        else:
            need(r['dtype'] in ('<f4', '<i8') and type(r['shape']) is list and type(r['stride']) is list
                 and len(r['shape']) == len(r['stride']) and all(type(x) is int and x >= 0 for x in r['shape']+r['stride'])
                 and type(r['storage_offset']) is int and r['storage_offset'] >= 0, 'exact dtype/shape/layout')
            n = 1
            for d in r['shape']: n *= d
            need(type(r['value_bytes']) is int and r['value_bytes'] == n*({'<f4': 4, '<i8': 8}[r['dtype']])
                 and re.fullmatch('[0-9a-f]{64}', r['value_sha256'] or ''), 'exact value SHA/bytes')
            need(k in r['tensor_aliases'] and k in r['storage_aliases'], 'self alias included')
    by = {r['key']: r for r in rows}
    for r in rows:
        for a in ('tensor_aliases', 'storage_aliases'):
            for key in r[a]:
                need(key in by and by[key][a] == r[a] and not by[key]['is_none'], 'complete symmetric alias classes')
    return by


def runtime_sources(model, torch):
    paths = {Path(torch.__file__).resolve(), Path(inspect.getfile(type(model))).resolve()}
    for node in model.modules(): paths.add(Path(inspect.getfile(type(node))).resolve())
    for module in ('torch.nn.functional', 'hydra.utils', 'omegaconf'):
        paths.add(Path(importlib.import_module(module).__file__).resolve())
    return {str(p): sha(p) for p in sorted(paths)}


def snapshot(model, torch):
    """MAIN data copies only, no tensor reconstruction/forward/dtype conversion."""
    need(all(not n.training for n in model.modules()), 'all graph eval')
    entries = []
    for prefix, node in model.named_modules(remove_duplicate=False):
        for role, values in (('parameter', node._parameters), ('buffer', node._buffers)):
            for name, value in values.items():
                entries.append((prefix+'.'+name if prefix else name, role,
                                role == 'parameter' or name not in node._non_persistent_buffers_set, value))
    identities, storages = {}, {}
    for key, _, _, x in entries:
        if x is not None:
            need(x.device == torch.device('cuda:0') and not x.requires_grad, 'frozen original cuda:0 tensor')
            identities.setdefault(id(x), []).append(key)
            storage = x.untyped_storage()
            storages.setdefault((str(x.device), storage.data_ptr(), storage.nbytes()), []).append(key)
    out = []
    for key, role, persistent, x in entries:
        r = dict(key=key, role=role, persistent=persistent, is_none=x is None,
                 dtype=None, shape=None, stride=None, storage_offset=None, value_bytes=None,
                 value_sha256=None, tensor_aliases=[], storage_aliases=[])
        if x is not None:
            version = x._version
            a = x.detach().cpu().contiguous().numpy()
            need(a.dtype.str in ('<f4', '<i8'), 'original little-endian f32/i64 only')
            b = a.tobytes(order='C'); storage = x.untyped_storage()
            r.update(dtype=a.dtype.str, shape=list(x.shape), stride=list(x.stride()), storage_offset=x.storage_offset(),
                value_bytes=len(b), value_sha256=hashlib.sha256(b).hexdigest(),
                tensor_aliases=sorted(identities[id(x)]),
                storage_aliases=sorted(storages[(str(x.device), storage.data_ptr(), storage.nbytes())]))
            need(x._version == version, 'no tensor mutation while copying')
        out.append(r)
    out.sort(key=lambda r: r['key']); validate_rows(out)
    need({r['key'] for r in out if not r['is_none'] and r['persistent']} == set(model.state_dict()),
         'complete runtime state_dict denominator including alias paths')
    return out


class LoaderStore:
    """Exact original loaded_model sink, in-memory receipts, no math replacement."""
    def __init__(self): self.events, self.receipts = [], {}
    def event(self, row): self.events.append(row)
    def put(self, name, row):
        need(name == 'loader.json' and name not in self.receipts, 'original single loader receipt')
        self.receipts[name] = row


def export_cuda(manifest, mh, output, *, main_authorized=False):
    need(main_authorized is True, 'MAIN authorization before any I/O or torch import')
    out = Path(output)
    need(out.parent.resolve() == ROOT/'results' and out.name.startswith(TAG+'_') and not out.exists(), 'exclusive own output directory')
    need(shutil.disk_usage(out.parent).free >= FLOOR+CAP, 'whole output cap/floor')
    out.mkdir(); start = time.monotonic(); model = None; hooks = []
    r = dict(schema=TAG+'.export', manifest=str(Path(manifest).resolve()), manifest_sha256=mh,
        completed=False, errors=[], rows_before=None, rows_after=None, model_before=None, model_after=None,
        runtime_sources_before=None, runtime_sources_after=None, loader=None, load_events=[], forward_attempts=0,
        sources_after=None, data_only=True, scientific_admission=False)
    def guard(): need(time.monotonic()-start <= WALL, 'cooperative total wall')
    def reject(*args):
        r['forward_attempts'] += 1
        raise ValueError('forward forbidden in data-only catalog')
    try:
        m = verify_prepare(manifest, mh); guard()
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        need(torch.cuda.is_available() and Path(torch.__file__).resolve() == ROOT/'.venv/lib/python3.12/site-packages/torch/__init__.py',
             'bound HOST CUDA runtime, no CPU neural fallback')
        torch.empty(0, device='cuda:0')
        owner = importlib.import_module(OWNER)
        need(sha(owner.__file__) == PINS[OWNER.replace('.', '/')+'.py'], 'original loaded_model code')
        store = LoaderStore()
        try:
            model, r['model_before'] = owner.loaded_model(m['loader_input'], torch, torch.device('cuda:0'), store, mh)
        finally:
            r['load_events'] = store.events; r['loader'] = store.receipts.get('loader.json')
        owner.validate_loader(r['loader'], m['loader_input'], mh)
        for n in model.modules(): hooks.append(n.register_forward_pre_hook(reject))
        r['runtime_sources_before'] = runtime_sources(model, torch); guard()
        r['rows_before'] = snapshot(model, torch); guard()
        r['rows_after'] = snapshot(model, torch)
        r['model_after'] = owner.enc.tensor_hash(model); owner.enc.frozen(model)
        r['runtime_sources_after'] = runtime_sources(model, torch)
        need(r['rows_before'] == r['rows_after'] and r['model_before'] == r['model_after'] == MODEL_HASH, 'all values/layout/roles frozen')
        need(r['runtime_sources_before'] == r['runtime_sources_after'], 'installed runtime source stable')
        r['sources_after'] = verify_prepare(manifest, mh)['sources']; guard()
        r['cuda'] = dict(device='cuda:0', torch=str(torch.__version__), CUDA=str(torch.version.cuda))
        r['completed'] = True
    except BaseException as exc:
        r['errors'].append(dict(type=type(exc).__name__, message=str(exc)[:2000]))
    finally:
        for hook in hooks: hook.remove()
        r['wall_seconds'] = time.monotonic()-start
        r['resources'] = dict(model_load_attempts=1 if r['load_events'] else 0,
            model_load_returns=int(model is not None), forward_attempts=r['forward_attempts'],
            catalog_passes_returned=int(r['rows_before'] is not None)+int(r['rows_after'] is not None),
            predict=0, encoder=0, backward=0, physics=0, FLOPs=None)
        result = write(out/'report.json', r)
    return result


def consume_catalog(path, h):
    r = read(path, h)
    need(r['schema'] == TAG+'.export' and r['completed'] is True and r['errors'] == []
         and r['forward_attempts'] == 0 and r['data_only'] is True, 'complete data-only export')
    m = verify_prepare(r['manifest'], r['manifest_sha256'])
    need(r['sources_after'] == m['sources'] and r['rows_before'] == r['rows_after']
         and r['model_before'] == r['model_after'] == MODEL_HASH, 'source/model/catalog before-after equality')
    rows = r['rows_before']; validate_rows(rows)
    state = [x for x in rows if not x['is_none'] and x['persistent']]
    # Match original tensor_hash over runtime state_dict names and raw C bytes via per-key SHA?
    # A concatenation hash cannot be rebuilt from individual digests; do NOT claim this.
    need(state and r['runtime_sources_before'] and r['runtime_sources_before'] == r['runtime_sources_after'], 'complete stable runtime evidence')
    for p, v in r['runtime_sources_before'].items(): need(sha(p) == v, 'current installed runtime source')
    loader = r['loader']
    need(loader['schema'] == 'agent_p2_equal144_baseline_cuda_20261028_v1.loader_receipt'
         and loader['prepare_sha256'] == r['manifest_sha256'] and loader['frozen_eval'] is True
         and loader['device'] == 'cuda:0' and loader['tensor_hash_before'] == MODEL_HASH
         and loader['missing'] == loader['unexpected'] == []
         and loader['adapter_source_sha256'] == PINS[OWNER.replace('.', '/')+'.py']
         and loader['legacy_loader_source_sha256'] == PINS['experiments/checkpoint_compat.py'], 'full original loader receipt')
    expected = {str(Path(m['loader_input']['policy'])/'config.json'): CONFIG_SHA,
                str(Path(m['loader_input']['policy'])/'weights.pt'): WEIGHTS_SHA}
    need(len(loader['files']) == 2 and {f['path'] for f in loader['files']} == set(expected), 'two consumed original model files')
    for f in loader['files']:
        need(f['same_buffer'] is True and all(f[k] == expected[f['path']] for k in
             ('sha256', 'hash_buffer_sha256', 'deserialize_buffer_sha256')), 'actual same-buffer checkpoint SHA')
    need(r['load_events'] == [event for f in loader['files'] for event in (
        dict(kind='load_intent', path=f['path'], expected_sha256=f['sha256']),
        dict(kind='consumed_buffer_frozen', receipt=f))], 'entire original load ledger')
    need(r['resources'] == dict(model_load_attempts=1, model_load_returns=1,
        forward_attempts=0, catalog_passes_returned=2, predict=0, encoder=0,
        backward=0, physics=0, FLOPs=None), 'actual data-only resource counts')
    return dict(schema=TAG+'.catalog_receipt', report_sha256=h, manifest_sha256=r['manifest_sha256'],
                rows=len(rows), tensor_pairing=False, independent_tensor_rehash=False), rows


def hi_rows(report):
    """Decode authenticated original Hi metadata, not any NPY or foreign class."""
    need(report['schema'] == 'agent_hilewm_storage_npy_export_20261028_v1'
         and report['status'] == 'literal_nn_data_exported_not_model_equivalence'
         and report['errors'] == [] and report['ir_sha256'] == IR_SHA
         and report['all_storage_crc_verified'] is True and report['byteorder'] == 'little'
         and report['module_alias_paths'] == report['node_alias_paths'] == report['storage_alias_paths'] == {},
         'exact known nonalias Hi export ABI')
    out = []
    need(len(report['rows']) == len(report['state_dict_paths']), 'Hi entry denominator')
    for index, (entry, rr) in enumerate(zip(report['state_dict_paths'], report['rows'], strict=True)):
        source, array = rr['source_entry'], rr['array']; meta = source['metadata']
        key = '.'.join(source['module_path']+[source['name']])
        need(rr['index'] == entry['index'] == index and entry['key'] == key
             and (array is None) == (meta is None), 'Hi exact key/None/index join')
        role = {'_parameters': 'parameter', '_buffers': 'buffer'}[source['collection']]
        persistent = not entry['nonpersistent_buffer']
        need(entry['include_in_literal_state_dict'] == (meta is not None and persistent), 'Hi persistence/state inclusion')
        row = dict(key=key, role=role, persistent=persistent, is_none=meta is None,
            dtype=None, shape=None, stride=None, storage_offset=None, value_bytes=None,
            value_sha256=None, tensor_aliases=[], storage_aliases=[])
        if meta is not None:
            need(array['shape'] == meta['shape'] and array['descr'] == {'float32': '<f4', 'int64': '<i8'}[meta['dtype_declaration']], 'Hi original shape/dtype')
            row.update(dtype=array['descr'], shape=meta['shape'], stride=meta['stride'], storage_offset=meta['storage_offset'],
                value_bytes=array['value_bytes'], value_sha256=array['value_sha256'], tensor_aliases=[key], storage_aliases=[key])
        out.append(row)
    validate_rows(out); return out


def join_catalogs(hi, p2):
    """Pure saved/RAM; all differences reported, never early-exit on first group."""
    validate_rows(hi); validate_rows(p2)
    out = []
    semantic = ('role', 'persistent', 'is_none', 'dtype', 'shape', 'value_bytes', 'value_sha256', 'tensor_aliases', 'storage_aliases')
    def mapped(key):
        head, _, tail = key.partition('.')
        return remap(GROUPS.get(head, head)+('.'+tail if tail else ''))
    for hg, pg in GROUPS.items():
        hr = {}; pr = {r['key']: r for r in p2 if r['key'].split('.')[0] == pg}
        for row in hi:
            if row['key'].split('.')[0] != hg: continue
            r = dict(row); r['key'] = mapped(row['key'])
            for a in ('tensor_aliases', 'storage_aliases'): r[a] = sorted(mapped(x) for x in row[a])
            need(r['key'] not in hr and all(len(r[a]) == len(set(r[a])) for a in ('tensor_aliases', 'storage_aliases')), 'mapping collision')
            hr[r['key']] = r
        missing, unexpected = sorted(hr.keys()-pr.keys()), sorted(pr.keys()-hr.keys())
        mismatch, layout = [], []
        for key in sorted(hr.keys() & pr.keys()):
            fields = [k for k in semantic if hr[key][k] != pr[key][k]]
            if fields: mismatch.append(dict(key=key, fields=fields, hi={k: hr[key][k] for k in fields}, p2={k: pr[key][k] for k in fields}))
            lf = [k for k in ('stride', 'storage_offset') if hr[key][k] != pr[key][k]]
            if lf: layout.append(dict(key=key, fields=lf))
        established = bool(hr and pr) and not (missing or unexpected or mismatch)
        out.append(dict(hi_group=hg, p2_group=pg, hi_entries=len(hr), p2_entries=len(pr), missing=missing,
                        unexpected=unexpected, mismatch=mismatch, layout_differences=layout,
                        status='tensor_catalog_equal' if established else 'unestablished'))
    return dict(schema=TAG+'.join', groups=out, all_five_tensor_catalog_equal=all(x['status'] == 'tensor_catalog_equal' for x in out),
        config_pairing=False, computation_pairing=False, normalizer_pairing=False,
        original_class_parity=False, scientific_admission=False, independent_roots=False)


def saved_join(p2, ph, hi, hh, receipt, rh, output):
    need(hh == HI_SHA, 'exact existing Hi export')
    accepted, rows = consume_catalog(p2, ph)
    report = read(hi, hh); rc = read(receipt, rh)
    need(rc == dict(status='saved_data_verified_not_model_equivalence', report_sha256=hh,
         arrays_verified=850, ir_sha256=IR_SHA, trusted_pickle_loading_allowed=False, scientific_admission=False), 'entire actual Hi exporter receipt')
    result = join_catalogs(hi_rows(report), rows)
    result['authorities'] = dict(p2_report=str(Path(p2).resolve()), p2_report_sha256=ph, p2_receipt=accepted,
        hi_report=str(Path(hi).resolve()), hi_report_sha256=hh, hi_receipt=str(Path(receipt).resolve()), hi_receipt_sha256=rh)
    return write(output, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'verify', 'export', 'consume', 'join', 'protocol'))
    p.add_argument('--manifest', type=Path); p.add_argument('--manifest-sha256')
    p.add_argument('--report', type=Path); p.add_argument('--report-sha256')
    p.add_argument('--hi-report', type=Path); p.add_argument('--hi-report-sha256')
    p.add_argument('--hi-receipt', type=Path); p.add_argument('--hi-receipt-sha256')
    p.add_argument('--output', type=Path); p.add_argument('--main-authorized', action='store_true')
    a = p.parse_args()
    if a.command == 'prepare': value = prepare(a.output)
    elif a.command == 'verify': value = dict(verified=bool(verify_prepare(a.manifest, a.manifest_sha256)))
    elif a.command == 'export': value = export_cuda(a.manifest, a.manifest_sha256, a.output, main_authorized=a.main_authorized)
    elif a.command == 'consume':
        value, _ = consume_catalog(a.report, a.report_sha256)
        if a.output: value = write(a.output, value)
    elif a.command == 'join': value = saved_join(a.report, a.report_sha256, a.hi_report, a.hi_report_sha256, a.hi_receipt, a.hi_receipt_sha256, a.output)
    else: value = protocol()
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__': main()
