"""RAM/saved metadata only. No torch, checkpoint arrays or upstream consumer."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p2_hilewm_pairing_catalog_20261029_v1 as m


def row(key, **changes):
    r = dict(key=key, role='parameter', persistent=True, is_none=False,
             dtype='<f4', shape=[2], stride=[1], storage_offset=0, value_bytes=8,
             value_sha256='a'*64, tensor_aliases=[key], storage_aliases=[key])
    r.update(changes); return r


def catalogs():
    hi = [row(g+'.weight') for g in m.GROUPS]
    p2 = [row(g+'.weight') for g in m.GROUPS.values()]
    return hi, p2


def test_all_five_equal_not_computation_or_config_pairing():
    r = m.join_catalogs(*catalogs())
    assert len(r['groups']) == 5 and r['all_five_tensor_catalog_equal']
    assert not any(r[k] for k in ('config_pairing', 'computation_pairing', 'normalizer_pairing', 'scientific_admission'))


@pytest.mark.parametrize('field,value', [('role', 'buffer'), ('dtype', '<i8'),
    ('shape', [1, 2]), ('value_sha256', 'b'*64), ('persistent', False), ('is_none', True)])
def test_semantic_mismatch_all_retained(field, value):
    hi, p2 = catalogs()
    if field == 'persistent': p2[0]['role'] = 'buffer'
    p2[0][field] = value
    if field == 'shape': p2[0]['stride'] = [2, 1]
    if field == 'dtype': p2[0]['value_bytes'] = 16
    if field == 'is_none':
        for f in ('dtype', 'shape', 'stride', 'storage_offset', 'value_bytes', 'value_sha256'): p2[0][f] = None
        p2[0]['tensor_aliases'] = p2[0]['storage_aliases'] = []
    r = m.join_catalogs(hi, p2)
    assert not r['all_five_tensor_catalog_equal']
    assert field in r['groups'][0]['mismatch'][0]['fields']
    assert len(r['groups']) == 5


def test_all_missing_and_unexpected_across_groups():
    hi, p2 = catalogs()
    p2 = [row(g+'.other') for g in m.GROUPS.values()]
    r = m.join_catalogs(hi, p2)
    assert all(len(g['missing']) == len(g['unexpected']) == 1 for g in r['groups'])
    assert not r['all_five_tensor_catalog_equal']


def test_absent_group_cannot_equal_empty_group():
    hi, p2 = catalogs(); hi.pop(); p2.pop()
    r = m.join_catalogs(hi, p2)
    assert r['groups'][-1]['status'] == 'unestablished'


def test_legacy_key_mapping_and_collision_rejection():
    hi, p2 = catalogs()
    hi[0] = row('encoder.encoder.layer.0.attention.attention.query.weight')
    p2[0] = row('encoder.layers.0.attention.q_proj.weight')
    assert m.join_catalogs(hi, p2)['all_five_tensor_catalog_equal']
    hi.append(row('encoder.layers.0.attention.q_proj.weight'))
    with pytest.raises(ValueError, match='collision'): m.join_catalogs(hi, p2)


def test_exact_remap_literal_from_frozen_original_source():
    p = m.ROOT/'experiments/checkpoint_compat.py'
    assert m.sha(p) == m.PINS['experiments/checkpoint_compat.py']
    tree = ast.parse(p.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'remap_legacy_vit_keys')
    replacements = next(n.value for n in fn.body if isinstance(n, ast.Assign) and
                        any(isinstance(t, ast.Name) and t.id == 'replacements' for t in n.targets))
    assert ast.literal_eval(replacements) == m.REPLACEMENTS


def test_bounded_source_closure_only_python_and_own_metadata():
    closed = m.sources()
    assert len(closed) <= 515
    assert not any(p.endswith(('.pt', '.npy', '.npz', '.pkl')) for p in closed)
    assert all(closed[str(m.ROOT/p)] == h for p, h in m.PINS.items())


def test_alias_mismatch_not_hidden_by_equal_values():
    hi, p2 = catalogs()
    hi.append(row('encoder.extra')); p2.append(row('encoder.extra'))
    for x in p2:
        if x['key'].startswith('encoder.'):
            x['storage_aliases'] = ['encoder.extra', 'encoder.weight']
    r = m.join_catalogs(hi, p2)
    assert not r['all_five_tensor_catalog_equal']
    assert all('storage_aliases' in q['fields'] for q in r['groups'][0]['mismatch'])


def test_layout_reported_separately_not_hidden_or_equal_compute():
    hi, p2 = catalogs(); p2[0]['stride'] = [2]
    r = m.join_catalogs(hi, p2)
    assert r['groups'][0]['layout_differences'] == [{'key': 'encoder.weight', 'fields': ['stride']}]
    assert r['computation_pairing'] is False


@pytest.mark.parametrize('change', ['duplicate', 'shape_bool', 'dtype', 'hash', 'asymmetric_alias', 'none_zero', 'parameter_nonpersistent'])
def test_malformed_catalog_rejected(change):
    hi, _ = catalogs()
    if change == 'duplicate': hi.append(deepcopy(hi[0]))
    if change == 'shape_bool': hi[0]['shape'] = [True]
    if change == 'dtype': hi[0]['dtype'] = '<f8'
    if change == 'hash': hi[0]['value_sha256'] = 'no'
    if change == 'asymmetric_alias': hi[0]['tensor_aliases'].append(hi[1]['key'])
    if change == 'none_zero': hi[0]['is_none'] = True
    if change == 'parameter_nonpersistent': hi[0]['persistent'] = False
    with pytest.raises(ValueError): m.validate_rows(hi)


def test_none_entries_preserved_distinct_from_zeros():
    hi, p2 = catalogs()
    for rows in (hi, p2):
        r = rows[0]; r['is_none'] = True
        for f in ('dtype', 'shape', 'stride', 'storage_offset', 'value_bytes', 'value_sha256'): r[f] = None
        r['tensor_aliases'] = r['storage_aliases'] = []
    assert m.join_catalogs(hi, p2)['all_five_tensor_catalog_equal']


def test_actual_hi_saved_metadata_census_without_arrays():
    p = m.ROOT/'results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json'
    report = m.read(p, m.HI_SHA); rows = m.hi_rows(report)
    assert sum(not r['is_none'] for r in rows) == 425
    assert [sum(not r['is_none'] and r['key'].split('.')[0] == g for r in rows) for g in m.GROUPS] == [198, 9, 6, 81, 9]
    assert all(r['role'] == 'buffer' for r in rows if r['key'].endswith('num_batches_tracked'))
    assert any(r['is_none'] for r in rows)


@pytest.mark.parametrize('mutation', ['alias', 'shape', 'none', 'index', 'persistent'])
def test_actual_hi_metadata_mutations(mutation):
    p = m.ROOT/'results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json'
    r = m.read(p, m.HI_SHA)
    if mutation == 'alias': r['storage_alias_paths'] = {'unexpected': [1]}
    if mutation == 'shape': r['rows'][0]['array']['shape'] = [99]
    if mutation == 'none': r['rows'][0]['array'] = None
    if mutation == 'index': r['state_dict_paths'][0]['index'] = 1
    if mutation == 'persistent': r['state_dict_paths'][0]['nonpersistent_buffer'] = True
    with pytest.raises(ValueError): m.hi_rows(r)


def test_export_unauthorized_no_io(tmp_path):
    with pytest.raises(ValueError, match='MAIN authorization'): m.export_cuda(None, None, None)
    assert not list(tmp_path.iterdir())


def test_loader_store_original_sink_no_substitution():
    s = m.LoaderStore(); x = {'anything': 'original'}
    s.event(x); s.put('loader.json', x)
    assert s.events == [x] and s.receipts['loader.json'] is x
    with pytest.raises(ValueError): s.put('loader.json', x)


def test_read_external_sha_duplicate_keys_and_exclusive_write(tmp_path):
    p = tmp_path/'data.json'; p.write_bytes(b'{"a":1,"a":2}')
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='duplicate'): m.read(p, h)
    with pytest.raises(ValueError, match='SHA'): m.read(p, '0'*64)
    p = tmp_path/'new.json'; m.write(p, {'ok': True})
    with pytest.raises(FileExistsError): m.write(p, {'ok': False})


def test_cli_protocol_cold_import_no_torch_or_foreign_code():
    code = 'import sys; from experiments.spectral_reachability import agent_p2_hilewm_pairing_catalog_20261029_v1 as m; assert "torch" not in sys.modules; print(m.protocol()["forward"])'
    p = subprocess.run([sys.executable, '-c', code], cwd=m.ROOT, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.strip() == '0'
    source = Path(m.__file__).read_text()
    assert 'owner.loaded_model(' in source and 'owner.validate_loader(' in source
    tree = ast.parse(source)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ('forward', 'fit', 'backward')]


class FakeValue:
    """A saved-array container, not a neural Tensor implementation."""
    def __init__(self, value, storage=None):
        from types import SimpleNamespace
        self.array = value
        self.shape, self._version, self.requires_grad = value.shape, 0, False
        self.device = 'cuda:0'
        self.storage = storage or SimpleNamespace(data_ptr=lambda: id(self.array), nbytes=lambda: self.array.nbytes)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.array
    def stride(self): return tuple(x//self.array.itemsize for x in self.array.strides)
    def storage_offset(self): return 0
    def untyped_storage(self): return self.storage


class FakeGraph:
    def __init__(self):
        import numpy as np
        self.training = False
        self._parameters = {'weight': FakeValue(np.array([1, 2], dtype='<f4')), 'bias': None}
        self._buffers = {'counter': FakeValue(np.array(3, dtype='<i8'))}
        self._non_persistent_buffers_set = {'counter'}
        self.hooks = []
    def modules(self): return iter([self])
    def named_modules(self, remove_duplicate=False): return iter([('encoder', self)])
    def state_dict(self): return {'encoder.weight': self._parameters['weight']}
    def register_forward_pre_hook(self, callback):
        from types import SimpleNamespace
        self.hooks.append(callback)
        return SimpleNamespace(remove=lambda: self.hooks.remove(callback))


def test_snapshot_data_only_preserves_none_nonpersistent_and_parameter_identity():
    from types import SimpleNamespace
    graph = FakeGraph(); before = graph._parameters['weight']
    rows = m.snapshot(graph, SimpleNamespace(device=lambda x: x))
    assert graph._parameters['weight'] is before
    by = m.validate_rows(rows)
    assert by['encoder.bias']['is_none'] is True
    assert by['encoder.counter']['persistent'] is False
    assert by['encoder.counter']['dtype'] == '<i8'
    assert by['encoder.weight']['value_sha256'] == hashlib.sha256(before.array.tobytes()).hexdigest()


def test_snapshot_alias_classes_and_denominator_failure():
    from types import SimpleNamespace
    graph = FakeGraph()
    graph._parameters['tied'] = graph._parameters['weight']
    graph.state_dict = lambda: {'encoder.weight': None, 'encoder.tied': None}
    rows = m.snapshot(graph, SimpleNamespace(device=lambda x: x))
    by = m.validate_rows(rows)
    assert by['encoder.weight']['tensor_aliases'] == ['encoder.tied', 'encoder.weight']
    assert by['encoder.tied']['storage_aliases'] == ['encoder.tied', 'encoder.weight']
    graph.state_dict = lambda: {}
    with pytest.raises(ValueError, match='denominator'): m.snapshot(graph, SimpleNamespace(device=lambda x: x))


@pytest.fixture
def export_fixture(tmp_path, monkeypatch):
    """Run the real control flow with explicitly non-neural load/copy callbacks."""
    from types import SimpleNamespace
    root = tmp_path/'repo'; (root/'results').mkdir(parents=True)
    torchfile = root/'.venv/lib/python3.12/site-packages/torch/__init__.py'
    torchfile.parent.mkdir(parents=True); torchfile.write_text('# RAM fixture only\n')
    fake_t = SimpleNamespace(__file__=str(torchfile), __version__='fixture_not_torch',
        version=SimpleNamespace(cuda='fixture_not_CUDA'), cuda=SimpleNamespace(is_available=lambda: True),
        empty=lambda *a, **kw: None, device=lambda x: x)
    monkeypatch.setitem(sys.modules, 'torch', fake_t)
    graph = FakeGraph()
    source = {str(Path(m.__file__).resolve()): m.sha(m.__file__)}
    manifest = dict(sources=source, loader_input=dict(policy='/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm'))
    monkeypatch.setattr(m, 'verify_prepare', lambda *a: manifest)
    monkeypatch.setattr(m, 'runtime_sources', lambda *a: source)
    monkeypatch.setattr(m, 'ROOT', root)
    mh = 'c'*64
    def load(p, t, device, store, h):
        files = []
        for name, digest in [('config.json', m.CONFIG_SHA), ('weights.pt', m.WEIGHTS_SHA)]:
            path = str(Path(p['policy'])/name)
            f = dict(path=path, bytes=123, same_buffer=True, sha256=digest,
                     hash_buffer_sha256=digest, deserialize_buffer_sha256=digest)
            files.append(f)
            store.event(dict(kind='load_intent', path=path, expected_sha256=digest))
            store.event(dict(kind='consumed_buffer_frozen', receipt=f))
        store.put('loader.json', dict(schema='agent_p2_equal144_baseline_cuda_20261028_v1.loader_receipt',
            prepare_sha256=h, adapter_source_sha256=m.PINS[m.OWNER.replace('.', '/')+'.py'],
            legacy_loader_source_sha256=m.PINS['experiments/checkpoint_compat.py'],
            files=files, frozen_eval=True, device='cuda:0', remapped_keys=0,
            missing=[], unexpected=[], tensor_hash_before=m.MODEL_HASH))
        return graph, m.MODEL_HASH
    fake_owner = SimpleNamespace(__file__=str(Path(__file__).resolve().parents[1]/(m.OWNER.replace('.', '/')+'.py')),
        loaded_model=load, validate_loader=lambda *args: None,
        enc=SimpleNamespace(tensor_hash=lambda *args: m.MODEL_HASH, frozen=lambda *args: None))
    old_import = m.importlib.import_module
    monkeypatch.setattr(m.importlib, 'import_module', lambda name: fake_owner if name == m.OWNER else old_import(name))
    return root, mh, graph, fake_owner


def test_export_and_saved_consume_control_path_non_neural(export_fixture):
    root, mh, graph, _ = export_fixture
    result = m.export_cuda(root/'fixture_manifest.json', mh, root/'results'/(m.TAG+'_fixture'), main_authorized=True)
    receipt, rows = m.consume_catalog(result['path'], result['sha256'])
    assert len(rows) == 3 and receipt['independent_tensor_rehash'] is False
    report = m.read(result['path'], result['sha256'])
    assert report['completed'] and report['rows_before'] == report['rows_after']
    assert graph.hooks == []


def test_partial_failure_terminal_preserved_and_hooks_removed(export_fixture, monkeypatch):
    root, mh, graph, _ = export_fixture
    real = m.snapshot
    def fail_second(*args):
        fail_second.n += 1
        if fail_second.n == 2: raise ValueError('fixture interrupted second capture')
        return real(*args)
    fail_second.n = 0
    monkeypatch.setattr(m, 'snapshot', fail_second)
    r = m.export_cuda(root/'fixture_manifest.json', mh, root/'results'/(m.TAG+'_partial'), main_authorized=True)
    saved = m.read(r['path'], r['sha256'])
    assert saved['rows_before'] and saved['rows_after'] is None and saved['errors']
    assert not saved['completed'] and not graph.hooks
    with pytest.raises(ValueError, match='complete'): m.consume_catalog(r['path'], r['sha256'])


@pytest.mark.parametrize('field', ['rows_after', 'model_after', 'loader', 'load_events', 'resources'])
def test_saved_export_receipt_mutations_rejected(export_fixture, tmp_path, field):
    root, mh, _, _ = export_fixture
    result = m.export_cuda(root/'fixture_manifest.json', mh, root/'results'/(m.TAG+'_mutation'), main_authorized=True)
    r = m.read(result['path'], result['sha256'])
    if field == 'rows_after': r[field][0]['value_sha256'] = 'b'*64
    if field == 'model_after': r[field] = 'd'*64
    if field == 'loader': r[field]['files'][0]['deserialize_buffer_sha256'] = 'd'*64
    if field == 'load_events': r[field] = []
    if field == 'resources': r[field]['encoder'] = 1
    out = m.write(tmp_path/'mutation.json', r)
    with pytest.raises(ValueError): m.consume_catalog(out['path'], out['sha256'])
