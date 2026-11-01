"""Saved arithmetic and callback fixtures, never CPU/GPU neural execution."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_hilewm_computation_parity_20261029_v1 as m


PRECISION = dict(deterministic=True, tf32_matmul=False, tf32_cudnn=False, autocast=False,
    grad=True, inference=False, math=True, flash=False, efficient=False, cudnn=False)


class Memory:
    def __init__(self): self.arrays = {}
    def put(self, x):
        a = np.asarray(x); key = 'saved_%d'%len(self.arrays)
        self.arrays[key] = a.copy(); return key
    def get(self, k): return self.arrays[k]
    def event(self, events, kind, inputs, output, grad=False):
        e = dict(index=len(events), parent=None, kind=kind, returned=True, precision=dict(PRECISION),
            inputs=[self.put(x) for x in inputs], layouts=[self.layout(x, grad) for x in inputs],
            output=self.put(output), output_layout=self.layout(output, grad))
        events.append(e); return e
    def layout(self, a, grad=False):
        a = np.asarray(a)
        return dict(shape=list(a.shape), stride=[s//a.itemsize for s in a.strides], storage_offset=0,
            contiguous=a.flags.c_contiguous is True or bool(a.flags.c_contiguous),
            dtype={'|u1': 'torch.uint8', '<f4': 'torch.float32', '<f8': 'torch.float64'}[a.dtype.str],
            device='cuda:0', requires_grad=grad)


def low_fixture(memory, slot, route):
    z = np.asarray(slot['z'], np.float32).reshape(1, 3, 192)
    a = np.asarray(slot['a'], np.float32).reshape(1, 2, 10)
    u, w = m.fixed_arrays(); item = dict(status='returned', events=[], checkpoints=[],
        z=memory.put(z), a=memory.put(a), future=memory.put(u))
    pred = []
    for step in range(3):
        outgoing = np.concatenate((a, u[:, :, step].reshape(1, 1, 10)), axis=1)
        # Deliberate synthetic zero callback outputs, NOT a model or derivative.
        emb = np.zeros((1, outgoing.shape[1], 192), np.float32)
        memory.event(item['events'], 'action_encoder', [outgoing], emb, True)
        h = np.zeros((1, 3, 192), np.float32)
        if route == 'p2': memory.event(item['events'], 'P2_predict_API', [z, emb], h, True)
        memory.event(item['events'], 'low_predictor', [z[:, -3:], emb[:, -3:]], h, True)
        memory.event(item['events'], 'low_pred_proj', [h.reshape(3, 192)], h.reshape(3, 192), True)
        nxt = h[:, -1:]
        z = np.concatenate((z[:, 1:] if route == 'p2' else z, nxt), axis=1)
        a = outgoing[:, 1:] if route == 'p2' else outgoing
        item['checkpoints'].append(dict(z=memory.put(z), a=memory.put(a)))
        pred.append(nxt[:, 0])
    prediction = np.stack(pred, axis=1).reshape(1, 1, 3, 192)
    scalar = np.array(0., np.float64); gradient = np.zeros_like(u)
    item.update(prediction=memory.put(prediction), scalar=memory.put(scalar), gradient=memory.put(gradient))
    memory.event(item['events'], 'scalar_VJP', [scalar, u], gradient, True)
    return item


def fixture():
    mem = Memory(); slots = []; rows = []
    for root in (0, 1):
        slot = dict(root=root, z=np.arange(576, dtype=np.float32).reshape(3, 192).tolist(),
                    a=np.arange(20, dtype=np.float32).reshape(2, 10).tolist(), pixel_sha256=[])
        row = dict(root=root, status='returned', pixels=[], routes={})
        for image in range(4):
            raw = np.full((224, 224, 3), root+image, np.uint8)
            slot['pixel_sha256'].append(hashlib.sha256(raw.tobytes()).hexdigest())
            panel = dict(image=image, rgb=mem.put(raw), paths={})
            for backend in ('p2', 'hi'):
                x = np.zeros((1, 3, 224, 224), np.float32); h = np.zeros((1, 257, 192), np.float32)
                events = []
                mem.event(events, 'encoder', [x], h)
                mem.event(events, 'pixel.patch', [x], np.zeros((1, 192, 16, 16), np.float32))
                for i in range(12): mem.event(events, 'pixel.layer'+str(i), [h], h)
                mem.event(events, 'pixel.finalnorm', [h], h)
                mem.event(events, 'projector', [h[:, 0]], h[:, 0])
                panel['paths'][backend] = dict(status='returned', preprocessed=mem.put(x),
                    preprocessed_layout=mem.layout(x), output=mem.put(h[:, 0]), events=events)
            row['pixels'].append(panel)
        row['routes'] = {route: low_fixture(mem, slot, route) for route in m.ROUTES}
        rows.append(row); slots.append(slot)
    return dict(slots=slots), dict(rows=rows), mem


@pytest.fixture(scope='module')
def saved(): return fixture()


def test_full_saved_denominator_and_resources(saved):
    manifest, report, mem = saved
    result = m.analyze(manifest, report, mem.get)
    assert result['all_values_exact']
    assert result['resources'] == dict(encoder=16, projector=16, action_encoder=18, low_predictor=18,
                                      low_pred_proj=18, P2_predict_API=6, scalar_VJP=6)
    assert not result['normalizer_pairing'] and not result['scientific_admission']


def test_low_full_history_not_truncated_to_p2_shape(saved):
    manifest, report, mem = saved
    r = report['rows'][0]['routes']
    assert [mem.get(c['z']).shape[1] for c in r['p2']['checkpoints']] == [3, 3, 3]
    assert [mem.get(c['z']).shape[1] for c in r['hi_whole']['checkpoints']] == [4, 5, 6]
    for route in m.ROUTES: m.validate_low(manifest['slots'][0], route, r[route], mem.get)


@pytest.mark.parametrize('field', ['append', 'AE_delivery', 'project_flatten', 'context_z', 'context_a', 'prediction', 'scalar', 'VJP_leaf', 'VJP_gradflag', 'precision', 'order'])
def test_saved_low_mutations_rejected(saved, field):
    manifest, report, mem = saved
    item = deepcopy(report['rows'][0]['routes']['hi_whole']); events = item['events']
    # New wrong descriptors only in the copy, leave shared baseline untouched.
    if field == 'append': events[0]['inputs'][0] = mem.put(np.zeros((1, 3, 10), np.float32))
    if field == 'AE_delivery': events[1]['inputs'][1] = mem.put(np.ones((1, 3, 192), np.float32))
    if field == 'project_flatten': events[2]['inputs'][0] = mem.put(np.ones((3, 192), np.float32))
    if field == 'context_z': item['checkpoints'][0]['z'] = mem.put(np.zeros((1, 4, 192), np.float32))
    if field == 'context_a': item['checkpoints'][0]['a'] = mem.put(np.zeros((1, 3, 10), np.float32))
    if field == 'prediction': item['prediction'] = mem.put(np.ones((1, 1, 3, 192), np.float32))
    if field == 'scalar': item['scalar'] = mem.put(np.array(1., np.float64))
    if field == 'VJP_leaf': events[-1]['inputs'][1] = mem.put(np.zeros((1, 1, 3, 10), np.float32))
    if field == 'VJP_gradflag': events[-1]['layouts'][1]['requires_grad'] = False
    if field == 'precision': events[0]['precision']['grad'] = False
    if field == 'order': events[0]['kind'] = 'low_predictor'
    with pytest.raises(ValueError): m.validate_low(manifest['slots'][0], 'hi_whole', item, mem.get)


def test_numerical_difference_is_diagnostic_not_early_abort(saved):
    manifest, report, mem = saved; report = deepcopy(report)
    # Change a real gradient witness AND its captured output consistently. No neural claim.
    item = report['rows'][1]['routes']['hi_segmented']
    wrong = mem.put(np.ones((1, 1, 3, 10), np.float32))
    item['gradient'] = wrong; item['events'][-1]['output'] = wrong
    d = m.analyze(manifest, report, mem.get)
    assert len(d['roots']) == 2 and not d['all_values_exact']
    assert d['roots'][0]['all_values_exact']
    assert any(not c['exact'] and c['name'] == 'hi_segmented.gradient' for c in d['roots'][1]['comparisons'])


@pytest.mark.parametrize('field', ['root', 'missing_pixel', 'missing_route', 'missing_module', 'pixel_RGB', 'pixel_delivery'])
def test_all_saved_root_and_pixel_bindings(saved, field):
    manifest, report, mem = saved; report = deepcopy(report)
    row = report['rows'][0]
    if field == 'root': row['root'] = 1
    if field == 'missing_pixel': row['pixels'].pop()
    if field == 'missing_route': row['routes'].pop('p2')
    if field == 'missing_module': row['pixels'][0]['paths']['p2']['events'].pop()
    if field == 'pixel_RGB': row['pixels'][0]['rgb'] = mem.put(np.ones((224, 224, 3), np.uint8))
    if field == 'pixel_delivery': row['pixels'][0]['paths']['hi']['output'] = mem.put(np.ones((1, 192), np.float32))
    with pytest.raises(ValueError): m.analyze(manifest, report, mem.get)


def test_actual_known2_saved_task_RGB_history_ABI_only():
    from experiments.spectral_reachability.p1_constructive_moving_history_encode_cuda_20261027_v1 import decode_blind
    original = m.cat.read(m.cat.PREP, m.cat.PREP_SHA)
    assert [(s['root_slot'], s['condition']) for s in original['slots']] == [(0, 'off'), (1, 'off')]
    for slot in original['slots']:
        data = m.task_input(slot, decode_blind)
        assert len(data['z']) == 3 and len(data['a']) == 2 and len(data['pixel_sha256']) == 4


def test_fixed_arrays_and_protocol_no_raw_normalization():
    a, w = m.fixed_arrays()
    assert a.shape == (1, 1, 3, 10) and w.shape == (192,)
    assert a.dtype == w.dtype == np.float32
    p = m.protocol()
    assert p['roots'] == [0, 1] and p['low_input_not_replaced_by_reencoded_pixels']
    assert not p['raw_inverse'] and p['new_resources']['explicit_scalar_VJP'] == 6


def test_authorization_before_any_production_io():
    with pytest.raises(ValueError, match='MAIN authorization'): m.run_cuda(None, None, None)


def test_compare_bitwise_no_tolerance():
    a = np.zeros(2, np.float32); b = a.copy(); b[0] = np.nextafter(np.float32(0), np.float32(1))
    assert not m.compare(a, b)['exact']
    b = np.array([-0., 0.], np.float32)
    assert not m.compare(a, b)['exact'] and m.compare(a, b)['maxabs'] == 0
    assert not m.compare(a, a.astype(np.float64))['same_ABI']


def test_census_includes_failed_attempt(saved):
    _, report, _ = saved; report = deepcopy(report)
    report['rows'][0]['routes']['p2']['events'][0]['returned'] = False
    c = m.census(report['rows'])
    assert c['action_encoder']['attempts'] == 18 and c['action_encoder']['returns'] == 17
    assert c['action_encoder']['input_tokens'] == 66


def test_array_writer_reader_tamper_and_exclusive(tmp_path, monkeypatch):
    root = tmp_path/'root'; (root/'results').mkdir(parents=True); monkeypatch.setattr(m, 'ROOT', root)
    s = m.Store(root/'results'/(m.TAG+'_ram'))
    name = s.array(np.arange(12, dtype=np.float32).reshape(3, 4))
    np.testing.assert_array_equal(m.load_array(s.path, name, s.arrays[name]), np.arange(12, dtype=np.float32).reshape(3, 4))
    desc = dict(s.arrays[name]); desc['data_sha256'] = '0'*64
    with pytest.raises(ValueError, match='data'): m.load_array(s.path, name, desc)
    with pytest.raises(ValueError): m.load_array(s.path, '../evil.npy', desc)
    with pytest.raises(ValueError): m.Store(s.path)


def test_manifest_closure_includes_constructor_metadata_and_runtime_without_weights():
    src = m.closure()
    for folder, name in [('tests', 'test_'+m.rgb.loader.TAG+'.py'), ('innerdocs', m.rgb.loader.TAG+'.md')]:
        p = str(m.ROOT/folder/name); assert p in src and src[p] == m.sha(p)
    assert not any(p.endswith('weights.pt') for p in src)


def test_source_no_hidden_model_fitting_or_global_monkeypatch():
    import ast
    tree = ast.parse(Path(m.__file__).read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ('fit', 'train', 'load_state_dict')]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)]
    assert 'model.predict(zs, emb)' in Path(m.__file__).read_text()


def test_partial_outputs_not_fill_or_select(saved):
    manifest, report, mem = saved; report = deepcopy(report)
    report['rows'][1]['status'] = 'not_started'
    with pytest.raises(ValueError): m.analyze(manifest, report, mem.get)


def test_exact_786_array_reference_census_fixture():
    manifest, report, mem = fixture(); referenced = set()
    def get(name): referenced.add(name); return mem.get(name)
    d = m.analyze(manifest, report, get)
    assert len(mem.arrays) == 786 and referenced == set(mem.arrays)
    assert len(d['roots'][0]['paid_history_checks']) == 6


class Callback:
    """Forward/hook control-flow double. No neural arithmetic."""
    def __init__(self, fn): self.fn = fn; self.pre = []; self.post = []
    def register_forward_pre_hook(self, fn):
        self.pre.append(fn); return NS(remove=lambda: self.pre.remove(fn))
    def register_forward_hook(self, fn):
        self.post.append(fn); return NS(remove=lambda: self.post.remove(fn))
    def __call__(self, *args, **kw):
        for fn in self.pre: fn(self, args)
        out = self.fn(*args, **kw)
        for fn in self.post: fn(self, args, out)
        return out


class CallbackTrace:
    def __init__(self): self.events = []
    def begin(self, kind, args):
        e = dict(kind=kind, inputs=args, returned=False); self.events.append(e); return e
    def end(self, e, value): e.update(output=value, returned=True); return value
    def call(self, kind, args, fn): return self.end(self.begin(kind, args), fn())


def pixel_callbacks(fail=False):
    patch = Callback(lambda x: np.zeros((1, 192, 16, 16), np.float32))
    layers = [Callback(lambda x: x) for _ in range(12)]
    norm = Callback(lambda x: x)
    def encode(x, **kw):
        assert kw == {'interpolate_pos_encoding': True}
        patch(x); value = np.zeros((1, 257, 192), np.float32)
        for layer in layers: value = layer(value)
        if fail: raise ValueError('RAM callback failure')
        return NS(last_hidden_state=norm(value))
    encoder = Callback(encode)
    encoder.embeddings = NS(patch_embeddings=NS(projection=patch))
    encoder.layers = layers; encoder.layernorm = norm
    projector = Callback(lambda x: x)
    model = NS(encoder=encoder, projector=projector)
    def original_encode(info):
        x = info['pixels']; hidden = encoder(x.reshape(1, 3, 224, 224), interpolate_pos_encoding=True).last_hidden_state
        return dict(emb=projector(hidden[:, 0])[:, None])
    model.encode = original_encode
    return model, [encoder, patch, *layers, norm, projector]


def test_actual_pixel_encode_callback_order_and_hook_cleanup():
    model, nodes = pixel_callbacks(); trace = CallbackTrace()
    out = m.p2_pixel(None, model, np.zeros((1, 3, 224, 224), np.float32), trace)
    assert out.shape == (1, 192)
    assert [e['kind'] for e in trace.events] == ['encoder']+list(m.PIXEL_KINDS[:-2])+['projector']
    assert all(e['returned'] for e in trace.events)
    assert all(not n.pre and not n.post for n in nodes)


def test_actual_pixel_callback_exception_cleans_all_hooks():
    model, nodes = pixel_callbacks(fail=True); trace = CallbackTrace()
    with pytest.raises(ValueError, match='RAM callback'): m.p2_pixel(None, model, np.zeros((1, 3, 224, 224), np.float32), trace)
    assert not trace.events[0]['returned'] and all(not n.pre and not n.post for n in nodes)


def low_callbacks(fail=False):
    encoder = Callback(lambda a: np.zeros((1, a.shape[1], 192), np.float32))
    predictor = Callback(lambda z, a: np.zeros(z.shape, np.float32))
    projector = Callback(lambda x: x)
    def original_predict(z, emb):
        out = predictor(z, emb)
        if fail: raise ValueError('RAM low callback failure')
        return projector(out.reshape(-1, 192)).reshape(1, 3, 192)
    return NS(action_encoder=encoder, predictor=predictor, pred_proj=projector, predict=original_predict), [predictor, projector]


def test_original_predict_control_path_rolling3_not_Hi_recursion():
    model, nodes = low_callbacks(); trace = CallbackTrace()
    t = NS(cat=lambda xs, dim: np.concatenate(xs, axis=dim), stack=lambda xs, dim: np.stack(xs, axis=dim))
    future, _ = m.fixed_arrays()
    out = m.p2_low(t, model, np.ones((1, 3, 192), np.float32), np.ones((1, 2, 10), np.float32), future, trace)
    assert out.prediction.shape == (1, 1, 3, 192)
    assert all(c.z.shape == (1, 3, 192) and c.a.shape == (1, 2, 10) for c in out.contexts)
    assert [e['kind'] for e in trace.events] == ['action_encoder', 'P2_predict_API', 'low_predictor', 'low_pred_proj']*3
    assert all(not n.pre and not n.post for n in nodes)


def test_original_predict_callback_failure_no_hook_leak():
    model, nodes = low_callbacks(fail=True); trace = CallbackTrace()
    t = NS(cat=lambda xs, dim: np.concatenate(xs, axis=dim), stack=lambda xs, dim: np.stack(xs, axis=dim))
    with pytest.raises(ValueError, match='RAM low'):
        m.p2_low(t, model, np.ones((1, 3, 192), np.float32), np.ones((1, 2, 10), np.float32), m.fixed_arrays()[0], trace)
    assert all(not n.pre and not n.post for n in nodes)


def test_complete_saved_consumer_fixture_and_tamper(tmp_path, monkeypatch):
    """Real saved consumer path, no original consumer/authority/model execution."""
    manifest, report, mem = fixture()
    root = tmp_path/'root'; (root/'results').mkdir(parents=True); monkeypatch.setattr(m, 'ROOT', root)
    own = {}
    for folder, name in (('experiments/spectral_reachability', m.rgb.loader.TAG+'.py'),
                         ('tests', 'test_'+m.rgb.loader.TAG+'.py'), ('innerdocs', m.rgb.loader.TAG+'.md')):
        path = root/folder/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('RAM metadata fixture, not real loader\n'); own[str(path)] = m.sha(path)
    store = m.Store(root/'results'/(m.TAG+'_ram'))
    mapping = {k: store.array(v) for k, v in mem.arrays.items()}
    def replace(value):
        if isinstance(value, dict): return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list): return [replace(v) for v in value]
        return mapping.get(value, value) if isinstance(value, str) else value
    report = replace(report)
    expected_hi = {'fixture_tensor_%d'%i: 'a'*64 for i in range(425)}
    before = dict(p2=m.cat.MODEL_HASH, p2_catalog=[], hi=expected_hi)
    manifest.update(sources=own, expected_hi_hashes=expected_hi, expected_p2_catalog=[], p2_loader={},
        construction_expected=dict(module_count=531, entry_count=491, state_key_count=425, ir_sha256=m.cat.IR_SHA),
        export=dict(report_sha256=m.cat.HI_SHA, receipt_sha256='a'*64))
    construction = dict(manifest['construction_expected'], own_sha256=own, tensor_value_hashes=expected_hi,
        schema=m.rgb.loader.TAG+'.construction', export_report_sha256=m.cat.HI_SHA, export_receipt_sha256='a'*64,
        tensor_copy_attempts=425, tensor_copy_returns=425, all_parameters_frozen=True, forward_calls=0, physics=0)
    loader = dict(files=[])
    report.update(schema=m.TAG+'.report', completed=True, errors=[], manifest='fixture_manifest.json', manifest_sha256='b'*64,
        arrays=store.arrays, model_before=before, model_after=deepcopy(before), construction=construction,
        sources_after=own, runtime_before=own, runtime_after=own, loader=loader, load_events=[],
        actual_call_census=m.census(report['rows']))
    report['diagnostics'] = m.analyze(manifest, report, lambda name: m.load_array(store.path, name, store.arrays[name]))
    store.put('report.json', m.encode(report), final=True)
    monkeypatch.setattr(m, 'verify', lambda *args: manifest)
    real_import = m.importlib.import_module
    calls = []
    monkeypatch.setattr(m.importlib, 'import_module', lambda name: NS(validate_loader=lambda *args: calls.append(args))
                        if name == m.cat.OWNER else real_import(name))
    out = root/'results'/'fixture_receipt.json'
    result = m.consume(store.path/'report.json', m.sha(store.path/'report.json'), out)
    assert calls and m.read(out, result['sha256'])['all_values_exact'] is True
    # Mutate a real persisted descriptor-bound trace: consumer rejects before receipt.
    report['construction']['own_sha256'] = {}
    path = store.path/'report.json'; path.write_bytes(m.encode(report))
    with pytest.raises(ValueError, match='complete own set'):
        m.consume(path, m.sha(path), root/'results'/'must_not_write.json')
