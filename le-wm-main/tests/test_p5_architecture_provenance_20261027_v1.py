"""Small local fixtures only; never scientific or official authentication."""
import copy
import json
from pathlib import Path

import pytest
from experiments.spectral_reachability import p5_architecture_provenance_20261027_v1 as m


def stats():
    return dict(ACTION_MEAN=[-.0087, .0068], ACTION_STD=[.2019, .2002],
                PROPRIO_MEAN=[1., 2., 3., 4.], PROPRIO_STD=[2., 3., 4., 5.])


def stats_source(data=None):
    return '\n'.join(f'{k} = torch.tensor({v!r})' for k, v in (data or stats()).items())


def history():
    s = stats()
    h = [[[float(i + j) / 100., -float(i + j) / 100.] for i in range(5)] for j in range(2)]
    norm = lambda packets: [[[m.project_action(a[i], s['ACTION_MEAN'][i], s['ACTION_STD'][i])
                              for i in range(2)] for a in p] for p in packets]
    return {'action_coordinates': dict(history_native_u64=h, history_normalized_f32=norm(h),
        nominal_native_u64=[[0., 0.]] * 5, nominal_normalized_f32=norm([[[0., 0.]] * 5])[0],
        action_mean_float32=[m.f32(x) for x in s['ACTION_MEAN']],
        action_std_float32=[m.f32(x) for x in s['ACTION_STD']], dataset_projection='fixture', inverse='fixture')}


def test_stats_without_torch():
    assert m.stats_literals('import never_executed\n' + stats_source()) == stats()


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'call', 'zero_std', 'bool', 'dimension', 'nan'])
def test_bad_stats(kind):
    s = stats()
    text = stats_source()
    if kind == 'duplicate':
        text += '\nACTION_MEAN = torch.tensor([0., 0.])'
    elif kind == 'missing':
        text = '\n'.join(text.splitlines()[1:])
    elif kind == 'call':
        text = text.replace('[-0.0087, 0.0068]', "__import__('os').system('never')")
    else:
        if kind == 'zero_std':
            s['ACTION_STD'][0] = 0.
        elif kind == 'bool':
            s['ACTION_MEAN'][0] = True
        elif kind == 'dimension':
            s['PROPRIO_MEAN'].pop()
        else:
            s['ACTION_MEAN'][0] = float('nan')
        text = stats_source(s)
    with pytest.raises((ValueError, TypeError)):
        m.stats_literals(text)


def test_history_float32_not_normalized_zero():
    r = m.verify_history(history(), stats())
    assert r['verified_history_scalars'] == 20
    assert r['verified_nominal_scalars'] == 10
    assert r['normalized_native_zero'] != [0., 0.]


@pytest.mark.parametrize('kind', ['shape', 'value', 'stats', 'zero'])
def test_bad_history(kind):
    rec = history()
    c = rec['action_coordinates']
    if kind == 'shape':
        c['history_native_u64'].pop()
    elif kind == 'value':
        c['history_normalized_f32'][0][0][0] += 1e-4
    elif kind == 'stats':
        c['action_std_float32'][0] *= 2.
    else:
        c['nominal_normalized_f32'] = [[0., 0.]] * 5
    with pytest.raises(ValueError):
        m.verify_history(rec, stats())


def config_source():
    tree = {}
    for field in m.CONFIG_FIELDS:
        cur = tree
        parts = field.split('.')
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = 'null' if field.endswith('iterations_per_epoch') else 'true' if field.endswith('freeze_encoder') else '1'
    def render(d, indent=0):
        out = []
        for k, v in d.items():
            out.append(' ' * indent + k + ':' + ('' if isinstance(v, dict) else ' ' + v))
            if isinstance(v, dict):
                out.extend(render(v, indent + 2))
        return out
    return '\n'.join(render(tree))


def test_config_selected_scalars_not_YAML_execution():
    d = m.selected_config(config_source())
    assert d['meta.freeze_encoder'] is True
    assert d['optimization.transition_model.iterations_per_epoch'] is None
    assert m.selected_config(config_source() + '\nunrelated:\n  - !!python/name:never') == d


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'tag', 'anchor', 'nonfinite', 'interpolation'])
def test_bad_config(kind):
    text = config_source()
    if kind == 'missing':
        text = text.replace('nodes: 1', '')
    elif kind == 'duplicate':
        text += '\nnodes: 1'
    else:
        val = {'tag': '!!python/object:never', 'anchor': '&a 1', 'nonfinite': '1e999',
               'interpolation': '$' + '{never}'}[kind]
        text = text.replace('nodes: 1', 'nodes: ' + val)
    with pytest.raises(ValueError):
        m.selected_config(text)


def checkpoint(tmp_path):
    p = tmp_path / 'weights.bin'
    p.write_bytes(b'not deserialized')
    shapes = {'predictor.module.predictor_embed.weight': [384, 384],
              'predictor.module.action_encoder.weight': [400, 10],
              'predictor.module.predictor_blocks.0.attn.qkv.weight': [1200, 400],
              'predictor.module.predictor_blocks.0.adaLN_modulation.1.weight': [2400, 400],
              'proprio_encoder.module.patch_embed.weight': [16, 4, 1]}
    comp = dict(exact_match=True, missing_keys=[], unexpected_keys=[], shape_mismatches=[], dtype_mismatches=[])
    return dict(task='pusht', path=str(p), expected_sha256=m.sha(p.read_bytes()), status='exact_source_schema_match',
        comparison={'predictor': comp, 'proprio_encoder': copy.deepcopy(comp)},
        native_interface={'proprio_dim': 4}, config={'model': {'visual_encoder': {'embed_dim': 384},
                                                           'proprio_encoder': {'proprio_emb_dim': 16}}},
        nested_metadata=[dict(path=k.split('.'), kind='tensor', shape=v) for k, v in shapes.items()] +
        [dict(path=['opt', 0, 'state'], kind='tensor', shape=[400]),
         dict(path=['epoch'], kind='int', value=50), dict(path=['stats', 'steps_seen'], kind='int', value=0)])


def test_current_weight_hash_saved_shapes_not_official(tmp_path):
    r = m.metadata_checkpoint(m.Reader(tmp_path), checkpoint(tmp_path))
    assert r['saved_checkpoint_counters'] == {'epoch': 50, 'stats.steps_seen': 0}
    assert r['official_authentication'] == 'unknown'
    assert 'no new tensor load' in r['tensor_scope']


@pytest.mark.parametrize('kind', ['hash', 'shape', 'comparison', 'interface'])
def test_bad_checkpoint(tmp_path, kind):
    row = checkpoint(tmp_path)
    if kind == 'hash':
        row['expected_sha256'] = '0' * 64
    elif kind == 'shape':
        row['nested_metadata'][0]['shape'] = [192, 192]
    elif kind == 'comparison':
        row['comparison']['predictor']['exact_match'] = 1
    else:
        row['native_interface']['proprio_dim'] = 2
    with pytest.raises(ValueError):
        m.metadata_checkpoint(m.Reader(tmp_path), row)


def test_source_revision_edits_bound_vs_imported(tmp_path, monkeypatch):
    prep, receipt, status = {'bindings': {}}, {'imported_sources': {}}, b''
    for i, path in enumerate(m.EDITS):
        p = tmp_path / m.JEPA / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('current')
        prep['bindings'][str(p)] = m.sha(b'current')
        status += (' M ' + path).encode() + b'\0'
        if i == 2:
            receipt['imported_sources']['module'] = str(p)
    def git(repo, args):
        return b'a' * 40 + b'\n' if args[0] == 'rev-parse' else status if args[0] == 'status' else b'original'
    monkeypatch.setattr(m, 'git_read', git)
    rec = m.source_identity(m.Reader(tmp_path), prep, receipt)
    assert rec['tracked_edits'] == 5
    assert sum(x['C1_module_file_imported'] for x in rec['edits']) == 1
    assert all(not x['semantic_equivalence_certified'] for x in rec['edits'])


def test_new_local_edit_requires_review(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'git_read', lambda repo, args: b'a' * 40 if args[0] == 'rev-parse' else b'?? new.py\0')
    with pytest.raises(ValueError, match='census'):
        m.source_identity(m.Reader(tmp_path), {'bindings': {}}, {'imported_sources': {}})


@pytest.mark.parametrize('args', [['fetch'], ['config', '--list'], ['show', 'HEAD:../../secret'], ['checkout', 'HEAD']])
def test_no_mutating_or_remote_git(tmp_path, args):
    with pytest.raises(ValueError, match='not allowed'):
        m.git_read(tmp_path, args)


def test_C1_source_seal_actual_bytes(tmp_path):
    seal = tmp_path / 'seal'
    (seal / 'sources').mkdir(parents=True)
    rows = []
    for i in range(6):
        raw = ('source' + str(i)).encode()
        p = tmp_path / f'original_{i}.py'
        p.write_bytes(raw)
        archive = f'sources/{i:03d}.source'
        (seal / archive).write_bytes(raw)
        rows.append(dict(source=str(p), archive=archive, sha256=m.sha(raw), size=len(raw)))
    (seal / 'args.json').write_bytes(b'{}')
    raw = json.dumps(dict(schema='d2p_prerun_source_seal.v1', files=rows, args_sha256=m.sha(b'{}'))).encode()
    (seal / 'manifest.json').write_bytes(raw)
    prep = {'source_seal': dict(output_dir=str(seal), manifest_sha256=m.sha(raw), source_count=6,
                               source_bytes=sum(r['size'] for r in rows))}
    assert m.verify_source_seal(m.Reader(tmp_path), prep)['source_count'] == 6
    (tmp_path / 'original_0.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        m.verify_source_seal(m.Reader(tmp_path), prep)


@pytest.mark.parametrize('raw', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '{bad'])
def test_strict_JSON(raw):
    with pytest.raises(ValueError):
        m.decode(raw)


@pytest.mark.parametrize('kind', ['absolute', 'escape', 'symlink', 'small_cap', 'blob_cap', 'total'])
def test_reader_limits(tmp_path, monkeypatch, kind):
    p = tmp_path / 'x'
    p.write_bytes(b'{}')
    rel, blob = 'x', False
    if kind == 'absolute':
        rel = str(p)
    elif kind == 'escape':
        rel = '../x'
    elif kind == 'symlink':
        (tmp_path / 'link').symlink_to(p)
        rel = 'link'
    elif kind == 'small_cap':
        monkeypatch.setattr(m, 'SMALL_CAP', 1)
    elif kind == 'blob_cap':
        monkeypatch.setattr(m, 'BLOB_CAP', 1)
        blob = True
    else:
        monkeypatch.setattr(m, 'TOTAL_CAP', 1)
    with pytest.raises(ValueError):
        m.Reader(tmp_path).raw(rel, blob=blob)


def test_blob_drift(tmp_path):
    p = tmp_path / 'x'
    p.write_bytes(b'one')
    r = m.Reader(tmp_path)
    r.raw('x', blob=True)
    p.write_bytes(b'two')
    with pytest.raises(ValueError):
        r.stable()


def test_unknowns_explicit_owner_and_zero_cost_boundary():
    rows = m.unknowns()
    assert len(rows) == len({r['id'] for r in rows}) == 6
    assert all(r['status'] == 'unknown' and r['needed_from'] and r['required_evidence'] for r in rows)
    assert 'zero counters are uninformative' in next(r for r in rows if r['id'] == 'actual_pretraining_cost')['required_evidence']


def test_disk_reserve_before_read(tmp_path, monkeypatch):
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda root: type('Disk', (), {'free': m.RESERVE - 1})())
    with pytest.raises(ValueError, match='200 MiB'):
        m.produce(tmp_path)


@pytest.mark.parametrize('key,value', [('official_authenticated', True), ('RGB_only', True),
                                     ('actual_GPU_hours', 0), ('official_authenticated', 0)])
def test_contract_tampering(key, value):
    current = dict(official_authenticated=False, RGB_only=False, actual_GPU_hours=None)
    saved = copy.deepcopy(current)
    saved[key] = value
    assert m.verify(current, current)
    with pytest.raises(ValueError):
        m.verify(saved, current)


def test_cli_incomplete_not_scientific_admission(monkeypatch, capsys):
    monkeypatch.setattr(m, 'produce', lambda: dict(official_authenticated=False, architecture_contract_complete=False))
    assert m.main([]) == 0
    assert json.loads(capsys.readouterr().out)['official_authenticated'] is False
    assert m.main(['--require-complete']) == 3


def test_stdlib_only():
    import ast
    tree = ast.parse(Path(m.__file__).read_text())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported.update(a.name.split('.')[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split('.')[0])
    assert imported <= {'argparse', 'ast', 'hashlib', 'json', 'math', 'pathlib', 're', 'shutil', 'stat', 'struct', 'subprocess'}
