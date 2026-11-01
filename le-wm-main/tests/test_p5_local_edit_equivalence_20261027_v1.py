"""Contract fixtures and real saved-source AST checks; never import a model."""
import ast
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'experiments/spectral_reachability'
sys.path.insert(0, str(SCRIPTS))
import p5_local_edit_equivalence_20261027_v1 as audit


def sources(suffix):
    prov = audit.provenance.decode((ROOT / audit.PROV).read_bytes())
    edit = next(e for e in prov['source_identity']['edits'] if e['path'].endswith(suffix))
    original = audit.provenance.git_read(ROOT / audit.provenance.JEPA,
        ['show', 'HEAD:' + edit['path'][len(audit.provenance.JEPA)+1:]]).decode()
    current = (ROOT / edit['path']).read_text()
    return original, current


@pytest.mark.parametrize('suffix,cls,attrs', [
    ('prop_embedding.py', 'ProprioceptiveEmbedding', {'use_mlp': False, 'tokens_per_step': 1}),
    ('/vit.py', 'Attention', {'use_sdpa': False}),
])
def test_actual_defaults(suffix, cls, attrs):
    result = audit.prove_defaults(*sources(suffix), cls, attrs)
    assert result['projected_full_AST_equal']
    assert result['numeric_kernel_pair_executed'] is False


@pytest.mark.parametrize('old,new', [
    ('getattr(self, "tokens_per_step", 1)', 'getattr(self, "tokens_per_step", 2)'),
    ('getattr(self, "use_mlp", False)', 'getattr(self, "use_mlp", True)'),
    ('self.use_mlp = use_mlp', 'self.use_mlp = False'),
    ('x = x.permute(0, 2, 1)', 'x = x.permute(0, 1, 2)'),
])
def test_default_mutations_rejected(old, new):
    base, current = sources('prop_embedding.py')
    assert old in current
    with pytest.raises(ValueError):
        audit.prove_defaults(base, current.replace(old, new), 'ProprioceptiveEmbedding',
                             {'use_mlp': False, 'tokens_per_step': 1})


def test_missing_attribute_is_not_equivalence():
    # Plain Python attribute lookup only, not a learned model object.
    obj = object()
    with pytest.raises(AttributeError):
        getattr(obj, 'use_sdpa')
    assert getattr(obj, 'use_sdpa', False) is False


def test_dataset_actual():
    result = audit.prove_dataset(*sources('pusht_dset.py'))
    assert result['symbolic_channel_checks'] == 72


def test_dataset_other_change_rejected():
    base, current = sources('pusht_dset.py')
    with pytest.raises(ValueError):
        audit.prove_dataset(base, current.replace('image / 255.0', 'image / 256.0'))


def test_dino_actual():
    assert audit.prove_dino(*sources('/dino.py'))['absent_local_full_AST_equal']


def test_dino_retained_fallback_change_rejected():
    base, current = sources('/dino.py')
    with pytest.raises(ValueError):
        audit.prove_dino(base, current.replace('"facebookresearch/dinov2"', '"other/repo"'))


def test_wrapper_actual():
    assert audit.prove_wrapper(*sources('vit_enc_preds.py'))['residual_full_AST_equal']


@pytest.mark.parametrize('old,new', [
    ('torch.zeros(', 'torch.ones('),
    ('[pad, act_feats]', '[act_feats, pad]'),
    ('dtype=act_feats.dtype', 'dtype=None'),
    ('act_feats[:, -context_len:]', 'act_feats[:, :context_len]'),
    ('act_feats.shape[1] < context_len', 'act_feats.shape[1] <= context_len'),
])
def test_wrapper_mutations_rejected(old, new):
    base, current = sources('vit_enc_preds.py')
    assert old in current
    with pytest.raises(ValueError):
        audit.prove_wrapper(base, current.replace(old, new))


@pytest.mark.parametrize('v,a,w,equal', [
    (3, 3, 3, True), (8, 5, 3, True), (2, 2, 5, True),
    (3, 1, 3, False), (1, 2, 2, False), (2, 3, 5, False),
])
def test_slice_edges(v, a, w, equal):
    old, new = audit.action_slices(v, a, w)
    assert (old == new) is equal


def test_all_symbol_slices():
    result = audit.enumerate_slices()
    assert result['counts'] == dict(equal=106, zero_padding_change=55, truncation_change=55)


@pytest.mark.parametrize('bad', [0, -1, 1.5, True])
def test_invalid_lengths(bad):
    with pytest.raises(ValueError):
        audit.action_slices(bad, 2, 3)


def tensor_records(dim):
    return [dict(path=[part, 'module.' + key], kind='tensor', shape=shape, dtype='torch.float32')
            for part, schema in audit.state_schema(dim).items() for key, shape in schema.items()]


@pytest.mark.parametrize('dim,numel', [(4, 80), (2, 48)])
def test_full_94_key_schema(dim, numel):
    result = audit.compare_metadata(tensor_records(dim), audit.state_schema(dim))
    assert result['predictor']['keys'] == 92
    assert result['predictor']['numel'] == 17626480
    assert result['proprio_encoder']['keys'] == 2
    assert result['proprio_encoder']['numel'] == numel


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'dtype', 'shape', 'prefix'])
def test_schema_rejects_mutation(change):
    records = tensor_records(4)
    if change == 'missing':
        records.pop()
    elif change == 'duplicate':
        records.append(copy.deepcopy(records[0]))
    elif change == 'dtype':
        records[0]['dtype'] = 'torch.float16'
    elif change == 'shape':
        records[0]['shape'] = [384, 385]
    else:
        records[0]['path'][1] = 'bad'
    with pytest.raises(ValueError):
        audit.compare_metadata(records, audit.state_schema(4))


def test_wall_cannot_use_pusht_proprio():
    with pytest.raises(ValueError):
        audit.compare_metadata(tensor_records(2), audit.state_schema(4))


def test_optimizer_records_ignored_not_loaded():
    records = tensor_records(4) + [
        dict(path=['opt', 'state', 0], kind='tensor', dtype='torch.int64', shape=[])]
    audit.compare_metadata(records, audit.state_schema(4))


def test_packet_order():
    assert audit.packet_indices() == [f't{t}d{d}' for t in range(5) for d in range(2)]


@pytest.mark.parametrize('field', ['model_forward', 'encoder_runtime_verified', 'P5_admission'])
def test_saved_strict_receipt_cannot_be_promoted(field):
    rec = audit.provenance.decode((ROOT / audit.STRICT / 'pusht_native_strict_cuda_main_v1.json').read_bytes())
    rec[field] = True
    with pytest.raises(ValueError):
        audit.check_receipt(rec, 'pusht', rec['checkpoint_sha256_before'],
                            audit.compare_metadata(tensor_records(4), audit.state_schema(4)))


def test_no_neural_imports_or_target_exec():
    tree = ast.parse((SCRIPTS / (audit.NAME + '.py')).read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [n.module for n in imports if isinstance(n, ast.ImportFrom)] + [
        a.name for n in imports if isinstance(n, ast.Import) for a in n.names]
    assert not any(s.split('.')[0] in {'torch', 'numpy', 'cv2', 'decord', 'transformers'} for s in names)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and
                   n.func.id in {'eval', 'exec', 'compile', '__import__'} for n in ast.walk(tree))


def test_reserve_refusal(monkeypatch):
    from collections import namedtuple
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(audit.shutil, 'disk_usage', lambda _: usage(1000, 999, 1))
    with pytest.raises(ValueError, match='reserve'):
        audit.produce()


def test_source_hash_mismatch(tmp_path):
    path = tmp_path / 'file'
    path.write_text('fixture')
    with pytest.raises(ValueError, match='SHA mismatch'):
        audit.provenance.Reader(tmp_path).raw('file', '0' * 64)


def test_actual_constructor_source():
    source = (SCRIPTS / 'p5_jepa_strict_metadata_20261027_v3.py').read_text()
    assert audit.constructor_contract(source)['predictor_feature_width'] == 400


@pytest.mark.parametrize('old,new', [
    ('num_heads=16', 'num_heads=8'), ('tokens_per_step=1', 'tokens_per_step=2'),
    ('action_dim=10', 'action_dim=2'), ('"wall": 2', '"wall": 4'),
    ('proprio_dim=PROPRIO_DIMS[task]', 'proprio_dim=4'),
])
def test_constructor_source_mutations(old, new):
    source = (SCRIPTS / 'p5_jepa_strict_metadata_20261027_v3.py').read_text()
    assert old in source
    with pytest.raises(ValueError):
        audit.constructor_contract(source.replace(old, new))
