"""Non-neural, non-physics regression tests for the new offline audit."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p1_numpy_toimage_callpath_audit_20261027_v1 as audit


@pytest.mark.parametrize('path,classification', [
    ('train.py', 'readers'),
    ('experiments/train_lewm_directional.py', 'readers'),
    ('experiments/latent_eigenspace_audit.py', 'readers'),
    ('experiments/audit_frozen_cross_environment_prediction.py', 'permute'),
    ('experiments/empirical_spectral_audit/run_external_action_permutation_audit.py', 'decord'),
])
def test_helper_classification(path, classification):
    assert classification in audit.classify(dict(path=path, resolved='utils.get_img_preprocessor'))


def test_alias_resolved_not_leaf_only(tmp_path):
    source = tmp_path/'x.py'
    source.write_text('from utils import get_img_preprocessor as prep\nprep("a", "b")\n')
    calls, imports = audit.callsites(source, tmp_path)
    assert calls[0]['expression'] == 'prep'
    assert calls[0]['resolved'] == 'utils.get_img_preprocessor'
    assert imports[0]['local'] == 'prep'


def test_unknown_call_fails_closed():
    with pytest.raises(ValueError, match='unclassified'):
        audit.classify(dict(path='new_online.py', resolved='stable_pretraining.data.transforms.ToImage'))


def test_cannot_conflate_online_classes():
    row = dict(path='eval.py', resolved='stable_pretraining.data.transforms.ToImage')
    with pytest.raises(ValueError, match='unclassified'):
        audit.classify(row)
    row['resolved'] = 'torchvision.transforms.v2.ToImage'
    assert 'torchvision' in audit.classify(row)


def test_scan_excludes_vendor_tests_and_symlinks(tmp_path):
    for folder in ('tests', 'external', 'results', 'experiments'):
        (tmp_path/folder).mkdir()
        (tmp_path/folder/'x.py').write_text('x=1\n')
    (tmp_path/'alias').symlink_to(tmp_path/'experiments', target_is_directory=True)
    assert [str(p.relative_to(tmp_path)) for p in audit.python_sources(tmp_path)] == ['experiments/x.py']


def test_real_tree_inventory_and_evidence():
    paths = audit.python_sources(audit.REPO)
    result = audit.inventory(paths, audit.REPO)
    assert result['scanned_python_files'] > 100
    assert len(result['calls']) >= 20
    assert all(row['classification'] for row in result['calls'])
    assert audit.evidence(paths)


def test_actual_cpu_policy_and_five_factories_non_square():
    rgb = np.arange(5*7*3, dtype=np.uint8).reshape(5,7,3)
    result = audit.runtime_checks([('asymmetric', rgb)])
    positive = result['positive_controls'][0]
    assert positive['bad_shape'] == [3,7,5]
    assert positive['correct_shape'] == [3,5,7]
    assert len(result['online']) == 5
    assert sum(row['policy_spy_calls'] for row in result['online']) == 60
    assert all(row['observed_transform_input']['shape'] == [3,5,7] for row in result['online'])
    assert all(row['policy_vs_CHW_max_abs'] == 0 for row in result['online'])
    assert len(result['sealed_feedback_preprocessing']) == 1
    assert result['sealed_feedback_preprocessing'][0]['max_abs_vs_CHW_reference'] == 0


def test_rgb_type_guard():
    with pytest.raises(ValueError, match='HWC uint8'):
        audit.runtime_checks([('wrong', np.zeros((3,5,7), dtype=np.float32))])


@pytest.mark.parametrize('lfs', [True, False])
def test_local_metadata_digest_and_tamper(tmp_path, lfs):
    content = tmp_path/'object'
    content.write_bytes(b'local object')
    digest = hashlib.sha256(content.read_bytes()).hexdigest() if lfs else hashlib.sha1(b'blob 12\0local object').hexdigest()
    meta = tmp_path/'object.metadata'
    meta.write_text('a'*40+'\n'+digest+'\n123\n')
    assert audit.prior.metadata_record(meta, content, lfs=lfs)['digest_verified']
    content.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='digest mismatch'):
        audit.prior.metadata_record(meta, content, lfs=lfs)


def test_no_reuse_existing_output(tmp_path):
    with pytest.raises(ValueError, match='exclusive NEW'):
        audit.audit(tmp_path)
