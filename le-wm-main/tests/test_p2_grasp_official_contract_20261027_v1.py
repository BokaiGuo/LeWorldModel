"""Non-neural, offline source and adaptation contract tests."""
import io
import pytest
from experiments.spectral_reachability import p2_grasp_official_contract_20261027_v1 as m


def test_bounded_source_read():
    assert m.bounded(io.BytesIO(b'123'), 3) == b'123'
    with pytest.raises(ValueError, match='limit'):
        m.bounded(io.BytesIO(b'1234'), 3)


@pytest.mark.parametrize('url', ['https://example.org/code',
    'https://raw.githubusercontent.com/other/grasp/main/a.py',
    'https://raw.githubusercontent.com/michael-psenka/grasp/main/a.py',
    'https://api.github.com/repos/michael-psenka/grasp/zipball',
    'https://raw.githubusercontent.com/michael-psenka/grasp/' + 'a'*40 + '/weights.pt'])
def test_no_mutable_code_refs_or_model_downloads(url):
    with pytest.raises(ValueError):
        m.check_url(url)


def test_pinned_raw_source_url_allowed():
    assert m.check_url('https://raw.githubusercontent.com/michael-psenka/grasp/' + m.COMMIT + '/grasp_pseudocode.py')


@pytest.fixture(scope='module')
def contract():
    return m.build_contract(m.ROOT / 'results' / m.NAME)


def test_three_official_repos_bound_without_paper_commit_imputation(contract):
    assert len(contract['repos']) == 3
    assert contract['paper_results_code_commit'] == 'unknown'
    assert contract['paper']['version'] == '2602.00475v1'
    assert contract['local_reproduction'] is False


def test_every_algorithm_claim_has_exact_fixed_source_lines(contract):
    for claim in contract['algorithm_mapping']:
        assert claim['anchors']
        for anchor in claim['anchors']:
            assert anchor['start_line_1based'] <= anchor['end_line_1based']
            assert len(anchor['source_sha256']) == 64
            assert len(anchor['excerpt_sha256']) == 64
            assert '#L' in anchor['url']


def test_no_false_native_full_repo_equivalence(contract):
    comparisons = contract['native_source_comparison']
    assert comparisons['app/plan_common/models/AdaLN_vit.py']['exact_git_blob_match'] is True
    assert comparisons['app/plan_common/models/prop_embedding.py']['exact_git_blob_match'] is False
    assert comparisons['app/plan_common/datasets/pusht_dset.py']['exact_git_blob_match'] is False
    assert contract['same_as_existing_real_H3_contract'] is False


def test_contract_distinguishes_pseudocode_base_and_adapters(contract):
    text = str(contract)
    assert 'pre-update' in text
    assert 'cylinder' in text
    assert 'global RNG' in text
    assert contract['proposed_core']['official_equivalence'] is False


def test_prepare_is_exclusive_and_model_free(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('network not allowed during prepare')
    monkeypatch.setattr(m, 'urlopen', forbidden)
    output = tmp_path / 'prepare'
    receipt=m.prepare(m.ROOT / 'results' / m.NAME, output)
    assert receipt['status'] == 'prepared_source_contract_not_executed'
    assert receipt['new_model_transitions'] == 0
    assert receipt['new_encodings'] == 0
    assert receipt['roots_bound'] == 0
    assert receipt['cuda_launch_authorized'] is False
    assert (output/'contract.json').exists()
    with pytest.raises(FileExistsError):
        m.prepare(m.ROOT / 'results' / m.NAME, output)


def test_pinned_source_corruption_is_rejected(tmp_path):
    (tmp_path/'sources').mkdir()
    (tmp_path/'sources'/'planning__grasp.py.source.txt').write_text('not official')
    with pytest.raises(ValueError,match='SHA'):
        m.read_source(tmp_path,'planning/grasp.py')
