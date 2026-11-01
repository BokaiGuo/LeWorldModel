"""Configuration provenance is not training identity; all tests are offline."""
from copy import deepcopy
import sys
import pytest
from experiments.spectral_reachability import agent_p1_configuration_provenance_20261028_v1 as m


def test_same_constants_cannot_certify_training_lineage():
    result=m.build()
    assert result['value_binding']['all_float32_bits_equal'] is True
    assert result['training_identity']['verified'] is False
    assert result['training_identity']['missing_fields']
    assert result['checkpoint']['weights']['sha256']==m.WEIGHTS_SHA
    assert result['checkpoint']['config']['sha256']==m.CONFIG_SHA
    assert result['archive_statistics']['recomputed_in_this_tool'] is False
    assert result['archive_statistics']['hash_scope']=='action_field_only_not_full_archive'
    assert result['current_training_snapshot']['historical_checkpoint_training_identity'] is False


def test_ast_extraction_never_executes_source(tmp_path):
    path=tmp_path/'literal.py'
    path.write_text("raise RuntimeError('do not run')\nACTION_MEAN=torch.tensor([-.0087,.0068])\nACTION_STD=torch.tensor([.2019,.2002])\n")
    values=m.constants(path,'ACTION_MEAN','ACTION_STD')
    assert values['mean']['float32_le_hex']=='728a0ebc89d2de3b'


@pytest.mark.parametrize('text',[
    'ACTION_MEAN=read_secret()\nACTION_STD=torch.tensor([1,2])',
    'ACTION_MEAN=torch.tensor([1,2])\nACTION_MEAN=torch.tensor([3,4])\nACTION_STD=torch.tensor([1,2])',
    'ACTION_MEAN=torch.tensor([1])\nACTION_STD=torch.tensor([1,2])',
    'ACTION_MEAN=torch.tensor([1,2])\nACTION_STD=torch.tensor([0,2])'])
def test_nonliteral_ambiguous_or_invalid_normalizer_rejected(tmp_path,text):
    path=tmp_path/'source.py';path.write_text(text)
    with pytest.raises(ValueError):m.constants(path,'ACTION_MEAN','ACTION_STD')


def test_identity_does_not_follow_cache_revision_or_readme_dataset_name():
    result=m.build()
    assert result['checkpoint']['cache_metadata']['same_download_revision'] is True
    assert result['checkpoint']['declared_dataset_names']==['quentinll/lewm-pusht']
    missing=result['training_identity']['missing_fields']
    for field in ('checkpoint_to_training_run','training_dataset_immutable_revision','normalizer_to_checkpoint_binding'):
        assert field in missing


def test_full_manifest_tampering_rejected():
    original=m.build();tampered=deepcopy(original)
    tampered['training_identity']['verified']=True
    with pytest.raises(ValueError):m.validate_saved(tampered,original)


def test_manifest_is_not_alias_of_sealed_normalization():
    result=m.build()
    assert result['sealed_normalization']['normalization']['training_provenance_verified'] is False
    assert result['training_identity']['verified'] is False
    assert result['operations']==dict(network=0,GPU=0,physics=0,training=0,normalizer_fit=0,weight_deserialization=0,dataset_array_reads=0)


def test_only_own_new_output_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    results=tmp_path/'results';results.mkdir()
    path=results/(m.NAME+'_fixture.json')
    m.write(path,{'test':True})
    with pytest.raises(FileExistsError):m.write(path,{'test':False})
    with pytest.raises(ValueError):m.write(results/'other.json',{})


def test_cli_verify_recomputes_whole_manifest_without_writing(tmp_path,monkeypatch,capsys):
    import json
    path=tmp_path/'saved.json';path.write_bytes(m.canonical(m.build()))
    monkeypatch.setattr(sys,'argv',[m.NAME,'verify','--manifest',str(path),'--manifest-sha',m.sha(path)])
    m.main();out=json.loads(capsys.readouterr().out)
    assert out['status']=='saved_manifest_verified_training_lineage_still_unestablished'
    assert out['training_identity_verified'] is False


def test_source_imports_no_neural_training_or_environment_modules():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(m.__file__).read_text())
    modules=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):modules.extend(x.name for x in node.names)
        elif isinstance(node,ast.ImportFrom):modules.append(node.module)
    assert all(not x.startswith(('torch','stable_worldmodel','train','experiments','numpy','h5py')) for x in modules)
