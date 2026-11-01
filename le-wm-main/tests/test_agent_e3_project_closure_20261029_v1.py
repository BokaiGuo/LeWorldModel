import pytest
from experiments.spectral_reachability import agent_e3_project_closure_20261029_v1 as c


@pytest.fixture
def tree(tmp_path):
    for d in c.DIRS:(tmp_path/d).mkdir()
    (tmp_path/'top.py').write_text('top')
    (tmp_path/'experiments'/'pkg').mkdir()
    (tmp_path/'experiments'/'pkg'/'__init__.py').write_text('init')
    (tmp_path/'external'/'native.so').write_bytes(b'elf')
    (tmp_path/'tests'/'__pycache__').mkdir()
    (tmp_path/'tests'/'__pycache__'/'dynamic.py').write_text('ignored')
    return tmp_path


def test_independent_scans_and_exclusions(tree):
    a=c.walk(tree);assert a==c.scan(tree) and len(a)==3
    assert all('__pycache__' not in p for p in a)
    assert c.stable(tree)==c.hash_files(a)


def test_symlink_source_and_directory_rejected(tree):
    link=tree/'scripts'/'alias.py';link.symlink_to(tree/'top.py')
    for scan in (c.walk,c.scan):
        with pytest.raises(ValueError,match='source'):scan(tree)
    link.unlink();(tree/'scripts'/'nested').symlink_to(tree/'experiments',target_is_directory=True)
    for scan in (c.walk,c.scan):
        with pytest.raises(ValueError,match='symlink'):scan(tree)


def test_changed_bytes_during_collection(tree,monkeypatch):
    original=c.hash_files; calls=[]
    def changing(paths):
        result=original(paths);calls.append(1)
        if len(calls)==1:(tree/'top.py').write_text('changed')
        return result
    monkeypatch.setattr(c,'hash_files',changing)
    with pytest.raises(ValueError,match='bytes changed'):c.stable(tree)


def test_added_source_during_collection(tree,monkeypatch):
    original=c.hash_files
    def changing(paths):
        result=original(paths);(tree/'scripts'/'new.py').write_text('new');return result
    monkeypatch.setattr(c,'hash_files',changing)
    with pytest.raises(ValueError,match='set changed'):c.stable(tree)


@pytest.mark.parametrize('base,extra', [({'/a':'bad'},{}), ({},{'/a':'bad'}), ({'/a':'ok'},{'/a':'bad'})])
def test_every_existing_pin_collision_rejected(base,extra):
    with pytest.raises(ValueError,match='collision'):
        c.reconstruct({'prepared':{'bindings_sha256':base}}, {'additional_sources':extra}, {'/a':'ok'})


def test_merge_preserves_prior_and_base():
    out=c.reconstruct({'prepared':{'bindings_sha256':{'/base':'a'}}},
        {'additional_sources':{'/old':'b'}},{'/base':'a','/new':'c'})
    assert out['additional_sources']=={'/old':'b','/new':'c'}


def test_original_census_refuses_unsealed(tmp_path):
    from types import SimpleNamespace
    p=tmp_path/'late.py';p.write_text('x')
    census=c.launch.module_census
    with pytest.raises(ValueError,match='not presealed'):
        census.snapshot({}, {'late':SimpleNamespace(__file__=str(p))},root=tmp_path)
