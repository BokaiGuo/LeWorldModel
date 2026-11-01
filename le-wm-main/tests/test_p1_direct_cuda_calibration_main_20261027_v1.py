from types import SimpleNamespace
import pytest
import numpy as np
from experiments.spectral_reachability import p1_direct_cuda_calibration_main_20261027_v1 as m


@pytest.mark.parametrize('method',m.METHODS)
def test_exact_config_budget(method):
    c=m.config(method);budget=m.screen.budget(c)
    assert budget['predict']==30 and budget['action_encoder']==30
    assert budget['jacobian']==3 and budget['vjp_directions']==576
    assert c['B1']==1 and c['B2']==3 and c['extra_line_search_trials']==0
    assert np.array_equal(c['initial_logits'],np.zeros((1,2,10)))


def test_no_unrecognized_method():
    with pytest.raises(ValueError):m.config('fake_GN')


def test_write_exclusive_small_json(tmp_path):
    path=tmp_path/'x.json';m.write(path,dict(test=True))
    with pytest.raises(FileExistsError):m.write(path,dict(test=False))


def test_no_sourcepack_no_neural_import(tmp_path):
    with pytest.raises(ValueError):m.material(tmp_path,'0'*64)


def test_space_guard_precedes_claim_and_model(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:{})
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=200*1024**2))
    with pytest.raises(ValueError):m.run(tmp_path/'prepared','fake',tmp_path/'out')
    assert not list(tmp_path.iterdir())
