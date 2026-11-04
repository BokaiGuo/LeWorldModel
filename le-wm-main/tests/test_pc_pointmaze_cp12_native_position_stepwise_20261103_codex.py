import json

import numpy as np
import pytest

from experiments import pc_pointmaze_cp12_native_position_stepwise_20261103_codex as audit
from experiments import pc_pointmaze_cp12_stepwise_backend_worker_20261103_codex as worker


def test_sealed_request_roster_and_actions():
    _,_,names = audit.prior.files()
    sources, actions, images = audit.build_inputs(names)
    assert sources.shape == (64,4) and actions.shape == (64,17,2) and images.shape == (64,18)
    assert np.array_equal(sources[0::2],sources[1::2])
    assert np.array_equal(actions[0::2,:12],actions[1::2,:12])
    worker.validate(sources,actions)


def test_worker_rejects_bad_contract():
    src, acts = np.zeros((64,4)), np.zeros((64,17,2))
    worker.validate(src,acts)
    acts[2,3,0] = 1.01
    with pytest.raises(ValueError):
        worker.validate(src,acts)
    with pytest.raises(ValueError):
        worker.validate(src[:-1],acts)


def test_no_overwrite(monkeypatch,tmp_path):
    existing = tmp_path/"report.json"
    existing.touch()
    monkeypatch.setattr(audit,"OUTPUT",existing)
    with pytest.raises(FileExistsError):
        audit.main()


def test_saved_stepwise_arrays_and_original_receipts():
    if not audit.OUTPUT.exists():
        pytest.skip("post-outcome audit not run")
    report = json.loads(audit.OUTPUT.read_text())
    assert report["arrays_sha256"] == audit.prior.sha(audit.ARRAYS)
    assert report["journal_sha256"] == audit.prior.sha(audit.JOURNAL)
    for path,digest in audit.EXPECTED.items():
        assert audit.prior.sha(path) == digest
    _,_,names = audit.prior.files()
    _,_,expected_rgb = audit.build_inputs(names)
    with np.load(audit.ARRAYS,allow_pickle=False) as data, \
         np.load(audit.prior.EVALUATION.with_suffix(".npz"),allow_pickle=False) as old:
        native,position = data["native_states"],data["position_states"]
        assert native.shape == position.shape == (64,18,4)
        np.testing.assert_array_equal(data["delta"],native-position)
        np.testing.assert_array_equal(position[:,-1],old["endpoints"])
        np.testing.assert_array_equal(data["native_rgb_sha256"],expected_rgb)
        np.testing.assert_array_equal(data["expected_rgb_sha256"],expected_rgb)
        np.testing.assert_allclose(data["native_times"],np.broadcast_to(np.arange(18)*.05,(64,18)),rtol=0,atol=1e-12)
        np.testing.assert_allclose(data["position_times"],data["native_times"],rtol=0,atol=1e-12)
        pd = np.linalg.norm((native-position)[:,:,:2],axis=2)
        vd = np.linalg.norm((native-position)[:,:,2:],axis=2)
        np.testing.assert_array_equal(data["position_l2"],pd)
        np.testing.assert_array_equal(data["velocity_l2"],vd)
        assert report["exact_state_count"] == int(np.all(native==position,axis=2).sum()) == 1152
        assert report["rgb_exact_count"] == 1152
        assert report["max_position_l2"] == pd.max()
        assert report["max_velocity_l2"] == vd.max()
        for i,row in enumerate(report["rows"]):
            assert row["name"] == names[i//2]
            assert row["arm"] == ("one","three")[i%2]
            assert row["final_commands"] == 17
            assert row["max_position_l2"] == pd[i].max()
            assert row["max_velocity_l2"] == vd[i].max()
            assert row["native_success"] == (np.linalg.norm(native[i,-1,:2]-[1.8,2.8]) < .5)
            assert row["position_success"] == (np.linalg.norm(position[i,-1,:2]-[1.8,2.8]) < .5)
