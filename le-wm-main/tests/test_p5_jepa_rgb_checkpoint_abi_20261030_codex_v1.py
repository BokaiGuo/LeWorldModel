import copy
import json

import pytest

from experiments.spectral_reachability import p5_jepa_rgb_checkpoint_abi_20261030_codex_v1 as m


@pytest.fixture(scope="module")
def data():
    return m.load_files()


def test_pinned_static_audit(data):
    result = m.build(data)
    assert result["model_config_from_pinned_loader"]["use_proprio"] is True
    assert result["selected_saved_weight_shapes"]["predictor_blocks.0.norm1.weight"] == [400]
    assert result["proprio_encoder_saved_weight_shape"] == [16, 4, 1]
    assert result["P5_admission"] is False
    assert m.check_descriptor(result["native_h1_predictor_input_abi"])["RGB_only"] is False


@pytest.mark.parametrize("error", ["missing_proprio", "none_proprio", "wrong_dim", "dtype", "extra", "wrong_time"])
def test_native_abi_rejects_bad_descriptors(data, error):
    desc = copy.deepcopy(m.build(data)["native_h1_predictor_input_abi"])
    if error == "missing_proprio":
        del desc["proprio"]
    elif error == "none_proprio":
        desc["proprio"] = None
    elif error == "wrong_dim":
        desc["proprio"]["shape"][-1] = 0
    elif error == "dtype":
        desc["visual"]["dtype"] = "float64"
    elif error == "extra":
        desc["goal"] = {}
    else:
        desc["action"]["shape"][1] = 2
    with pytest.raises(ValueError):
        m.check_descriptor(desc)


@pytest.mark.parametrize("error", ["false_default", "false_config", "changed_width"])
def test_constructor_drift_rejected(data, error):
    loader, source = data["loader"], data["model_source"]
    if error == "false_default":
        source = source.replace("use_proprio=True,", "use_proprio=False,", 1)
    elif error == "false_config":
        loader = loader.replace("embed_dim=384,predictor_embed_dim=384,depth=6",
                                "embed_dim=384,predictor_embed_dim=384,use_proprio=False,depth=6", 1)
    else:
        loader = loader.replace('proprio_emb_dim=16,proprio_tokens=0', 'proprio_emb_dim=8,proprio_tokens=0', 1)
    with pytest.raises(ValueError):
        m.extract_config(loader, source)


@pytest.mark.parametrize("error", ["checkpoint_shape", "receipt_rgb_only", "permission", "denominator"])
def test_evidence_mismatch_fails_closed(data, error):
    changed = dict(data)
    if error == "checkpoint_shape":
        d = json.loads(changed["manifest"])
        d["expected_dynamics_schema"]["predictor"]["predictor_norm.weight"] = [384]
        changed["manifest"] = json.dumps(d)
    elif error in ("receipt_rgb_only", "denominator"):
        d = json.loads(changed["h1_receipt"])
        d["RGB_only" if error == "receipt_rgb_only" else "pairs"] = True if error == "receipt_rgb_only" else 79
        changed["h1_receipt"] = json.dumps(d)
    else:
        d = json.loads(changed["permission_contract"])
        d["arms"]["jepa_native_h1"]["observation_channels"] = ["rgb_history"]
        changed["permission_contract"] = json.dumps(d)
    with pytest.raises(ValueError):
        m.build(changed)


def test_duplicate_and_nonfinite_json():
    with pytest.raises(ValueError):
        m.strict_json('{"x": 1, "x": 2}')
    with pytest.raises(ValueError):
        m.strict_json('{"x": NaN}')
