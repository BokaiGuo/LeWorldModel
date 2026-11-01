"""Static, pinned JEPA PushT checkpoint input-ABI audit; never loads a model."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEM = "p5_jepa_rgb_checkpoint_abi_20261030_codex_v1"
FILES = {
    "loader": ("experiments/spectral_reachability/p5_jepa_native_action_response_20261027_v8.py",
               "fc915ddee2e53ed58366994684fe2190da5eaaeb8323ad608ffe3292534c936a"),
    "model_source": ("external/jepa-wms/app/plan_common/models/AdaLN_vit.py",
                     "77e20661849a33199c581b4ab47e2d0417ef2e81571e4998a30cc242b68abe97"),
    "manifest": ("results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json",
                 "0588f825c29b884350e97ed0dc47d8872bb539d0ca7a7faf045190eee5f3340f"),
    "checkpoint": ("data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar",
                   "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"),
    "h1_receipt": ("results/agent_p5_raw164_jepa_cuda_20261029_v1_consume_MAIN01.json",
                   "4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759"),
    "permission_contract": ("results/p5_observation_permission_contract_20261030_codex_v1_MAIN01.json",
                            "73c88ad0ff522c6bf21d729895c0d9985a5ec167c2ea6bc82061ee82e54702a8"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_files():
    data = {}
    for key, (relative, expected) in FILES.items():
        path = ROOT / relative
        require(digest_file(path) == expected, f"file drift: {key}")
        if key == "checkpoint":
            continue
        require(path.stat().st_size <= 8 * 1024 * 1024, f"file too large: {key}")
        data[key] = path.read_text()
    return data


def strict_json(text):
    def unique(items):
        result = {}
        for k, v in items:
            require(k not in result, "duplicate JSON key")
            result[k] = v
        return result
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def extract_config(loader, model):
    loader_ast = ast.parse(loader)
    load = next(n for n in loader_ast.body if isinstance(n, ast.FunctionDef) and n.name == "load_cuda")
    calls = [n for n in ast.walk(load) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "vit_predictor_AdaLN"]
    require(len(calls) == 1, "unique native predictor constructor")
    config = {kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords}
    expected = {"embed_dim": 384, "predictor_embed_dim": 384, "depth": 6, "num_heads": 16,
                "action_dim": 10, "proprio_dim": 4, "proprio_encoder_inpred": False,
                "action_encoder_inpred": True, "proprio_encoding": "feature",
                "proprio_emb_dim": 16, "proprio_tokens": 0}
    require(all(config.get(k) == v for k, v in expected.items()) and "use_proprio" not in config,
            "native loader constructor mismatch")
    tree = ast.parse(model)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VisionTransformerAdaLN")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):], init.args.defaults, strict=True))
    require(ast.literal_eval(defaults["use_proprio"]) is True, "native default use_proprio")
    return {**expected, "use_proprio": True, "use_proprio_source": "constructor_default"}


def check_descriptor(descriptor):
    """Validate a declared native H1 descriptor; never substitutes for tensor hooks."""
    require(type(descriptor) is dict and set(descriptor) == {"visual", "action", "proprio"}, "input fields")
    for key, shape in (("visual", [1, 3, 1, 16, 16, 384]),
                       ("action", [1, 3, 10]), ("proprio", [1, 3, 256, 16])):
        require(descriptor[key] == {"shape": shape, "dtype": "float32"}, f"native {key} ABI")
    return {"status": "descriptor_only_not_runtime_evidence", "RGB_only": False}


def build(data):
    config = extract_config(data["loader"], data["model_source"])
    manifest, receipt = strict_json(data["manifest"]), strict_json(data["h1_receipt"])
    contract = strict_json(data["permission_contract"])
    require(manifest["schema"] == "p5_jepa_native_action_response_20261027_v8"
            and manifest["status"] == "prepared_non_neural", "manifest ABI")
    require(manifest["seals"][str(ROOT / FILES["checkpoint"][0])] == FILES["checkpoint"][1],
            "checkpoint manifest identity")
    pred, prop = manifest["expected_dynamics_schema"]["predictor"], manifest["expected_dynamics_schema"]["proprio_encoder"]
    shapes = {"action_encoder.weight": [400, 10], "predictor_blocks.0.norm1.weight": [400],
              "predictor_blocks.0.adaLN_modulation.1.weight": [2400, 400],
              "predictor_norm.weight": [400], "predictor_embed.weight": [384, 384],
              "predictor_proj.weight": [384, 384]}
    require(all(pred[k] == shape for k, shape in shapes.items())
            and prop["patch_embed.weight"] == [16, 4, 1], "saved strict schema mismatch")
    require(receipt["status"] == "all164_factual_H1_and80_pairs_verified" and receipt["RGB_only"] is False
            and receipt["requests"] == 164 and receipt["pairs"] == 80 and receipt["P5_admission"] is False,
            "existing H1 receipt")
    require(contract["arms"]["jepa_native_h1"]["observation_channels"] ==
            ["rgb_history", "agent_position", "agent_velocity"]
            and contract["P5_admission"] is False, "permission contract")
    abi = {"visual": {"shape": [1, 3, 1, 16, 16, 384], "dtype": "float32"},
           "action": {"shape": [1, 3, 10], "dtype": "float32"},
           "proprio": {"shape": [1, 3, 256, 16], "dtype": "float32"}}
    check_descriptor(abi)
    return {
        "schema": STEM, "status": "static_native_abi_and_rgb_only_gap_recorded",
        "pinned_sha256": {k: x[1] for k, x in FILES.items()},
        "model_config_from_pinned_loader": config, "selected_saved_weight_shapes": shapes,
        "proprio_encoder_saved_weight_shape": prop["patch_embed.weight"],
        "native_h1_predictor_input_abi": abi,
        "native_h1_output_abi_inherited": {"visual": [1, 3, 256, 384], "middle": None,
                                           "proprio": [1, 3, 256, 16]},
        "analysis": {
            "none_argument": "Signature accepts None, but with feature-proprio enabled the no-proprio branch leaves visual width 384 while block norm1 requires 400; statically incompatible, no forward executed.",
            "use_proprio_false": "Changes internal width 400 to 384 including action encoder and block weights; pinned 400-wide checkpoint schema cannot strict-load into that architecture.",
            "constant_feature": "A separately specified constant 16-wide input could be an intervention or fixed padding, but its validity, training support and actual execution are unverified. Do not call it native RGB-only or equate it with native JEPA.",
            "same_permission_status": "blocked: no verified RGB-only native checkpoint interface or matched-permission runtime receipt in this fixed evidence set; goal/history alignment also unresolved.",
        },
        "boundary": "Pinned bytes and saved metadata only; no deserialization, forward, GPU or checkpoint strict-load in this audit; H1 receipt is inherited, not re-run. Descriptor validation checks declarations, not live tensors. Not proof all alternative software or checkpoints are impossible.",
        "new_neural_calls": 0, "new_physics_calls": 0, "training_calls": 0,
        "P1_admission": False, "P5_admission": False,
    }


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--artifact", type=Path, required=True)
    v.add_argument("--sha256", required=True)
    args = parser.parse_args()
    data = canonical(build(load_files()))
    for key, (path, expected) in FILES.items():
        require(digest_file(ROOT / path) == expected, f"input drift: {key}")
    if args.command == "write":
        require(args.output.name.startswith(STEM + "_"), "exclusive output stem")
        with args.output.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps({"status": "written", "sha256": hashlib.sha256(data).hexdigest()}))
    else:
        require(digest_file(args.artifact) == args.sha256, "artifact SHA mismatch")
        require(canonical(strict_json(args.artifact.read_text())) == data, "artifact mismatch")
        print(json.dumps({"status": "same_implementation_reconstruction_verified",
                          "sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    main()
