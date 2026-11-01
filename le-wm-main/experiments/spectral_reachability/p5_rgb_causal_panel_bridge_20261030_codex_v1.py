"""CPU-only checkpoint strict-load and fixed saved physical RGB input preparation."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
STEM = "p5_rgb_causal_panel_bridge_20261030_codex_v1"
PHYSICAL_DIR = ROOT / "results/agent_observed_raw_coordinate_physical_20261028_v1_MAIN01"
FILES = {
    "checkpoint": ("results/causal_visual_pusht_phaseutility_formaltrain30_20261009_seed20261009/causal_visual_world_model.pt", "ca9b51ed2b98923bebc59941ba24612bc5ab8f59d7afdf6239334b550350d0dd"),
    "training_report": ("results/causal_visual_pusht_phaseutility_formaltrain30_20261009_seed20261009/causal_visual_world_model.json", "9cb27f7824460e28f1286aa468d5784b552483eebd127c8767a538464623264a"),
    "model_source": ("experiments/run_causal_visual_pusht_world_model.py", "261f609e91da9110f8ba5254033529527314d6bcf38e058c3a237c920d3c4190"),
    "loader_source": ("experiments/evaluate_causal_visual_checkpoint.py", "f73a3830be6c75f6ca7f81130107bf54245d6057ef33d0885fec8a80b2c2eeef"),
    "physical_receipt": ("results/agent_observed_raw_coordinate_physical_20261028_v1_consume_MAIN01.json", "8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb"),
    "panel": ("results/p5_crossmodel_panel_binding_20261030_codex_v1_MAIN01.json", "80ed10d3b8f623d361cc25e2764d390bc9cc758178c89e7467e99da3d785ef98"),
}


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_fixed(name, max_bytes=128 * 1024 * 1024):
    relative, expected = FILES[name]
    path = ROOT / relative
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= max_bytes, f"file {name}")
    data = path.read_bytes()
    require(digest(data) == expected, f"SHA {name}")
    return data


def read_json(data):
    def unique(items):
        out = {}
        for k, v in items:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def cpu_load(checkpoint):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU isolation requires CUDA_VISIBLE_DEVICES empty")
    import torch
    from experiments.run_causal_visual_pusht_world_model import Normalization, VisualCausalWorldModel
    from experiments.evaluate_causal_visual_checkpoint import checkpoint_adapter_architecture, checkpoint_utility_mode

    payload = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
    require(set(payload) == {"model", "normalization", "args"}, "checkpoint schema")
    args = payload["args"]
    require(args["latent_dim"] == 96 and args["vision_encoder"] == "mask_moments"
            and args["temporal_context"] is True and args["action_repeat"] == 1
            and args["image_size"] == 64, "checkpoint input configuration")
    feature_flags = ("structured_readout", "contact_frame_dynamics", "mode_conditioned_dynamics",
                     "current_only_readout", "torque_features", "temporal_dynamics",
                     "previous_action_dynamics", "explicit_angle_effect", "explicit_angle_readout",
                     "explicit_goal_angle_readout", "rich_temporal_dynamics", "invariant_angle_readout",
                     "contact_interface_dynamics", "angle_readout_current_only")
    require(all(args.get(k, False) is False for k in feature_flags), "unknown architecture flags")
    mode, architecture = checkpoint_utility_mode(payload), checkpoint_adapter_architecture(payload)
    require((mode, architecture) == ("phase", "legacy"), "checkpoint architecture")
    norm = Normalization(**payload["normalization"])
    require(len(norm.mean) == len(norm.std) == 6 and all(math.isfinite(x) for x in norm.mean + norm.std)
            and all(x > 0 for x in norm.std), "checkpoint normalization")
    with torch.device("cpu"):
        model = VisualCausalWorldModel(96, "mask_moments", True, mode, architecture)
        status = model.load_state_dict(payload["model"], strict=True)
        model.set_state_normalization(*norm.tensors("cpu"))
        model.eval().requires_grad_(False)
    require(not status.missing_keys and not status.unexpected_keys
            and all(p.device.type == "cpu" and not p.requires_grad for p in model.parameters())
            and not model.training, "CPU frozen strict load")
    return {"device": "cpu", "weights_only": True, "strict_load": True, "eval": True,
            "requires_grad": False, "state_dict_entries": len(payload["model"]),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "utility_mode": mode, "architecture": architecture, "normalization": payload["normalization"],
            "torch_version": torch.__version__}


def input_image(frame, panel_dir=PHYSICAL_DIR, return_array=False):
    name = frame["png"]
    require(Path(name).name == name, "flat image name")
    path = panel_dir / name
    require(path.is_file() and not path.is_symlink(), "history image")
    raw = path.read_bytes()
    require(digest(raw) == frame["file_sha256"], "history PNG SHA")
    with Image.open(io.BytesIO(raw)) as image:
        require(image.mode == "RGB" and image.size == (224, 224), "RGB 224 image")
        pixels = np.asarray(image, dtype=np.uint8).copy()
    require(digest(pixels.tobytes()) == frame["pixel_sha256"], "pixel SHA")
    chw = np.moveaxis(cv2.resize(pixels, (64, 64), interpolation=cv2.INTER_AREA), -1, 0).astype(np.uint8)
    f32 = chw.astype(np.float32) / np.float32(255)
    require(chw.shape == (3, 64, 64) and np.isfinite(f32).all(), "preprocess ABI")
    record = {"path": str(path), "file_sha256": frame["file_sha256"], "pixel_sha256": frame["pixel_sha256"],
            "resized_chw_uint8_sha256": digest(chw.tobytes(order="C")),
            "model_input_chw_float32_sha256": digest(f32.astype("<f4").tobytes(order="C"))}
    return (record, f32) if return_array else record


def materialize_inputs(row):
    """Only history RGB and action, never returns any evaluation target."""
    inputs = []
    for key in ("previous_rgb", "current_rgb"):
        expected = row[key]
        path = Path(expected["path"])
        require(path.parent == PHYSICAL_DIR and path.name not in ("", ".", ".."), "input directory")
        frame = {"png": path.name, "file_sha256": expected["file_sha256"],
                 "pixel_sha256": expected["pixel_sha256"]}
        actual, arr = input_image(frame, return_array=True)
        require(actual == expected, "prepared image identity")
        inputs.append(arr[None, ...])
    action = np.asarray(row["action_raw_float32"], dtype="<f4")
    require(action.shape == (2,) and np.isfinite(action).all()
            and digest(action.tobytes()) == row["action_raw_float32_sha256"], "prepared action identity")
    return {"previous_image": inputs[0], "image": inputs[1], "action": action[None, :]}


def bind_row(index, physical, panel_row, panel_dir=PHYSICAL_DIR):
    slot = index // 41
    root, condition = slot // 2, ("off", "on")[slot % 2]
    keys = ("index", "root_slot", "condition", "task_id", "selection_digest")
    require(all(physical[k] == panel_row[k] for k in keys) and
            (physical["index"], physical["root_slot"], physical["condition"]) == (index, root, condition),
            "identity schedule")
    require(physical["status"] == "horizon_completed" and physical["own_returned"] == 10
            and physical["initial_exact"] is True, "physical complete")
    plan = physical["plan"]
    require(all(plan[k] == physical[k] for k in keys), "plan identity")
    commands = np.asarray(plan["commands"], dtype="<f4")
    require(commands.shape == (10, 2) and np.isfinite(commands).all()
            and digest(commands.tobytes()) == panel_row["raw10_sha256"] == plan["raw_float32_sha256"],
            "raw10 identity")
    overlay = int(condition == "on")
    selected = []
    for step in (4, 5, 6):
        matches = [x for x in physical["trajectory"] if
                   (x["phase"], x["step"], x["overlay_index"], x["condition"]) ==
                   ("plan", step, overlay, condition)]
        require(len(matches) == 1, "exact physical frame")
        selected.append(matches[0])
    require(selected[1]["file_sha256"] == panel_row["history_rgb"][2]["file_sha256"],
            "current frame shared with JEPA H1")
    previous, current = (input_image(f, panel_dir) for f in selected[:2])
    future = selected[2]
    # plan step6 is a future evaluation target: its bytes are not opened here.
    return {**{k: physical[k] for k in keys}, "previous_global_step": 14, "current_global_step": 15,
            "target_global_step": 16, "previous_rgb": previous, "current_rgb": current,
            "action_raw_float32": commands[5].tolist(), "action_raw_float32_sha256": digest(commands[5].tobytes()),
            "target_identity_only": {"file_sha256": future["file_sha256"],
                                     "pixel_sha256": future["pixel_sha256"], "plan_step": 6}}


def prepare(data, model_receipt):
    panel, physical, report = (read_json(data[k]) for k in ("panel", "physical_receipt", "training_report"))
    require(panel["status"] == "saved_input_physical_identity_bound_not_scientific_admission"
            and panel["branch_denominator"] == 164 and panel["pair_denominator"] == 80, "panel ABI")
    require(physical["status"] == "observed_raw164_physical_recorded_not_admitted"
            and physical["fixed_denominator"] == 164, "physical receipt")
    protocol = report["protocol"]
    require(protocol["policy_runtime_state_access"] is False and protocol["temporal_context"] is True
            and protocol["vision_encoder"] == "mask_moments" and protocol["image_size"] == 64
            and protocol["action_repeat"] == 1, "training report ABI")
    require(len(physical["rows"]) == len(panel["rows"]) == 164 and len(panel["pairs"]) == 80, "fixed denominator")
    rows = [bind_row(i, p, r) for i, (p, r) in enumerate(zip(physical["rows"], panel["rows"], strict=True))]
    for i, p in enumerate(panel["pairs"]):
        slot, local = divmod(i, 20)
        require((p["slot"], p["coord"], p["h"], p["minus_index"], p["plus_index"]) ==
                (slot, local % 10, (0.01, 0.001)[local // 10], 41 * slot + 1 + 2 * local,
                 41 * slot + 2 + 2 * local), "pair schedule")
    return {"schema": STEM, "status": "cpu_strict_load_and_input_only_prepared_not_prediction",
            "input_sha256": {k: v[1] for k, v in FILES.items()}, "model": model_receipt, "rows": rows,
            "pairs": panel["pairs"], "branch_denominator": 164, "pair_denominator": 80,
            "baseline_indices": [0, 41, 82, 123], "known_roots": 2, "certified_fresh_roots": 0,
            "image_abi": "RGB224 uint8 -> OpenCV INTER_AREA 64 -> CHW uint8 -> float32 / 255",
            "input_abi": "previous RGB14, current RGB15, raw action6; no goal, state or target image to forward",
            "target": "plan6/global16 label metadata only; not opened",
            "provenance": "locally intervention-supervised PushT model; physical state used in prior training labels and evaluator, not runtime observation",
            "boundary": "single raw action versus JEPA 5-action packet; not matched horizon, prediction or control benefit; known panel already observed; no P5 admission. Input SHA is not proof of actual future model inputs.",
            "opencv_version": cv2.__version__, "new_model_forward_calls": 0, "new_physics_calls": 0,
            "new_training_calls": 0, "gpu_calls": 0, "P1_admission": False, "P5_admission": False}


def canonical(data):
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write")
    w.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--artifact", type=Path, required=True)
    v.add_argument("--sha256", required=True)
    args = parser.parse_args()
    data = {k: read_fixed(k) for k in FILES}
    prepared = canonical(prepare(data, cpu_load(data["checkpoint"])))
    for k in FILES:
        read_fixed(k)
    if args.command == "write":
        require(args.output.name.startswith(STEM + "_"), "exclusive output stem")
        with args.output.open("xb") as f:
            f.write(prepared)
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps({"status": "prepared", "sha256": digest(prepared), "branches": 164, "pairs": 80}))
    else:
        require(args.artifact.is_file() and not args.artifact.is_symlink(), "artifact file")
        existing = args.artifact.read_bytes()
        require(digest(existing) == args.sha256 and canonical(read_json(existing)) == prepared, "verify mismatch")
        print(json.dumps({"status": "same_implementation_reconstruction_verified", "sha256": digest(prepared)}))


if __name__ == "__main__":
    main()
