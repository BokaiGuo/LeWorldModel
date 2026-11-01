"""Offline binding of the already-opened physical panel to the JEPA input bridge."""
import argparse
import hashlib
import json
import math
import numbers
import os
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STEM = "p5_crossmodel_panel_binding_20261030_codex_v1"
PANEL_DIR = ROOT / "results/agent_observed_raw_coordinate_physical_20261028_v1_MAIN01"
FILES = {
    "bridge": ("results/agent_p5_raw164_jepa_input_bridge_20261029_v1_prepare_MAIN01.json",
               "1e042999e6970261882324c8adfc45ef6303decd047e2ac7f01f88942488586f"),
    "physical": ("results/agent_observed_raw_coordinate_physical_20261028_v1_consume_MAIN01.json",
                 "8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb"),
    "join": ("results/agent_p5_fixed164_physical_model_persist_20261029_v1_MAIN01.json",
             "25cb6846263027b1ceedd39b9060c3158c6cc6a095d323b012be30728e7511c3"),
}


def require(ok, what):
    if not ok:
        raise ValueError(what)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_pinned(path, expected, cap=128 * 1024 * 1024):
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= cap, "input file")
    data = path.read_bytes()
    require(sha(data) == expected, f"input hash: {path}")
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def packed(values, fmt):
    values = list(values)
    require(all(isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))
                and math.isfinite(x) for x in values), "nonfinite numeric input")
    return b"".join(struct.pack(fmt, x) for x in values)


def shape(value, expected):
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == expected and np.isfinite(array).all(), "numeric shape/finite")
    return array


def bind_row(i, req, physical, joined, stats, panel_dir=PANEL_DIR, verify_images=True):
    slot = i // 41
    root, condition = slot // 2, ("off", "on")[slot % 2]
    keys = ("index", "root_slot", "condition", "task_id", "selection_digest")
    require(all(req[k] == physical[k] == joined[k] for k in keys), "branch identity")
    require((req["index"], req["root_slot"], req["condition"]) == (i, root, condition), "schedule")
    require(physical["status"] == "horizon_completed" and physical["initial_exact"] is True
            and physical["own_returned"] == 10 and physical["warmup_returned"] == 10
            and physical["prefix51"]["complete"] is True, "physical completeness")
    plan = physical["plan"]
    require(all(plan[k] == req[k] for k in keys), "plan identity")
    raw = shape(req["commands"], (15, 2))
    own = shape(plan["commands"], (10, 2))
    require(packed(raw[5:].flat, "<f") == packed(own.flat, "<f"), "own physical actions")
    require(packed(raw[:5].flat, "<f") == packed(([0.0, 0.10000000149011612] * 5), "<f"),
            "warmup action ABI")
    raw_sha = sha(packed(own.flat, "<f"))
    require(raw_sha == plan["raw_float32_sha256"] == joined["raw_float32_sha256"]
            and plan["producer_job_sha256"] == joined["producer_job_sha256"], "physical action binding")
    require(req["context_global_frames"] == [5, 10, 15] and req["successor_global_frame"] == 20,
            "time ABI")
    require(len(req["images"]) == 3, "three history images")
    recs = req["images"] + [req["label"]]
    frames = []
    for record, (phase, step) in zip(recs, (("warmup", 5), ("plan", 0), ("plan", 5), ("plan", 10)), strict=True):
        overlay = int(condition == "on")
        matches = [x for x in physical["trajectory"] if
                   (x["phase"], x["step"], x["overlay_index"]) == (phase, step, overlay)]
        require(len(matches) == 1, "unique physical frame")
        frame = matches[0]
        require((record["phase"], record["step"], record["condition"], record["overlay_index"]) ==
                (phase, step, condition, overlay) and frame["condition"] == condition, "frame time/condition")
        require(all(record[k] == frame[k] for k in ("file_sha256", "pixel_sha256")), "frame hash metadata")
        require(Path(frame["png"]).name == frame["png"], "flat image name")
        path = panel_dir / frame["png"]
        require(record["path"] == str(path), "physical image path")
        if verify_images:
            require(not path.is_symlink() and path.is_file() and sha(path.read_bytes()) == record["file_sha256"],
                    "physical PNG bytes")
        frames.append(frame)
    require(recs[3]["file_sha256"] == joined["plan10_png_sha256"], "common physical successor")
    for frame, prop in zip(frames[:3], req["raw_proprio"], strict=True):
        state = shape(frame["state"], (7,))
        snap = frame["snapshot"]
        agent = snap["bodies"]["agent"]
        require(packed(state, "<d") == packed(shape(snap["state"], (7,)), "<d")
                and packed(state, "<d").hex() == snap["state_float64_bits"], "state snapshot bits")
        actual = agent["position"] + agent["velocity"]
        require(packed(shape(actual, (4,)), "<d") == packed(state[[0, 1, 5, 6]], "<d")
                == packed(shape(prop, (4,)), "<d"), "observed proprio identity")
    p = shape(req["raw_proprio"], (3, 4)).astype(np.float32)
    am = shape(stats["ACTION_MEAN"], (2,)).astype(np.float32)
    ast = shape(stats["ACTION_STD"], (2,)).astype(np.float32)
    pm = shape(stats["PROPRIO_MEAN"], (4,)).astype(np.float32)
    ps = shape(stats["PROPRIO_STD"], (4,)).astype(np.float32)
    require(np.all(ast > 0) and np.all(ps > 0), "positive normalizer")
    normalized_p = (p - pm) / ps
    normalized_u = (((100. * raw).astype(np.float32) / np.float32(100.) - am) / ast).reshape(3, 10)
    require(normalized_p.tobytes() == shape(req["normalized_proprio"], (3, 4)).astype(np.float32).tobytes()
            and normalized_u.tobytes() == shape(req["packets"], (3, 10)).astype(np.float32).tobytes(),
            "normalized input bytes")
    return {**{k: req[k] for k in keys}, "raw10_sha256": raw_sha,
            "physical_producer_job_sha256": plan["producer_job_sha256"],
            "history_rgb": req["images"], "successor_rgb_identity_only": req["label"],
            "raw15_float32_sha256": sha(packed(raw.flat, "<f")),
            "normalized_packets_float32_sha256": sha(normalized_u.astype("<f4").tobytes()),
            "raw_proprio_float64_sha256": sha(packed(shape(req["raw_proprio"], (3, 4)).flat, "<d")),
            "normalized_proprio_float32_sha256": sha(normalized_p.astype("<f4").tobytes()),
            "lewm_model_task_binding": plan["model_task_binding"]}


def build(bridge, physical, join, panel_dir=PANEL_DIR, verify_images=True):
    require(bridge["schema"] == "agent_p5_raw164_jepa_input_bridge_20261029_v1.prepare"
            and bridge["status"] == "saved_inputs_prepared_not_cuda_executed", "bridge ABI")
    require(physical["schema"] == "agent_observed_raw_coordinate_physical_20261028_v1.consume"
            and physical["status"] == "observed_raw164_physical_recorded_not_admitted"
            and physical["fixed_denominator"] == 164, "physical ABI")
    a = join["analysis"]
    require(join["schema"] == "agent_p5_fixed164_physical_model_persist_20261029_v1.result"
            and a["status"] == "fixed164_saved_identity_join_only"
            and a["input_sha256"]["physical"] == FILES["physical"][1], "join ABI/physical binding")
    proto = bridge["protocol"]
    require(proto["requests"] == 164 and proto["P5_admission"] is False
            and join["P5_admission"] is False and a["P5_admission"] is False, "admission boundary")
    require(bridge["bindings"][str(ROOT / FILES["physical"][0])] == FILES["physical"][1],
            "bridge physical receipt binding")
    requests, rows, joined = bridge["requests"], physical["rows"], a["rows"]
    require(len(requests) == len(rows) == len(joined) == 164, "fixed denominator")
    bound = [bind_row(i, q, p, j, bridge["stats"], panel_dir, verify_images)
             for i, (q, p, j) in enumerate(zip(requests, rows, joined, strict=True))]
    pairs = proto["pairs"]
    require(len(pairs) == len(a["pairs"]) == 80, "fixed pairs")
    for i, (p, j) in enumerate(zip(pairs, a["pairs"], strict=True)):
        slot, local = divmod(i, 20)
        expected = {"slot": slot, "coord": local % 10, "h": (0.01, 0.001)[local // 10],
                    "minus_index": 41 * slot + 1 + 2 * local, "plus_index": 41 * slot + 2 + 2 * local}
        require(all(p[k] == j[k] == v for k, v in expected.items())
                and p["root_slot"] == slot // 2 and p["condition"] == ("off", "on")[slot % 2],
                "pair schedule")
    return {"schema": STEM, "status": "saved_input_physical_identity_bound_not_scientific_admission",
            "input_sha256": {k: v[1] for k, v in FILES.items()}, "rows": bound, "pairs": pairs,
            "branch_denominator": 164, "pair_denominator": 80, "baseline_indices": [0, 41, 82, 123],
            "known_roots": 2, "fresh_roots_certified": 0, "P1_admission": False, "P5_admission": False,
            "permissions": {"jepa": "history RGB plus agent position and velocity",
                            "lewm": "RGB-only; model task binding inherited from physical receipt"},
            "boundary": "Saved input identity, not identical model inputs, budget, model predictions, task benefit or independent confirmation. PNG file bytes checked; pixel hashes inherited from receipt, not decoded. Upstream full strict and loaded model bytes not revalidated. Successor metadata only, never decoded.",
            "new_neural_calls": 0, "new_physics_calls": 0, "training_calls": 0}


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    out = sub.add_parser("write")
    out.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--artifact", required=True, type=Path)
    verify.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "write":
        require(args.output.name.startswith(STEM + "_"), "exclusive output stem")
    sources = {k: read_pinned(ROOT / path, h) for k, (path, h) in FILES.items()}
    data = canonical(build(**sources))
    for path, h in FILES.values():
        require(sha((ROOT / path).read_bytes()) == h, "source changed during binding")
    if args.command == "write":
        with args.output.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps({"status": "written", "sha256": sha(data), "rows": 164, "pairs": 80}))
    else:
        old = read_pinned(args.artifact, args.sha256, 8 * 1024 * 1024)
        require(canonical(old) == data, "binding differs")
        print(json.dumps({"status": "same_implementation_reconstruction_verified", "sha256": sha(data)}))


if __name__ == "__main__":
    main()
