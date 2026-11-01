"""Independently verify saved render conflicts; no neural inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.reacher_d2p_context import unpack_rgb_observation, rgb_digest


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_difference(first, repeat):
    a, b = unpack_rgb_observation(first), unpack_rgb_observation(repeat)
    if a.shape != b.shape: raise ValueError("same-shape complete saved RGB pair required")
    return {"changed_components": int(np.count_nonzero(a != b)),
        "changed_coordinates": np.argwhere(a != b).tolist(),
        "max_pixel_error": int(np.max(np.abs(a.astype(np.int16)-b.astype(np.int16))))}


def verify(path):
    report = json.loads(path.read_text())
    p = report["protocol"]
    if (report.get("schema") != "reacher_d2p_render_replay.v1" or report.get("P1_admission") is not False
            or report.get("frozen_model_unchanged") is not True
            or json.loads(path.with_suffix(".protocol.json").read_text()) != p
            or p.get("all17") is not True or p.get("P1_admission") is not False):
        raise ValueError("complete sealed diagnostic report required")
    source = path.with_suffix(".implementation")/"auditor.py"
    if not source.exists(): source = Path(__file__).with_name("audit_reacher_d2p_render_replay.py")
    if digest(source) != p["implementation_sha256"]:
        raise ValueError("exact archived or current generator source required")
    for name, sha in p["bindings"].items():
        if digest(Path(name)) != sha: raise ValueError("bound source changed")
    records = report["records"]
    if [r["index"] for r in records] != list(range(17)):
        raise ValueError("all17 original ordered programs required")
    n = {"full_drive": 20, "shoulder_only": 10, "wrist_only": 10}[p["task"]]
    if report["physical_futures"] != 17*(2+4*n): raise ValueError("complete original physical probe accounting required")
    draws = report["renders"]
    if [r["render"] for r in draws] != list(range(len(draws))): raise ValueError("complete ordered render records required")
    exact, nonexact = [], []
    for r in records:
        a, b = r["first"], r["repeat"]
        x, y = np.asarray(a["observed"], float), np.asarray(b["observed"], float)
        if x.shape != (192,) or y.shape != (192,) or not np.isfinite([x, y]).all():
            raise ValueError("complete finite saved192 observations required")
        for endpoint in (a, b):
            if endpoint["render"] != draws[endpoint["render"]["render"]]:
                raise ValueError("saved endpoint does not match the render stream")
            key = hashlib.sha256(json.dumps(endpoint["snapshot"], sort_keys=True).encode()).hexdigest()
            if key != endpoint["render"]["state_sha256"]: raise ValueError("endpoint complete physical state binding changed")
        details = {"index": r["index"], "state_exact": a["snapshot"] == b["snapshot"],
            "history_rgb_exact": a["history_rgb_sha256"] == b["history_rgb_sha256"],
            "rgb_exact": a["rgb_sha256"] == b["rgb_sha256"],
            "embedding_exact": a["observed"] == b["observed"], "embedding_max_error": float(np.max(np.abs(x-y)))}
        if details["rgb_exact"] and details["embedding_exact"]: exact.append(r["index"])
        else: nonexact.append(details)
    conflicts = []
    for c in report["conflicts"]:
        if c["first"] != draws[c["first"]["render"]] or c["repeat"] != draws[c["repeat"]["render"]]:
            raise ValueError("conflict render identity changed")
        if c["first"]["state_sha256"] != c["repeat"]["state_sha256"]:
            raise ValueError("conflict is not the same complete state")
        measured = pixel_difference(c["first_rgb"], c["repeat_rgb"])
        if any(c[k] != v for k, v in measured.items()) or measured["changed_components"] < 1:
            raise ValueError("saved RGB conflict magnitude or coordinates changed")
        for endpoint in ("first", "repeat"):
            if rgb_digest(unpack_rgb_observation(c[endpoint+"_rgb"])) != c[endpoint]["rgb_sha256"]:
                raise ValueError("conflict image hash does not match render")
        derived = [k for k in c["first"]["derived"] if c["first"]["derived"][k] != c["repeat"]["derived"][k]]
        if derived != c["derived_differences"]: raise ValueError("derived-state attribution changed")
        scenes_equal = None
        if p.get("inspect_scene"):
            for endpoint in ("first", "repeat"):
                sha = hashlib.sha256(json.dumps(c[endpoint+"_scene"], sort_keys=True).encode()).hexdigest()
                if sha != c[endpoint]["scene_sha256"]: raise ValueError("scene fingerprint changed")
            scenes_equal = c["first_scene"] == c["repeat_scene"]
        conflicts.append({**measured, "derived_differences": derived, "recorded_scene_equal": scenes_equal})
    gl = report.get("gl_calls", [])
    if p.get("gl_policy") == "disable-dither" and (not gl or any(r["after_enabled"] is not False for r in gl)):
        raise ValueError("dithering intervention was not actually active")
    if p.get("gl_policy") == "warm-render" and (not gl or any(
            r.get("raw_render_passes") != 2 or r.get("physical_state_exact") is not True
            or not 0 <= r.get("warmup_retained_max_pixel_error", -1) <= 255 for r in gl)):
        raise ValueError("every fixed warm/retained render must preserve full physical state")
    return {"path": str(path.resolve()), "sha256": digest(path), "generator_source": str(source.resolve()),
        "policy": p.get("gl_policy", "original"), "inspect_scene": p.get("inspect_scene", False),
        "all_original_prefixes_exact": all(r["original_prefix_exact"] is True for r in records),
        "physical_futures": report["physical_futures"], "renders": len(draws),
        "zero_exact_programs": exact, "zero_nonexact_programs": nonexact, "conflicts": conflicts,
        "gl_calls": len(gl), "raw_render_passes_recorded": sum(r.get("raw_render_passes", 1) for r in gl) if gl else None,
        "warmup_retained_differences": sum(r.get("warmup_retained_max_pixel_error", 0) > 0 for r in gl),
        "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("new immutable analysis output required")
    result = {"schema": "reacher_d2p_render_replay_analysis.v1", "reports": [verify(p) for p in args.report],
        "implementation_sha256": digest(Path(__file__)), "neural_inference": False, "P1_admission": False,
        "scope": "Independent saved-file validation of render-layer diagnostics, not a P1 admission or an efficacy claim."}
    with args.output.open("x") as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
