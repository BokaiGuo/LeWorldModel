"""Bind new CUDA-planned suffixes and execute the fixed physical roster."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_diverse_adam_cp5_20261030_main as planner
from . import p1_diverse_physical_runner_20261030_main as previous

NAME = "p1_diverse_adam_physical_20261030_main"
ROOT, runner = previous.ROOT, previous.runner
PLAN = ROOT / "results/p1_diverse_adam_cp5_20261030_main_RUN01.json"
PLAN_SHA = "f4b297e53dd82fac41089eb2751cd7636b6c9500710d3af02af2630623e6f9aa"
PLAN_PREP = ROOT / "results/p1_diverse_adam_cp5_20261030_main_prepare_RUN01.json"
PLAN_PREP_SHA = "cee64cf13f17197b81e1a95a659f76d1cbf880b58638dbdb5adff725e37100c9"
JOIN = ROOT / "results/p1_diverse_adam_physical_source_20261030_main_RUN01.json"
PROTOCOL = ROOT.parent / "docs/D2P_P1_DIVERSE_ADAM_PHYSICAL_PROTOCOL_20261030_MAIN.md"
ORIGINAL_JOIN_SHA = "bbf2bdb0fcb02308e7a98615c8a526c612833e9afd83a5c64cdb42259256a06f"
need, sha, native = runner.require, runner.sha, runner.native


def source():
    need(sha(PLAN) == PLAN_SHA and sha(PLAN_PREP) == PLAN_PREP_SHA and
         sha(previous.JOIN) == ORIGINAL_JOIN_SHA, "input SHA")
    plan = native.read(PLAN, cap=8*1024**2)
    prep = native.read(PLAN_PREP, cap=64*1024**2)
    need(plan["prepare"] == str(PLAN_PREP) and plan["prepare_sha256"] == PLAN_PREP_SHA
         and plan["model_before"] == plan["model_after"] == planner.base.MODEL_HASH
         and plan["calls"] == dict(encoder=3, projector=3, action_encoder=504, predictor=504, backward=240)
         and plan["physical_execution"] is False and plan["P1_admission"] is False
         and all(sha(f) == d for f, d in prep["bindings"].items()), "plan source closure")
    joined = native.read(previous.JOIN, cap=32*1024**2)
    joined = copy.deepcopy(joined)
    need(len(plan["rows"]) == len(prep["rows"]) == 3 and len(joined["rows"]) == 16, "roster")
    for source_index, bundle, selected in zip((5, 13, 15), prep["rows"], plan["rows"], strict=True):
        row = joined["rows"][source_index]
        need(row["role"] == "model_selected" and selected["index"] == bundle["index"]
             and selected["logical_pair"] == bundle["logical_pair"] and selected["role"] == bundle["role"]
             and selected["prefix_raw_f32_hex"] == bundle["prefix_raw_f32_hex"] == row["raw15_float32_hex"][:80]
             and selected["cp5_pixel_sha256"] == bundle["cp5_pixel_sha256"]
             and selected["selected_index"] == planner.prior.pick(selected["candidate_scores"]), "planned identity")
        initial = planner.prior.initial_actions(bundle["logical_pair"])
        need(hashlib.sha256(initial.tobytes()).hexdigest() == selected["initial_f32_sha256"], "initialization")
        raw = np.asarray(selected["selected_actions"], dtype="<f4")
        need(raw.shape == (2, 5, 2) and np.isfinite(raw).all() and (np.abs(raw) <= 1).all()
             and hashlib.sha256(raw.tobytes()).hexdigest() == selected["selected_action_sha256"], "suffix bytes")
        row["raw15_float32_hex"] = row["raw15_float32_hex"][:80] + raw.tobytes().hex()
    joined.update(diverse_original_source_sha256=ORIGINAL_JOIN_SHA, adam_plan_sha256=PLAN_SHA,
                  adam_plan_prepare_sha256=PLAN_PREP_SHA, replaced_rows=[5, 13, 15],
                  replay_scope="original CUDA replay covers unchanged prefixes, not newly planned suffixes",
                  limitation="Known roots and post-outcome exploration; no independent confirmation or P1 admission.")
    return joined


def configure(join_sha):
    native.actual_hash(join_sha)
    runner.NAME, runner.JOIN, runner.JOIN_SHA = NAME, JOIN, join_sha


def material(join_sha):
    configure(join_sha)
    p, data = runner.material()
    joined = native.read(JOIN, cap=32*1024**2)
    need(joined == source(), "source reconstruction")
    prep = native.read(PLAN_PREP, cap=64*1024**2)
    files = dict(prep["bindings"])
    for path in (Path(__file__).resolve(), PROTOCOL, PLAN, PLAN_PREP, previous.JOIN,
                 Path(planner.__file__).resolve(), Path(previous.__file__).resolve(),
                 Path(previous.consumer.__file__).resolve()):
        files[str(path)] = sha(path)
    for path, digest in files.items():
        need(sha(path) == digest and (path not in p["bindings"] or p["bindings"][path] == digest), "source drift")
        p["bindings"][path] = digest
    p["adam_plan_sha256"] = PLAN_SHA
    return p, data


def verify(path, digest, join_sha):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path) == digest, "prepare SHA")
    saved = native.read(path, cap=64*1024**2)
    current, data = material(join_sha)
    need(saved == current and sha(path) == digest, "prepare drift")
    return saved, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("source", "prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--join-sha")
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--report-sha")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    path = Path(args.output)
    if args.command == "source":
        need(path == JOIN and not path.exists(), "exclusive source output")
        result = dict(sha256=planner.base.write(path, source()), physical_authorized=False)
    elif args.command == "prepare":
        need(path.is_absolute() and path.parent == ROOT / "results" and
             path.name.startswith(NAME+"_prepare_") and not path.exists(), "exclusive prepare")
        p, _ = material(args.join_sha)
        result = dict(sha256=planner.base.write(path, p), physical_authorized=False)
    elif args.command == "run":
        need(args.authorize_physics, "physics authorization")
        p, data = verify(args.prepared, args.prepare_sha, args.join_sha)
        native.constructor_contract()

        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

        print("Execute all fixed 16 rows including the three new CUDA-planned suffixes.", flush=True)
        r = runner.execute(p, data, runner.Store(path), factory, args.prepared, args.prepare_sha,
                           authorization=runner.AUTH)
        result = dict(status=r["status"], report_sha256=sha(path / "report.json"), errors=r["errors"])
    else:
        need(path.is_absolute() and path.is_dir() and not path.is_symlink() and
             sha(path / "report.json") == args.report_sha, "report SHA")
        report = native.read(path / "report.json", cap=64*1024**2)
        binding = report["execution_binding"]
        p, _ = verify(binding["prepared"], binding["prepare_sha256"], args.join_sha)
        need(binding == native.read(path / "execution_binding.json") and
             binding["join_sha256"] == args.join_sha, "execution binding")
        result = previous.consumer.check(path, report, p)
    print(json.dumps(result, sort_keys=True))
    if result.get("status") == "partial_or_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
