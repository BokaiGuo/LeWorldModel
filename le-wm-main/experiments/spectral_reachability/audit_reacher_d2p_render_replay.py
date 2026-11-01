"""Replay the failing root's complete probe order and inspect render state.

All model encoding/prediction runs on host CUDA. Physical state and rendering
instrumentation are evaluator-only. No tolerances or successful P1 are inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
from pathlib import Path

from experiments.spectral_reachability import audit_reacher_d2p_prefix_capability as core

DERIVED = ("xpos", "xquat", "xmat", "geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat", "light_xpos", "light_xdir")


def array_fingerprint(value):
    a = core.np.ascontiguousarray(value)
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def numeric_fields(obj):
    result = {}
    for name in dir(obj):
        if name.startswith("_"): continue
        value = getattr(obj, name)
        if isinstance(value, numbers.Integral):
            result[name] = int(value)
        elif isinstance(value, numbers.Real):
            if not math.isfinite(float(value)): raise ValueError("nonfinite render scene scalar")
            result[name] = float(value).hex()
        elif isinstance(value, core.np.ndarray) and value.dtype.kind in "biuf" and value.size <= 64:
            result[name] = {**array_fingerprint(value), "values": value.tolist()}
    return result


def scene_record(scene):
    return {"scene": numeric_fields(scene),
        "geoms": [numeric_fields(scene.geoms[i]) for i in range(scene.ngeom)],
        "lights": [numeric_fields(scene.lights[i]) for i in range(scene.nlight)],
        "camera": [numeric_fields(c) for c in scene.camera]}


def differing_paths(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        return [path for key in sorted(set(left) | set(right))
            for path in differing_paths(left.get(key), right.get(key), prefix+"/"+str(key))]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [path for i, (a, b) in enumerate(zip(left, right)) for path in differing_paths(a, b, prefix+"/"+str(i))]
    return [] if left == right else [prefix]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task", choices=tuple(core.chart.TASKS), default="full_drive")
    parser.add_argument("--refresh-derived", action="store_true")
    parser.add_argument("--gl-policy", choices=("original", "disable-dither", "finish", "warm-render"), default="original")
    parser.add_argument("--inspect-scene", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".protocol.json").exists():
        raise FileExistsError("new immutable render diagnostic output required")
    core.verify_files(args.reference)
    ref = json.loads(args.reference.read_text())
    pp = args.reference.with_suffix(".predictions.json")
    old = {r["seed"]: r for r in json.loads(pp.read_text())["rows"]}[args.seed]
    original_actual = {r["seed"]: r for r in ref["rows"]}[args.seed]
    shared, execution, np, torch = core.shared, core.execution, core.np, core.torch
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if (source_sha != ref["sealed_protocol"]["source_sha256_receipt"]
            or core.digest(args.policy/"weights.pt") != ref["sealed_protocol"]["weights_sha256"]
            or core.digest(args.adapter_artifact) != ref["sealed_protocol"]["adapter_sha256"]):
        raise ValueError("exact original source/model/statistics required")
    protocol = {"seed": args.seed, "task": args.task, "all17": True, "scales": list(core.SCALES),
        "refresh_derived": args.refresh_derived, "gl_policy": args.gl_policy, "inspect_scene": args.inspect_scene,
        "weights_sha256": ref["sealed_protocol"]["weights_sha256"],
        "bindings": {str(p.resolve()): core.digest(p) for p in (args.reference, pp, args.adapter_artifact)},
        "implementation_sha256": core.digest(Path(__file__)), "P1_admission": False,
        "scope": "Known-failure render-layer diagnosis only. All17 prefixes and original complete probe order retained. Refresh, if requested, preserves exact integration state; it is not permission to reuse old RGB efficacy evidence."}
    core.write_new(args.output.with_suffix(".protocol.json"), protocol)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    normalizer = shared.ActionNormalizer(mean, std)
    processor = shared._Preprocessor(normalizer)
    mean_t, std_t = (torch.tensor(v, device="cuda") for v in (normalizer.block_mean, normalizer.block_std))
    model = shared.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameters, buffers = shared._parameter_hash(model), {k: v.clone() for k, v in model.named_buffers()}
    env = core.goals.goal_wrapper(ref["sealed_protocol"]["goal_contract"])(
        core.gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
    physics = env.unwrapped.env.physics
    original_render = env.render
    original_raw_render = env.unwrapped.render
    scene = None
    from dm_control.mujoco import engine
    original_camera_update = engine.Camera.update
    if args.inspect_scene:
        def camera_update(camera, *a, **kw):
            nonlocal scene
            result = original_camera_update(camera, *a, **kw)
            scene = scene_record(camera._scene)
            return result
        engine.Camera.update = camera_update
    gl_calls = []
    if args.gl_policy != "original":
        from OpenGL import GL
        def controlled_raw_render(*a, **kw):
            def disable():
                was_enabled = bool(GL.glIsEnabled(GL.GL_DITHER))
                if args.gl_policy == "disable-dither": GL.glDisable(GL.GL_DITHER)
                elif args.gl_policy == "finish": GL.glFinish()
                return was_enabled
            with physics.contexts.gl.make_current() as context:
                before = context.call(disable)
            warm = None
            if args.gl_policy == "warm-render":
                raw_state = execution.capture_execution_snapshot(env)
                warm = np.asarray(original_raw_render(*a, **kw), dtype=np.uint8).copy()
                if raw_state != execution.capture_execution_snapshot(env):
                    raise RuntimeError("warm render changed complete physical state")
            image = original_raw_render(*a, **kw)
            if warm is not None and raw_state != execution.capture_execution_snapshot(env):
                raise RuntimeError("retained render changed complete physical state")
            with physics.contexts.gl.make_current() as context:
                if args.gl_policy == "finish": context.call(GL.glFinish)
                after = bool(context.call(GL.glIsEnabled, GL.GL_DITHER))
            record = {"before_enabled": before, "after_enabled": after}
            if warm is not None:
                record.update(raw_render_passes=2, physical_state_exact=True,
                    warmup_rgb_sha256=execution.rgb_digest(warm), retained_rgb_sha256=execution.rgb_digest(image),
                    warmup_retained_max_pixel_error=int(np.max(np.abs(warm.astype(np.int16)-np.asarray(image).astype(np.int16)))))
            gl_calls.append(record)
            return image
        env.unwrapped.render = controlled_raw_render
    renders, conflicts, registry = [], [], {}
    def instrumented_render(*a, **kw):
        before = execution.capture_execution_snapshot(env)
        if args.refresh_derived:
            packed = np.asarray(before["physics_integration_state"], dtype=np.float64)
            spec = before["integration_state_spec"]
            physics.forward()
            physics.set_state(packed, sig=spec)
            execution.assert_exact_execution_state(before, execution.capture_execution_snapshot(env))
        derived = {k: array_fingerprint(getattr(physics.data, k)) for k in DERIVED}
        frame = np.asarray(original_render(*a, **kw), dtype=np.uint8).copy()
        if before != execution.capture_execution_snapshot(env):
            raise RuntimeError("render instrumentation changed complete integration/scene state")
        state_key = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
        rgb = execution.rgb_digest(frame)
        record = {"render": len(renders), "state_sha256": state_key, "derived": derived, "rgb_sha256": rgb,
            "scene_sha256": hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest() if args.inspect_scene else None}
        prior = registry.get(state_key)
        if prior is not None and prior["record"]["rgb_sha256"] != rgb:
            changed = np.argwhere(prior["frame"] != frame)
            conflicts.append({"first": prior["record"], "repeat": record,
                "derived_differences": [k for k in DERIVED if prior["record"]["derived"][k] != derived[k]],
                "max_pixel_error": int(np.max(np.abs(prior["frame"].astype(np.int16)-frame.astype(np.int16)))),
                "changed_components": len(changed), "changed_coordinates": changed.tolist(),
                "scene_differences": differing_paths(prior["scene"], scene),
                "first_scene": prior["scene"], "repeat_scene": scene,
                "first_rgb": execution.pack_rgb_observation(prior["frame"]), "repeat_rgb": execution.pack_rgb_observation(frame)})
        if prior is None:
            registry[state_key] = {"record": record, "frame": frame, "scene": scene}
        renders.append(record)
        return frame
    env.render = instrumented_render
    records, futures = [], 0
    def prepare(c):
        return core.p0._prepare(processor, c["frames"], c["root_frame"], c["goal"],
            np.asarray(c["history_action_blocks"], np.float32), torch.device("cuda"))
    try:
        ctx = shared._source_state_context(env, dataset=dataset, seed=args.seed)
        ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
        if ctx["execution_snapshot"] != old["initial_snapshot"]:
            core.write_new(args.output.with_suffix(".failure.json"), {
                "reason": "original_root_snapshot_changed", "protocol": protocol,
                "changed_keys": [k for k in set(ctx["execution_snapshot"]) | set(old["initial_snapshot"])
                    if ctx["execution_snapshot"].get(k) != old["initial_snapshot"].get(k)],
                "gl_calls": gl_calls, "P1_admission": False})
            raise ValueError("original root snapshot changed")
        pixels, _, history = prepare(ctx)
        with torch.no_grad(): root_emb = core.p0._latent_context(model, pixels, history)
        width = len(core.chart.active_columns(args.task))
        for bank in old["tasks"][args.task]["bank"]:
            index = bank["index"]
            raw = np.asarray(bank["variants"]["cem_then_gn4"]["raw_actions"], np.float32)[:5]
            frame, goal = execution.restore_execution_snapshot(env, seed=args.seed,
                snapshot=ctx["execution_snapshot"], root_identity="sealed_observation_exact_state")
            core.p0._verify_execution_identity(frame, goal, ctx["root_frame"], ctx["goal"])
            current = execution.advance_executed_context(env, context=ctx, raw_action_block=raw,
                action_block=5, verified_root_frame=ctx["root_frame"])
            current["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(current["root_frame"])
            pix, _, hist = prepare(current)
            with torch.no_grad(): emb = core.p0._latent_context(model, pix, hist).detach()
            expected = original_actual["tasks"][args.task]["bank_"+str(index)]["cem_then_gn4"]["packets"][0]
            prefix_exact = (emb[0, -1].tolist() == expected["observed_vector"]
                and current["root_rgb_sha256"] == expected["root_rgb_sha256"])
            prefix = torch.tensor(raw.ravel()[core.chart.active_columns(args.task)], device="cuda").reshape(1, 1, width)
            pe, ph, _ = core.compiler.after_prefix(model, root_emb, history, prefix, task=args.task, mean=mean_t, std=std_t)
            def predicted(x): return core.chart.terminal(model, pe, ph, x.reshape(1, 2, width), task=args.task, mean=mean_t, std=std_t)[0]
            def terminal(x): return core.chart.terminal(model, emb, hist, x.reshape(1, 2, width), task=args.task, mean=mean_t, std=std_t)[0]
            zero = torch.zeros(2*width, device="cuda", requires_grad=True)
            with torch.no_grad(): predicted(zero)
            torch.autograd.functional.jacobian(predicted, zero, vectorize=True)
            def observe(x):
                nonlocal futures
                actions = core.chart.expand_commands(x.reshape(1, 2, width), args.task)[0].reshape(10, 2).detach().cpu().numpy()
                f, g = execution.restore_execution_snapshot(env, seed=args.seed,
                    snapshot=current["execution_snapshot"], root_identity="sealed_observation_exact_state")
                core.p0._verify_execution_identity(f, g, current["root_frame"], current["goal"])
                advanced = current
                for j in range(2):
                    advanced = execution.advance_executed_context(env, context=advanced,
                        raw_action_block=actions[5*j:5*(j+1)], action_block=5, verified_root_frame=advanced["root_frame"])
                px, _, h = prepare(advanced)
                if px.device.type != "cuda" or h.device.type != "cuda": raise RuntimeError("CPU inference forbidden")
                with torch.no_grad(): obs = core.p0._latent_context(model, px, h)[0, -1].tolist()
                futures += 1
                return {"observed": obs, "snapshot": advanced["execution_snapshot"],
                    "rgb_sha256": advanced["root_rgb_sha256"], "history_rgb_sha256": advanced["history_rgb_sha256"], "render": renders[-1]}
            with torch.no_grad():
                first, repeat = observe(zero), observe(zero)
                terminal(zero)
            torch.autograd.functional.jacobian(terminal, zero, vectorize=True)
            for scale in core.SCALES:
                for _, command in core.probe_commands(args.task, scale):
                    for v in (command, -command):
                        x = torch.tensor(v, device="cuda")
                        with torch.no_grad(): observe(x); terminal(x)
            records.append({"index": index, "original_prefix_exact": prefix_exact, "first": first, "repeat": repeat})
            print(json.dumps({"index": index, "original_prefix_exact": prefix_exact,
                "zero_rgb_exact": first["rgb_sha256"] == repeat["rgb_sha256"],
                "zero_state_exact": first["snapshot"] == repeat["snapshot"],
                "render_conflicts": len(conflicts), "physical_futures": futures}), flush=True)
    finally:
        env.render = original_render
        env.unwrapped.render = original_raw_render
        engine.Camera.update = original_camera_update
        env.close()
    unchanged = shared._parameter_hash(model) == parameters and all(torch.equal(v, buffers[k]) for k, v in model.named_buffers())
    if not unchanged or any(m.training for m in model.modules()): raise RuntimeError("frozen model changed")
    core.write_new(args.output, {"schema": "reacher_d2p_render_replay.v1", "protocol": protocol,
        "cuda_host": host, "records": records, "renders": renders, "conflicts": conflicts,
        "physical_futures": futures, "frozen_model_unchanged": unchanged, "gl_calls": gl_calls,
        "P1_admission": False})


if __name__ == "__main__": main()
