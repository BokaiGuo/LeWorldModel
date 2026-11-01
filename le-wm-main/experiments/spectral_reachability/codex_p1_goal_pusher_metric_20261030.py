"""Exploratory RGB-derived pusher tangent metric and fixed candidate CUDA replay.

This is a post-outcome model-only diagnostic. It uses edited goal pixels, not
simulator state or a trained readout; no physical or scientific admission.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from . import codex_p1_fixed_prefix_suffix_cuda_20261030_main as original
from . import codex_p1_fixed_prefix_suffix_recheck_20261030_main as recheck
from . import p1_backtrack_physical_runner_20261030 as physical
from . import preflight_pusht_d2p_p0 as p0

NAME = "codex_p1_goal_pusher_metric_20261030"
need = original.spec.require
ALPHA = 16.0
SHIFT = 8


def translated_goal(rgb, dx, dy):
    """Translate blue sprite and anti-aliased edges onto white; reject overlaps."""
    rgb = np.asarray(rgb)
    need(rgb.dtype == np.uint8 and rgb.shape == (224, 224, 3), "RGB format")
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    mask = (b - g > 2 * (g - r)) & (b - g > 8) & (g - r > 4)
    need(np.any(np.all(rgb == [65, 105, 225], axis=-1)), "canonical pusher pixels")
    y, x = np.nonzero(mask)
    need(len(x) >= 8 and type(dx) is int and type(dy) is int and (dx or dy), "visible pusher")
    nx, ny = x + dx, y + dy
    need(np.all((nx >= 0) & (nx < 224) & (ny >= 0) & (ny < 224)), "translation clipped")
    # No claim of accurate reconstruction under the original sprite.
    out = rgb.copy()
    out[mask] = 255
    need(np.all(out[ny, nx] == 255), "translated sprite overlaps nonwhite scene")
    out[ny, nx] = rgb[y, x]
    return out


def run(output):
    output = Path(output)
    need(output.is_absolute() and output.parent == original.ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists() and not output.is_symlink(),
         "exclusive output")
    verified = recheck.check()
    need(verified["candidate_checks"] == 32, "saved candidate check")
    report = original.source.saved.read_verified(recheck.DIRECTORY / "report.json", recheck.REPORT_SHA)
    binding = physical.native.read(recheck.DIRECTORY / "binding.json", cap=64 * 1024**2)
    need(physical.sha(recheck.DIRECTORY / "binding.json") == report["binding_sha256"], "binding SHA")
    source_report, prepared, original_rows = original.inputs()
    need(binding["rows"] == original_rows, "original task contexts")
    _, native_data = physical.material()
    joined = physical.native.read(physical.JOIN, cap=32 * 1024**2)
    goals = []
    for slot in range(4):
        entry = joined["task_rows"][slot]
        goals.append(np.asarray(native_data["roots"][entry["root_slot"]]["packets"][entry["condition"]]["goal_rgb"]))
    variants = [[translated_goal(rgb, dx, dy) for dx, dy in ((-SHIFT, 0), (SHIFT, 0),
                                                             (0, -SHIFT), (0, SHIFT))] for rgb in goals]
    traces = []
    for i, row in enumerate(report["rows"]):
        path = recheck.DIRECTORY / f"node_{i:02d}/trace.jsonl"
        need(physical.sha(path) == row["trace_sha256"], "trace SHA")
        trace = [json.loads(line) for line in path.read_text().splitlines()]
        traces.append(np.asarray(trace[-1]["payload"]["actions"], dtype=np.float32))
        need(traces[-1].shape == (4, 3, 5, 2), "fixed plan shape")
    source_files = {str(Path(__file__).resolve()): physical.sha(Path(__file__).resolve())}
    started = time.monotonic()
    with original.locking.gpu_lock(original.LOCK):
        output.mkdir()
        results, failure, model_before, model_after, host = [], None, None, None, None
        calls = dict(encoder=0, projector=0, action_encoder=0, predictor=0)
        handles = []
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host = original.gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            model = original.checkpoint_compat.load_compatible_pretrained(prepared["policy"], device=device)
            model.eval().requires_grad_(False)
            for name in calls:
                def count_call(_module, _args, key=name):
                    calls[key] += 1
                handles.append(getattr(model, name).register_forward_pre_hook(count_call))
            model_before = original.source.host.signal.model_hash(model)
            need(model_before == source_report["model_before"], "frozen model hash")
            transform = p0.image_transform()
            tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                bases = []
                for goal, edits in zip(goals, variants, strict=True):
                    images = [goal] + edits
                    vectors = [model.encode({"pixels": p0._image_tensor(x, transform=transform, device=device)})["emb"][0, 0]
                               for x in images]
                    directions = torch.stack((vectors[2] - vectors[1], vectors[4] - vectors[3]), dim=1)
                    u, s, _ = torch.linalg.svd(directions, full_matrices=False)
                    need(bool((s > 1e-6).all().item()), "rank-two pusher perturbations")
                    bases.append((vectors[0], u, s))
                for i, (row, model_row, actions) in enumerate(zip(binding["rows"], report["rows"], traces, strict=True)):
                    need(time.monotonic() - started < 600, "wall cap")
                    task = row["task"]
                    goal, basis, singular = bases[row["slot"]]
                    need(torch.allclose(goal, tensor(task["goal"]), rtol=1e-5, atol=1e-5), "RGB goal latent identity")
                    mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
                    values, baselines = [], []
                    for action in actions:
                        z = tensor(task["root_context"]["latent_history"])[None]
                        past = tensor(task["root_context"]["action_history"])[None]
                        for packet in action:
                            normalized = ((tensor(packet) - mean) / std).reshape(1, 1, 10)
                            context = torch.cat((past, normalized), dim=1)
                            prediction = model.predict(z, model.action_encoder(context))
                            z = torch.cat((z, prediction[:, -1:]), dim=1)[:, -3:]
                            past = context[:, -2:]
                        residual = z[0, -1] - goal
                        baseline = residual.square().mean()
                        adjusted = baseline + ALPHA * (basis.T @ residual).square().sum() / residual.numel()
                        baselines.append(float(baseline.item()))
                        values.append(float(adjusted.item()))
                    need(np.allclose(baselines, model_row["scores"], rtol=1e-4, atol=1e-5),
                         "exact same action baseline replay")
                    results.append(dict(slot=row["slot"], arm=row["arm"], indices=row["indices"],
                                        baseline=baselines, adjusted=values,
                                        baseline_winner=int(np.argmin(baselines)),
                                        adjusted_winner=int(np.argmin(values)),
                                        perturb_singular_values=singular.cpu().tolist()))
            torch.cuda.synchronize(device)
            model_after = original.source.host.signal.model_hash(model)
            need(model_after == model_before, "model changed")
            need(all(physical.sha(p) == h for p, h in source_files.items()), "source drift")
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc)[:1000])
        finally:
            for handle in handles:
                handle.remove()
        summary = dict(schema=NAME + ".report", status="complete" if failure is None else "failed",
                       failure=failure, host=host, model_before=model_before, model_after=model_after,
                       rows=results, alpha=ALPHA, shift_pixels=SHIFT, source_report_sha256=recheck.REPORT_SHA,
                       edited_images="blue-dominant sprite and anti-alias translation on white; synthetic RGB",
                       actual_module_calls=calls, P1_admission=False, physical_execution=False,
                       independent_confirmation=False, fairness_proven=False)
        digest = original.write_new(output / "report.json", summary)
        print(json.dumps(dict(status=summary["status"], sha256=digest, failure=failure)))
        return 0 if failure is None else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    raise SystemExit(run(parser.parse_args().output))
