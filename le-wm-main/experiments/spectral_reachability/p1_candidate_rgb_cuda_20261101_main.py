"""Encode all saved candidate-intervention frames on host CUDA for diagnosis."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_candidate_intervention_analysis_20261101_main as analysis
from . import p1_q_return_shard_20261101_main as q
from . import e3_cp10_target_cuda_20261031 as source

NAME = "p1_candidate_rgb_cuda_20261101_main"
DIRECTORY = source.ROOT / "results/p1_candidate_intervention_20261101_main_run_MAIN01"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    out = Path(output)
    if not out.is_absolute() or out.parent != source.ROOT / "results" or out.exists():
        raise ValueError("exclusive output")
    checked = analysis.analyze(DIRECTORY)
    if source.base.sha(q.METRIC) != q.METRIC_SHA:
        raise ValueError("metric hash")
    metric = json.loads(q.METRIC.read_bytes())
    goal = np.asarray(metric["goal_latent"], np.float32)
    matrix = np.asarray(metric["sqrt_matrix"], np.float64)
    report = json.loads((DIRECTORY / "report.json").read_bytes())
    sources = dict(checked["sources"])
    sources.update(source.base.bindings())
    for path in (q.METRIC, Path(__file__).resolve(), Path(analysis.__file__).resolve()):
        sources[str(path)] = source.base.sha(path)
    frames = []
    for row in report["rows"]:
        observations = list(row["observations"])
        if row["status"] == "censored_out_of_domain":
            observations.append(row["last_observation"])
        for cp in observations:
            sources[cp["path"]] = cp["file_sha256"]
            frames.append(dict(branch=row["branch"], candidate=row["candidate"],
                               suffix=row["suffix_role"], step=cp["step"],
                               path=cp["path"], pixel_sha256=cp["pixel_sha256"]))
    with source.base.locking.gpu_lock(source.base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(source.base.MODEL), device=device).eval().requires_grad_(False)
        before = source.base.model_hash(model)
        if before != source.base.MODEL_HASH:
            raise ValueError("frozen checkpoint")
        encoded = {}
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            transform = p0.image_transform()
            for frame in frames:
                key = frame["pixel_sha256"]
                if key in encoded:
                    continue
                with Image.open(frame["path"]) as image:
                    pixels = np.asarray(image).copy()
                tensor = p0._image_tensor(pixels, transform=transform, device=device)
                latent = model.encode({"pixels": tensor})["emb"][0, 0]
                if latent.shape != (192,) or not bool(torch.isfinite(latent).all()):
                    raise ValueError("finite frozen embedding")
                vector = latent.cpu().numpy()
                residual = (vector - goal).astype(np.float64)
                encoded[key] = dict(latent=vector.tolist(), q_score=float(np.mean((residual @ matrix.T) ** 2)),
                                    euclidean_score=float(np.mean(residual ** 2)))
        torch.cuda.synchronize(device)
        after = source.base.model_hash(model)
        if after != before or any(source.base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model/source drift")
        pairs = []
        for a in range(8):
            for b in range(a + 1, 8):
                ends = [next((f for f in frames if f["candidate"] == i and f["suffix"] == "own"
                              and f["step"] == 15 and next(r for r in checked["rows"]
                                  if r["candidate"] == i and r["suffix"] == "own")["endpoint15"] is not None),
                             None) for i in (a, b)]
                if None in ends:
                    continue
                predicted = checked["predicted_scores"][a] - checked["predicted_scores"][b]
                observed = encoded[ends[0]["pixel_sha256"]]["q_score"] - encoded[ends[1]["pixel_sha256"]]["q_score"]
                physical = [next(r for r in checked["rows"] if r["candidate"] == i and r["suffix"] == "own")
                            ["endpoint15"]["joint_xy"] for i in (a, b)]
                pairs.append(dict(candidates=[a, b], predicted_q_margin=predicted,
                                  actual_rgb_q_margin=observed, joint_xy_margin=physical[0]-physical[1],
                                  model_rgb_reversal=predicted * observed < 0,
                                  rgb_joint_reversal=observed * (physical[0]-physical[1]) < 0))
        result = dict(status="complete", host=host, frames=frames, encoded=encoded, pairs=pairs,
                      encoder_calls=len(encoded), frame_occurrences=len(frames),
                      exact_pixel_deduplication=True, sources=sources,
                      model_before=before, model_after=after, P1_admission=False,
                      independent_confirmation=False, physical_outcomes_used_for_deployment=False,
                      caution="Pairs share one known root; censored endpoints excluded, never imputed.")
        digest = source.base.write(out, result)
    print(json.dumps(dict(output=str(out), sha256=digest, encoder_calls=len(encoded),
                          comparable_pairs=len(pairs))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    run(args.output)
