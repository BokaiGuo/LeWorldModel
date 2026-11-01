"""Post-outcome development RGB feedback/model scoring interface; no executor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_dev303_cp5_encoded_v13_replan_20261102_exclusive as frozen
from . import p1_rgb_chunk_approach_interface_20261102_exclusive as approach

NAME = "p1_dev303_rgb_feedback_scored_tail_20261102_codex_exclusive"
ROOT = frozen.ROOT
engine = frozen.engine
require = frozen.require
common = engine.base.common
prior = approach.prior
rgb = approach.rgb


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def protocol():
    source = frozen.protocol()
    physical_report = ROOT / "results/p1_dev303_cp5_tail_physical_20261102_exclusive_RUN_DEV01/report.json"
    physical_sha = "f9f367dff0913ab85287461178d089ad0aa214af28beb6af0d0fc75491ae838a"
    require(sha(physical_report.read_bytes()) == physical_sha, "design provenance report SHA")
    paths = [Path(__file__), Path(approach.__file__), Path(rgb.__file__),
             Path(approach.geometry.__file__), Path(approach.motion.__file__),
             Path(prior.__file__), Path(common.__file__)]
    return {
        "schema": NAME + ".protocol.v1",
        "classification": "post_outcome_development_only_not_holdout",
        "source_encode_report_sha256": frozen.ENCODE_REPORT_SHA,
        "source_tail_physical_report_sha256": physical_sha,
        "source_bindings": source["bindings"],
        "implementation_sha256": {str(p.resolve()): sha(p.read_bytes()) for p in paths},
        "model_hash": source["model_hash"], "task": source["task"], "policy": source["policy"],
        "history": "11 consecutive trusted RGB frames and 10 actual actions; model uses frames 0,5,10 and two five-action packets",
        "next_step": "one action only, then require next real RGB and exact actual-action receipt",
        "candidate": "each iteration includes one bounded RGB approach first action followed by zeros",
        "selection": "V13 five-variant robust model score among one-step RGB-safe candidates",
        "budget_per_replan": {"rgb_encode_calls": 3, "action_encoder_calls": 10,
                              "predict_calls": 10, "predict_samples": 3200,
                              "iterations": 5, "candidates": 320, "variants": 1600},
        "seed": frozen.parent.SEED,
        "safety": "old RGB single-object-component, pusher bounds, near-object abstention, disk and predicted segment guards",
        "model_five_step_spacing_requires_consecutive_history": True,
        "receipt_trust": "trusted caller must establish actual actions and real RGB; interface cannot authenticate camera provenance",
        "max_actions_per_instance": 5,
        "maximum_five_replans": {"rgb_encode_calls": 15, "predict_calls": 50,
                                 "action_encoder_calls": 50, "predict_samples": 16000},
        "no_state_or_outcome_selection": True, "physical_execution_authorized": False,
        "cuda_execution_authorized": False, "P1_admission": False,
        "independent_confirmation": False,
    }


def raw_actions(value, shape):
    a = np.asarray(value, np.float64)
    require(a.shape == shape and np.isfinite(a).all() and np.all(np.abs(a) <= 1),
            "actual action shape or bounds")
    out = a.astype("<f4")
    require(np.array_equal(out.astype(np.float64), a), "actual action precision")
    return out


def geometry(image, previous):
    image = rgb._image(image)
    _, pusher = rgb._mask(image, rgb.BLUE, 8, single=False)
    mask, obj = rgb._mask(image, rgb.BLOCK, 8, single=False)
    require(obj["components"] == 1, "fragmented object")
    x0, y0, x1, y1 = pusher["bounds_xyxy"]
    require(pusher["components"] <= prior.MOVING_PUSHER_COMPONENTS_MAX
            and x1-x0+1 <= prior.MOVING_PUSHER_BBOX_MAX_PIXELS
            and y1-y0+1 <= prior.MOVING_PUSHER_BBOX_MAX_PIXELS, "pusher segmentation")
    y, x = np.nonzero(mask)
    pixels = np.stack((x, y), axis=1).astype(np.float64)
    position = np.asarray(pusher["centroid_xy"], np.float64)
    delta = pixels - position
    distances = np.linalg.norm(delta, axis=1)
    nearest = int(np.argmin(distances))
    gap = float(distances[nearest])
    require(np.isfinite(gap), "invalid geometry")
    if gap <= prior.TARGET_GAP_PIXELS:
        return None
    direction = delta[nearest] / gap
    magnitude = min(prior.MAX_RAW_MAGNITUDE,
                    (gap-prior.TARGET_GAP_PIXELS)*prior.NATIVE_PER_PIXEL/prior.RAW_TARGET_SCALE_NATIVE)
    candidate = np.asarray(direction * magnitude, dtype="<f4")

    def safe(action):
        action = np.asarray(action, np.float64)
        if not np.isfinite(action).all() or np.linalg.norm(action) > prior.MAX_RAW_MAGNITUDE + 1e-7:
            return False
        displacement = (approach.motion.NATIVE_DELTA_PER_CURRENT_ACTION * action
                        + approach.motion.NATIVE_DELTA_PER_PREVIOUS_VELOCITY
                        * approach.motion.NATIVE_VELOCITY_PER_PREVIOUS_ACTION * previous
                        ) * approach.motion.PIXELS_PER_NATIVE
        end = position + displacement
        radius = approach.geometry.PUSHER_RADIUS_PIXELS
        return (all(radius <= point[0] <= 223-radius and radius <= point[1] <= 223-radius
                    for point in (position, end))
                and approach._distance_segment(position, end, pixels) >= prior.TARGET_GAP_PIXELS)

    require(safe(candidate), "approach fails one-step guard")
    return candidate, safe, gap


def search(candidate, safe, score):
    """Run exact V13 sampling and robust score; backend must meter model calls."""
    mean = np.zeros((2, 5, 2), np.float32)
    std = np.full_like(mean, engine.INITIAL_STD)
    seed = mean.copy()
    seed[0, 0] = candidate
    rng = np.random.default_rng(frozen.parent.SEED)
    records, iterations = [], []
    for iteration in range(engine.ITERATIONS):
        bank = common.make_bank(rng, mean, std, np.stack([seed, mean]), iteration)
        bank[0] = seed  # reserved in-budget approach candidate at every iteration
        variants = np.stack([common.action_variants(c) for c in bank])
        scores = np.asarray(score(variants), np.float64)
        require(scores.shape == (64, 5) and np.isfinite(scores).all(), "finite model scores")
        robust = common.robust_scores(scores)
        feasible = np.asarray([safe(c[0, 0]) for c in bank])
        require(feasible[0], "approach candidate no longer safe")
        order = sorted(np.flatnonzero(feasible).tolist(), key=lambda i: (robust[i], i))
        records.extend((float(robust[i]), iteration, i, bank[i].copy()) for i in order)
        elites = (order * ((engine.ELITES + len(order)-1)//len(order)))[:engine.ELITES]
        selected = bank[elites]
        mean = selected.mean(axis=0, dtype=np.float32)
        std = np.clip(selected.std(axis=0, dtype=np.float32), engine.base.MIN_STD, 1)
        iterations.append({"iteration": iteration, "approach_score": float(robust[0]),
                           "feasible_candidates": int(feasible.sum())})
    best = min(records, key=lambda x: (x[0], x[1], x[2]))
    return best[3][0, 0].copy(), {"selected_score": best[0], "iteration": best[1],
                                   "candidate": best[2], "iterations": iterations}


class CUDARealScorer:
    """Actual frozen-model scoring backend; instantiated only by an authorized integrator."""
    def __init__(self, model, task, expected_model_hash):
        import torch
        self.torch, self.model, self.task = torch, model, task
        frozen_source = frozen.protocol()
        require(task == frozen_source["task"] and expected_model_hash == frozen_source["model_hash"],
                "frozen task and model binding")
        require(not model.training, "evaluation mode required")
        require(next(model.parameters()).device.type == "cuda", "CUDA model required")
        self.before = engine.base.blind.host.signal.model_hash(model)
        require(self.before == expected_model_hash, "model hash")
        self.counts = {"rgb_encode_calls": 0, "action_encoder_calls": 0,
                       "predict_calls": 0, "predict_samples": 0}
        self.hooks = {"action_encoder": 0, "predict": 0}
        self.handles = [
            model.action_encoder.register_forward_pre_hook(
                lambda *_: self.hooks.__setitem__("action_encoder", self.hooks["action_encoder"] + 1)),
            model.predictor.register_forward_pre_hook(
                lambda *_: self.hooks.__setitem__("predict", self.hooks["predict"] + 1)),
        ]

    def begin(self, frames, actions):
        torch, model, task = self.torch, self.model, self.task
        device = next(model.parameters()).device
        self.device = device
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        self.mean, self.std = tensor(task["action_mean"]), tensor(task["action_std"])
        self.goal = tensor(task["goal"])[None]
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            latents = []
            for image in (frames[0], frames[5], frames[10]):
                image_tensor = engine.base.p0._image_tensor(
                    image, transform=engine.base.p0.image_transform(), device=device)
                latents.append(model.encode({"pixels": image_tensor})["emb"][0, 0])
                self.counts["rgb_encode_calls"] += 1
            self.states = torch.stack(latents)[None]
            self.history = ((tensor(actions) - self.mean) / self.std).reshape(1, 2, 10)
            require(self.states.shape == (1, 3, task["latent_dim"])
                    and bool(torch.isfinite(self.states).all()), "encoded actual RGB history")

    def score(self, variants):
        torch, model = self.torch, self.model
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            flat = torch.as_tensor(variants.reshape(320, 2, 5, 2), dtype=torch.float32, device=self.device)
            states, past = self.states.repeat(320, 1, 1), self.history.repeat(320, 1, 1)
            for packet in range(2):
                context = torch.cat((past, ((flat[:, packet] - self.mean) / self.std).reshape(320, 1, 10)), dim=1)
                encoded = model.action_encoder(context)
                self.counts["action_encoder_calls"] += 1
                predicted = model.predict(states, encoded)
                self.counts["predict_calls"] += 1
                self.counts["predict_samples"] += 320
                require(predicted.shape == states.shape and bool(torch.isfinite(predicted).all()), "finite model output")
                states = torch.cat((states, predicted[:, -1:]), dim=1)[:, -3:]
                past = context[:, -2:]
            return ((states[:, -1] - self.goal).square().mean(dim=1).reshape(64, 5).cpu().numpy())

    def finish(self):
        self.torch.cuda.synchronize(self.device)
        require(self.counts == {"rgb_encode_calls": 3, "action_encoder_calls": 10,
                                "predict_calls": 10, "predict_samples": 3200}
                and self.hooks == {"action_encoder": 10, "predict": 10}
                and self.before == engine.base.blind.host.signal.model_hash(self.model), "budget or model drift")
        return dict(self.counts)

    def close(self):
        for handle in self.handles:
            handle.remove()


class Feedback:
    """Trusted caller supplies chronological actual RGB/actions; no physical dispatch."""
    def __init__(self, frames, actions, *, frame_ids):
        require(len(frames) == 11 and frame_ids == list(range(frame_ids[0], frame_ids[0] + 11))
                and type(frame_ids[0]) is int and frame_ids[0] >= 0, "consecutive RGB history")
        self.frames = [rgb._image(x).copy() for x in frames]
        self.actions = raw_actions(actions, (10, 2))
        self.frame_id = frame_ids[-1]
        self.pending = None
        self.count = 0
        self.closed = False

    def plan(self, backend):
        try:
            require(not self.closed and self.pending is None, "receipt required or controller closed")
            require(isinstance(backend, CUDARealScorer), "frozen CUDA scorer required")
            require(self.count < 5, "action limit")
            features = geometry(self.frames[-1], self.actions[-1])
            if features is None:
                self.closed = True
                return {"status": "near_object_abstain_no_contact_claim"}
            candidate, safe, gap = features
            backend.begin(self.frames, self.actions)
            action, result = search(candidate, safe, backend.score)
            budget = backend.finish()
            self.pending = action.copy()
            self.count += 1
            return {"status": "proposal_only_wait_for_real_rgb", "action": action.tolist(),
                    "action_sha256": sha(action.tobytes()), "frame_id": self.frame_id,
                    "rgb_sha256": sha(self.frames[-1].tobytes()),
                    "actual_history_sha256": sha(self.actions.tobytes()),
                    "gap_pixels": gap, "approach_candidate": candidate.tolist(),
                    "model_scoring": result, "budget": budget,
                    "P1_admission": False, "physical_execution_authorized": False}
        except Exception:
            self.closed = True
            raise
        finally:
            backend.close()

    def accept(self, frame, *, frame_id, actual_action):
        try:
            require(not self.closed and self.pending is not None and type(frame_id) is int
                    and frame_id == self.frame_id + 1, "fresh consecutive receipt")
            actual = raw_actions([actual_action], (1, 2))[0]
            require(np.array_equal(actual, self.pending), "actual action mismatch")
            image = rgb._image(frame).copy()
            self.frames = self.frames[1:] + [image]
            self.actions = np.concatenate((self.actions[1:], actual[None]), axis=0)
            self.frame_id = frame_id
            self.pending = None
        except Exception:
            self.closed = True
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", required=True, type=Path)
    args = parser.parse_args()
    require(args.prepare.name.startswith(NAME + "_PREP_"), "exclusive contract name")
    raw = (json.dumps(protocol(), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.prepare.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    print(json.dumps({"prepared": str(args.prepare.resolve()), "sha256": sha(raw)}))


if __name__ == "__main__":
    main()
