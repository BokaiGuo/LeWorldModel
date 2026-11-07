"""Compare true observed history to sealed single-frame directional predictions."""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_rgb_directional_alignment_20261107_codex as source
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_single_frame_contract_20261107_codex import model_digest
from experiments.pc_rgb_timed_history_contract_20261107_codex import TimedHistoryRGB

ROOT = source.source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_HISTORY_DIRECTIONAL_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_history_directional_prediction_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_history_directional_result_20261107_codex.json"


def history_window(episodes, steps, pixels, actions, row):
    if row < 10:
        raise ValueError("insufficient history")
    ep = np.asarray(episodes[row - 10:row + 1])
    clock = np.asarray(steps[row - 10:row + 1])
    if len(ep) != 11 or not np.all(ep == ep[0]):
        raise ValueError("episode discontinuity")
    if len(clock) != 11 or not np.array_equal(clock, np.arange(clock[0], clock[0] + 11)):
        raise ValueError("clock discontinuity")
    hist = np.asarray(actions[row - 10:row])
    if hist.shape != (10, 2) or not np.isfinite(hist).all() or np.abs(hist).max() > 1:
        raise ValueError("invalid action history")
    return np.stack([pixels[row - 10], pixels[row - 5], pixels[row]]), hist.copy()


def pair_counts(predicted, truth):
    p, t = np.asarray(predicted, dtype=float), np.asarray(truth, dtype=float)
    if p.shape != t.shape or p.ndim != 1 or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("invalid pair differences")
    ties = (p == 0) | (t == 0)
    agree = (~ties) & (np.sign(p) == np.sign(t))
    return {"agree": int(agree.sum()), "disagree": int((~ties & ~agree).sum()), "tie": int(ties.sum())}


def predict():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    old = json.loads(source.PLAN.read_text())
    if (old["input_sha256"] != file_hash(source.source.inputs.INPUT)
            or old["runner_sha256"] != file_hash(source.__file__)
            or old["protocol_sha256"] != file_hash(source.PROTOCOL)):
        raise ValueError("original prediction drift")
    model, scaler, transform, action_hash = source.source.common.model_setup()
    if model_fingerprint(model) != old["model_sha256"] or action_hash != old["normalizer_sha256"]:
        raise ValueError("checkpoint or normalizer mismatch")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, source.PLAN, source.source.inputs.INPUT,
                         source.source.common.CHECKPOINT / "weights.pt",
                         source.source.common.CHECKPOINT / "config.json"],
                         preprocessing="ImageNet 224; original action scaler", clock="two observed and five future packets")
    guard = TimedHistoryRGB(model, audit, expected_model_digest=model_digest(model),
                            action_mean=scaler.mean_, action_scale=scaler.scale_, packet_steps=5,
                            physical_low=-1., physical_high=1., max_frames=3)
    actions = np.asarray(old["raw_actions"], dtype=np.float32)
    expected = np.stack([source.directions()[d] * h * sign for d, h, sign in old["configs"]])
    np.testing.assert_array_equal(actions, expected)
    candidates = torch.tensor(scaler.transform(actions.reshape(-1, 2)).reshape(1, 24, 5, 10),
                              dtype=torch.float32, device="cuda")
    rows = []
    with h5py.File(source.source.common.DATA, "r") as dataset, \
            np.load(source.source.inputs.INPUT, allow_pickle=False) as data, torch.inference_mode():
        for row in old["rows"]:
            index = int(row["row"])
            selected = np.flatnonzero(data["rows"] == index)
            if len(selected) != 1:
                raise ValueError("input identity mismatch")
            frames, history = history_window(dataset["ep_idx"], dataset["step_idx"], dataset["pixels"],
                                             dataset["action"], index)
            np.testing.assert_array_equal(frames[-1], data["current"][selected.item()])
            pixels = torch.stack([transform(frame) for frame in frames])[None, None].cuda()
            goal = transform(data["goal"][selected.item()])[None, None, None].cuda()
            hist = torch.tensor(scaler.transform(history).reshape(1, 1, 2, 10),
                                dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            values, context = guard.score_timed(pixels=pixels, goal=goal, history=hist, future=candidates,
                                                frame_steps=(0, 5, 10), current_step=10)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rows.append({"root": row["root"], "row": index, "scores": values[0].cpu().tolist(),
                         "predicted_features": context["predicted_emb"][0, :, -1].cpu().tolist(),
                         "history_actions": history.tolist(), "elapsed_seconds": elapsed})
            print(f"root={row['root']} three-frame predictions sealed", flush=True)
    guard.verify_unchanged()
    with PLAN.open("x") as stream:
        json.dump({"rows": rows, "original_plan_sha256": file_hash(source.PLAN),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "model_sha256": old["model_sha256"], "normalizer_sha256": action_hash,
                   "scope": "RGB-only historical input, exposed offline development"},
                  stream, allow_nan=False)


def analyze():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    history, old, truth = (json.loads(path.read_text()) for path in (PLAN, source.PLAN, source.PHYSICAL))
    if (history["original_plan_sha256"] != file_hash(source.PLAN)
            or history["protocol_sha256"] != file_hash(PROTOCOL)
            or history["runner_sha256"] != file_hash(Path(__file__))
            or truth["plan_sha256"] != file_hash(source.PLAN)):
        raise ValueError("sealed artifact mismatch")
    rows = []
    for h, s, real in zip(history["rows"], old["rows"], truth["rows"], strict=True):
        if (h["root"], h["row"]) != (s["root"], s["row"]) or (s["root"], s["row"]) != (real["root"], real["row"]):
            raise ValueError("root mismatch")
        indices = real["indices"]
        if len(indices) != 16 or len(set(indices)) != 16:
            raise ValueError("candidate coverage")
        single = np.asarray(s["predicted_features"])[indices]
        multi = np.asarray(h["predicted_features"])[indices]
        observed = np.asarray(real["true_features"])
        if single.shape != multi.shape or single.shape != observed.shape:
            raise ValueError("feature dimensions")
        lookup = {idx: i for i, idx in enumerate(indices)}
        pairs = [(d * 6 + 2 * k, d * 6 + 2 * k + 1) for d in range(4) for k in (1, 2)]
        physical = [real["traces"][lookup[a]][-1]["distance"] -
                    real["traces"][lookup[b]][-1]["distance"] for a, b in pairs]
        single_scores = np.asarray(s["predicted_costs"]).reshape(-1)
        multi_scores = np.asarray(h["scores"])
        metrics = {name: pair_counts([scores[a] - scores[b] for a, b in pairs], physical)
                   for name, scores in (("single", single_scores), ("history", multi_scores))}
        rows.append({"root": h["root"], "row": h["row"],
                     "single_mse": float(np.square(single - observed).mean()),
                     "history_mse": float(np.square(multi - observed).mean()),
                     "single_physical_pairs": metrics["single"], "history_physical_pairs": metrics["history"],
                     "elapsed_seconds": h["elapsed_seconds"]})
    first = np.asarray([r["single_mse"] for r in rows])
    second = np.asarray([r["history_mse"] for r in rows])
    single_agree = sum(r["single_physical_pairs"]["agree"] for r in rows)
    history_agree = sum(r["history_physical_pairs"]["agree"] for r in rows)
    summary = {"roots": len(rows), "single_mean_mse": float(first.mean()), "history_mean_mse": float(second.mean()),
               "history_lower": int((second < first).sum()), "history_higher": int((second > first).sum()),
               "history_equal": int((second == first).sum()), "single_agree": single_agree,
               "history_agree": history_agree,
               "passes_development_exit": bool((second < first).sum() >= 7 and
                                               second.mean() <= .9 * first.mean() and history_agree >= single_agree)}
    with RESULT.open("x") as stream:
        json.dump({"rows": rows, "summary": summary, "prediction_sha256": file_hash(PLAN),
                   "physical_sha256": file_hash(source.PHYSICAL), "protocol_sha256": file_hash(PROTOCOL)},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("predict", "analyze"):
        raise SystemExit("expected predict|analyze")
    {"predict": predict, "analyze": analyze}[sys.argv[1]]()
