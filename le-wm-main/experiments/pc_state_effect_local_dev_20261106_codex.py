"""Frozen State-assisted local consequences: AD/FD, held-out scales, real outcomes."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pc_action_effect_descriptor import ActionWitness, describe, central_derivative, quadratic_prediction
from experiments.pc_input_permissions import StateAssistedPointMaze, manifest, file_hash
from experiments.pc_state_track_smoke_20261106_codex import load_model, ROOT
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PROTOCOL = ROOT.parent / "docs/PC_STATE_EFFECT_LOCAL_DEV_PROTOCOL_20261106_CODEX.md"
PREFIX = ROOT / "results/pc_state_effect_local_dev_20261106_codex"
BANK = np.array([-.1, -.03, .03, .1])
EPS = .003


def witness(name, x):
    return ActionWitness(name, tuple((x, 0.) for _ in range(5)))


def rgb_tensor(image):
    return torch.tensor(image.copy(), device="cuda").permute(2, 0, 1)[None, None]


def main():
    output, features_path = PREFIX.with_suffix(".json"), PREFIX.with_suffix(".npz")
    if output.exists() or features_path.exists():
        raise FileExistsError(PREFIX)
    print("State-assisted local consequence validation; fixed held-out amplitudes, no optimizer.", flush=True)
    model, dataset, audit = load_model()
    fingerprint = model_fingerprint(model)
    wrapper = StateAssistedPointMaze(model, audit)
    source = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz"
    source_audit = json.loads((source.parent / (source.stem + "_audit.json")).read_text())
    source_meta = source.with_suffix(".json")
    if source_audit["report_sha256"] != file_hash(source_meta):
        raise ValueError("physical audit drift")
    if json.loads(source_meta.read_text())["payload_sha256"] != file_hash(source):
        raise ValueError("physical payload drift")
    rows, features = [], {}
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with np.load(source, allow_pickle=False) as data:
        for root in range(4):
            start, target = 2 * root, 2 * root + 1
            for camera in (0, 1):
                key = f"r{root}_v{camera}"
                inputs = {"visual": rgb_tensor(data["images"][start, 0, camera, 0]),
                          "proprio": torch.tensor(data["traces"][start, 0, camera, 0].copy(),
                                                  device="cuda", dtype=torch.float32)[None, None]}
                with torch.no_grad():
                    goal = model.encode(rgb_tensor(data["images"][target, 0, camera, 0]))[0]
                features[key + "_goal"] = goal.cpu().numpy()

                def evaluate(raw):
                    packet = ((raw - mean) / std).reshape(1, 1, 10)
                    pred = wrapper.predict(inputs, packet)["visual"][-1]
                    return (pred - goal).square().mean(), pred

                # Derivative in physical x units; all five controls vary together.
                x = torch.tensor(0., device="cuda", requires_grad=True)
                raw = torch.stack([x, x * 0])[None].expand(5, 2)
                cost, _ = evaluate(raw)
                derivative = float(torch.autograd.grad(cost, x)[0])
                ad_parent = float(cost.detach())
                del cost, raw, x

                def score(array):
                    return evaluate(torch.tensor(array, device="cuda", dtype=torch.float32))[0]

                with torch.no_grad():
                    descriptor = describe(score, witness("zero", 0.),
                                          [witness("minus_probe", -EPS), witness("plus_probe", EPS)])
                    parent, minus, plus = descriptor["costs"].cpu().numpy().astype(float)
                    np.testing.assert_allclose(parent, ad_parent, atol=1e-6, rtol=1e-5)
                    fd = []
                    for eps in (.001, EPS):
                        lo = minus if eps == EPS else float(score(witness("fd_minus", -eps).array()))
                        hi = plus if eps == EPS else float(score(witness("fd_plus", eps).array()))
                        value = float(central_derivative(lo, hi, eps))
                        fd.append({"epsilon": eps, "derivative": value,
                                   "absolute_error": abs(value - derivative),
                                   "within_tolerance": abs(value - derivative) <= .001 + .05 * abs(derivative)})
                    labels = []
                    for displacement in BANK:
                        value, pred = evaluate(torch.tensor(witness("heldout", displacement).array(),
                                                            device="cuda", dtype=torch.float32))
                        labels.append(float(value))
                        features[key + f"_pred_{displacement}"] = pred.cpu().numpy()
                    predictions = {"parent": np.full(4, parent), "probe_min": np.full(4, min(parent, minus, plus)),
                                   "linear_ad": parent + derivative * BANK,
                                   "quadratic": quadratic_prediction(parent, minus, plus, EPS, BANK)}
                    labels = np.array(labels)
                    maes = {name: float(np.abs(values - labels).mean()) for name, values in predictions.items()}
                    regrets = {name: float(labels[int(np.argmin(values))] - labels.min())
                               for name, values in predictions.items()}
                    # Evaluation only: true successor imagery and positions never enter evaluate().
                    encoded_costs, physical_costs = [], []
                    goal_position = data["traces"][target, 0, camera, 0, :2]
                    for action_index in range(3):
                        encoded = model.encode(rgb_tensor(data["images"][start, action_index, camera, -1]))[0]
                        features[key + f"_real_{action_index}"] = encoded.cpu().numpy()
                        encoded_costs.append(float((encoded - goal).square().mean()))
                        physical_costs.append(float(np.linalg.norm(
                            data["traces"][start, action_index, camera, -1, :2] - goal_position)))
                    predicted_costs = [parent, float(labels[-1]), float(labels[0])]
                rows.append({"root": 1880 + root, "camera": camera, "ad_derivative": derivative, "fd": fd,
                             "descriptor_actions": descriptor["actions"],
                             "descriptor_effects": descriptor["effects"].cpu().tolist(),
                             "heldout_raw_x": BANK.tolist(), "heldout_model_costs": labels.tolist(),
                             "predictions": {k: v.tolist() for k, v in predictions.items()},
                             "model_mae": maes, "model_selection_regret": regrets,
                             "predicted_costs_zero_plus_minus": predicted_costs,
                             "real_encoded_costs_zero_plus_minus": encoded_costs,
                             "physical_distances_zero_plus_minus": physical_costs,
                             "model_physical_regret": physical_costs[int(np.argmin(predicted_costs))] - min(physical_costs),
                             "encoding_physical_regret": physical_costs[int(np.argmin(encoded_costs))] - min(physical_costs),
                             "zero_physical_regret": physical_costs[0] - min(physical_costs)})
                print(f"{key}: AD={derivative:.6g}, FD_errors={[f['absolute_error'] for f in fd]}, MAE={maes}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if fingerprint != model_fingerprint(model):
        raise ValueError("frozen weights changed")
    root_rows = []
    for root in range(1880, 1884):
        subset = [r for r in rows if r["root"] == root]
        root_mae = {k: float(np.mean([r["model_mae"][k] for r in subset])) for k in predictions}
        root_rows.append({"root": root, "mae": root_mae,
                          "quadratic_beats_both_constants": all(root_mae["quadratic"] < root_mae[k]
                                                               for k in ("parent", "probe_min"))})
    np.savez_compressed(features_path, **features)
    mean_mae = {k: float(np.mean([r["mae"][k] for r in root_rows])) for k in predictions}
    record = manifest(audit, algorithm_version="state_local_effect_v1", protocol_path=PROTOCOL,
                      task_manifest_path=source, receipts=wrapper.receipts)
    record.update(rows=rows, root_summary=root_rows, raw_action_support="constant x, five environment steps",
                  numerical_check_pass=all(any(f["within_tolerance"] for f in r["fd"]) for r in rows),
                  mean_root_mae=mean_mae,
                  local_prediction_dev_exit=(sum(r["quadratic_beats_both_constants"] for r in root_rows) >= 3
                                             and all(mean_mae["quadratic"] < mean_mae[k]
                                                     for k in ("parent", "probe_min"))),
                  local_prediction_winning_roots=sum(r["quadratic_beats_both_constants"] for r in root_rows),
                  features_sha256=file_hash(features_path), frozen_fingerprint=fingerprint,
                  measurement={"predict_calls": len(wrapper.receipts), "backward_calls": len(rows),
                               "extra_encoder_calls": 4 * len(rows), "synchronized_stage_seconds": elapsed},
                  dependencies={str(p): file_hash(p) for p in (
                      Path(__file__), Path(__file__).with_name("pc_action_effect_descriptor.py"),
                      source_meta, source.parent / (source.stem + "_audit.json"))},
                  scope="state-assisted exposed-root diagnostic; no independent or RGB-only promotion")
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({k: record[k] for k in ("numerical_check_pass", "local_prediction_dev_exit",
                                            "local_prediction_winning_roots", "measurement")}, indent=2))


if __name__ == "__main__":
    main()
