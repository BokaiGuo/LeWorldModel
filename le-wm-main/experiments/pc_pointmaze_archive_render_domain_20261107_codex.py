"""Two-stage state-assisted archive/render domain diagnostic."""
import json
import sys

import numpy as np
import torch

from experiments.pc_input_permissions import StateAssistedPointMaze, file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import ROOT, load_model
from experiments import pc_pointmaze_same_renderer_three_way_20261107_codex as panel

PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_ARCHIVE_RENDER_DOMAIN_PROTOCOL_20261107_CODEX.md"
PREFIX = ROOT / "results/pc_pointmaze_archive_render_domain_20261107_codex"
INPUT = PREFIX.with_name(PREFIX.name + "_input.npz")
PRED = PREFIX.with_name(PREFIX.name + "_prediction.npz")
PLAN = PREFIX.with_name(PREFIX.name + "_plan.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")


def archive_path(root):
    return ROOT / f"data/point_maze_tail1800_1899/obses/episode_{root:03d}.pth"


def prepare():
    if INPUT.exists():
        raise FileExistsError(INPUT)
    folder = ROOT / "data/point_maze_tail1800_1899"
    states = torch.load(folder / "states.pth", map_location="cpu", weights_only=True).float()
    actions = torch.load(folder / "actions.pth", map_location="cpu", weights_only=True).float()
    images = np.stack([torch.load(archive_path(root), map_location="cpu", weights_only=True)[0].numpy()
                       for root in range(1880, 1884)])
    with INPUT.open("xb") as f:
        np.savez_compressed(f, images=images, states=states[1880:1884, 0].numpy(),
                            actions=actions[1880:1884, :5].numpy(),
                            archive_hashes=np.array([file_hash(archive_path(root)) for root in range(1880, 1884)]))
    print("Sealed frame-zero inputs; no future frames copied.")


def predict():
    if PRED.exists() or PLAN.exists():
        raise FileExistsError("prediction already exists")
    model, dataset, audit = load_model()
    fingerprint = model_fingerprint(model)
    old = json.loads(panel.PLAN.read_text())
    if fingerprint != old["model_sha256"] or audit.fingerprint != old["input_audit_hash"]:
        raise ValueError("sealed model or input audit drift")
    wrapper = StateAssistedPointMaze(model, audit)
    arrays, rows = {}, []
    with np.load(panel.INPUT, allow_pickle=False) as fixed, np.load(INPUT, allow_pickle=False) as source, torch.inference_mode():
        for index, root in enumerate(range(1880, 1884)):
            image = source["images"][index]
            state = torch.tensor(source["states"][index], dtype=torch.float32)
            np.testing.assert_allclose(state.numpy(), fixed["proprio"][index, 0], atol=1e-6, rtol=0)
            inputs = {"visual": rgb_tensor(image), "proprio": state.cuda()[None, None]}
            archive_current = model.encode(inputs)["visual"][:, 0]
            arrays[f"r{index}_context"] = archive_current.cpu().numpy()
            archive_preds = []
            for x in panel.ACTIONS:
                action = torch.tensor([[x, 0.]] * 5)
                packet = model.preprocessor.normalize_actions(action).reshape(1, 1, 10).cuda()
                archive_preds.append(wrapper.predict(inputs, packet)["visual"][-1].cpu().numpy())
            arrays[f"r{index}_counterfactual"] = np.stack(archive_preds)
            raw_action = torch.tensor(source["actions"][index], dtype=torch.float32)
            packet = model.preprocessor.normalize_actions(raw_action).reshape(1, 1, 10).cuda()
            arrays[f"r{index}_factual"] = wrapper.predict(inputs, packet)["visual"][-1].cpu().numpy()
            diffs = []
            for camera in range(2):
                observed = fixed["current"][index, camera].astype(np.float32)
                diffs.append(float(np.abs(image.astype(np.float32) - observed).mean()))
            rows.append({"root": root, "archive_frame0_vs_render_mae_255": diffs,
                         "image_file_sha256": str(source["archive_hashes"][index])})
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    with PRED.open("xb") as f:
        np.savez_compressed(f, **arrays)
    plan = {"track": audit.track_id, "model_sha256": fingerprint, "audit_hash": audit.fingerprint,
            "protocol_sha256": file_hash(PROTOCOL), "script_sha256": file_hash(__file__),
            "fixed_input_sha256": file_hash(panel.INPUT), "archive_input_sha256": file_hash(INPUT),
            "prior_plan_sha256": file_hash(panel.PLAN),
            "prediction_sha256": file_hash(PRED), "rows": rows,
            "actions_sha256": file_hash(dataset.data_path / "actions.pth"),
            "states_sha256": file_hash(dataset.data_path / "states.pth"),
            "receipts": wrapper.receipts}
    with PLAN.open("x") as f:
        json.dump(plan, f, indent=2, allow_nan=False)
    print(f"Sealed predictions for {len(rows)} exposed roots.")


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan = json.loads(PLAN.read_text())
    for key, path in (("prediction_sha256", PRED), ("protocol_sha256", PROTOCOL),
                      ("script_sha256", __file__), ("fixed_input_sha256", panel.INPUT),
                      ("archive_input_sha256", INPUT), ("prior_plan_sha256", panel.PLAN)):
        if plan[key] != file_hash(path):
            raise ValueError(f"sealed {key} drift")
    for row in plan["rows"]:
        if row["image_file_sha256"] != file_hash(archive_path(row["root"])):
            raise ValueError("archive image drift")
    old = json.loads(panel.RESULT.read_text())
    if old["prediction_sha256"] != file_hash(panel.PRED) or old["actual_sha256"] != file_hash(panel.ACTUAL):
        raise ValueError("prior prediction or truth drift")
    model, _, audit = load_model()
    if model_fingerprint(model) != plan["model_sha256"] or audit.fingerprint != plan["audit_hash"]:
        raise ValueError("model drift")
    rows = []
    with (np.load(PRED, allow_pickle=False) as pred, np.load(panel.PRED, allow_pickle=False) as rendered_pred,
          np.load(panel.ACTUAL, allow_pickle=False) as actual, torch.inference_mode()):
        for item in plan["rows"]:
            root, index = item["root"], item["root"] - 1880
            archive = torch.load(archive_path(root), map_location="cpu", weights_only=True)
            factual = model.encode(rgb_tensor(archive[5].numpy()))[0].cpu().numpy()
            mse = lambda a, b: float(np.mean((np.asarray(a, dtype=np.float64) - b) ** 2))
            comparisons = []
            for camera in range(2):
                key = f"r{index}_v{camera}"
                truth = actual[key]
                archive_cf = pred[f"r{index}_counterfactual"]
                render_cf = rendered_pred[key + "_pred"]
                comparisons.append({"camera": camera,
                                    "archive_prediction_to_render_successor_mse": mse(archive_cf, truth),
                                    "render_prediction_to_render_successor_mse": mse(render_cf, truth)})
            rows.append({"root": root, "pixels": item["archive_frame0_vs_render_mae_255"],
                         "factual_archive_prediction_mse": mse(pred[f"r{index}_factual"], factual),
                         "factual_archive_persistence_mse": mse(pred[f"r{index}_context"], factual),
                         "cameras": comparisons})
    if model_fingerprint(model) != plan["model_sha256"]:
        raise ValueError("model changed during evaluation")
    with RESULT.open("x") as f:
        json.dump({"track": "official_state_assisted", "rows": rows, "plan_sha256": file_hash(PLAN),
                   "actual_sha256": file_hash(panel.ACTUAL), "prior_result_sha256": file_hash(panel.RESULT),
                   "scope": "exposed diagnostic, actual encoded outcomes evaluation only"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "predict", "evaluate"):
        raise SystemExit("expected prepare|predict|evaluate")
    {"prepare": prepare, "predict": predict, "evaluate": evaluate}[sys.argv[1]]()
