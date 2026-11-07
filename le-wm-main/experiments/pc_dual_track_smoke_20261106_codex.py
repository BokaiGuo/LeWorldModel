"""Separate-process invocation per track; direct-call parity, not control evidence."""
import argparse
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from experiments.pc_input_permissions import (
    RGB, STATE, RGBLeWM, StateAssistedPointMaze, freeze_audit, manifest,
)
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_DUAL_TRACK_IMPLEMENTATION_PROTOCOL_20261106_CODEX.md"


def rgb():
    import h5py
    from sklearn.preprocessing import StandardScaler
    from eval import img_transform
    from experiments.checkpoint_compat import load_compatible_pretrained, _resolve_local

    data = ROOT / "results/pc_geometric_contact_equivalence_dev_20261106_codex_images.npz"
    stats = Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5")
    weights, config = _resolve_local("pusht/lewm")
    model = load_compatible_pretrained("pusht/lewm", device="cuda")
    before = model_fingerprint(model)
    with h5py.File(stats) as stream:
        values = np.asarray(stream["action"])
    scaler = StandardScaler().fit(values[~np.isnan(values).any(axis=1)])
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    audit = freeze_audit(RGB, "lewm", [weights, config, stats, Path(__file__),
                        Path(inspect.getfile(RGBLeWM)), Path(inspect.getfile(type(model))),
                        Path(inspect.getfile(load_compatible_pretrained)), Path(inspect.getfile(img_transform))],
                        preprocessing="img_transform 224; full finite train action StandardScaler",
                        clock="five environment actions per 10D packet; explicit initial empty history placeholder")
    wrapper = RGBLeWM(model, audit)
    rows = []
    with np.load(data, allow_pickle=False) as images, torch.inference_mode():
        for name, action in (("zero", 0.), ("plus", .1), ("minus", -.1)):
            key = f"r0_white_-1_{name}"
            current = transform(images[key + "_initial"]).cuda()
            goal = transform(images["r0_white_goal"]).cuda()
            actions = np.tile([action, 0.], (5, 1))
            packet = torch.tensor(scaler.transform(actions).reshape(1, 1, 1, 10),
                                  device="cuda", dtype=torch.float32)
            inputs = {"pixels": current[None, None, None], "goal": goal[None, None, None],
                      "action": torch.zeros_like(packet)}
            cost, _ = wrapper.score(inputs, packet)
            direct = model.get_cost(dict(inputs), packet)
            torch.testing.assert_close(cost, direct, atol=1e-5, rtol=1e-5)
            if not torch.isfinite(cost).all():
                raise ValueError("nonfinite RGB cost")
            rows.append({"action": name, "cost": cost.cpu().tolist(),
                         "direct_max_error": float((cost - direct).abs().max())})
    if before != model_fingerprint(model):
        raise ValueError("weights changed")
    return wrapper, data, rows, before


def state():
    import yaml
    sys.path.insert(0, str(ROOT / "external/jepa-wms"))
    original_load = torch.hub.load
    local = ROOT / "external/local_dinov2"

    def local_load(repo, model, *args, **kwargs):
        if repo == "facebookresearch/dinov2":
            return original_load(str(local), model, *args, source="local", **kwargs)
        raise ValueError("unapproved model hub dependency")

    torch.hub.load = local_load
    try:
        from app.plan_common.datasets.point_maze_dset import PointMazeDataset
        from app.plan_common.datasets.preprocessor import Preprocessor
        from app.plan_common.datasets.transforms import make_transforms
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

        folder = ROOT / "data/point_maze_tail1800_1899"
        weights = ROOT / "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
        config = ROOT / ("external/jepa-wms/configs/vjepa_wm/mz_sweep/"
                         "mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml")
        cfg = yaml.safe_load(config.read_text())
        dataset = PointMazeDataset(data_path=str(folder), normalize_action=True)
        transform = make_transforms(img_size=224, normalize=cfg["data_aug"]["normalize"],
                                    random_horizontal_flip=False, random_resize_aspect_ratio=(1., 1.),
                                    random_resize_scale=(1., 1.), reprob=0., auto_augment=False,
                                    motion_shift=False)
        pre = Preprocessor(dataset.action_mean, dataset.action_std, dataset.state_mean, dataset.state_std,
                           dataset.proprio_mean, dataset.proprio_std, transform)
        model = init_module(str(weights.parent), weights.name, cfg["model"], torch.device("cuda"),
                            action_dim=2, proprio_dim=4, preprocessor=pre, cfgs_data=cfg["data"],
                            wrapper_kwargs={"ctxt_window": 1}).eval().requires_grad_(False)
    finally:
        torch.hub.load = original_load
    before = model_fingerprint(model)
    audit = freeze_audit(STATE, "pointmaze_jepa",
                        [weights, config, folder / "states.pth", folder / "actions.pth",
                         folder / "seq_lengths.pth", Path(__file__), Path(inspect.getfile(StateAssistedPointMaze)),
                         Path(inspect.getfile(type(model))), Path(inspect.getfile(type(model.model))),
                         Path(inspect.getfile(PointMazeDataset)), Path(inspect.getfile(Preprocessor)),
                         Path(inspect.getfile(make_transforms)), local / "hubconf.py"],
                        proprio_fields=("x", "y", "vx", "vy"),
                        preprocessing="raw RGB /255 then official transform; official dataset archive statistics",
                        clock="context 1; 5 environment steps = 0.25s per normalized 10D packet")
    wrapper = StateAssistedPointMaze(model, audit)
    data = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz"
    rows = []
    with np.load(data, allow_pickle=False) as payload, torch.inference_mode():
        for action_index, action in enumerate((0., .1, -.1)):
            raw = torch.tensor([[action, 0.]] * 5)
            packet = ((raw - dataset.action_mean) / dataset.action_std).reshape(1, 1, 10).cuda()
            for camera in (0, 1):
                image = payload["images"][0, action_index, camera, 0]
                current = payload["traces"][0, action_index, camera, 0]
                inputs = {"visual": torch.tensor(image.copy()).permute(2, 0, 1)[None, None].cuda(),
                          "proprio": torch.tensor(current.copy(), dtype=torch.float32)[None, None].cuda()}
                predicted = wrapper.predict(inputs, packet)["visual"]
                direct = model.unroll(model.encode(dict(inputs)), act_suffix=packet)["visual"]
                torch.testing.assert_close(predicted, direct, atol=1e-5, rtol=1e-5)
                if not torch.isfinite(predicted).all():
                    raise ValueError("nonfinite state-assisted prediction")
                rows.append({"action": action_index, "camera": camera, "shape": list(predicted.shape),
                             "direct_max_error": float((predicted - direct).abs().max())})
    if before != model_fingerprint(model):
        raise ValueError("weights changed")
    return wrapper, data, rows, before


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=(RGB, STATE), required=True)
    args = parser.parse_args()
    output = ROOT / "results/pc_dual_track_smoke_20261106_codex" / args.track / "result.json"
    if output.exists():
        raise FileExistsError(output)
    print(f"Input boundary and direct-call parity only: {args.track}", flush=True)
    wrapper, data, rows, fingerprint = rgb() if args.track == RGB else state()
    record = manifest(wrapper.audit, algorithm_version="input_boundary_smoke_v1",
                      protocol_path=PROTOCOL, task_manifest_path=data, receipts=wrapper.receipts)
    record.update(rows=rows, unchanged_model_fingerprint=fingerprint,
                  scope="interface parity only; no scientific gate promotion")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"track": args.track, "rows": rows, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
