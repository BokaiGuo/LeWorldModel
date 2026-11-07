"""State-assisted smoke with explicit local DINO allowlist.

Separate entry preserves the already-sealed RGB smoke source and evidence.
"""
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from experiments.pc_dual_track_smoke_20261106_codex import ROOT, PROTOCOL
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, freeze_audit, manifest
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint


def load_model():
    sys.path.insert(0, str(ROOT / "external/jepa-wms"))
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
    local = (ROOT / "external/local_dinov2").resolve()
    original = torch.hub.load

    def local_load(repo, name, *args, **kwargs):
        if repo != "facebookresearch/dinov2" and Path(repo).resolve() != local:
            raise ValueError("unapproved hub dependency")
        kwargs["source"] = "local"
        return original(str(local), name, *args, **kwargs)

    torch.hub.load = local_load
    try:
        model = init_module(str(weights.parent), weights.name, cfg["model"], torch.device("cuda"),
                            action_dim=2, proprio_dim=4, preprocessor=pre, cfgs_data=cfg["data"],
                            wrapper_kwargs={"ctxt_window": 1}).eval().requires_grad_(False)
    finally:
        torch.hub.load = original
    paths = [weights, config, folder / "states.pth", folder / "actions.pth", folder / "seq_lengths.pth",
             Path(__file__), Path(inspect.getfile(StateAssistedPointMaze)), Path(inspect.getfile(type(model))),
             Path(inspect.getfile(type(model.model))), Path(inspect.getfile(PointMazeDataset)),
             Path(inspect.getfile(Preprocessor)), Path(inspect.getfile(make_transforms)), local / "hubconf.py"]
    audit = freeze_audit(STATE, "pointmaze_jepa", paths, proprio_fields=("x", "y", "vx", "vy"),
                        preprocessing="raw RGB /255 then official transform; full archive statistics",
                        clock="context 1; five environment steps = 0.25s per normalized 10D packet")
    return model, dataset, audit


def main():
    output = ROOT / "results/pc_dual_track_smoke_20261106_codex" / STATE / "result.json"
    if output.exists():
        raise FileExistsError(output)
    print("State-assisted GPU input parity; not RGB-only or planning evidence.", flush=True)
    model, dataset, audit = load_model()
    before = model_fingerprint(model)
    wrapper = StateAssistedPointMaze(model, audit)
    source = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz"
    rows = []
    with np.load(source, allow_pickle=False) as data, torch.inference_mode():
        for index, action in enumerate((0., .1, -.1)):
            raw = torch.tensor([[action, 0.]] * 5)
            packet = ((raw - dataset.action_mean) / dataset.action_std).reshape(1, 1, 10).cuda()
            for camera in (0, 1):
                image = data["images"][0, index, camera, 0].copy()
                state = data["traces"][0, index, camera, 0].copy()
                inputs = {"visual": torch.tensor(image).permute(2, 0, 1)[None, None].cuda(),
                          "proprio": torch.tensor(state, dtype=torch.float32)[None, None].cuda()}
                predicted = wrapper.predict(inputs, packet)["visual"]
                direct = model.unroll(model.encode(dict(inputs)), act_suffix=packet)["visual"]
                torch.testing.assert_close(predicted, direct, atol=1e-5, rtol=1e-5)
                if not torch.isfinite(predicted).all():
                    raise ValueError("nonfinite predictions")
                rows.append({"action": index, "camera": camera, "shape": list(predicted.shape),
                             "direct_max_error": float((predicted - direct).abs().max())})
    if before != model_fingerprint(model):
        raise ValueError("weights changed")
    record = manifest(audit, algorithm_version="input_boundary_smoke_v1", protocol_path=PROTOCOL,
                      task_manifest_path=source, receipts=wrapper.receipts)
    record.update(rows=rows, unchanged_model_fingerprint=before,
                  scope="state-assisted interface parity only; local compatibility path, not official equivalence")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
