"""State-assisted v2 parity using the currently installed local DINO path."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from experiments import pc_input_permissions as v1
from experiments import pc_input_permissions_v2_20261107_codex as v2
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_INPUT_RECEIPT_V2_SMOKE_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_input_receipt_v2_state_smoke_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sys.path.insert(0, str(ROOT / "external/jepa-wms"))
    from app.plan_common.datasets.point_maze_dset import PointMazeDataset
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    folder = ROOT / "data/point_maze_tail1800_1899"
    weights = ROOT / "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar"
    config = ROOT / ("external/jepa-wms/configs/vjepa_wm/mz_sweep/"
                     "mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml")
    local = ROOT / "external/local_dinov2"
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
    fingerprint = model_fingerprint(model)
    import inspect
    from app.plan_common.models import dino
    audit = v1.freeze_audit(v1.STATE, "pointmaze_jepa",
                            [weights, config, folder / "states.pth", folder / "actions.pth",
                             folder / "seq_lengths.pth", __file__, v1.__file__, v2.__file__,
                             inspect.getfile(type(model)), inspect.getfile(type(model.model)),
                             inspect.getfile(PointMazeDataset), inspect.getfile(Preprocessor),
                             inspect.getfile(make_transforms), inspect.getfile(dino), local / "hubconf.py"],
                            proprio_fields=("x", "y", "vx", "vy"),
                            preprocessing="raw RGB /255 and official transform; official archive statistics",
                            clock="context 1; 5 environment steps per normalized 10D packet")
    wrapper = v2.StateAssistedPointMazev2(model, audit)
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
                actual = wrapper.predict(inputs, packet)["visual"]
                direct = model.unroll(model.encode(dict(inputs)), act_suffix=packet)["visual"]
                torch.testing.assert_close(actual, direct, atol=1e-5, rtol=1e-5)
                rows.append({"action": action_index, "camera": camera,
                             "direct_max_error": float((actual - direct).abs().max())})
    if fingerprint != model_fingerprint(model) or len(rows) != len(wrapper.receipts):
        raise ValueError("model or call count changed")
    record = v2.call_manifest(audit, algorithm_version="input_boundary_v2_state_smoke",
                              protocol_path=PROTOCOL, task_manifest_path=data, receipts=wrapper.receipts)
    record.update(rows=rows, unchanged_model_fingerprint=fingerprint,
                  scope="state-assisted GPU parity only; not an episode or gate")
    with OUTPUT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(OUTPUT), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
