"""Compare actual PointMaze data transitions with official simulator semantics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_pointmaze_packet_clock_replay_20261106_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_PACKET_CLOCK_REPLAY_PROTOCOL_20261106_CODEX.md"


def backend(request_path, xml_path, output_path):
    import mujoco_py
    with np.load(request_path, allow_pickle=False) as request:
        states, actions = request["states"], request["actions"]
    model = mujoco_py.load_model_from_xml(Path(xml_path).read_text())
    sim = mujoco_py.MjSim(model)
    traces, clocks = [], []
    for state, sequence in zip(states, actions):
        sim.reset()
        sim.data.qpos[:] = state[:2]
        sim.data.qvel[:] = state[2:]
        sim.forward()
        trace, clock = [], []
        for action in sequence:
            sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
            sim.forward()
            sim.data.ctrl[:] = np.clip(action, -1, 1)
            for _ in range(5):
                sim.step()
            trace.append(np.r_[sim.data.qpos, sim.data.qvel].copy())
            clock.append(float(sim.data.time))
        traces.append(trace)
        clocks.append(clock)
    np.savez_compressed(output_path, traces=traces, clocks=clocks,
                        simulator_dt=float(model.opt.timestep))


def main():
    import torch
    import yaml
    from experiments.pc_input_permissions import file_hash
    from experiments.a0_pointmaze_native_render_probe_20261102 import DATA, official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    if OUT.exists():
        raise FileExistsError(OUT)
    sys.path.insert(0, str(ROOT / "external/jepa-wms"))
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.point_maze_dset import PointMazeDataset
    from app.plan_common.datasets.transforms import make_transforms
    source_states, source_actions = DATA / "states.pth", DATA / "actions.pth"
    archive = torch.load(source_states, map_location="cpu", weights_only=True)
    actions = torch.load(source_actions, map_location="cpu", weights_only=True)
    states = archive[1884:1892, :16].numpy()
    raw_actions = actions[1884:1892, :15].numpy()
    print("Replay 8 real action histories and compare each recorded next state.", flush=True)
    with tempfile.TemporaryDirectory(prefix="pc_clock_") as tmp:
        folder = Path(tmp)
        request, xml, response = folder / "request.npz", folder / "scene.xml", folder / "replay.npz"
        np.savez_compressed(request, states=states[:, 0], actions=raw_actions)
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(Path(__file__).resolve()), "backend",
                        str(request), str(xml), str(response)], env=env, check=True, timeout=180)
        with np.load(response, allow_pickle=False) as data:
            difference = np.abs(data["traces"] - states[:, 1:])
            clocks = data["clocks"].copy()
            dt = float(data["simulator_dt"])
    dataset = PointMazeDataset(data_path=str(DATA), normalize_action=True)
    official = get_data_stats("pointmaze")
    stats = {}
    for name in ("action_mean", "action_std", "proprio_mean", "proprio_std"):
        stats[name] = float(np.max(np.abs(getattr(dataset, name).numpy() - np.asarray(official[name]))))
    normalized = (torch.from_numpy(raw_actions) - dataset.action_mean) / dataset.action_std
    packets = normalized.reshape(8, 3, 10)
    restored = packets.reshape(8, 15, 2) * dataset.action_std + dataset.action_mean
    roundtrip = float((restored - torch.from_numpy(raw_actions)).abs().max())
    cfg_path = ROOT / ("external/jepa-wms/configs/vjepa_wm/mz_sweep/"
                       "mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    kwargs = dict(img_size=224, normalize=cfg["data_aug"]["normalize"], random_horizontal_flip=False,
                  random_resize_aspect_ratio=(1., 1.), reprob=0., auto_augment=False, motion_shift=False)
    local = make_transforms(random_resize_scale=(1., 1.), **kwargs)
    configured = make_transforms(random_resize_scale=tuple(cfg["data_aug"]["random_resize_scale"]), **kwargs)
    sample = torch.load(DATA / "obses/episode_1887.pth", map_location="cpu", weights_only=True)[[0, 5, 10]]
    frames = sample.permute(0, 3, 1, 2).float() / 255.
    torch.manual_seed(12345)
    first = local(frames.clone())
    torch.manual_seed(12345)
    second = configured(frames.clone())
    rows = [{"root": 1884 + i, "max_abs_state_error": float(difference[i].max()),
             "per_step_max_abs_error": difference[i].max(axis=1).tolist()} for i in range(8)]
    result = {"rows": rows, "max_abs_state_error": float(difference.max()),
              "strict_replay_atol_1e_4": bool(difference.max() <= 1e-4),
              "dt_substep": dt, "max_clock_error": float(np.abs(clocks - np.arange(1, 16)*5*dt).max()),
              "stats_max_abs_difference": stats, "action_roundtrip_max_abs_error": roundtrip,
              "configured_resize_scale": cfg["data_aug"]["random_resize_scale"],
              "local_resize_scale": [1., 1.], "transform_max_abs_difference": float((first-second).abs().max()),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
              "source_states_sha256": file_hash(source_states), "source_actions_sha256": file_hash(source_actions),
              "scope": "state-assisted exposed-data engineering diagnostic, not online control"}
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "backend":
        backend(*sys.argv[2:])
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit("unexpected arguments")
