"""Compare native MuJoCo rendering with fixed, exposed archive frames.

This is a visual compatibility probe, not an environment or planner result.
It extracts the official XML builder without importing legacy gym/d4rl, then
renders recorded states using a fixed camera; it does not tune to these frames.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "external/jepa-wms/evals/simu_env_planning/envs/pointmaze_env"
DATA = ROOT / "data/point_maze_tail1800_1899"
OUTPUT = ROOT / "results/a0_pointmaze_native_render_probe_20261102_DEV01.json"
ROOTS = (1880, 1881, 1882, 1883)
FRAMES = (0, 5, 10)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_xml() -> str:
    """Use the released builder and its dependency, without loading gym."""
    spec = importlib.util.spec_from_file_location("pointmaze_dynamic_mjc", ENV / "dynamic_mjc.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official XML builder dependency")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (ENV / "maze_model.py").read_text()
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"parse_maze", "point_maze"}]
    if {node.name for node in functions} != {"parse_maze", "point_maze"}:
        raise RuntimeError("official XML builder not found")
    namespace = {"np": np, "MJCModel": module.MJCModel, "WALL": 10, "EMPTY": 11, "GOAL": 12}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(ENV / "maze_model.py"), "exec"), namespace)
    generated = namespace["point_maze"]("#####\\#GOO#\\###O#\\#OOO#\\#####")
    with generated.asfile() as stream:
        return stream.read()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    xml = official_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    states = torch.load(DATA / "states.pth", map_location="cpu", weights_only=True).numpy()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = 90
    camera.elevation = -90
    rows = []
    with mujoco.Renderer(model, height=224, width=224) as renderer:
        for root in ROOTS:
            image_path = DATA / "obses" / f"episode_{root:03d}.pth"
            images = torch.load(image_path, map_location="cpu", weights_only=True).numpy()
            for frame in FRAMES:
                state = states[root, frame]
                data.qpos[:] = state[:2]
                data.qvel[:] = state[2:]
                mujoco.mj_forward(model, data)
                # Source with_target=False moves the target offscreen.
                site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
                if site < 0:
                    raise RuntimeError("official target site missing")
                data.site_xpos[site] = [11.0, 11.0, 0.0]
                renderer.update_scene(data, camera=camera)
                image = renderer.render().copy()
                reference = images[frame]
                difference = np.abs(image.astype(np.int16) - reference.astype(np.int16))
                rows.append({
                    "root": root, "frame": frame,
                    "archive_sha256": digest(image_path),
                    "render_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "archive_rgb_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                    "mean_absolute_pixel_error": float(difference.mean()),
                    "fraction_nonidentical_pixels": float(np.any(difference != 0, axis=2).mean()),
                    "max_absolute_channel_error": int(difference.max()),
                })
    result = {
        "schema": "a0_pointmaze_native_render_probe.v1",
        "scope": "exposed archive states; visual compatibility only; no action selection or P1 admission",
        "camera": {"source": "mujoco.mjv_defaultFreeCamera plus official prepare_for_render azimuth/elevation",
                   "azimuth": camera.azimuth, "elevation": camera.elevation,
                   "distance": camera.distance, "lookat": camera.lookat.tolist()},
        "backend": {"mujoco": mujoco.__version__, "official_legacy_binding_identical": False},
        "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "source_sha256": digest(ENV / "maze_model.py"),
        "dependency_sha256": digest(ENV / "dynamic_mjc.py"),
        "states_sha256": digest(DATA / "states.pth"),
        "script_sha256": digest(Path(__file__)),
        "rows": rows,
    }
    raw = (json.dumps(result, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest(),
                      "mae_range": [min(row["mean_absolute_pixel_error"] for row in rows),
                                    max(row["mean_absolute_pixel_error"] for row in rows)]}))


if __name__ == "__main__":
    main()
