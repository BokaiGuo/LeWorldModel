"""Native-geometry contact-ready pairs, physical collection, then frozen scoring."""
import argparse
import hashlib
import inspect
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "results/pc_geometric_contact_equivalence_dev_20261106_codex"
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")
IMAGES = PREFIX.with_name(PREFIX.name + "_images.npz")
MODEL = PREFIX.with_name(PREFIX.name + "_model.json")
FEATURES = PREFIX.with_name(PREFIX.name + "_features.npz")
PROTOCOL = ROOT.parent / "docs/PC_GEOMETRIC_CONTACT_EQUIVALENCE_DEV_PROTOCOL_20261106_CODEX.md"
COLORS = {"white": (255, 255, 255), "gray": (245, 245, 245)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)


def tasks():
    return [{"position": list(position), "angle": float(angle)}
            for position, angle in itertools.product(
                ((220., 220.), (280., 220.), (220., 280.), (280., 280.)),
                (0., np.pi / 4, np.pi / 2, 3 * np.pi / 4))]


def pose_cost(state, goal):
    state, goal = np.asarray(state), np.asarray(goal)
    delta = np.arctan2(np.sin(state[4] - goal[4]), np.cos(state[4] - goal[4]))
    return float(np.sum(((state[2:4] - goal[2:4]) / 20) ** 2) + (delta / (np.pi / 9)) ** 2)


def geometry(actor):
    circles = list(actor.agent.shapes)
    if len(circles) != 1 or not hasattr(circles[0], "radius"):
        raise ValueError("requires one circular pusher")
    circle = circles[0]
    clearance = min(shape.point_query(actor.agent.position).distance for shape in actor.block.shapes) - circle.radius
    overlaps = []
    for body in (actor.agent, actor.block):
        for shape in body.shapes:
            for hit in actor.space.shape_query(shape):
                if hit.shape.body is body:
                    continue
                depths = [float(point.distance) for point in hit.contact_point_set.points]
                if any(depth < -1e-6 for depth in depths):
                    overlaps.append({"body": "agent" if body is actor.agent else "block",
                                     "other": "agent" if hit.shape.body is actor.agent else
                                     "block" if hit.shape.body is actor.block else "wall",
                                     "depths": depths})
    return {"clearance": float(clearance), "radius": float(circle.radius), "overlaps": overlaps}


def state_for_side(actor, side):
    if side not in (-1, 1):
        raise ValueError("side must be +/-1")
    radius = geometry(actor)["radius"]
    polygons = [np.asarray([[v.x, v.y] for v in shape.get_vertices()]) for shape in actor.block.shapes]
    bar = max(polygons, key=lambda vertices: np.ptp(vertices[:, 0]))
    edge_x = float(bar[:, 0].min() if side == -1 else bar[:, 0].max())
    edge_y = float((bar[:, 1].min() + bar[:, 1].max()) / 2)
    local_center = (edge_x + side * (radius + 1.), edge_y)
    world = actor.block.local_to_world(local_center)
    state = np.asarray(actor._get_obs(), dtype=float)
    state[:2], state[-2:] = [world.x, world.y], 0
    return state


def collect():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from experiments.pc_pusht_reset_hidden_audit_20261105_codex import snapshot

    if PHYSICAL.exists() or IMAGES.exists():
        raise FileExistsError(PREFIX)
    env = gym.make("swm/PushT-v1", resolution=224, relative=True, render_action=False, with_target=True)
    actor = env.unwrapped
    original_callback = actor._handle_collision
    events = []

    def callback(arbiter, space, data):
        original_callback(arbiter, space, data)
        bodies = [shape.body for shape in arbiter.shapes]
        if any(body is actor.agent for body in bodies) and any(body is actor.block for body in bodies):
            events.append(len(arbiter.contact_point_set.points))

    actor._handle_collision = callback
    images, rows = {}, []
    dependencies = {str(path): sha(path) for path in
                    (Path(__file__), PROTOCOL, Path(inspect.getfile(type(actor))), Path(inspect.getfile(snapshot)))}
    print("Geometry-only preflight: 16 roots, no model-based filtering.", flush=True)
    try:
        for root, task in enumerate(tasks()):
            x, y = task["position"]
            theta = task["angle"]
            base = np.array([80., 80., x, y, theta, 0., 0.])
            goal = np.array([80., 80., x + 20., y, theta + np.pi / 12, 0., 0.])
            env.reset(seed=881001 + root, options={"state": base.copy(), "goal_state": goal.copy()})
            side_states = {side: state_for_side(actor, side) for side in (-1, 1)}
            row = {"root": root, "task": task, "goal": goal.tolist(), "conditions": [], "invalid_reasons": []}
            for color_name, color in COLORS.items():
                env.reset(seed=881001 + root, options={"state": goal.copy(), "goal_state": goal.copy()})
                actor.variation_space["background"]["color"].set_value(np.asarray(color, np.uint8))
                before = snapshot(actor)
                goal_rgb = actor.render().copy()
                if not np.array_equal(before, snapshot(actor)) or geometry(actor)["overlaps"]:
                    row["invalid_reasons"].append(f"{color_name}:invalid_goal")
                images[f"r{root}_{color_name}_goal"] = goal_rgb
                for side, desired in side_states.items():
                    condition = {"side": side, "appearance": color_name, "requested_state": desired.tolist(),
                                 "actions": {}, "invalid_reasons": []}
                    for name, sign in (("zero", 0), ("plus", 1), ("minus", -1)):
                        env.reset(seed=881001 + root, options={"state": desired.copy(), "goal_state": goal.copy()})
                        actor.variation_space["background"]["color"].set_value(np.asarray(color, np.uint8))
                        initial, geo = snapshot(actor), geometry(actor)
                        actual = np.asarray(actor._get_obs())
                        delta = actual - desired
                        delta[4] = np.arctan2(np.sin(delta[4]), np.cos(delta[4]))
                        if not np.allclose(delta, 0, atol=1e-6, rtol=0):
                            condition["invalid_reasons"].append(f"{name}:reset_state")
                        if geo["overlaps"] or abs(geo["clearance"] - 1.) > 1e-6:
                            condition["invalid_reasons"].append(f"{name}:geometry")
                        if not np.allclose(initial[[2, 3, 4, 7, 8, 10]], 0, atol=1e-6, rtol=0):
                            condition["invalid_reasons"].append(f"{name}:initial_velocity")
                        initial_rgb = actor.render().copy()
                        if not np.array_equal(snapshot(actor), initial):
                            raise ValueError("render mutated state")
                        action = (sign * .1 * np.array([np.cos(theta), np.sin(theta)])).astype(np.float32)
                        trace = []
                        for _ in range(5):
                            events.clear()
                            obs, _, terminated, truncated, _ = actor.step(action.copy())
                            if truncated:
                                raise ValueError("unexpected truncation")
                            trace.append({"state": obs["state"].tolist(), "snapshot": snapshot(actor).tolist(),
                                          "agent_block_contacts": sum(events), "success": bool(terminated)})
                        final_rgb = actor.render().copy()
                        if name == "zero" and not np.allclose(snapshot(actor), initial, atol=1e-6, rtol=0):
                            condition["invalid_reasons"].append("zero:passive_drift")
                        if sign == -side and not any(item["agent_block_contacts"] for item in trace):
                            condition["invalid_reasons"].append("inward:no_contact")
                        key = f"r{root}_{color_name}_{side}_{name}"
                        images[key + "_initial"], images[key + "_final"] = initial_rgb, final_rgb
                        condition["actions"][name] = {"initial_snapshot": initial.tolist(), "geometry": geo,
                            "physical_action": np.repeat(action[None], 5, axis=0).tolist(), "trace": trace,
                            "initial_sha256": array_sha(initial_rgb), "final_sha256": array_sha(final_rgb),
                            "object_pose_cost": pose_cost(trace[-1]["state"], goal),
                            "full_state_distance": float(np.linalg.norm(np.asarray(trace[-1]["state"]) - goal))}
                    row["conditions"].append(condition)
            for side in (-1, 1):
                w, g = [c for c in row["conditions"] if c["side"] == side]
                for name in ("zero", "plus", "minus"):
                    if w["actions"][name]["trace"] != g["actions"][name]["trace"]:
                        row["invalid_reasons"].append(f"side{side}:appearance_changes_physics")
            for color in COLORS:
                left, right = [c for c in row["conditions"] if c["appearance"] == color]
                a = np.asarray(left["actions"]["zero"]["initial_snapshot"])
                b = np.asarray(right["actions"]["zero"]["initial_snapshot"])
                if not np.allclose(a[5:], b[5:], atol=1e-6, rtol=0):
                    row["invalid_reasons"].append(f"{color}:block_hidden_state_mismatch")
            row["valid"] = not row["invalid_reasons"] and not any(c["invalid_reasons"] for c in row["conditions"])
            rows.append(row)
            print(f"root {root}: physical contract valid={row['valid']}", flush=True)
    finally:
        env.close()
    if dependencies != {path: sha(path) for path in dependencies}:
        raise ValueError("source drift")
    with IMAGES.open("xb") as file:
        np.savez_compressed(file, **images)
    save(PHYSICAL, {"dependencies": dependencies, "images_sha256": sha(IMAGES),
                    "scope": "geometry-generated development roots, not source-qualified independent tasks", "rows": rows})


def analyze():
    import h5py
    import torch
    from sklearn.preprocessing import StandardScaler
    from eval import img_transform
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

    if MODEL.exists() or FEATURES.exists():
        raise FileExistsError(MODEL)
    source = json.loads(PHYSICAL.read_text())
    if source["images_sha256"] != sha(IMAGES):
        raise ValueError("image drift")
    if source["dependencies"] != {path: sha(path) for path in source["dependencies"]}:
        raise ValueError("physical dependency drift")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as file:
        all_actions = np.asarray(file["action"][:])
    scaler = StandardScaler().fit(all_actions[~np.isnan(all_actions).any(axis=1)])
    images = np.load(IMAGES, allow_pickle=False)
    rows, features = [], {}
    print("Frozen model scores validated images/actions only; physical labels used after prediction.", flush=True)
    with torch.inference_mode():
        for row in source["rows"]:
            if not row["valid"]:
                rows.append({"root": row["root"], "valid": False, "conditions": []})
                continue
            root, results = row["root"], []
            for c in row["conditions"]:
                color, side = c["appearance"], c["side"]
                goal = transform(images[f"r{root}_{color}_goal"]).to("cuda")
                goal_emb = model.encode({"pixels": goal[None, None]})["emb"][0, 0]
                actions = {}
                for name, record in c["actions"].items():
                    key = f"r{root}_{color}_{side}_{name}"
                    current = transform(images[key + "_initial"]).to("cuda")
                    realized = transform(images[key + "_final"]).to("cuda")
                    packet = scaler.transform(np.asarray(record["physical_action"])).reshape(1, 1, 1, 10)
                    context = {"pixels": current[None, None, None], "goal": goal[None, None, None],
                               "action": torch.zeros((1, 1, 1, 10), device="cuda")}
                    cost = model.get_cost(context, torch.tensor(packet, dtype=torch.float32, device="cuda"))
                    predicted = context["predicted_emb"][0, 0, -1]
                    real_emb = model.encode({"pixels": realized[None, None]})["emb"][0, 0]
                    initial_emb = context["emb"][0, 0, 0]
                    direct = (predicted - goal_emb).square().sum()
                    if not torch.allclose(cost[0, 0], direct, atol=1e-4, rtol=1e-5):
                        raise ValueError("cost/prediction alignment mismatch")
                    features[key] = torch.stack((initial_emb, predicted, real_emb, goal_emb)).cpu().numpy()
                    actions[name] = {"predicted_cost": float(cost[0, 0]),
                                     "real_encoded_cost": float((real_emb - goal_emb).square().sum()),
                                     "feature_prediction_mse": float((predicted - real_emb).square().mean())}
                results.append({"appearance": color, "side": side, "actions": actions})
            rows.append({"root": root, "valid": True, "conditions": results})
            print(f"root {root}: frozen features and scores recorded", flush=True)
    if fingerprint != model_fingerprint(model):
        raise ValueError("model drift")
    with FEATURES.open("xb") as file:
        np.savez_compressed(file, **features)
    save(MODEL, {"physical_sha256": sha(PHYSICAL), "model_sha256": fingerprint,
                 "features_sha256": sha(FEATURES), "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "analyze"))
    args = parser.parse_args()
    {"collect": collect, "analyze": analyze}[args.mode]()
