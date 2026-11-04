"""Short RGB-guarded physical feedback versus fixed and zero packet controls."""
import argparse
import importlib
import inspect
import os

import numpy as np

from . import p1_dev305_contact_packet_physical_20261102 as prior

base, need, ROOT = prior.base, prior.need, prior.ROOT
NAME = "p1_dev305_contact_rgb_feedback_20261102"
AUTHORITY = "DEV305_CONTACT_RGB_FEEDBACK_PHYSICAL"
SOURCE_SHA = "f847c3d1699d2607aa9d7c6c2db5e5dec8ab21ea47102c7d456c00dcc804bb57"


def choose_action(mode, current_direction, initial_direction):
    if mode not in ("rgb_feedback", "fixed", "zero"):
        raise ValueError("mode")
    direction = np.asarray(current_direction if mode == "rgb_feedback" else initial_direction, dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1, atol=1e-5):
        raise ValueError("direction")
    scale = {"rgb_feedback": .2, "fixed": .5, "zero": 0.0}[mode]
    return np.asarray(scale * direction, dtype="<f4")


def protocol():
    old = base.read(ROOT / "results" / f"{prior.NAME}_PREP_DEV01.json", SOURCE_SHA)
    need(old == prior.protocol(), "source")
    roles = []
    for item in old["roles"]:
        if item["candidate_origin"] != "toward":
            continue
        initial = np.asarray(item["tail"][0], dtype=float) * 2
        need(np.isclose(np.linalg.norm(initial), 1, atol=1e-5), "initial direction")
        for mode in ("rgb_feedback", "fixed", "zero"):
            roles.append({"arm": item["arm"].split("__")[0] + "__" + mode,
                          "mode": mode, "initial_direction": initial.tolist(),
                          "history": item["history"], "states": item["states"],
                          "checkpoint": item["checkpoint"]})
    need(len(roles) == 9, "roster")
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "source_sha256": SOURCE_SHA, "seed": old["seed"], "roles": roles,
            "rule": "up to five steps; RGB visibility checked at every frame; stop on failure; feedback=.2 current direction; fixed=.5 initial direction; zero=0",
            "candidate_scope": "RGB-only geometric development control, no simulator state or contact event in action selection",
            "classification": "post-outcome known root; not a model-derived program or confirmation",
            "p1_admission": False, "independent_confirmation": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol() and not os.path.lexists(output), "contract")
    spec = base.read(path, digest)
    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    native = prior.engine.planner.prior.replay.compare.compass.source.base.base
    need(base.sha(inspect.getsourcefile(cls)) == native.native.ENV_SHA, "environment")
    store = native.Store(output)
    store.emit("started", contract_sha256=digest)
    rows, errors = [], []
    for role in spec["roles"]:
        row = {"arm": role["arm"], "mode": role["mode"], "status": "running",
               "transitions": [], "observations": []}
        env = None
        try:
            env = cls(render_mode="rgb_array", resolution=224, with_target=False)
            env.reset(seed=spec["seed"])
            for action in np.zeros((10, 2), dtype="<f4"):
                _, _, term, trunc, _ = env.step(action.copy())
                need(not term and not trunc and np.array_equal(env.latest_action, action), "warmup")
            for i, action in enumerate(np.asarray(role["history"], dtype="<f4")):
                obs, _, term, trunc, _ = env.step(action.copy())
                need(not term and not trunc and np.array_equal(env.latest_action, action)
                     and np.array_equal(obs["state"], role["states"][i]), "history replay")
            row["observations"].append(native.capture(store, env, role["arm"], "feedback", 0, role["checkpoint"]))
            for step in range(1, 6):
                pixels = np.asarray(env.render())
                screen = prior.rgb_source.visibility.inspect(pixels)
                if not screen["eligible"]:
                    row["status"] = "visibility_guard_stop"
                    break
                geometry = prior.rgb_source.run.coherent.base.first.engine._geometry(pixels)
                action = choose_action(role["mode"], geometry["contact_direction_raw_xy"], role["initial_direction"])
                obs, reward, term, trunc, _ = env.step(action.copy())
                need(np.array_equal(env.latest_action, action), "dispatch")
                row["transitions"].append({"step": step, "action": action.tolist(), "state": obs["state"].tolist(),
                                           "reward": float(reward), "terminated": bool(term), "truncated": bool(trunc),
                                           "native_contact_points": int(env.n_contact_points),
                                           "direction_from_rgb": geometry["contact_direction_raw_xy"]})
                row["observations"].append(native.capture(store, env, role["arm"], "feedback", step))
                if term or trunc:
                    row["status"] = "native_early_stop"
                    break
            else:
                row["status"] = "fixed_horizon_completed"
            row["final_visibility"] = prior.rgb_source.visibility.inspect(np.asarray(env.render()))
        except Exception as exc:
            row.update({"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)[:1000]}})
            errors.append({"arm": role["arm"], **row["error"]})
        finally:
            if env is not None:
                env.close()
        store.json(role["arm"] + ".row.json", row)
        store.emit("branch_closed", arm=role["arm"], status=row["status"])
        rows.append(row)
    store.emit("finished", errors=errors)
    return store.json("report.json", {"schema": NAME + ".report.v1", "contract_sha256": digest,
                                      "status": "recorded" if not errors else "failed",
                                      "errors": errors, "rows": rows,
                                      "journal_sha256": base.sha(store.path / "journal.jsonl"),
                                      "p1_admission": False, "independent_confirmation": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contract-sha")
    parser.add_argument("--authority")
    args = parser.parse_args()
    need(args.tag.isascii() and args.tag.isalnum(), "tag")
    path = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(path, protocol()))
    else:
        print(execute(path, args.contract_sha, ROOT / "results" / f"{NAME}_RUN_{args.tag}", args.authority))
