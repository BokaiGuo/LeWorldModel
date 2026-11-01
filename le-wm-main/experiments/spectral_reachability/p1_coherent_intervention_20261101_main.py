"""Five frozen prefixes crossed with a common15 suffix and zero15."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_centered_target_cuda_20261101_main as target
from . import p1_constructive_moving_history_source_20261027_v1 as native_source
from . import p1_centered_panel_20261101_main as panel
from . import p1_return_variants_analysis_20261101_main as metrics

NAME = "p1_coherent_intervention_20261101_main"
DOC = panel.ROOT.parent / "docs/D2P_P1_COHERENT_INTERVENTION_PROTOCOL_20261101_MAIN.md"


def run(output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    output = Path(output)
    if not output.is_absolute() or output.parent != panel.ROOT / "results" or output.exists():
        raise ValueError("exclusive output")
    entry, images, sources = target.collect()
    root = json.loads((target.DIRECTORY / "report.json").read_bytes())
    summary_path = panel.ROOT / "results/p1_centered_panel_20261101_main_COHERENTMAIN01.json"
    summary = json.loads(summary_path.read_bytes())
    if [r["arm"] for r in summary["rows"]] != list(panel.ARMS):
        raise ValueError("all four source arms")
    prefixes, complete = {}, {}
    for item in summary["rows"]:
        path = Path(item["report"])
        if panel.saved.sha(path) != item["report_sha256"]:
            raise ValueError("sealed report")
        panel.saved.check(path.parent, item["report_sha256"])
        row = json.loads(path.read_bytes())["rows"][0]
        raw = np.asarray(row["executed_raw"], np.float32)
        if raw.shape != (20, 2) or not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1:
            raise ValueError("complete issued actions")
        prefixes[item["arm"]] = raw[:5].copy()
        complete[item["arm"]] = raw
        sources[str(path)] = item["report_sha256"]
    prefixes["zero"] = np.zeros((5, 2), np.float32)
    suffixes = dict(common_structured=complete["structured"][5:].copy(),
                    zero=np.zeros((15, 2), np.float32))
    for path in (summary_path, DOC, Path(__file__).resolve(), Path(native_source.__file__).resolve()):
        sources[str(path)] = panel.saved.sha(path)
    schedule = [dict(prefix=name, suffix=kind,
                     commands=np.concatenate((prefix, suffix)).tolist())
                for name, prefix in prefixes.items() for kind, suffix in suffixes.items()]
    output.mkdir()
    with (output / "protocol.json").open("x") as stream:
        json.dump(dict(schedule=schedule, sources=sources, P1_admission=False,
                       diagnostic_only=True, neural_calls=0), stream, indent=2)
    results = []
    for slot in schedule:
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                       with_target=False, render_action=False)
        row = dict(prefix=slot["prefix"], suffix=slot["suffix"], status="failed",
                   checkpoints=[], executed_raw=[])
        try:
            body = env.unwrapped
            env.reset(seed=root["reset_seed"])
            env.reset(seed=0, options=dict(state=np.asarray(root["requested_reset"]),
                                          goal_state=np.asarray(root["placeholder_goal"])))
            for step in range(11):
                if step:
                    action = np.asarray(root["raw_history"][step-1], np.float32)
                    _, _, stop, trunc, _ = env.step(action.copy())
                    if stop or trunc or not np.array_equal(body.latest_action, action):
                        raise ValueError("warmup stop/action mismatch")
                if step in (0, 5, 10) and not np.array_equal(env.render(), images[step//5]):
                    raise ValueError("exact warmup RGB")
            native_source.attach_goal(body, np.asarray(entry["goal_state"]), images[-1])

            def capture(step):
                state = np.asarray(body._get_obs()).copy()
                pixels = np.asarray(env.render()).copy()
                path = output / f"{slot['prefix']}_{slot['suffix']}_{step:02}.png"
                Image.fromarray(pixels).save(path)
                native = metrics.qualification.native(entry["goal_state"], state)
                row["checkpoints"].append(dict(step=step, state=state.tolist(),
                    metrics=dict(**metrics.metrics(state, entry["goal_state"]),
                                 joint_xy=native["joint_xy_error"], native_success=native["initial_success"]),
                    png=str(path), file_sha256=panel.saved.sha(path),
                    pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
                return state

            state = capture(0)
            if not np.array_equal(state, entry["root_state"]):
                raise ValueError("exact translated root")
            for step, action in enumerate(np.asarray(slot["commands"], np.float32), 1):
                obs, _, stop, trunc, info = env.step(action.copy())
                if not np.array_equal(body.latest_action, action) or not np.array_equal(info["goal_state"], entry["goal_state"]):
                    raise ValueError("actual action/goal")
                row["executed_raw"].append(action.tolist())
                state = np.asarray(obs["state"])
                outside = not bool(np.all((state[:4] >= 0) & (state[:4] <= 512)))
                if step % 5 == 0 or stop or trunc or outside:
                    capture(step)
                if stop or trunc or outside:
                    row.update(status="native_stopped" if stop or trunc else "canvas_censored",
                               terminated=bool(stop), truncated=bool(trunc))
                    break
            else:
                row["status"] = "completed20"
            row["steps"] = len(row["executed_raw"])
        except Exception as exc:
            row["error"] = dict(type=type(exc).__name__, message=str(exc))
        finally:
            env.close()
        results.append(row)
        with (output / f"{slot['prefix']}_{slot['suffix']}.json").open("x") as stream:
            json.dump(row, stream, indent=2, allow_nan=False)
        print(json.dumps({k: row.get(k) for k in ("prefix", "suffix", "status", "steps", "error")}), flush=True)
    if any(panel.saved.sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    with (output / "report.json").open("x") as stream:
        json.dump(dict(rows=results, sources=sources, protocol_sha256=panel.saved.sha(output / "protocol.json"),
                       P1_admission=False, independent_confirmation=False, neural_calls=0),
                  stream, indent=2, allow_nan=False)
    return all(r["status"] != "failed" for r in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", required=True, action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
