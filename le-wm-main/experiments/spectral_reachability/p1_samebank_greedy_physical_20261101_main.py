"""Execute the certified same-bank greedy prefix with common and zero tails."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_coherent_intervention_20261101_main as common

NAME = "p1_samebank_greedy_physical_20261101_main"
ROOT = common.panel.ROOT
DOC = ROOT.parent / "docs/D2P_P1_SAMEBANK_GREEDY_PHYSICAL_20261101_MAIN.md"
GREEDY = ROOT / "results/p1_coherent_same_bank_greedy_20261101_codex_RUN_MAIN01/report.json"
GREEDY_SHA = "df39207feda1a2239c2b2bcee120a78680e99a7c91b9393c7572b53731e9cc76"


def run(output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    saved = common.panel.saved
    output = Path(output)
    if not output.is_absolute() or output.parent != ROOT / "results" or output.exists():
        raise ValueError("exclusive output")
    if saved.sha(GREEDY) != GREEDY_SHA:
        raise ValueError("fixed greedy report")
    report = json.loads(GREEDY.read_bytes())
    prepared = Path(report["prepared"])
    binding = GREEDY.parent / "binding.json"
    if saved.sha(prepared) != report["prepare_sha256"] or saved.sha(binding) != report["binding_sha256"]:
        raise ValueError("greedy binding")
    value = json.loads(binding.read_bytes())
    if value != json.loads(prepared.read_bytes()) or report["status"] != "complete" or report["error"]:
        raise ValueError("complete greedy")
    if report["model_before"] != report["model_after"] or report["model_before"] != value["expected_model_hash"]:
        raise ValueError("frozen model")
    result = report["result"]
    scores = np.asarray(result["scores"], float)
    if scores.shape != (8,) or not np.isfinite(scores).all():
        raise ValueError("eight finite scores")
    index = min(range(8), key=lambda i: (scores[i], i))
    prefix = np.asarray(result["selected_prefix"], np.float32)
    if (index != result["greedy_index"] or result["prefixes"] != value["prefixes"]
            or not np.array_equal(prefix, np.asarray(value["prefixes"][index], np.float32))):
        raise ValueError("same-bank greedy argmin")
    ledger = result["ledger"]
    account = saved.consumer.consume(ledger["journal"], ledger["sha256"], result["contract_sha256"],
                                     "structured", "node", "same_bank")
    if account != result["account"] or account["charged_upper_bounds"] != value["expected_cost"]:
        raise ValueError("greedy actual ledger")
    goal, images, sources = common.target.collect()
    sources.update(value["sources"])
    original = json.loads((common.target.DIRECTORY / "report.json").read_bytes())
    panel_path = ROOT / "results/p1_centered_panel_20261101_main_COHERENTMAIN01.json"
    panel = json.loads(panel_path.read_bytes())
    item = next(r for r in panel["rows"] if r["arm"] == "structured")
    source = Path(item["report"])
    saved.check(source.parent, item["report_sha256"])
    tail = np.asarray(json.loads(source.read_bytes())["rows"][0]["executed_raw"], np.float32)[5:]
    schedule = [dict(prefix="same_bank_greedy", suffix=name,
                     commands=np.concatenate((prefix, actions)).tolist())
                for name, actions in (("common", tail), ("zero", np.zeros((15, 2), np.float32)))]
    for path in (GREEDY, prepared, binding, Path(ledger["journal"]), panel_path, source, DOC, Path(__file__).resolve()):
        sources[str(path)] = saved.sha(path)
    if any(saved.sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    output.mkdir()
    with (output / "protocol.json").open("x") as stream:
        json.dump(dict(schedule=schedule, sources=sources, greedy_index=index, neural_calls=0,
                       P1_admission=False), stream, indent=2)
    rows = []
    for slot in schedule:
        row = dict(prefix=slot["prefix"], suffix=slot["suffix"], checkpoints=[], executed_raw=[], status="failed")
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                       with_target=False, render_action=False)
        try:
            body = env.unwrapped
            env.reset(seed=original["reset_seed"])
            env.reset(seed=0, options=dict(state=np.asarray(original["requested_reset"]),
                                          goal_state=np.asarray(original["placeholder_goal"])))
            for step in range(11):
                if step:
                    action = np.asarray(original["raw_history"][step-1], np.float32)
                    _, _, stop, trunc, _ = env.step(action.copy())
                    if stop or trunc or not np.array_equal(body.latest_action, action):
                        raise ValueError("warmup action/stop")
                if step in (0, 5, 10) and not np.array_equal(env.render(), images[step//5]):
                    raise ValueError("warmup RGB")
            common.native_source.attach_goal(body, np.asarray(goal["goal_state"]), images[-1])

            def capture(step):
                pixels = np.asarray(env.render()).copy()
                state = np.asarray(body._get_obs()).copy()
                path = output / f"{slot['suffix']}_{step:02}.png"
                Image.fromarray(pixels).save(path)
                native = common.metrics.qualification.native(goal["goal_state"], state)
                row["checkpoints"].append(dict(step=step, state=state.tolist(),
                    metrics=dict(**common.metrics.metrics(state, goal["goal_state"]),
                                 joint_xy=native["joint_xy_error"], native_success=native["initial_success"]),
                    png=str(path), file_sha256=saved.sha(path),
                    pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
                return state

            if not np.array_equal(capture(0), goal["root_state"]):
                raise ValueError("same root")
            for step, action in enumerate(np.asarray(slot["commands"], np.float32), 1):
                obs, _, stop, trunc, info = env.step(action.copy())
                if not np.array_equal(body.latest_action, action) or not np.array_equal(info["goal_state"], goal["goal_state"]):
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
        except Exception as exc:
            row["error"] = dict(type=type(exc).__name__, message=str(exc))
        finally:
            env.close()
        row["steps"] = len(row["executed_raw"])
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("prefix", "suffix", "status", "steps", "error")}), flush=True)
    if any(saved.sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    with (output / "report.json").open("x") as stream:
        json.dump(dict(rows=rows, sources=sources, protocol_sha256=saved.sha(output / "protocol.json"),
                       neural_calls=0, P1_admission=False, independent_confirmation=False),
                  stream, indent=2, allow_nan=False)
    return all(r["status"] != "failed" for r in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", required=True, action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
