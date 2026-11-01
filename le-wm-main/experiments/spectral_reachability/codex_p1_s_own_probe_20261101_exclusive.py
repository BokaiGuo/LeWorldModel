"""Retrospective known-root structured S-prefix / pre-carry H3 own-tail probe."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_common_structured_cp10_host_20261101_exclusive as cp_host
from . import p1_prospective_mechanism_physical_20261101_codex as physical

ROOT = physical.ROOT
NAME = "codex_p1_s_own_probe_20261101_exclusive"
RUN = ROOT / "results/codex_p1_common_structured_cp10_host_20261101_exclusive_RUN_CODEX01"
PREP = ROOT / "results/codex_p1_common_structured_cp10_host_20261101_exclusive_PREP_CODEX01.json"
JOURNAL = ROOT / "results/p1_constructive_moving_history_source_20261027_v1_main01/journal.jsonl"
PIN = {
    PREP: "8509f02776e9d6650e605d4128a76bce1148ebdc4f2e6dc3b068265de8b74b75",
    RUN / "report.json": "6c49744a051ab327ca91c22f0aa5c19261eacff51c472fdf0b1cd4915d80f4cd",
    JOURNAL: "f30790272a2d2bd1eaa0d533e1767577c69c8608d98b4fb3a3705bd9b70490e9",
}


def require(test, message):
    if not test:
        raise ValueError(message)


def choose(rows):
    kinds = {row["kind"]: row["payload"] for row in rows}
    selection = kinds["final_selection"]
    scores = np.asarray(selection["scores"], dtype=np.float64)
    raw = np.asarray(kinds["optimized_candidates"]["raw"], dtype=np.float32)
    require(scores.shape == (8,) and np.isfinite(scores).all(), "scores")
    require(raw.shape == (8, 3, 5, 2) and np.isfinite(raw).all()
            and np.all(np.abs(raw) <= 1), "candidate actions")
    index = selection["index"]
    require(type(index) is int and index == int(np.argmin(scores))
            and np.count_nonzero(scores == scores[index]) == 1, "ambiguous selection")
    require(kinds["first_feedback_selection"]["selected_source"] == "carry", "carry identity")
    return raw[index].reshape(15, 2).tolist(), index, scores.tolist()


def build():
    physical.check_sources({str(k): v for k, v in PIN.items()})
    prepared = json.loads(PREP.read_bytes())
    report = json.loads((RUN / "report.json").read_bytes())
    binding = prepared["cp10_identity"]
    require(binding == report["cp10_goal_binding"], "binding mismatch")
    data = cp_host.host.p3.source_data(prepared["source"])
    rebound = cp_host.cp10.rebind(deepcopy(prepared), data)
    require(rebound == binding, "rebind mismatch")
    ref = data["references"][0]
    row = report["rows"][0]
    points = {p["step"]: p for p in row["checkpoints"]}
    cp0, cp5 = points[0], points[5]
    require(cp0["goal"] == cp5["goal"] == ref["goal"].tolist(), "goal state mismatch")
    require(cp0["goal_rgb_sha256"] == binding["goal_pixel_sha256"], "goal pixel mismatch")
    nodes = report["nodes"]["s0_structured"]
    n0, n5 = nodes[0]["receipt"], nodes[1]["receipt"]
    prefix = row["executed_raw"][:5]
    require(prefix == n0["commands"], "prefix mismatch")
    seal = json.loads((RUN / "s0_structured_b05_seal.json").read_bytes())
    require(seal["identity"]["raw10"][-5:] == prefix
            and seal["identity"]["observation"]["pixel_sha256"] == cp5["pixel_sha256"], "cp5 binding")
    require(seal["receipt"] == n5, "sealed receipt differs from report")
    trace = Path(n5["trace"])
    ledger = Path(n5["ledger"]["journal"])
    require(physical.file_sha(trace) == n5["trace_sha256"]
            and physical.file_sha(ledger) == n5["ledger"]["sha256"], "trace or ledger drift")
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    require(rows[0]["payload"]["task_sha256"] == seal["task_sha256"], "task mismatch")
    own, index, scores = choose(rows)
    require(n5["selected_index"] == 8 and n5["scores"][8] < n5["scores"][0], "actual carry mismatch")
    recipes = [json.loads(line) for line in JOURNAL.read_text().splitlines()]
    recipes = [r for r in recipes if r["kind"] == "placement_recipe"
               and r["phase"] == "construction" and r["slot"] == ref["slot"]]
    require(len(recipes) == 1, "reset recipe mismatch")
    recipe = recipes[0]
    digest = physical.legacy.panel.saved.spec.digest
    require(np.array_equal(np.asarray(cp0["state"]), np.asarray(ref["states"][-1])), "root mismatch")
    require(hashlib.sha256(ref["goal_rgb"].tobytes()).hexdigest() == binding["goal_pixel_sha256"], "goal pixels")
    root = dict(reset_seed=recipe["slot"]["reset_seed"], requested_reset=recipe["requested"],
                placeholder_goal=recipe["original_goal"], goal_state=ref["goal"].tolist(),
                root_state=cp0["state"], root_dynamics_sha256=digest(cp0["snapshot"]),
                warmup_raw=np.asarray(ref["raw_history"], np.float32).tolist())
    images = [np.asarray(x, np.uint8).copy() for x in ref["history"]]
    require(len(images) == 3 and all(x.shape == (224, 224, 3) for x in images), "history RGB")
    slot = dict(role="S_structured_retrospective", tail="precarry_h3",
                commands=prefix + own, cp5_pixel_sha256=cp5["pixel_sha256"],
                cp5_state_identity_sha256=digest(cp5["state"]),
                cp5_dynamics_sha256=digest(cp5["snapshot"]))
    sources = {str(k): v for k, v in PIN.items()}
    for path in (trace, ledger, RUN / "s0_structured_b05_seal.json",
                 Path(__file__).resolve(), Path(physical.__file__).resolve()):
        sources[str(path)] = physical.file_sha(path)
    sources[str(trace)] = n5["trace_sha256"]
    sources[str(ledger)] = n5["ledger"]["sha256"]
    manifest = dict(schema=NAME, root=root, slot=slot, binding=binding, sources=sources,
        warmup_pixel_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in images],
        original_selection=dict(precarry_h3_index=index, scores=scores,
                                actual_carry_index=n5["selected_index"], actual_carry_scores=n5["scores"]),
        original_cp5_cost=n5["account"]["charged_upper_bounds"],
        original_account_status=n5["account"]["status"],
        meaning="retrospective pre-carry full-H3 candidate, not actual carry policy or coherent S",
        prospectively_sealed=False, P1_admission=False, independent_confirmation=False,
        neural_calls_this_probe=0, historical_cost_unknowns=prepared["source"]["historical_cost_unknowns"])
    return manifest, images


def run(output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    manifest, images = build()
    output.mkdir()
    with (output / "manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    sealed_hash = physical.file_sha(output / "manifest.json")
    physical.check_sources(manifest["sources"])
    env = None
    try:
        env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                       with_target=False, render_action=False)
        row = physical.run_branch(env, manifest["root"], images, manifest["slot"], output)
    except Exception as exc:
        row = dict(status="failed", error=dict(type=type(exc).__name__, message=str(exc)), steps=0)
    finally:
        if env is not None:
            env.close()
    physical.check_sources(manifest["sources"])
    require(physical.file_sha(output / "manifest.json") == sealed_hash, "manifest changed")
    report = dict(schema=NAME + ".report", manifest_sha256=sealed_hash, row=row,
                  independent_confirmation=False, P1_admission=False, neural_calls=0)
    with (output / "report.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({"status": row["status"], "steps": row["steps"], "error": row.get("error"),
                      "manifest_sha256": sealed_hash}), flush=True)
    return row["status"] != "failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Retrospective single-branch physical probe; no neural inference.", flush=True)
    with physical.locking.gpu_lock(physical.LOCK):
        ok = run(args.output)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
