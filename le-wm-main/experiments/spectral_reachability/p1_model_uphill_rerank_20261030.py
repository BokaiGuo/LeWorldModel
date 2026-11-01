"""Post-hoc model-only selection on a fixed, saved CUDA candidate inventory."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/p1_joint_prefix_suffix_cuda_20261030_RUN01.json"
SOURCE_SHA = "de579ff2c8e4873e74dfa89bf0ebf86f4af8fc1a6803bbe904f6427a95a94d54"
REPORT = ROOT / "results/agent_p1_joint_vreach_host_cuda_20261029_v1_cuda_MAIN01/report.json"
REPORT_SHA = "25eb16c1e2515f98d4e8ace4fafbf1a8d5cbc74d79762421ba248387d1f272c5"
PROTOCOL = ROOT.parent / "docs/D2P_P1_MODEL_UPHILL_RERANK_PROTOCOL_20261030_MAIN.md"
OUT = ROOT / "results/p1_model_uphill_rerank_20261030_RUN01.json"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path, digest):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "input path")
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == digest, "input hash")
    return json.loads(raw)


def mse(state, goal):
    need(len(state) == len(goal) == 192, "latent shape")
    need(all(math.isfinite(x) for x in state + goal), "finite latent")
    return math.fsum((x - y) ** 2 for x, y in zip(state, goal, strict=True)) / 192


def choose(group, root_score):
    feasible = [r for r in group if r["prefix_mse"] >= root_score]
    return min(feasible, key=lambda r: (r["terminal_mse"], r["candidate_index"], r["iteration"])) if feasible else None


def build(saved, report, prepared):
    need(saved["schema"] == "p1_joint_prefix_suffix_cuda_20261030"
         and saved["source_report_sha256"] == REPORT_SHA and saved["training"] is False
         and saved["P1_admission"] is False and saved["model_before"] == saved["model_after"]
         and saved["calls"] == {"predict": 2400, "action_encoder": 2400, "reverse": 640},
         "source contract")
    need(len(saved["rows"]) == 160 and len(saved["winners"]) == 8
         and len(report["rows"]) == 8 and len(prepared["requests"]) == 4, "roster")
    groups = []
    for arm_index in range(8):
        source_arm = report["rows"][arm_index]
        task = prepared["requests"][arm_index // 2]["task"]
        goal = task["goal"]
        root_score = mse(task["root_context"]["latent_history"][-1], goal)
        group = []
        for index, candidate in enumerate(saved["rows"][arm_index * 20:(arm_index + 1) * 20]):
            need(candidate["candidate_index"] == index
                 and (candidate["slot"], candidate["arm"]) == (source_arm["slot"], source_arm["arm"])
                 and len(candidate["trajectory"]) == 5, "candidate identity")
            for iteration, record in enumerate(candidate["trajectory"]):
                need(record["iteration"] == iteration, "iteration identity")
                actions = record["actions"]
                need(len(actions) == 3 and all(len(packet) == 5 for packet in actions)
                     and all(len(action) == 2 and all(math.isfinite(v) and -1 <= v <= 1 for v in action)
                             for packet in actions for action in packet), "bounded actions")
                terminal = mse(record["terminal_latent"], goal)
                need(math.isclose(terminal, record["score"], rel_tol=2e-6, abs_tol=1e-7),
                     "terminal score mismatch")
                group.append({"candidate_index": index, "iteration": iteration,
                              "prefix_mse": mse(record["prefix_latent"], goal),
                              "terminal_mse": terminal, "actions": actions})
        selected = choose(group, root_score)
        unconstrained = min(group, key=lambda r: (r["terminal_mse"], r["candidate_index"], r["iteration"]))
        winner = saved["winners"][arm_index]
        need((winner["slot"], winner["arm"], winner["candidate_index"], winner["best_iteration"]) ==
             (source_arm["slot"], source_arm["arm"], unconstrained["candidate_index"], unconstrained["iteration"]),
             "unconstrained winner disagreement")
        groups.append({"slot": source_arm["slot"], "arm": source_arm["arm"],
                       "root_mse": root_score, "candidate_records": group,
                       "feasible_count": sum(r["prefix_mse"] >= root_score for r in group),
                       "selected": selected, "unconstrained": unconstrained})
    return {"schema": "p1_model_uphill_rerank_20261030", "groups": groups,
            "source_sha256": SOURCE_SHA, "source_report_sha256": REPORT_SHA,
            "P1_admission": False, "physical_release": False, "independent_confirmation": False,
            "limitations": "Post-hoc known-root model-only latent screening; not task non-progress or capability evidence."}


def main():
    saved, report = read(SOURCE, SOURCE_SHA), read(REPORT, REPORT_SHA)
    prepared = read(report["prepared"], report["prepare_sha256"])
    result = build(saved, report, prepared)
    result["protocol_sha256"] = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUT.open("xb") as f:
        f.write(raw)
    print(json.dumps({"path": str(OUT), "sha256": hashlib.sha256(raw).hexdigest(),
                      "feasible_counts": [g["feasible_count"] for g in result["groups"]]}))


if __name__ == "__main__":
    main()
