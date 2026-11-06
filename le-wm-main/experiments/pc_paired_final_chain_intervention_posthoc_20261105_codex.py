"""Common-suffix and equal-budget replanning for every previously selected chain."""
import json
import time
import traceback
from pathlib import Path

import numpy as np

from experiments import pc_paired_segment_repair_dev_20261105_codex as source

ROOT, fixed = source.ROOT, source.fixed
OUT = ROOT / "results/pc_paired_final_chain_intervention_posthoc_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_PAIRED_FINAL_CHAIN_INTERVENTION_POSTHOC_PROTOCOL_20261105_CODEX.md"
CONTROLS = ("selected", "parent", "zero")


def intervene(predict, task, chain, parent, root, repeat):
    chain, parent = np.asarray(chain, np.float32), np.asarray(parent, np.float32)
    if chain.shape != (15, 2) or parent.shape != (15, 2):
        raise ValueError("incorrect chain shape")
    prefixes = {"selected": chain[:5], "parent": parent[:5], "zero": np.zeros((5, 2), np.float32)}
    _, chain_scores = predict(np.stack((chain, parent)))
    _, prefix_score = predict(chain[:5][None])
    initial_distance = float(np.linalg.norm(np.asarray(task["start"][:2]) - np.asarray(task["goal"])))
    controls = {}
    for name in CONTROLS:
        prefix = prefixes[name]
        tail_zero = np.zeros((10, 2), np.float32)
        common = np.stack((np.concatenate((prefix, tail_zero)), np.concatenate((prefix, chain[5:]))))
        _, scores = predict(common)
        runs = []
        for k in range(3):
            seed = 2051500000 + root*100 + repeat*10 + k
            opt = fixed.optimizer.optimize(lambda actions: predict(actions)[1], prefix,
                                           tail_zero[None], 120, seed=seed, batch_size=12, elites=4)
            runs.append({"seed": seed, "score": float(opt["score"]), "actions": opt["actions"].tolist()})
        controls[name] = {"prefix": prefix.tolist(), "common_scores": scores.tolist(),
                          "tail_improvement": float(scores[0]-scores[1]), "runs": runs,
                          "mean_replan": float(np.mean([r["score"] for r in runs]))}
    result = {"chain_score": float(chain_scores[0]), "parent_score": float(chain_scores[1]),
              "prefix_score": float(prefix_score[0]), "initial_distance": initial_distance,
              "nonprogress_001": bool(prefix_score[0] >= initial_distance-.01),
              "selected_chain": chain.tolist(), "parent_chain": parent.tolist(), "controls": controls}
    for name in ("parent", "zero"):
        common_delta = controls["selected"]["tail_improvement"] - controls[name]["tail_improvement"]
        replan_delta = controls[name]["mean_replan"] - controls["selected"]["mean_replan"]
        result[name + "_comparison"] = {"common_improvement_delta": float(common_delta),
            "replan_score_improvement": float(replan_delta),
            "common_at_least_001": bool(common_delta >= .01),
            "replan_at_least_001": bool(replan_delta >= .01)}
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    audit = json.loads((source.OUT / "audit.json").read_text())
    if audit["status"] != "passed" or audit["rows"] != 64 or \
            audit["summary_sha256"] != source.old.prior.digest(source.OUT / "summary.json"):
        raise ValueError("source audit mismatch")
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    seal, journal = fixed.prior.prior.prior.seal, fixed.prior.prior.prior.journal
    seal.verify(manifest["dependencies"])
    seal.verify(manifest["source_dependencies"])
    source_rows = json.loads((source.OUT / "summary.json").read_text())["rows"]
    selected = [r for r in source_rows if r["arm"] == "pair"]
    if len(selected) != 32 or any(r["status"] != "completed" for r in selected):
        raise ValueError("source selection incomplete")
    paths = [Path(__file__), PROTOCOL, Path(source.__file__), Path(fixed.optimizer.__file__),
             source.OUT / "manifest.json", source.OUT / "summary.json", source.OUT / "audit.json"]
    paths += [source.OUT / f"root{r['root']:02d}_rep{r['repeat']}_pair.json" for r in selected]
    deps = seal.capture(paths)
    OUT.mkdir()
    journal.write_json(OUT / "manifest.json", {"dependencies": deps, "source_dependencies":
                                               manifest["source_dependencies"], "tasks": manifest["tasks"]})
    rows = []
    for r in selected:
        root, rep = r["root"], r["repeat"]
        name = f"root{root:02d}_rep{rep}"
        if json.loads((source.OUT / f"{name}_pair.json").read_text()) != r:
            raise ValueError("source row mismatch")
        parent_source = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        parent = fixed.starts_from_result(parent_source["result"])[0][0]
        recorder = journal.PredictionJournal(
            OUT / f"{name}_calls", fixed.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][root]),
            action_dim=2, packet_length=5, token_cap=10000)
        row, started = {"root": root, "repeat": rep}, time.monotonic()
        try:
            data = intervene(recorder, manifest["tasks"][root], r["result"]["final_actions"], parent, root, rep)
            if abs(data["chain_score"] - r["result"]["final_score"]) > 1e-12:
                raise ValueError("selected chain score differs from source")
            row.update(status="completed", result=data)
        except Exception:
            row.update(status="failed", error=traceback.format_exc())
        row.update(tokens=recorder.tokens, calls=recorder.calls, elapsed_seconds=time.monotonic()-started)
        rows.append(row)
        journal.write_json(OUT / f"{name}.json", row)
        print(name, row["status"], "tokens", row["tokens"], flush=True)
    seal.verify(deps)
    seal.verify(manifest["source_dependencies"])
    journal.write_json(OUT / "summary.json", {"rows": rows})


if __name__ == "__main__":
    print("Posthoc matched suffix and replanning for every selected chain.", flush=True)
    main()
