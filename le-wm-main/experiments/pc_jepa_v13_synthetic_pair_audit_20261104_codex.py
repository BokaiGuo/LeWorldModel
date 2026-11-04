"""Check actual initial-bank, carry and final-pair wiring in sealed smoke outputs."""
import json

import numpy as np

from experiments import pc_jepa_v13_synthetic_smoke_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    out = source.OUT / "pair_audit.json"
    if out.exists():
        raise FileExistsError(out)
    manifest, runs, audit = (json.loads((source.OUT / name).read_text())
                             for name in ("manifest.json", "runs.json", "audit.json"))
    seal.verify(manifest["dependencies"])
    if audit["runs_sha256"] != seal.digest(source.OUT / "runs.json"):
        raise ValueError("smoke audit binding")
    rows = []
    for run in runs["rows"]:
        arm, step = run["arm"], run["step"]
        if run["returncode"] != 0:
            rows.append({"arm": arm, "step": step, "status": "worker_failed"})
            continue
        base = source.OUT / f"{arm}_{step}"
        path = base / "output/decision.json"
        if seal.digest(path) != run["decision_sha256"]:
            raise ValueError("worker output hash")
        data = json.loads(path.read_text())
        pair = data["evaluations"][-1]
        actions, scores = np.asarray(pair["actions"], np.float32), np.asarray(pair["scores"], float)
        chosen = np.asarray(data["decision"]["selected"]["chain"], np.float32)
        if actions.shape != (2, 15-5*step, 2):
            raise ValueError("pair shape")
        if not np.array_equal(actions[0, 5:], actions[1, 5:]) or np.any(actions[1, :5] != 0):
            raise ValueError("zero-prefix pair mismatch")
        if arm == "response":
            compiler = data["decision"]["compiler"]
            index = int(compiler["abstained"])
            np.testing.assert_array_equal(actions[0], np.asarray(compiler["incumbent"], np.float32))
            if compiler["paired_scores"] != pair["scores"]:
                raise ValueError("response pair scores")
            if compiler["global_proposals"]["choice"] != int(np.argmin(data["evaluations"][0]["scores"])):
                raise ValueError("response bank choice")
            if step:
                with np.load(base / "input/context.npz", allow_pickle=False) as f:
                    carry = f["carry"]
                np.testing.assert_array_equal(np.asarray(data["evaluations"][0]["actions"], np.float32)[1], carry)
        else:
            index = data["decision"]["pair_choice"]
            np.testing.assert_array_equal(actions[0], np.asarray(data["decision"]["incumbent"], np.float32))
            if data["decision"]["paired_scores"] != pair["scores"]:
                raise ValueError("CEM pair scores")
            if data["decision"]["global_bank"]["choice"] != int(np.argmin(data["evaluations"][0]["scores"])):
                raise ValueError("CEM bank choice")
        if index not in (0, 1) or not np.array_equal(chosen, actions[index]):
            raise ValueError("selected chain does not match pair")
        # Equal-valued but identical chains allow either label; otherwise the
        # implementation must select a minimal pair score.
        if scores[index] > scores.min():
            raise ValueError("nonminimum pair selected")
        rows.append({"arm": arm, "step": step, "status": "passed", "pair_choice": index})
    source.write(out, {"runs_sha256": seal.digest(source.OUT / "runs.json"),
        "audit_sha256": seal.digest(source.OUT / "audit.json"), "rows": rows,
        "scope": "saved synthetic interface decisions; no physical control"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
