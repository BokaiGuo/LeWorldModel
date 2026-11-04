"""Separate predicate and stored-action audit of the post-hoc intervention."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v22_modecount_same_chain_posthoc_20261104_codex as run
from experiments.pc_v5_suffix_budget_curve_diag_20261104_codex import save_json
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

RESULT = run.OUT / "interventions.json"
REBUILD = run.OUT / "audit.json"
OUT = run.OUT / "predicate_audit.json"
sha, exact = seal.digest, run.mechanism.exact
names = ("selected", "zero", "greedy", "lookahead", "random")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    result, audit = json.loads(RESULT.read_text()), json.loads(REBUILD.read_text())
    if audit["interventions_sha256"] != sha(RESULT) or audit["attempted"] != 8:
        raise ValueError("rebuild audit mismatch")
    deps = seal.capture([Path(__file__), RESULT, REBUILD, run.PROTOCOL])
    roots = []
    for binding in result["rows"]:
        root = binding["root"]
        if binding["status"] != "completed":
            roots.append({"root": root, "base": False, "all": False})
            continue
        array_path = run.OUT / binding["arrays"]
        row = binding["data"]
        if sha(array_path) != binding["arrays_sha256"]:
            raise ValueError("root binding mismatch")
        goal, controls, response = np.asarray(row["task"]["goal"]), row["controls"], row["response"]
        with np.load(array_path, allow_pickle=False) as data:
            if data["own_actions"].shape != (5, 3, 481, 15, 2) or data["common_actions"].shape != (5, 6, 15, 2):
                raise ValueError("wrong stored array dimensions")
            for index, name in enumerate(names):
                prefix = np.asarray(controls[name]["prefix"], np.float32)
                own = data["own_actions"][index]
                np.testing.assert_array_equal(own[:, :, :5], np.broadcast_to(prefix, (3, 481, 5, 2)))
                np.testing.assert_array_equal(data["common_actions"][index, :, :5], np.broadcast_to(prefix, (6, 5, 2)))
                for rep in range(3):
                    pos = exact.endpoints(row["task"], own[rep])
                    scores = np.linalg.norm(pos-goal, axis=1)
                    np.testing.assert_allclose(scores, data["own_scores"][index, rep], atol=1e-12, rtol=0)
                    if float(np.min(scores)) != controls[name]["own"][rep]["score"]:
                        raise ValueError("CEM best differs from scored actions")
                common = exact.endpoints(row["task"], data["common_actions"][index])
                values = np.linalg.norm(common-goal, axis=1)
                np.testing.assert_allclose(values, controls[name]["common_distances"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(values-values[0], controls[name]["relative_zero"], atol=1e-12, rtol=0)

        def check(others):
            if not (response["nonprogress"] and response["selected_prefix_nonzero"] and response["selected_tail_nonzero"]):
                return False
            for name in others:
                if controls["selected"]["relative_zero"][-1] >= controls[name]["relative_zero"][-1] - 1e-10:
                    return False
                if controls["selected"]["mean_own"] >= controls[name]["mean_own"] - 1e-10:
                    return False
                for eps in ("0.01", "0.05"):
                    selected = response["controls"]["selected"]["scales"][eps]
                    other = response["controls"][name]["scales"][eps]
                    matrix = float(np.linalg.norm(np.asarray(selected["matrix"])-np.asarray(other["matrix"])))
                    if selected["distance_gradient"] is None or other["distance_gradient"] is None:
                        return False
                    gradient = float(np.linalg.norm(np.asarray(selected["distance_gradient"])-np.asarray(other["distance_gradient"])))
                    if matrix <= 1e-6 or gradient <= 1e-6:
                        return False
            return all(response["controls"][name]["scales"][eps]["branch_stable"]
                       for name in ("selected", *others) for eps in ("0.01", "0.05"))

        base_pass, all_pass = check(("zero", "greedy")), check(("zero", "greedy", "lookahead", "random"))
        if base_pass != row["base_intersection"] or all_pass != row["all_intersection"]:
            raise ValueError("predicate differs from independently reconstructed conditions")
        roots.append({"root": root, "base": base_pass, "all": all_pass})
    if len(roots) != 8:
        raise ValueError("incomplete roots")
    seal.verify(deps)
    save_json(OUT, {"dependencies": deps, "result_sha256": sha(RESULT), "verified_roots": len(roots),
                    "base_pass": sum(x["base"] for x in roots), "all_pass": sum(x["all"] for x in roots),
                    "scope": "separate predicate code and stored-action replay, same exact backend"})
    print(f"Verified 8 mechanism predicates; base={sum(x['base'] for x in roots)}, all={sum(x['all'] for x in roots)}.")


if __name__ == "__main__":
    main()
