"""Separate replay and predicate audit for privileged V5 path diagnostics."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v5_e0_oracle_path_injection_diag_20261104_codex as diag
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUTPUT = diag.OUTPUT.with_name(diag.OUTPUT.stem + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    inputs = [Path(__file__), Path(diag.__file__), diag.PROTOCOL, diag.OUTPUT, diag.WITNESSES,
              diag.source.PLANS, Path(diag.source.exact.__file__), Path(diag.source.exact.wall.__file__)]
    deps = seal.capture(inputs)
    result = json.loads(diag.OUTPUT.read_text())
    seal.verify(result["dependencies"])
    plans = json.loads(diag.source.PLANS.read_text())["rows"]
    witnesses = json.loads(diag.WITNESSES.read_text())["rows"]
    rows = []
    for entry, original, oracle in zip(result["rows"], plans, witnesses, strict=True):
        if entry["root"] != original["root"] or entry["root"] != oracle["root"]:
            raise ValueError("root mismatch")
        task, goal = original["task"], np.asarray(original["task"]["goal"])
        witness = np.asarray(oracle["witnesses"][0]["float32_actions"], dtype=np.float32)
        chain = np.asarray(entry["selected"]["chain"], dtype=np.float32)
        if not np.array_equal(chain, witness) or np.any(np.abs(chain) > 1):
            raise ValueError("witness not dispatched unchanged")
        endpoint = diag.source.exact.endpoints(task, chain[None])[0]
        score = float(np.linalg.norm(endpoint-goal))
        np.testing.assert_allclose(score, entry["dispatch_score"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(endpoint, entry["dispatch_endpoint"], atol=1e-12, rtol=0)
        if score >= .05 or entry["counts"]["prediction_token_samples"] != 1920:
            raise ValueError("score or token mismatch")
        bank = np.asarray([item["prefix"] for item in original["result"]["first_bank"]])
        distances = np.linalg.norm((bank-witness[:5]).reshape(len(bank), -1), axis=1)
        if not np.isclose(np.linalg.norm(diag.source.exact.endpoints(task, witness[None, :5])[0]-goal),
                          entry["prefix_score"], rtol=0, atol=1e-12):
            raise ValueError("prefix score mismatch")
        structure = {"svd": 0, "fd_action_samples": 0, "svd_seconds": 0.0}

        def predict(actions):
            pos = diag.source.exact.endpoints(task, np.asarray(actions, dtype=np.float32))
            return pos, np.linalg.norm(pos-goal, axis=1)

        _, response = diag.v4.response(diag.core.NOMINALS[0], witness[:5], goal, predict, structure)
        for key in ("matrix", "linear_reach_value"):
            np.testing.assert_allclose(response[key], entry["witness_response"][key], atol=1e-12, rtol=0)
        retained = diag.v4.retain(original["result"]["prefix_scores"] + [entry["prefix_score"]],
                                  original["result"]["successor_responses"] + [response])
        if retained != entry["retained_augmented"] or (len(bank) in retained) != entry["prefix_retained"]:
            raise ValueError("retention mismatch")
        rows.append({"root": entry["root"], "distance_to_nearest_generated_prefix_l2": float(distances.min()),
                     "prefix_retained": entry["prefix_retained"], "dispatch_score": score})
    seal.verify(deps)
    diag.source.write(OUTPUT, {"dependencies": deps, "source_sha256": diag.source.sha(diag.OUTPUT),
                              "scope": "separate replay and retention recalculation; shared exact backend",
                              "rows": rows})
    print(f"Audited {len(rows)} injected chains and augmented retention decisions.", flush=True)


if __name__ == "__main__":
    main()
