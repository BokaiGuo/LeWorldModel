"""Post-result local-response attribution; does not revise prospective outcome."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_fixed_terminal_e0_20261104_codex as feedback
from experiments import pc_response_temporal_v3_prospective_mechanism_20261104_codex as source

PROTOCOL = source.ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_MECHANISM_RESPONSE_ATTRIBUTION_PROTOCOL_20261104_CODEX.md"
OUTPUT = source.PREFIX.with_name(source.PREFIX.name + "_response_attribution.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = json.loads(source.INTERVENTION.read_text())
    rebuild = json.loads(source.REBUILD.read_text())
    if (data["dependencies"] != source.dependencies()
            or rebuild["intervention_sha256"] != source.base.sha(source.INTERVENTION)):
        raise ValueError("source seal mismatch")
    axes = np.eye(10).reshape(10, 5, 2)
    rows = []
    for row in data["rows"]:
        if not row["nonprogress"]:
            continue
        task, chain = row["task"], np.asarray(row["selected_chain_float32"])
        nominals = {"zero": np.zeros((5, 2)), "selected_suffix_clipped": np.clip(chain[5:10], -.9, .9)}
        controls = {}
        for name in ("selected", "zero", "greedy"):
            prefix = np.asarray(row["controls"][name]["prefix"])
            state, blocked = feedback.step_physics(task["start"], prefix)
            matrices = {}
            for nominal_name, nominal in nominals.items():
                for epsilon in (.01, .05, .1):
                    probes = np.concatenate((nominal[None] + epsilon * axes, nominal[None] - epsilon * axes))
                    if np.max(np.abs(probes)) > 1.00000001:
                        raise ValueError("probe outside action domain")
                    tails = np.concatenate((probes, np.zeros((20, 5, 2))), axis=1)
                    joined = np.concatenate((np.broadcast_to(prefix, (20, 5, 2)), tails), axis=1)
                    endpoints = source.base.exact.endpoints(task, joined.astype(np.float32))
                    matrices[f"{nominal_name}/{epsilon}"] = ((endpoints[:10] - endpoints[10:]) / (2 * epsilon)).T.tolist()
            np.testing.assert_allclose(matrices["zero/0.05"], row["controls"][name]["successor_response"],
                                       rtol=0, atol=1e-12)
            controls[name] = {"successor_state": state, "blocked": blocked, "matrices": matrices}
        gaps = {name: {key: float(np.linalg.norm(np.asarray(matrix) - np.asarray(controls[name]["matrices"][key])))
                       for key, matrix in controls["selected"]["matrices"].items()}
                for name in ("zero", "greedy")}
        rows.append({"root": row["root"], "controls": controls, "response_gaps": gaps})
    if len(rows) != 6:
        raise ValueError("unexpected nonprogress root count")
    source.base.write(OUTPUT, {"protocol_sha256": source.base.sha(PROTOCOL),
                              "source_sha256": source.base.sha(source.INTERVENTION),
                              "source_rebuild_sha256": source.base.sha(source.REBUILD),
                              "script_sha256": source.base.sha(Path(__file__)), "rows": rows})
    print("Saved post-result multi-scale local-response attribution for all six roots.", flush=True)


if __name__ == "__main__":
    main()
