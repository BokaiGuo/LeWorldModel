"""Recompute saved replay normalization, contexts and endpoint comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
from . import p1_joint_cuda_replay_20261030 as replay


def verify(value, manifest, rows, joint, backtrack, report, prepared):
    if (value["schema"] != replay.NAME or value["calls"] != {"predict": 96, "action_encoder": 96}
            or value["model_before"] != value["model_after"]
            or value["model_before"] != report["model_before"] or manifest["rows"] != rows
            or manifest["files"] != replay.files() or len(value["rows"]) != 16
            or value["training"] is not False or value["physical_release"] is not False
            or value["P1_admission"] is not False):
        raise ValueError("replay contract")
    for i, (row, source) in enumerate(zip(value["rows"], rows, strict=True)):
        if any(row[k] != v for k, v in source.items()):
            raise ValueError("row identity")
        arm, index = i // 2, row["candidate_index"]
        task = prepared["requests"][arm // 2]["task"]
        raw = np.frombuffer(bytes.fromhex(row["raw15_le_float32_hex"]), dtype="<f4").reshape(3, 10)
        mean = np.tile(np.asarray(task["action_mean"], np.float32), 5)
        std = np.tile(np.asarray(task["action_std"], np.float32), 5)
        tokens = np.asarray(row["action_tokens"], np.float32)
        ends = np.asarray(row["segmented_endpoints"], np.float32)
        whole = np.asarray(row["whole_endpoints"], np.float32)
        if (ends.shape != (3, 192) or whole.shape != ends.shape or tokens.shape != (3, 10)
                or not np.isfinite(ends).all() or not np.isfinite(whole).all()
                or not np.isfinite(tokens).all() or (std <= 0).any()
                or not np.array_equal(tokens, (raw - mean) / std) or len(row["contexts"]) != 3):
            raise ValueError("saved arrays or normalization")
        z = np.asarray(task["root_context"]["latent_history"], np.float32)
        h = np.asarray(task["root_context"]["action_history"], np.float32)
        for j, context in enumerate(row["contexts"]):
            z = np.concatenate((z[1:], ends[j:j+1]))
            h = np.concatenate((h[1:], tokens[j:j+1]))
            if not np.array_equal(z, np.asarray(context["latent_history"], np.float32)) or not np.array_equal(
                    h, np.asarray(context["action_history"], np.float32)):
                raise ValueError("saved context")
        if row["role"] == "incumbent":
            prefix = report["rows"][arm]["result"]["candidates"][index]["mid"]["latent_history"][-1]
            terminal = backtrack["rows"][20 * arm + index]["final"]["endpoint"]
        else:
            selected = joint["rows"][20 * arm + index]
            record = selected["trajectory"][selected["best_iteration"]]
            prefix, terminal = record["prefix_latent"], record["terminal_latent"]
        exact = {"whole_segmented": bool(np.array_equal(whole, ends)),
                 "source_prefix": bool(np.array_equal(ends[0], np.asarray(prefix, np.float32))),
                 "source_endpoint": bool(np.array_equal(ends[-1], np.asarray(terminal, np.float32)))}
        if row["exact"] != exact or not all(exact.values()):
            raise ValueError("replay comparison mismatch")
    if value["complete"] is not True:
        raise ValueError("incomplete")
    return {"rows": 16, "all_exact": True, "normalization_and_context_verified": True,
            "network_recomputed": False, "physical_release": False, "P1_admission": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--prepare-sha", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    path = Path(a.output)
    if not path.is_absolute() or path.parent != replay.ROOT / "results" or path.exists():
        raise ValueError("exclusive output")
    value = replay.old.one.saved.read_verified(a.result, a.sha256)
    manifest = replay.old.one.saved.read_verified(a.prepared, a.prepare_sha)
    if value["manifest"] != a.prepared or value["manifest_sha256"] != a.prepare_sha:
        raise ValueError("manifest binding")
    rows, joint, backtrack, report, prepared = replay.sources()
    result = verify(value, manifest, rows, joint, backtrack, report, prepared)
    result.update(result_sha256=a.sha256, prepare_sha256=a.prepare_sha,
                  code_sha256=replay.old.digest(Path(__file__)))
    print(json.dumps({"sha256": replay.old.one.saved.write(path, result), **result}))


if __name__ == "__main__":
    main()
