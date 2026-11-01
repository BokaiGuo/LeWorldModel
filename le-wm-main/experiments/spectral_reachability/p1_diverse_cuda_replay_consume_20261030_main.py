"""Independently check saved diverse CUDA replay arrays and source bindings."""
import argparse
import json

import numpy as np
from . import p1_diverse_cuda_replay_20261030_main as replay


def verify(value, manifest, rows, backtrack, report, prepared):
    if (value["schema"] != replay.NAME or value["calls"] != dict(predict=96, action_encoder=96)
            or value["model_before"] != report["model_before"] or value["model_before"] != value["model_after"]
            or value["training"] is not False or value["physical_release"] is not False
            or value["P1_admission"] is not False or manifest["rows"] != rows
            or manifest["files"] != replay.bindings() or len(value["rows"]) != 16):
        raise ValueError("replay contract")
    for i, (row, source) in enumerate(zip(value["rows"], rows, strict=True)):
        if any(row[k] != v for k, v in source.items()):
            raise ValueError("row identity")
        arm, index = i // 2, row["candidate_index"]
        candidate = report["rows"][arm]["result"]["candidates"][index]
        saved = backtrack["rows"][20 * arm + index]
        task = prepared["requests"][arm // 2]["task"]
        ends = np.asarray(row["segmented_endpoints"], np.float32)
        whole = np.asarray(row["whole_endpoints"], np.float32)
        tokens = np.asarray(row["action_tokens"], np.float32)
        raw = np.frombuffer(bytes.fromhex(row["raw15_le_float32_hex"]), dtype="<f4").reshape(3, 10)
        mean = np.tile(np.asarray(task["action_mean"], np.float32), 5)
        std = np.tile(np.asarray(task["action_std"], np.float32), 5)
        if (ends.shape != (3, 192) or whole.shape != ends.shape or tokens.shape != (3, 10)
                or not np.isfinite(ends).all() or not np.isfinite(whole).all()
                or not np.isfinite(tokens).all() or np.any(std <= 0)
                or not np.array_equal(tokens, (raw - mean) / std) or len(row["contexts"]) != 3):
            raise ValueError("saved arrays or normalization")
        z = np.asarray(task["root_context"]["latent_history"], np.float32)
        h = np.asarray(task["root_context"]["action_history"], np.float32)
        for j, context in enumerate(row["contexts"]):
            z = np.concatenate((z[1:], ends[j:j + 1]))
            h = np.concatenate((h[1:], tokens[j:j + 1]))
            if not np.array_equal(z, np.asarray(context["latent_history"], np.float32)) or not np.array_equal(
                    h, np.asarray(context["action_history"], np.float32)):
                raise ValueError("saved context")
        exact = dict(whole_segmented=bool(np.array_equal(whole, ends)),
                     source_prefix=bool(np.array_equal(ends[0], np.asarray(candidate["mid"]["latent_history"][-1], np.float32))),
                     source_endpoint=bool(np.array_equal(ends[2], np.asarray(saved["final"]["endpoint"], np.float32))),
                     source_prefix_action_history=bool(np.array_equal(
                         np.asarray(row["contexts"][0]["action_history"], np.float32),
                         np.asarray(candidate["mid"]["action_history"], np.float32))))
        if row["exact"] != exact:
            raise ValueError("saved equality mismatch")
    all_exact = all(all(row["exact"].values()) for row in value["rows"])
    if value["complete"] is not all_exact:
        raise ValueError("completion mismatch")
    return dict(rows=16, all_exact=all_exact, independently_checked_whole_arrays=True,
                network_recomputed=False, physical_release=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha", required=True)
    args = parser.parse_args()
    value = replay.prior.one.saved.read_verified(args.result, args.sha256)
    manifest = replay.prior.one.saved.read_verified(args.prepared, args.prepare_sha)
    if value["manifest"] != args.prepared or value["manifest_sha256"] != args.prepare_sha:
        raise ValueError("manifest binding")
    rows, backtrack, report, prepared = replay.sources()
    print(json.dumps(verify(value, manifest, rows, backtrack, report, prepared), sort_keys=True))


if __name__ == "__main__":
    main()
