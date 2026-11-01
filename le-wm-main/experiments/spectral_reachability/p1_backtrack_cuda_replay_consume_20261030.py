"""Recheck saved replay identities and arithmetic, without rerunning the network."""
import argparse
import json

import numpy as np
from experiments.spectral_reachability import p1_backtrack_cuda_replay_20261030 as replay


def verify(value, manifest, action, result, report, prepared):
    if (value["schema"] != replay.NAME or value["calls"] != dict(predict=96, action_encoder=96)
            or value["model_before"] != value["model_after"]
            or value["model_before"] != report["model_before"]
            or value["physical_release"] is not False or value["P1_admission"] is not False
            or value["training"] is not False or manifest["files"] != replay.files()):
        raise ValueError("replay contract")
    if len(value["rows"]) != 16:
        raise ValueError("fixed roster")
    checked = []
    for i, (row, item) in enumerate(zip(value["rows"], action["rows"])):
        if (row["slot"], row["arm"], row["role"], row["candidate_index"]) != (
                item["slot"], item["arm"], item["role"], item["candidate_index"]):
            raise ValueError("roster mismatch")
        arm, candidate_index = i // 2, item["candidate_index"]
        candidate = report["rows"][arm]["result"]["candidates"][candidate_index]
        saved = result["rows"][20 * arm + candidate_index]
        task = prepared["requests"][arm // 2]["task"]
        end = np.asarray(row["endpoints"], np.float32)
        tokens = np.asarray(row["action_tokens"], np.float32)
        contexts = row["contexts"]
        if (end.shape != (3, 192) or tokens.shape != (3, 10) or len(contexts) != 3
                or not np.isfinite(end).all() or not np.isfinite(tokens).all()):
            raise ValueError("saved shape")
        z = np.asarray(task["root_context"]["latent_history"], np.float32)
        h = np.asarray(task["root_context"]["action_history"], np.float32)
        for j in range(3):
            z = np.concatenate((z[1:], end[j:j+1]))
            h = np.concatenate((h[1:], tokens[j:j+1]))
            if (not np.array_equal(z, np.asarray(contexts[j]["latent_history"], np.float32))
                    or not np.array_equal(h, np.asarray(contexts[j]["action_history"], np.float32))):
                raise ValueError("saved context mismatch")
        exact = row["exact"]
        if (exact["source_prefix"] != bool(np.array_equal(end[0], np.asarray(candidate["mid"]["latent_history"][-1], np.float32)))
                or exact["source_endpoint"] != bool(np.array_equal(end[2], np.asarray(saved["final"]["endpoint"], np.float32)))
                or exact["source_prefix_action_history"] != bool(np.array_equal(
                    np.asarray(contexts[0]["action_history"], np.float32),
                    np.asarray(candidate["mid"]["action_history"], np.float32)))
                or type(exact["whole_segmented"]) is not bool):
            raise ValueError("saved replay equality mismatch")
        checked.append(exact)
    if value["complete"] is not all(all(v.values()) for v in checked):
        raise ValueError("completion mismatch")
    return dict(rows=16, all_exact=value["complete"], network_recomputed=False,
                whole_path_arrays_independently_checked=False, physical_release=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha", required=True)
    args = parser.parse_args()
    value = replay.one.saved.read_verified(args.result, args.sha256)
    manifest = replay.one.saved.read_verified(args.prepared, args.prepare_sha)
    if value["manifest"] != args.prepared or value["manifest_sha256"] != args.prepare_sha:
        raise ValueError("manifest binding")
    action, result, report, prepared = replay.sources()
    print(json.dumps(verify(value, manifest, action, result, report, prepared), sort_keys=True))


if __name__ == "__main__":
    main()
