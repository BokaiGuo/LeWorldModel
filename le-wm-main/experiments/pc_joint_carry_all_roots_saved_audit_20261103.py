"""Offline binding checks for all-root joint-plan carry follow-up."""
import hashlib
import json

import numpy as np

from experiments import pc_joint_carry_all_roots_20261103 as run

EXPECTED = "f5977bd71e636a316935ecfce1045968adbcf6c70b100de5c825bfc20f2fe849"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify(report, entries, arrays):
    need(len(report["rows"]) == 51 and len(entries) == 17, "roster")
    mean = np.asarray(report["action_mean"], dtype=np.float32)
    std = np.asarray(report["action_std"], dtype=np.float32)
    need(mean.shape == std.shape == (2,) and np.isfinite(mean).all() and np.all(std > 0), "normalizer")
    need(report["model_before"] == report["model_after"], "model identity")
    for i, entry in enumerate(entries):
        root, role = entry["root"], entry["role"]
        for j, mode in enumerate(("old", "zero_init", "carry_init")):
            row = report["rows"][3*i+j]
            need((row["root"], row["role"], row["mode"]) == (root, role, mode), "row order")
            name = f"r{root}_{role}_{mode}"
            actions, history = arrays[name+"_actions"], arrays[name+"_history"]
            need(np.array_equal(actions, arrays["sequences"][3*i+j]), "sequence binding")
            need(np.array_equal(arrays["starts"][3*i+j], entry["start"]), "start binding")
            need(np.array_equal(actions[:7], entry["actions"][:7]), "prefix actions")
            need(np.array_equal(history[:8], entry["history"][:8]), "prefix RGB")
            if mode == "old":
                need(np.array_equal(actions[7:], entry["plan"][5:]), "old tail")
            else:
                base = f"r{root}_{role}_{mode}_batch3"
                candidates, scores = arrays[base+"_candidates"], arrays[base+"_scores"]
                need(candidates.shape == (1, 2, 1, 20) and scores.shape == (2,), "final batch")
                need(float(scores[0]) == row["prior_score"] and float(scores[1]) == row["refined_score"], "scores")
                winner = int(scores[1] < scores[0])
                need(bool(winner) == row["refined"], "CEM decision")
                decoded = np.clip(candidates[0, winner, 0].reshape(10, 2) * std + mean, -1, 1).astype(np.float32)
                need(np.array_equal(decoded, actions[7:]), "decoded CEM actions")
                if mode == "zero_init":
                    need(np.array_equal(actions, entry["actions"]) and np.array_equal(history, entry["history"]),
                         "source zero-init replay")
            need(bool(np.array_equal(actions[7:], entry["plan"][5:])) == row["same_as_old"], "old action flag")
            for t in (2, 7, 17):
                pos = arrays[f"cp{t}_eval_only"][3*i+j]
                distance = float(np.linalg.norm(pos[:2] - run.h.GOAL[:2]))
                need(abs(distance - row[f"cp{t}_distance"]) < 1e-12, "evaluation distance")
    return {"branches": len(entries), "trajectories": len(report["rows"]), "feedback_decisions": 34,
            "status": "passed"}


def main():
    need(sha(run.OUTPUT) == EXPECTED, "result receipt hash")
    report, source = (json.loads(p.read_text()) for p in (run.OUTPUT, run.SOURCE))
    need(sha(run.SOURCE) == report["source_sha256"], "source receipt hash")
    need(sha(run.ARRAYS) == report["arrays_sha256"], "result arrays hash")
    need(sha(run.SOURCE.with_suffix(".npz")) == source["arrays_sha256"], "source arrays hash")
    need(sha(run.PROTOCOL) == report["protocol_sha256"] and sha(run.Path(run.__file__)) == report["script_sha256"],
         "code/protocol hash")
    with np.load(run.SOURCE.with_suffix(".npz"), allow_pickle=False) as original, \
            np.load(run.ARRAYS, allow_pickle=False) as arrays:
        print(json.dumps(verify(report, run.roster(source, original), arrays), sort_keys=True))


if __name__ == "__main__":
    main()
