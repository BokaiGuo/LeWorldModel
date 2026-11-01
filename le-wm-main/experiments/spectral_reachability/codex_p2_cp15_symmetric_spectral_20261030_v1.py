"""Prepare same-observed-context spectral pair; no CUDA or physical launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import codex_p2_feedback_shared_candidate_20261030_v1 as source
from experiments.spectral_reachability import codex_p2_symmetric_h3_run_20261030_v2 as h3

NAME = "codex_p2_cp15_symmetric_spectral_20261030_v1"
ROOT = Path(__file__).resolve().parents[2]
ARMS = ("joint", "suffix")


def require(value, message):
    if not value:
        raise ValueError(message)


def bind(jobs, pairs):
    require(len(jobs) == 4 and len(pairs) == 4, "four fixed feedback bases")
    result = []
    for index, (job, pair) in enumerate(zip(jobs, pairs, strict=True)):
        require(job["base_index"] == pair["base"] == index and job["policy"] == "F"
                and job["root_slot"] == pair["root"] == index // 2
                and job["arm"] == pair["source_arm"] and job["task_id"] == pair["task_id"]
                and job["task_sha256"] == pair["task_sha256"], "base identity")
        context, goal, prefix = job["input_context"], job["goal"], job["prefix_raw"]
        require(source.source.digest(context) == pair["feedback_context_sha256"], "feedback identity")
        z = np.asarray(context["latent_history"])
        a = np.asarray(context["action_history"])
        g = np.asarray(goal)
        raw = source.source.array(prefix, (15, 2), True).astype("<f4")
        require(z.shape == (3, 192) and a.shape == (2, 10) and g.shape == (192,) and
                np.isfinite(z).all() and np.isfinite(a).all() and np.isfinite(g).all()
                and np.all(np.abs(raw) <= 1), "context, goal or raw domain")
        result.append(dict(base=index, root=pair["root"], parent_arm=pair["source_arm"],
                           task_id=job["task_id"], task_sha256=job["task_sha256"],
                           feedback_context_sha256=pair["feedback_context_sha256"],
                           context=context, goal=goal, prefix_raw=raw.astype(float).tolist(),
                           prefix_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                           inherited_model_resources=pair["inherited_model_resources"]))
    return result


def contract():
    return dict(schema=NAME + ".contract", base_order=[0, 1, 2, 3], arms=list(ARMS),
                initial_logits=[0.0]*30, metric="mean192 identity", radius=h3.CAP,
                kernel_contract=h3.contract(), planning_scope="saved observed control15 context",
                composition_scope="packet suffix freezing, not proven reusable-program composition",
                host_cuda_driver_ready=False, physical_ready=False,
                same_total_compute=False, scientific_pass=False)


def build():
    shared = source.build()
    prepare = source.source.read(str(source.PATHS["prepare"]), source.source.PINS["prepare"])
    jobs = [prepare["jobs"][2*i+1] for i in range(4)]
    files = [Path(__file__).resolve(), ROOT / "tests/test_codex_p2_cp15_symmetric_spectral_20261030_v1.py",
             ROOT / "innerdocs/codex_p2_cp15_symmetric_spectral_20261030_v1.md",
             Path(h3.__file__).resolve(), Path(source.__file__).resolve()]
    return dict(schema=NAME + ".prepare", contract=contract(), cases=bind(jobs, shared["pairs"]),
                source_pins=shared["source_pins"], source_summary_sha256=shared["source_summary_sha256"],
                own_sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                inherited_upstream_authentication=True, independently_rerun_model=False,
                physical_execution=False, scientific_pass=False)


def execute_arm(backend, case, arm, meter):
    """Backend must supply real, metered operations; caller owns durable lifecycle."""
    require(arm in ARMS, "fixed arm")
    require(case["feedback_context_sha256"] == source.source.digest(case["context"]),
            "feedback context drift")
    inputs = {"context": case["context"], "goal": case["goal"],
              "projection": np.eye(192).tolist()}
    return h3.run_arm(backend, inputs, arm, meter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-sha")
    args = parser.parse_args()
    path = Path(args.output)
    require(path.is_absolute() and path.parent == ROOT / "results" and
            path.name.startswith(NAME + "_PREP_") and not path.is_symlink(), "exclusive path")
    value = build()
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    sha = hashlib.sha256(raw).hexdigest()
    if args.verify_sha:
        require(hashlib.sha256(path.read_bytes()).hexdigest() == args.verify_sha == sha,
                "prepare reconstruction")
        print(json.dumps({"status": "prepared_not_run", "sha256": sha}))
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps({"status": "prepared_not_run", "sha256": sha}))


if __name__ == "__main__":
    main()
