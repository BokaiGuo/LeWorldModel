"""Fixed root1 DLS candidate adaptation of known-root robust feedback contrast."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from . import codex_p1_root1_candidate_consumer_20261101_exclusive as consumer
from . import codex_p1_cp10_robust_feedback_20261101_exclusive as base
from . import codex_p1_diag32_fixed_physical_20261101_exclusive as physical
from . import e3_cp10_four_arm_root1_20261031 as cp10

ROOT = base.ROOT
NAME = "codex_p1_root1_robust_feedback_20261101_exclusive"
ROLES = ("selected", "zero", "greedy")
POLICIES, IDENTITY = base.POLICIES, base.IDENTITY
actions, choose, digest_actions = base.actions, base.choose, base.digest_actions
select_cp5, select_cp10, cp10_context = base.select_cp5, base.select_cp10, base.cp10_context
SOURCE = consumer.CONTRACT
SOURCE_SHA = consumer.CONTRACT_SHA
RECEIPT = ROOT / "results/codex_p1_root1_candidate_consumer_20261101_exclusive_CODEX01.json"
RECEIPT_SHA = "f2f843f1fd4e1e1a8ee07c25da5eca91337f3ea8895e19a92fb8337c1c325013"
CANDIDATES = json.loads(SOURCE.read_bytes())


def inputs():
    require = physical.require
    require(physical.sha(SOURCE) == SOURCE_SHA and physical.sha(RECEIPT) == RECEIPT_SHA, "candidate digest")
    verification = consumer.verify()
    require(verification == {**json.loads(RECEIPT.read_bytes())}, "candidate consumer mismatch")
    entry = CANDIDATES["candidates"]["selected"]
    source = entry["source"]
    require(physical.sha(source["prepared"]) == source["prepare_sha256"], "prepared hash")
    prepared = json.loads(Path(source["prepared"]).read_bytes())
    data = physical.engine.p3.source_data(prepared["source"])
    rebound = cp10.rebind(prepared, data)
    require(rebound == CANDIDATES["goal_binding"], "root1 target rebind")
    report = json.loads(Path(source["report"]).read_bytes())
    root = next(x for x in report["rows"][0]["checkpoints"] if x["step"] == 0)
    task = deepcopy(prepared["source"]["tasks"][1])
    require(root["pixel_sha256"] == CANDIDATES["root_pixel_sha256"]
            and root["goal_rgb_sha256"] == rebound["goal_pixel_sha256"]
            and root["goal"] == data["references"][1]["goal"].tolist()
            and task["binding"]["root_sha256"] == rebound["root_sha256"]
            and physical.engine.digest(task["goal"]) == rebound["goal_latent_sha256"], "root1 identity")
    mean, std = np.asarray(task["action_mean"]), np.asarray(task["action_std"])
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std > 0).all(), "normalizer")
    return prepared, data, root, task


def contract():
    from . import codex_p1_root1_robust_feedback_runner_20261101_exclusive as runtime
    prepared, _, root, _ = inputs()
    paths = [Path(__file__), Path(consumer.__file__), SOURCE, RECEIPT,
             ROOT / "experiments/spectral_reachability/codex_p1_root1_robust_feedback_runner_20261101_exclusive.py",
             ROOT / "experiments/spectral_reachability/codex_p1_root1_robust_feedback_verify_20261101_exclusive.py",
             ROOT / "tests/test_codex_p1_root1_robust_feedback_20261101_exclusive.py",
             ROOT / "tests/test_codex_p1_root1_candidate_consumer_20261101_exclusive.py",
             Path(base.__file__), Path(physical.__file__), Path(cp10.__file__),
             *(Path(m.__file__) for m in (runtime.old, runtime.old.fixed, runtime.old.fixed.blind,
                                          runtime.old.p0, runtime.engine, runtime.risk, runtime.physical.locking))]
    bindings = dict(prepared["sources"])
    for path in paths:
        path = path.resolve()
        digest = physical.sha(path)
        physical.require(str(path) not in bindings or bindings[str(path)] == digest, "source collision")
        bindings[str(path)] = digest
    physical.require(all(physical.sha(p) == h for p, h in bindings.items()), "source drift")
    return {"schema": NAME + ".v1", "roles": list(ROLES), "policies": list(POLICIES),
            "candidates": CANDIDATES["candidates"], "goal_binding": CANDIDATES["goal_binding"],
            "root": {k: root[k] for k in IDENTITY}, "model_sha256": CANDIDATES["model"]["original_state_dict"],
            "protocol": {**base.protocol(), "root1_candidate_method": CANDIDATES["method_difference"]},
            "physical_caps_total": {k: v * 6 for k, v in physical.CAPS.items()},
            "wall_seconds": 1800, "cuda_allocated_cap_bytes": 4 * 1024**3,
            "cuda_reserved_cap_bytes": 6 * 1024**3, "bindings": bindings,
            "independent_confirmation": False, "P1_admission": False}
