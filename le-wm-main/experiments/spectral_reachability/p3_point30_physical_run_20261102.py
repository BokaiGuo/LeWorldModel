"""Execute a previously sealed point-30 plan on the known PushT roots."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import p3_point30_physical_prepare_20261102 as source

AUTH = "P3_POINT30_KNOWN_ROOT_FIXED_WINNERS_PHYSICAL"


def run(plan_path, plan_sha, output, authority):
    prior, moving = source.prior, source.prior.moving
    prior.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
                  "physical authority and hidden CUDA required")
    prior.require(moving.sha(plan_path) == plan_sha, "plan hash")
    plan = moving.read(plan_path)
    prior.require(plan == source.material(), "plan or source drift")
    out = Path(output)
    prior.require(out.is_absolute() and out.parent == source.ROOT / "results"
                  and out.name.startswith(source.NAME + "_RUN_") and not os.path.lexists(out),
                  "exclusive output")
    data = moving.load_source(prior.prior.source.SOURCEPACK, prior.prior.source.SOURCEPACK_SHA,
                              prior.prior.source.SOURCE_PREPARE, prior.prior.source.SOURCE_PREPARE_SHA)
    store = moving.Store(out)
    store.put("execution_binding.json", {"prepared": str(Path(plan_path).resolve()),
                                         "prepare_sha256": plan_sha})

    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                        with_target=False, render_action=False)

    return moving.execute(plan, data, store, factory, authorization=moving.AUTH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "sha256", "output", "authority"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = run(args.plan, args.sha256, args.output, args.authority)
    print(json.dumps({"status": result["status"], "errors": result["errors"]}))


if __name__ == "__main__":
    main()
