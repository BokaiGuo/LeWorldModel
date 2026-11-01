"""Run one fixed H3 batch against reconstructed inputs and verify saved results."""
import argparse
import json
import os
from pathlib import Path
import shutil

from . import p3_raw_h3_physical_source_20261031 as source

batch, host, require, ROOT = source.batch, source.host, source.require, source.ROOT
NAME = "p3_raw_h3_physical_run_20261031"


def run(index, output):
    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith(NAME + "_") and not os.path.lexists(output),
            "exclusive output")
    require(shutil.disk_usage(output.parent).free >= host.helper.CAP + host.helper.FLOOR,
            "disk reservation")
    p, data = source.reconstruct(index)
    store = host.helper.Store(output)
    store.put("prepare.json", p)

    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                        with_target=False, render_action=False)

    result = batch.execute(p, data, store, factory)
    report_sha, prepare_sha = host.sha(output / "report.json"), host.sha(output / "prepare.json")
    if result["status"] == "recorded_not_admitted":
        receipt = source.consume(output, report_sha, prepare_sha, index)
        print(json.dumps({"status": receipt["status"], "batch": index,
                          "report_sha256": report_sha, "prepare_sha256": prepare_sha}),
              flush=True)
    else:
        print(json.dumps({"status": result["status"], "batch": index,
                          "report_sha256": report_sha, "prepare_sha256": prepare_sha}),
              flush=True)
    return result["status"] == "recorded_not_admitted"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physics", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.batch, args.output) else 1)
