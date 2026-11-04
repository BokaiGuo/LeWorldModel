"""Verify prospective source receipts and paired repeat against old development run."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_successor_rgb_bound_runner_20261103_codex as bound

OUTPUT = bound.PREFIX.with_name(bound.PREFIX.name + "_verification.json")
PHYSICAL = bound.PREFIX.with_name(bound.PREFIX.name + "_physical.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    before = json.loads(bound.PRE.read_text())
    after = json.loads(bound.POST.read_text())
    if after["source_before_sha256"] != bound.digest(bound.PRE):
        raise ValueError("source receipt mismatch")
    if before["sources"] != bound.snapshot(bound.source_paths()):
        raise ValueError("declared source drift")
    bound.runner.PREFIX = bound.PREFIX
    output_paths = [p for root in range(4) for arm in bound.pilot.ARMS
                    for group in bound.runner.paths(root, arm)
                    for p in (group if isinstance(group, list) else [group])]
    if after["outputs"] != bound.snapshot(output_paths):
        raise ValueError("output receipt mismatch")
    physical = json.loads(PHYSICAL.read_text())
    if len(physical["rows"]) != 8 or physical["protocol_sha256"] != bound.digest(bound.PROTOCOL):
        raise ValueError("physical receipt mismatch")
    rows = []
    for root in range(4):
        for arm in bound.pilot.ARMS:
            bound.runner.PREFIX = bound.PREFIX
            new_paths, execution, images = bound.runner.paths(root, arm)
            receipt = json.loads(execution.read_text())
            if receipt["images_sha256"] != bound.digest(images):
                raise ValueError("image receipt mismatch")
            bound.runner.PREFIX = bound.pilot.PREFIX
            old_paths, _, old_images = bound.runner.paths(root, arm)
            with np.load(images, allow_pickle=False) as new_file, np.load(old_images, allow_pickle=False) as old_file:
                image_equal = bool(np.array_equal(new_file["rgb"], old_file["rgb"]))
            for step, (new_path, old_path) in enumerate(zip(new_paths, old_paths, strict=True)):
                new = json.loads(new_path.read_text())
                old = json.loads(old_path.read_text())
                if receipt["rows"][step]["decision_sha256"] != bound.digest(new_path):
                    raise ValueError("decision receipt mismatch")
                new_action = np.asarray(new["result"]["selected"]["chain"])
                old_action = np.asarray(old["result"]["selected"]["chain"])
                rows.append({"root": root, "arm": arm, "step": step, "rgb_archive_equal": image_equal,
                             "selected_chain_equal": bool(np.array_equal(new_action, old_action)),
                             "max_action_difference": float(np.max(np.abs(new_action - old_action))),
                             "score_abs_difference": abs(float(new["result"]["selected"]["terminal_score"])
                                                         - float(old["result"]["selected"]["terminal_score"])),
                             "old_decision_sha256": bound.digest(old_path), "new_decision_sha256": bound.digest(new_path)})
    with OUTPUT.open("x") as stream:
        json.dump({"source_before_sha256": bound.digest(bound.PRE), "source_after_sha256": bound.digest(bound.POST),
                   "physical_sha256": bound.digest(PHYSICAL), "runner_sha256": bound.digest(Path(__file__)),
                   "scope": "same-hardware, same-roots repeat; not independent confirmation",
                   "rows": rows}, stream, indent=2, allow_nan=False)
    print(f"Verified {len(before['sources'])} declared sources, {len(output_paths)} files, and {len(rows)} paired decisions.")


if __name__ == "__main__":
    main()
