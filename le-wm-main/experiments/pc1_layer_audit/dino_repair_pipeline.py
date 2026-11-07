"""One-command stage resume; completed runs verify without loading CUDA models."""
import argparse
import json
import subprocess
import sys

from experiments.pc1_layer_audit import dino_repair_run as base, dino_repair_v2_run as v2


def expected(stage):
    return ([f"{stage}_r{r}.json" for r in base.camera.ROOTS] if stage in ("predict", "evaluate")
            else [{"parity": "parity.json", "summarize": "summary.json", "audit": "array_audit.json"}[stage]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    base.OUT = v2.OUT
    if (base.OUT/"manifest.json").exists():
        base.read_record("manifest.json")
    for stage in ("parity", "predict", "evaluate", "summarize", "audit"):
        files = expected(stage)
        complete = all((base.OUT/name).exists() for name in files)
        if complete:
            for name in files:
                base.read_record(name)
            print(f"verified; skip {stage}", flush=True)
            continue
        if args.verify_only:
            raise ValueError(f"stage incomplete: {stage}")
        module = "experiments.pc1_layer_audit.dino_repair_" + ("audit" if stage == "audit" else "v2_run")
        command = [sys.executable, "-m", module] + ([] if stage == "audit" else [stage])
        subprocess.run(command, check=True, cwd=base.ROOT)
    print(json.dumps({"bounded_repair_complete": True, "full_research_complete": False,
                      "state_assisted_only": True, "all_stage_artifacts_verified": True}))


if __name__ == "__main__":
    main()
