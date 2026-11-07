"""Inspect the selected open-syscall trace; not a filesystem access sandbox."""
import json
from pathlib import Path

from experiments.pc_input_permissions import file_hash

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "results/pc_rgb_gradient_public_plan_filetrace_host_20261107_codex.log"
OUT = ROOT / "results/pc_rgb_gradient_public_filetrace_audit_20261107_codex.json"
REQUIRED = ("pc_rgb_gradient_public_input_20261107_codex.npz",
            "/checkpoints/tworoom/lewm/weights.pt", "/checkpoints/tworoom/lewm/config.json")
FORBIDDEN = ("tworoom.h5", "pc_tworoom_precise_spectral_development_plan_20261107_codex.json",
             "pc_tworoom_precise_spectral_development_physical_20261107_codex.json",
             "pc_tworoom_goal_gradient_development_plan_20261107_codex.json",
             "pc_tworoom_goal_gradient_development_physical_20261107_codex.json")


def inspect(lines):
    # Record attempts even when the syscall failed; success is required for positive controls.
    required = {needle: sum(needle in line and " = -1 " not in line for line in lines) for needle in REQUIRED}
    forbidden = {needle: sum(needle in line for line in lines) for needle in FORBIDDEN}
    return {"required_open_counts": required, "forbidden_open_attempts": forbidden,
            "pass": all(required.values()) and not any(forbidden.values())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    with TRACE.open() as stream:
        result = inspect(stream.readlines())
    with OUT.open("x") as stream:
        json.dump({"trace_sha256": file_hash(TRACE), "runner_sha256": file_hash(__file__), **result,
                   "scope": "literal path matching on traced open syscalls; not OS access control"},
                  stream, indent=2, allow_nan=False)
    print(f"Selected-path trace audit: {result['pass']}")


if __name__ == "__main__":
    main()
