"""Retry sandbox-inaccessible GPU preflight to separate, non-overwriting outputs."""
from pathlib import Path

from experiments import pc_response_temporal_v3_reacher_adapter_preflight_20261104_codex as run

PREFIX = run.ROOT / "results/pc_response_temporal_v3_reacher_adapter_preflight_host_retry_20261104_codex"
original_dependencies = run.dependencies


def dependencies(source):
    return {**original_dependencies(source), str(Path(__file__).resolve()): run.host.bridge.sha_file(Path(__file__))}


if __name__ == "__main__":
    run.PREFIX = PREFIX
    run.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    run.CONTEXT = PREFIX.with_name(PREFIX.name + "_context.npz")
    run.RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
    run.dependencies = dependencies
    run.main()
