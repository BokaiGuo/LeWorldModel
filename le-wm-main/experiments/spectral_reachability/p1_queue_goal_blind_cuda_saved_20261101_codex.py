"""Save a read-only recheck of the development CUDA result."""
import argparse
import os
from pathlib import Path

from . import p1_queue_goal_blind_cuda_20261101_codex as run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha256", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    run.b.require(not output.exists(), "exclusive audit")
    result = run.verify_saved(args.prepared, args.prepare_sha256, args.result)
    result.update(schema="p1_queue_goal_blind_cuda_saved_20261101_codex.v1",
                  prepared=str(Path(args.prepared).resolve()),
                  prepare_sha256=args.prepare_sha256, result=str(Path(args.result).resolve()),
                  auditor_sha256=run.b.sha(Path(__file__)),
                  runner_sha256=run.b.sha(Path(run.__file__)))
    with output.open("xb") as f:
        f.write(run.b.canonical(result))
        f.flush()
        os.fsync(f.fileno())
    print(run.b.canonical(dict(path=str(output), sha256=run.b.sha(output), **result)).decode().strip())


if __name__ == "__main__":
    main()
