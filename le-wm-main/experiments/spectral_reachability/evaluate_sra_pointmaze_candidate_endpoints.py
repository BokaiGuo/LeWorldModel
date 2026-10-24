"""Evaluate candidate PointMaze action sequences in an isolated MuJoCo backend."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from audit_sra_pointmaze_replay_backend import replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite endpoints: {args.output}")
    import mujoco

    request_bytes = args.input.read_bytes()
    xml_bytes = args.xml.read_bytes()
    raw = np.load(args.input, allow_pickle=False)
    sources, sequences = raw["sources"], raw["sequences"]
    if sources.ndim != 2 or sources.shape[1] != 4 or sequences.ndim != 3 or sequences.shape[:1] != sources.shape[:1] or sequences.shape[2] != 2:
        raise ValueError("need sources[N,4] and sequences[N,T,2]")
    model = mujoco.MjModel.from_xml_string(xml_bytes.decode("utf-8"))
    endpoints = np.stack([replay(model, source, sequence) for source, sequence in zip(sources, sequences)])
    replay_backend = Path(__file__).with_name("audit_sra_pointmaze_replay_backend.py")
    np.savez_compressed(
        args.output,
        endpoints=endpoints,
        request_sha256=np.asarray(hashlib.sha256(request_bytes).hexdigest()),
        xml_sha256=np.asarray(hashlib.sha256(xml_bytes).hexdigest()),
        endpoint_script_sha256=np.asarray(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        replay_backend_script_sha256=np.asarray(hashlib.sha256(replay_backend.read_bytes()).hexdigest()),
    )


if __name__ == "__main__":
    main()
