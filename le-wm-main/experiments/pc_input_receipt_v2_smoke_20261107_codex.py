"""Run the existing two-track parity fixtures against the versioned boundary."""
import argparse
import json
from pathlib import Path

from experiments import pc_dual_track_smoke_20261106_codex as smoke
from experiments import pc_input_permissions as v1
from experiments import pc_input_permissions_v2_20261107_codex as v2

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_INPUT_RECEIPT_V2_SMOKE_PROTOCOL_20261107_CODEX.md"


def main(track):
    output = ROOT / f"results/pc_input_receipt_v2_smoke_20261107_codex_{track}.json"
    if output.exists():
        raise FileExistsError(output)
    freeze = smoke.freeze_audit

    def bind_sources(track_id, family, paths, **kwargs):
        paths = list(paths)
        for path in (v1.__file__, v2.__file__, __file__):
            if Path(path).resolve() not in {Path(p).resolve() for p in paths}:
                paths.append(path)
        return freeze(track_id, family, paths, **kwargs)

    smoke.freeze_audit = bind_sources
    smoke.RGBLeWM = v2.RGBLeWMv2
    smoke.StateAssistedPointMaze = v2.StateAssistedPointMazev2
    try:
        wrapper, data, rows, fingerprint = smoke.rgb() if track == v1.RGB else smoke.state()
    finally:
        smoke.freeze_audit = freeze
    if len(wrapper.receipts) != len(rows):
        raise ValueError("successful-call count mismatch")
    record = v2.call_manifest(wrapper.audit, algorithm_version="input_boundary_v2_smoke",
                              protocol_path=PROTOCOL, task_manifest_path=data, receipts=wrapper.receipts)
    record.update(rows=rows, unchanged_model_fingerprint=fingerprint,
                  scope="GPU call parity only; not episode completion or gate promotion")
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"track": track, "output": str(output), "rows": rows}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=(v1.RGB, v1.STATE), required=True)
    main(parser.parse_args().track)
