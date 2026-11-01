"""Recompute fixed DLS first-packet scores and physical metrics from saved vectors."""
import argparse
import json
from pathlib import Path

from . import e3_cp10_dls_first_packet_20261031 as experiment
from . import e3_cp10_first_packet_saved_20261031 as saved

NAME = "e3_cp10_dls_first_packet_saved_20261031"


def check(path, expected_sha):
    if experiment.base.sha(path) != expected_sha:
        raise ValueError("result hash")
    data = json.loads(Path(path).read_bytes())
    experiment.original.NAME = experiment.NAME
    rows = saved.verify(data)
    if not all(experiment.base.sha(path) == sha for path, sha in data["sources"].items()):
        raise ValueError("source drift")
    return {"schema": NAME, "status": "verified", "result_sha256": expected_sha,
            "rows": rows, "physical_event_chain_independently_checked": False,
            "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if (not out.is_absolute() or out.parent != experiment.ROOT / "results"
            or not out.name.startswith(NAME + "_") or out.exists() or out.is_symlink()):
        raise ValueError("exclusive output")
    result = check(args.input, args.sha256)
    digest = experiment.base.write(out, result)
    print(json.dumps({"status": "verified", "sha256": digest, "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
