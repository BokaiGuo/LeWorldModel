"""Post-hoc observed-H10 comparison, preserving the original stop status."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p1_uphill_complete_pairing_20261030 as source

NAME = "p1_uphill_common_h10_observation_20261031"
PATH = source.ROOT / "results/p1_uphill_complete_pairing_20261030_RUN01.json"
SHA = "e3efb9815f5f4154036e34331d8da87b8a2dab007fd5db3288cfaeec6a8538ed"
need = source.require


def observed(item):
    n, status = item["returned_suffix_steps"], item["status"]
    need(isinstance(n, int) and 0 <= n <= 10, "step count")
    if status == "probe_completed":
        need(n == 10 and item["endpoint"] is not None
             and source.same_metrics(item["endpoint"], item["last_accepted_metrics"])
             and item["last_accepted_terminated"] is False
             and item["last_accepted_truncated"] is False, "completed endpoint")
        return item["endpoint"]
    need(status == "native_stopped" and item["endpoint"] is None
         and (item["last_accepted_terminated"] or item["last_accepted_truncated"]),
         "stopped endpoint")
    return item["last_accepted_metrics"] if n == 10 else None


def build(data):
    need(data["schema"] == "p1_uphill_complete_pairing_20261030"
         and data["P1_admission"] is False and len(data["common_suffix_pairs"]) == 336,
         "fixed panel")
    groups = {}
    for pair in data["common_suffix_pairs"]:
        key = (pair["logical_pair"], pair["probe"])
        need(key not in groups and set(pair["roles"]) == set(source.ROLES), "pair identity")
        groups[key] = pair
    result = []
    for (logical, probe), pair in groups.items():
        baseline = groups[(logical, "zero")]
        ends = {r: observed(pair["roles"][r]) for r in source.ROLES}
        zeros = {r: observed(baseline["roles"][r]) for r in source.ROLES}
        for role in source.ROLES:
            own_response = source.difference(ends[role], zeros[role])
            zero_response = source.difference(ends["zero"], zeros["zero"])
            result.append({"logical_pair": logical, "probe": probe, "role": role,
                           "status": pair["roles"][role]["status"],
                           "returned_suffix_steps": pair["roles"][role]["returned_suffix_steps"],
                           "h10_metrics": ends[role],
                           "endpoint_minus_zero_prefix": source.difference(ends[role], ends["zero"]),
                           "endpoint_minus_own_zero_suffix": own_response,
                           "response_minus_zero_prefix": source.difference(own_response, zero_response)})
    need(len(result) == 1008, "fixed rows")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == source.ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    data = source.read_checked(PATH, SHA)
    rows = build(data)
    need(source.hashlib.sha256(PATH.read_bytes()).hexdigest() == SHA, "source drift")
    payload = {"schema": NAME, "rows": rows, "input_sha256": SHA,
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "P1_admission": False, "independent_confirmation": False,
               "limitations": "Post-hoc observation audit on dependent known roots; "
                              "does not replace sealed status-based results or prove mechanism."}
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"rows": len(rows), "observed": sum(x["h10_metrics"] is not None for x in rows),
                      "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
