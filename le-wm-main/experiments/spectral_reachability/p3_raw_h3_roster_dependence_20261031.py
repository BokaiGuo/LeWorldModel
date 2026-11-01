"""Index exact root/action duplicates in the fixed H3 roster, without outcomes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from . import p3_raw_h3_physical_source_20261031 as source

require = source.require


def index_roster(plans):
    require(len(plans) == 72, "fixed roster")
    for start in range(0, 72, 6):
        source.batch.validate_plans(plans[start:start + 6])
    executions, prefixes, pairs = defaultdict(list), defaultdict(list), defaultdict(list)
    identical_pair_count = 0
    for i, plan in enumerate(plans):
        actions = source.batch.branch.raw20(plan["commands"])
        root = (plan["root_sha256"], plan["task_id"])
        executions[root + (actions.tobytes(),)].append(i)
        prefixes[root + (actions[:15].tobytes(),)].append(i)
    for i in range(0, 72, 2):
        old, new = plans[i:i + 2]
        old_actions = source.batch.branch.raw20(old["commands"]).tobytes()
        new_actions = source.batch.branch.raw20(new["commands"]).tobytes()
        identical_pair_count += int(old_actions == new_actions)
        pairs[(old["root_sha256"], old["task_id"], old_actions, new_actions)].append(i // 2)

    def groups(mapping):
        return sorted(({"indices": rows, "multiplicity": len(rows)} for rows in mapping.values()),
                      key=lambda group: group["indices"][0])

    return {"logical_executions": 72, "unique_root_full_action_executions": len(executions),
            "logical_pairs": 36, "unique_root_old_new_action_pairs": len(pairs),
            "unique_root_prefixes": len(prefixes), "identical_old_new_pairs": identical_pair_count,
            "execution_groups": groups(executions), "prefix_groups": groups(prefixes),
            "pair_groups": groups(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == source.ROOT / "results"
            and output.name.startswith("p3_raw_h3_roster_dependence_20261031_")
            and not output.exists() and not output.is_symlink(), "exclusive output")
    source.reconstruct(0)
    require(source.host.sha(source.MANIFEST) == source.MANIFEST_SHA, "manifest hash")
    manifest = json.loads(source.MANIFEST.read_bytes())
    result = {"schema": "p3_raw_h3_roster_dependence_20261031", "manifest_sha256": source.MANIFEST_SHA,
              "analysis_sha256": source.host.sha(Path(__file__).resolve()), **index_roster(manifest["plans"]),
              "uses_physical_outcomes": False, "independent_roots_certified": False,
              "note": "Exact root/task/action groups describe duplicate inputs, not statistical independence."}
    require(source.host.sha(source.MANIFEST) == source.MANIFEST_SHA, "manifest drift")
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"sha256": hashlib.sha256(raw).hexdigest(),
                      "unique_executions": result["unique_root_full_action_executions"],
                      "unique_pairs": result["unique_root_old_new_action_pairs"],
                      "unique_prefixes": result["unique_root_prefixes"]}))


if __name__ == "__main__":
    main()
