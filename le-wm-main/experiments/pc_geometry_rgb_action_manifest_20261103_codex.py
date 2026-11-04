"""Export only actions from the exposed diagnostic, without outcome fields."""
import json

from experiments import pc_geometry_native_suffix_20261103_codex as source
from experiments.spectral_reachability import pc_geometry_persistent_feasibility_20261103_codex as base

OUTPUT = base.ROOT / "results/pc_geometry_rgb_action_manifest_20261103_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    doc = json.loads(source.OUTPUT.read_text())
    rows = [{"root_index": row["root_index"], "prefix_name": row["prefix_name"],
             "prefix_actions": row["prefix_actions"]} for row in doc["rows"]]
    if len(rows) != 48:
        raise ValueError("incomplete source bank")
    with OUTPUT.open("x") as handle:
        json.dump({"schema": "pc_geometry_rgb_action_manifest.v1",
                   "source_sha256": base.digest(source.OUTPUT), "rows": rows}, handle, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
