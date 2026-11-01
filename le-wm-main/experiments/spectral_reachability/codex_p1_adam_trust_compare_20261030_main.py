"""Pair trust-regularized physical results against the original fixed Adam plan."""
import json
from pathlib import Path

from . import codex_p1_adam_feedback_compare_20261030_main as compare

NAME = "codex_p1_adam_trust_compare_20261030_main"
ROOT = compare.base.ROOT


def build():
    compare.DIR = ROOT / "results/codex_p1_adam_trust_physical_20261030_main_RUN01"
    compare.REPORT_SHA = "cc8c0d35abc287d97abb141fb4fb12a70e41c72659cf9b6a96b5111536c4601b"
    compare.SAVED = ROOT / "results/codex_p1_adam_trust_physical_20261030_main_SAVED_RUN01.json"
    compare.SAVED_SHA = "60f3874081d873c9ff7bd8d7f18d6726261db22b12d6d88d1552bd3ac6e5405c"
    result = compare.build()
    result["schema"] = NAME
    result["comparison_code_sha256"] = compare.base.sha(Path(compare.__file__).resolve())
    result["wrapper_code_sha256"] = compare.base.sha(Path(__file__).resolve())
    return result


if __name__ == "__main__":
    result = build()
    out = ROOT / f"results/{NAME}_RUN01.json"
    print(json.dumps({"rows": len(result["rows"]), "sha256": compare.base.write(out, result)}), flush=True)
