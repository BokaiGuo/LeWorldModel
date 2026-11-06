"""Record the actual returned means of the prefix-conditional planner."""
import json
from pathlib import Path

from experiments.pc_native_prefix_conditional_20261105_codex import NativePrefixConditional


class NativePrefixTrace(NativePrefixConditional):
    def __init__(self, *args, chain_path, **kwargs):
        self.chain_path = Path(chain_path)
        self.chain_path.open("x", encoding="utf-8").close()
        self.chain_solve = 0
        super().__init__(*args, **kwargs)

    def solve(self, info_dict, init_action=None):
        result = super().solve(info_dict, init_action)
        with self.chain_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"solve": self.chain_solve,
                                     "actions": result["actions"].tolist()},
                                    allow_nan=False) + "\n")
        self.chain_solve += 1
        return result
