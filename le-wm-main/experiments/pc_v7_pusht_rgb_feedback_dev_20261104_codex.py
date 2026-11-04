"""Run v7 response compiler in the guarded PushT RGB feedback executor."""

from pathlib import Path

import torch

from experiments import pc_v5_pusht_rgb_feedback_dev_20261104_codex as prior
from experiments import pc_v7_reacher_rgb_live_20261104_codex as adapter
from experiments.spectral_reachability import pc_target_reach_action_dim_v6 as v6
from experiments.spectral_reachability import pc_target_reach_paired_abstention_v7 as v7

live, ROOT = prior.live, prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_PUSHT_RGB_FEEDBACK_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_pusht_rgb_feedback_dev_20261104_codex"


class SealAdapter:
    @staticmethod
    def capture(files):
        return prior.SealAdapter.capture([*files, Path(__file__), PROTOCOL, Path(adapter.__file__),
                                          Path(v6.__file__), Path(v7.__file__)])

    verify = staticmethod(prior.SealAdapter.verify)


def configure():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prior.configure()
    live.saved.method = adapter
    live.PROTOCOL, live.PREFIX = PROTOCOL, PREFIX
    live.MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
    live.RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
    live.seal = SealAdapter


if __name__ == "__main__":
    configure()
    live.main()
