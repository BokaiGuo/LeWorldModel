"""Run same-setting two-start CEM on the exposed v7 PushT episode."""
import json
from pathlib import Path

from experiments import pc_v7_pusht_rgb_feedback_dev_20261104_codex as v7

live, ROOT = v7.live, v7.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_PUSHT_MATCHED_WARM_CEM_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_pusht_matched_warm_cem_dev_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
RESULT = PREFIX.with_name(PREFIX.name + "_result.json")
V7_AUDIT = v7.PREFIX.with_name(v7.PREFIX.name + "_audit.json")
V7_MANIFEST = v7.PREFIX.with_name(v7.PREFIX.name + "_manifest.json")
V7_RESULT = v7.PREFIX.with_name(v7.PREFIX.name + "_result.json")


class SealAdapter:
    @staticmethod
    def capture(files):
        return v7.SealAdapter.capture([*files, Path(__file__), PROTOCOL, V7_AUDIT, V7_MANIFEST, V7_RESULT])

    verify = staticmethod(v7.SealAdapter.verify)


def configure():
    v7.configure()
    sha = live.HOST.bridge.sha_file
    audit = json.loads(V7_AUDIT.read_text())
    if audit["manifest_sha256"] != sha(V7_MANIFEST) or audit["result_sha256"] != sha(V7_RESULT):
        raise ValueError("v7 source audit mismatch")
    SealAdapter.verify(json.loads(V7_MANIFEST.read_text())["seal"])
    live.PROTOCOL, live.PREFIX, live.MANIFEST, live.RESULT = PROTOCOL, PREFIX, MANIFEST, RESULT
    live.ARMS, live.seal = ("smooth_cem",), SealAdapter


if __name__ == "__main__":
    configure()
    live.main()
