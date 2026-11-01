"""Execute each sealed original suffix under each fixed prefix."""
import hashlib
from pathlib import Path

from . import p1_uphill_cp5_physical20_20261030 as physical

NAME = "p1_uphill_fixedtail_physical_20261030"
PROTOCOL = physical.ROOT.parent / "docs/D2P_P1_UPHILL_FIXEDTAIL_PHYSICAL_PROTOCOL_20261030_MAIN.md"
original_build = physical.build
original_configure = physical.configure


def configure():
    original_configure()
    physical.engine.NAME = NAME


def build():
    value = original_build()
    fixed = physical.design.uphill.runner.native.read(physical.bundle.PREP, cap=64 * 1024**2)
    for i, item in enumerate(value["roster"]):
        anchor = fixed["anchors"][i]
        physical.need(item["index"] == anchor["index"] == i, "anchor order")
        tails = [x for x in fixed["probe_banks"][anchor["logical_pair"]]
                 if x["name"] == "original_selected_tail"]
        physical.need(len(tails) == 1, "tail roster")
        raw = bytes.fromhex(tails[0]["raw10_float32_hex"])
        physical.need(len(raw) == 80 and hashlib.sha256(raw).hexdigest() == tails[0]["raw10_sha256"],
                      "tail bytes")
        item["raw10_float32_hex"] = raw.hex()
        item["raw10_sha256"] = tails[0]["raw10_sha256"]
        item.pop("plan_row_sha256")
        item["tail_source"] = "sealed_h10_original_selected_tail"
    for path in (Path(__file__).resolve(), PROTOCOL):
        value["bindings"][str(path)] = physical.sha(path)
    value["schema"] = NAME + ".prepare"
    value["source_description"] = "common fixed original tail; no suffix optimization"
    return value


def main():
    physical.NAME = NAME
    physical.configure, physical.build = configure, build
    return physical.main()


if __name__ == "__main__":
    raise SystemExit(main())
