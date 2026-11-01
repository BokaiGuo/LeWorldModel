"""Prepare pinned saved control5 authorities for fixed cells 05, 06, 07, 14 and 15.

CPU-only preparation: does not run the feedback host, model, or physics.
"""
import argparse
from copy import deepcopy
from pathlib import Path

from . import e3_cell10_feedback_prepare_codex_20261030_v1 as previous

NAME = "e3_uncovered_fixed_inputs_20261101_exclusive"
ROOT = previous.ROOT
need = previous.need
PINS = {
    5: ("r0.g1.h1.dDx", "83f622283b777689c933662e143bfaeff09fa501b54b1a5f4384f7faec09ad80", "8d90608652fc795c18abe3894031886eadd5c22208f00c9d35d70ecdbd490eff", "47bc066186582f1b416c866e55a3ca8ce4e3178d80e6ac91993fef0cf7c8af62"),
    6: ("r0.g1.h3.dDxy", "3c30d1fa01191c88b1d4c68b17bb5f76ccc8cab0487372deb789826130e9b458", "b1ed494076cc57347807c2d5f6f82ce2fb2eae1ee74c979702ab85085a46352c", "098297ebca42c804dbc36a2aa98a05b228165fb597d6caa921e3e61c73239248"),
    7: ("r0.g1.h3.dDx", "1129fbf03afaff710cb980eed0e44b35523976e10e38a7c7356c48679df6e621", "d0a918e7c112bfc483bf5de1c064709c0aae5c0743f97929c5d6fc7d62b27ffc", "00b97d87e4316c48a3e0589c7e6dd0b96359bc86abbf67b75c2ce1fbfe636c93"),
    14: ("r1.g1.h3.dDxy", "f0b4dffc3794598d6dd5892b574f9b370b6a8a6d64c42fe04e72c97f712947e0", "94fddcf4bdd3eb5f877189385a4fe345f6d029f7591eaa766fa4f8d3fbc9e66a", "4800dfc21a6fde40b1f7dde8080383ee7fc30db0aa1f30086e945925c0f680b5"),
    15: ("r1.g1.h3.dDx", "9d008f8331f5bc7eb07379ae178850b01fe68253b655c1ddb11358af30b4ed23", "9d9c442db0e58bd477e2fc20115a7ff0bad7ec68bf7e7871eaacdf256fa949f8", "2446ea4d9aadc69d6090a5e25bb5ab64956be222306ea927ae2720c2625c72ee"),
}


def build(cell):
    need(type(cell) is int and cell in PINS, "cell not in fixed scope")
    task, report, completion, receipt = PINS[cell]
    need(previous.resolver.roster()[cell]["cell_id"] == task, "fixed task identity")
    previous.host.authority(previous.BASE, previous.BASE_SHA)
    base = previous.host.launch.host.read(previous.BASE, previous.BASE_SHA)
    refs = deepcopy(base["control5"])
    folder = ROOT / f"results/agent_e3_gd_host_cuda_20261029_v1_cell{cell:02d}_MAIN01"
    refs["report"] = {"path": str(folder / "report.json"), "sha256": report}
    refs["completion"] = {"path": str(folder / "completion.json"), "sha256": completion}
    refs["receipt"] = {"path": str(ROOT / f"results/agent_e3_gd_host_cuda_20261029_v1_consume_cell{cell:02d}_MAIN01.json"), "sha256": receipt}
    fresh = previous.identity.build(refs, base["resolver"], cell)
    need(base["cell"] == 0 and fresh["cell"] == cell and fresh["schema"] == base["schema"]
         and fresh["resolver"] == base["resolver"] and fresh["control5"] == refs
         and refs["prepare"] == base["control5"]["prepare"], "source identity")
    sources = dict(base["additional_sources"])
    for path, digest in fresh["additional_sources"].items():
        need(path in sources and sources[path] == digest, "inherited source pins: " + path)
    need(previous.host.launch.host.sha(previous.BASE) == previous.BASE_SHA, "base drift")
    return {"schema": base["schema"], "cell": cell, "control5": refs,
            "resolver": deepcopy(base["resolver"]), "additional_sources": sources}


def verify(cell, path, digest):
    need(previous.host.launch.host.read(path, digest) == build(cell), "authority reconstruction")
    binding = previous.host.authority(path, digest)
    old = previous.host.launch.episode.old
    joined = old.authority.load(*binding["args"])
    setup = old.setup_authority.resolve(binding["resolver"]["path"],
                                        binding["resolver"]["sha256"], cell, joined["body"])
    task = PINS[cell][0]
    horizon = int(task.split(".")[2][1:])
    domain = task.split(".")[3][1:]
    need(binding["cell"] == joined["cell"] == cell
         and joined["body"]["horizon_packets"] == horizon
         and joined["body"]["domain"]["name"] == domain
         and setup["expected_cp5"]["step"] == 5, "native setup identity")
    return {"verified": True, "cell": cell, "task": task,
            "native_saved_setup_resolved": True, "physical_execution": False,
            "production_authorized": False, "scientific_admission": False}


def prepare(cell, path):
    p = Path(path)
    need(p.is_absolute() and p.parent == ROOT / "results"
         and p.name.startswith(f"{NAME}_cell{cell:02d}_")
         and p.resolve() == p and not p.exists() and not p.is_symlink(), "exclusive output")
    digest = previous.startup.write_new(str(p), build(cell))
    result = verify(cell, str(p), digest)
    result.update(path=str(p), sha256=digest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, required=True, choices=sorted(PINS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(previous.host.launch.host.raw(prepare(args.cell, args.output)).decode())


if __name__ == "__main__":
    main()
