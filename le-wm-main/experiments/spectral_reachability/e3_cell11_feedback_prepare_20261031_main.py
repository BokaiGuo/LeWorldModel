"""Prepare the fixed cell11 feedback authority; no neural or physical execution."""
import argparse
from copy import deepcopy
from pathlib import Path

from . import e3_cell10_feedback_prepare_codex_20261030_v1 as prior

NAME = "e3_cell11_feedback_prepare_20261031_main"
ROOT = prior.ROOT
CELL, TASK = 11, "r1.g0.h3.dDx"
SHA = {"report": "3369a067fb4c4244e1ae1ac82da9e217d6e134526980180e296c908473d1cb4d",
       "completion": "17b581220bd6e4cc1d325f97f1044e97681a1e2278dc9c9d6f8f0b2eca6e073e",
       "receipt": "7f90f15e5e477d48629ee851f38e03007cf53251de0b6025a97ccc056d423a03"}
need = prior.need


def build():
    need(prior.resolver.roster()[CELL]["cell_id"] == TASK, "cell11 task")
    prior.host.authority(prior.BASE, prior.BASE_SHA)
    base = prior.host.launch.host.read(prior.BASE, prior.BASE_SHA)
    refs = deepcopy(base["control5"])
    for kind in ("report", "completion"):
        refs[kind] = {"path": str(ROOT / f"results/agent_e3_gd_host_cuda_20261029_v1_cell11_MAIN01/{kind}.json"),
                      "sha256": SHA[kind]}
    refs["receipt"] = {"path": str(ROOT / "results/agent_e3_gd_host_cuda_20261029_v1_consume_cell11_MAIN01.json"),
                       "sha256": SHA["receipt"]}
    fresh = prior.identity.build(refs, base["resolver"], CELL)
    need(base["cell"] == 0 and fresh["cell"] == CELL and fresh["schema"] == base["schema"]
         and fresh["resolver"] == base["resolver"] and fresh["control5"] == refs
         and refs["prepare"] == base["control5"]["prepare"], "source identity")
    sources = dict(base["additional_sources"])
    for path, digest in fresh["additional_sources"].items():
        need(path in sources and sources[path] == digest, "inherited source pins")
    need(prior.host.launch.host.sha(prior.BASE) == prior.BASE_SHA, "base drift")
    return {"schema": base["schema"], "cell": CELL, "control5": refs,
            "resolver": deepcopy(base["resolver"]), "additional_sources": sources}


def verify(path, digest):
    need(prior.host.launch.host.read(path, digest) == build(), "authority reconstruction")
    binding = prior.host.authority(path, digest)
    old = prior.host.launch.episode.old
    joined = old.authority.load(*binding["args"])
    setup = old.setup_authority.resolve(binding["resolver"]["path"],
                                        binding["resolver"]["sha256"], CELL, joined["body"])
    need(binding["cell"] == joined["cell"] == CELL and joined["body"]["horizon_packets"] == 3
         and joined["body"]["domain"]["name"] == "Dx" and setup["expected_cp5"]["step"] == 5,
         "native setup")
    return {"verified": True, "cell": CELL, "task": TASK, "native_saved_setup_resolved": True,
            "freshness": False, "production_authorized": False, "scientific_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    need(path.is_absolute() and path.parent == ROOT / "results" and path.name.startswith(NAME + "_")
         and path.resolve() == path and not path.exists() and not path.is_symlink(), "exclusive output")
    digest = prior.startup.write_new(str(path), build())
    result = verify(str(path), digest)
    result.update(path=str(path), sha256=digest)
    print(prior.host.launch.host.raw(result).decode())


if __name__ == "__main__":
    main()
