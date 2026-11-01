"""Prepare fixed cell10 control5 inputs for the existing E3 feedback host.

Preparation does not authorize or launch a model, GPU, or physical episode.
"""
import argparse
from copy import deepcopy
from pathlib import Path

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_identity_authority_prepare_20261029_v1 as identity
from experiments.spectral_reachability import agent_e3_startup_closure_20261029_v1 as startup
from experiments.spectral_reachability import agent_e3_frozen_model_task_resolver_20261029_v1 as resolver

NAME = "e3_cell10_feedback_prepare_codex_20261030_v1"
ROOT = host.launch.module_census.ROOT
BASE = str(ROOT / "results/agent_e3_observed_authority_20261029_v1_MAIN02.json")
BASE_SHA = "ea2d7d747e02c0054eca90092ca947e136cdba10aca3a204dfa85d8b01ec64e9"
CELL, TASK = 10, "r1.g0.h3.dDxy"
SHA = {
    "report": "c2da98188ce5b0ba121226e48321bbce51b3c24bf05730fdc1ecc09c65d25b95",
    "completion": "da6a9bac1b5e7f87ba612594138036da2eb5c6e1d156d713f47a831778c6abf7",
    "receipt": "878af55744906ef4cc5c052ed1462e2c0d2416c79dcac5f57e40735a2a958801",
}
need = host.need


def references(prior):
    refs = deepcopy(prior["control5"])
    for key in ("report", "completion"):
        refs[key] = dict(path=str(ROOT / f"results/agent_e3_gd_host_cuda_20261029_v1_cell10_MAIN01/{key}.json"),
                         sha256=SHA[key])
    refs["receipt"] = dict(path=str(ROOT / "results/agent_e3_gd_host_cuda_20261029_v1_consume_cell10_MAIN01.json"),
                           sha256=SHA["receipt"])
    return refs


def combine(prior, fresh):
    refs = references(prior)
    need(prior["cell"] == 0 and fresh["cell"] == CELL and prior["schema"] == fresh["schema"] and
         prior["resolver"] == fresh["resolver"] and
         prior["control5"]["prepare"] == fresh["control5"]["prepare"] and fresh["control5"] == refs,
         "fixed cell10 upstream identity")
    sources = dict(prior["additional_sources"])
    for path, sha in fresh["additional_sources"].items():
        need(path in sources and sources[path] == sha, "strict inherited source pins: " + path)
    return dict(schema=prior["schema"], cell=CELL, control5=deepcopy(refs),
                resolver=deepcopy(prior["resolver"]), additional_sources=sources)


def build():
    need(resolver.roster()[CELL]["cell_id"] == TASK, "fixed cell10 task")
    host.authority(BASE, BASE_SHA)
    prior = host.launch.host.read(BASE, BASE_SHA)
    fresh = identity.build(references(prior), prior["resolver"], CELL)
    value = combine(prior, fresh)
    need(host.launch.host.sha(BASE) == BASE_SHA, "stable base authority")
    return value


def verify(path, sha):
    need(host.launch.host.read(path, sha) == build(), "whole cell10 authority reconstruction")
    binding = host.authority(path, sha)
    need(binding["cell"] == CELL and host.launch.host.sha(path) == sha and
         host.launch.host.sha(BASE) == BASE_SHA, "final strict source and cell")
    old = host.launch.episode.old
    joined = old.authority.load(*binding["args"])
    setup = old.setup_authority.resolve(binding["resolver"]["path"],
                                        binding["resolver"]["sha256"], CELL, joined["body"])
    need(joined["cell"] == CELL and joined["body"]["horizon_packets"] == 3 and
         joined["body"]["domain"]["name"] == "Dxy" and setup["expected_cp5"]["step"] == 5,
         "cell10 native setup")
    return dict(verified=True, cell=CELL, task=TASK, freshness=False,
                native_saved_setup_resolved=True,
                production_authorized=False, scientific_admission=False)


def prepare(path):
    p = Path(path)
    need(p.is_absolute() and p.parent == ROOT / "results" and p.name.startswith(NAME + "_") and
         p.resolve() == p and not p.exists() and not p.is_symlink(), "own exclusive result")
    sha = startup.write_new(str(p), build())
    verify(str(p), sha)
    return dict(path=str(p), sha256=sha, cell=CELL, task=TASK, freshness=False,
                production_authorized=False, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--output", required=True)
    p = sub.add_parser("verify"); p.add_argument("--authority", required=True); p.add_argument("--authority-sha256", required=True)
    args = parser.parse_args()
    result = prepare(args.output) if args.command == "prepare" else verify(args.authority, args.authority_sha256)
    print(host.launch.host.raw(result).decode())


if __name__ == "__main__":
    main()
