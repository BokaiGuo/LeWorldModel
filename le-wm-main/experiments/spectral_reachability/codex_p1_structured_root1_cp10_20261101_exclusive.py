"""Known root1 structured feedback; process-local adapter, development only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from . import codex_four_arm_common_rgb_root1_20261030_main as host
from . import codex_p1_common_structured_host_20261101_exclusive as structured
from . import e3_cp10_four_arm_root1_20261031 as cp10
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

NAME = "codex_p1_structured_root1_cp10_20261101_exclusive"
REPORT = host.ROOT / "results/p1_common_budget_cem_host_20261101_leibniz_root1_RUN_MAIN02/report.json"
PINS = {
    REPORT: "b9f2d1a5a5b7653ec0ddaaebe2608bd72efd1a886d72441d4a4cb1b9de503479",
    Path(host.__file__).resolve(): "3379971996ec834ed52f4aacd91e2045998686c48db60d2826ccbce913c3f71b",
    Path(structured.__file__).resolve(): "fc2fbeebfc5e6694c32081187d094e3183926cc740b985f1b740a1f10c02d01c",
    Path(structured.adapter.__file__).resolve(): "994fdc70825e87e79b7e0cdfb32712026844b46aec7dd799943a2a518ff4fc59",
    Path(cp10.__file__).resolve(): "254239a3a79f61cf9e1f8067b54775fda9550dfbed2d22556802b64eec5da501",
    Path(locking.__file__).resolve(): "e4ca4fca376b0d111cc53db40a83f51dd8958745ca3eee5aa9444ad9f0b9c748",
}


def check_pins():
    for path, expected in PINS.items():
        host.require(path.is_file() and not path.is_symlink() and host.sha(path) == expected,
                     "pinned source drift: " + str(path))


def check_closure(prepared):
    for name, expected in prepared["sources"].items():
        path = Path(name)
        host.require(path.is_file() and host.sha(path) == expected, "source closure drift: " + name)


def bind(prepared, data):
    check_pins()
    cem = json.loads(REPORT.read_bytes())
    host.require(cem["status"] == "recorded_known_root_not_admitted"
                 and cem["protocol"]["root_slot"] == 1 and len(cem["rows"]) == 1
                 and cem["rows"][0]["source_slot"] == 1, "CEM root identity")
    entry, target, sources = cp10.inputs()
    identity = cp10.rebind(prepared, data)
    task = prepared["source"]["tasks"][1]
    zero = [x for x in cem["rows"][0]["checkpoints"] if x["step"] == 0]
    host.require(identity == cem["cp10_goal_binding"] and len(zero) == 1
                 and zero[0]["goal"] == entry["goal_state"] == target["goal_state"]
                 and zero[0]["goal_rgb_sha256"] == entry["pixel_sha256"]
                 and np.array_equal(np.asarray(zero[0]["state"]), np.asarray(entry["root_state"]))
                 and host.digest(task["goal"]) == identity["goal_latent_sha256"]
                 and task["binding"]["root_sha256"] == identity["root_sha256"]
                 and prepared["source"]["expected_model_hash"] == cem["model_before"]["original_state_dict"]
                 == cem["model_after"]["original_state_dict"], "CEM binding mismatch")
    return identity, sources


def configure():
    check_pins()
    original_build = host.build

    def prebound():
        value = original_build()
        data = host.p3.source_data(value["source"])
        original_root1_task = deepcopy(value["source"]["tasks"][1])
        identity, sources = bind(value, data)
        # Structured protocol builder is hardwired to task[0]; execution and
        # visual source stay explicitly root1 in the root1 host.
        value["source"]["tasks"][0] = deepcopy(value["source"]["tasks"][1])
        value["sources"].update(sources)
        value["sources"].update({str(p): h for p, h in PINS.items()})
        value["sources"][str(Path(__file__).resolve())] = host.sha(Path(__file__).resolve())
        value["cp10_identity"] = identity
        value["original_root1_task"] = original_root1_task
        return value

    host.build = prebound
    structured.host = host
    structured.configure()
    host.NAME = NAME
    old_build, old_protocol, old_execute = host.build, host.protocol, host.execute

    def protocol():
        value = old_protocol()
        value.update(root_slot=1, goal="root1_previously_observed_cp10",
                     target_rebinding_before_protocol=True, claim="known_root_development_only",
                     fairness_proven=False, independent_confirmation=False, P1_admission=False)
        return value

    def build():
        value = old_build()
        identity = value["cp10_identity"]
        common = value["common_node_protocol"]["base_protocol"]["identity"]
        host.require(common["root"] == identity["root_sha256"]
                     and common["goal"] == identity["goal_latent_sha256"]
                     and value["source"]["tasks"][0] == value["source"]["tasks"][1],
                     "structured protocol root1 mismatch")
        return value

    def execute(prepared, data, store, visual, runtime, factory):
        check = deepcopy(prepared)
        check["source"]["tasks"][1] = deepcopy(prepared["original_root1_task"])
        observed, _ = bind(check, data)
        host.require(observed == prepared["cp10_identity"]
                     and check["source"]["tasks"][1] == prepared["source"]["tasks"][1],
                     "runtime root1 binding drift")
        store.put("cp10_goal_binding.json", {**observed, "cem_report": str(REPORT),
                                             "cem_report_sha256": PINS[REPORT],
                                             "goal_state": data["references"][1]["goal"].tolist()})
        result = old_execute(prepared, data, store, visual, runtime, factory)
        result["cp10_goal_binding"] = observed
        return result

    host.build, host.protocol, host.execute = build, protocol, execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    host.require(args.tag.isalnum(), "alphanumeric tag")
    configure()
    prefix = host.ROOT / "results" / NAME
    prepared = Path(str(prefix) + "_PREP_" + args.tag + ".json")
    output = Path(str(prefix) + "_RUN_" + args.tag)
    host.require(not prepared.exists() and not prepared.is_symlink() and
                 not output.exists() and not output.is_symlink(), "exclusive outputs")
    seal = host.prepare(str(prepared))
    print(json.dumps({"prepared": seal}), flush=True)
    if args.run:
        with locking.gpu_lock(host.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            check_closure(host.verify(str(prepared), seal["sha256"]))
            result = host.run(str(prepared), seal["sha256"], str(output), host.AUTH)
        print(json.dumps(result), flush=True)
        return 0 if result["status"] == "recorded_known_root_not_admitted" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
