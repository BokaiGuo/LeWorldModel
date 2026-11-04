"""Four fixed P3 winners on CPU; exact moving-root guards precede each plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from . import p3_venv_winner_physical_prepare_20261102_codexexclusive as selected
from . import p1_constructive_moving_history_physical_20261027_v1 as moving
from . import p1_moving_extended_baseline_physical_20261027_v1 as checks

ROOT = selected.ROOT
NAME = "p3_venv_winner_cpu_physical_20261102_codexexclusive"
SOURCE = ROOT / "results/p3_venv_winner_physical_prepare_20261102_codexexclusive_01.json"
SOURCE_SHA = "7f0d3698c324304551e4b9ab11bce7a871ba2463511dd1bf4aa733029c1ef889"
AUTH = "P3_KNOWN_ROOT_FOUR_WINNER_CPU_PHYSICS"


def require(value, message):
    if not value:
        raise ValueError(message)


def material():
    require(selected.host.host.prior.sha(SOURCE) == SOURCE_SHA, "winner preparation SHA")
    sealed = json.loads(SOURCE.read_text())
    require(sealed == selected.build(), "winner preparation reconstruction")
    require([(x["arm"], x["root_slot"]) for x in sealed["plans"]] ==
            [("signal", 0), ("signal", 1), ("reverse", 0), ("reverse", 1)], "fixed roster")
    data = moving.load_source(selected.source.SOURCEPACK, selected.source.SOURCEPACK_SHA,
                              selected.source.SOURCE_PREPARE, selected.source.SOURCE_PREPARE_SHA)
    plans = []
    for item in sealed["plans"]:
        payload = {k: v for k, v in item.items() if k != "selection_sha256"}
        require(selected.journal.digest(payload) == item["selection_sha256"], "selection SHA")
        raw = selected.host.host.prior.prior.raw_commands(item["commands"])
        require(raw.shape == (10, 2) and hashlib.sha256(raw.tobytes()).hexdigest()
                == item["raw_float32_sha256"], "action bits")
        plan = dict(item)
        plan["selection_digest"] = item["selection_sha256"]
        plans.append(plan)
    native, native_sha = checks.native_constructor_contract()
    bindings = dict(data["bindings"])
    bindings[str(SOURCE)] = SOURCE_SHA
    bindings[native] = native_sha
    for path in (Path(__file__), Path(selected.__file__), Path(moving.__file__),
                 Path(moving.low.__file__), Path(moving.moving.__file__)):
        bindings[str(path.resolve())] = moving.sha(path)
    for path, expected in bindings.items():
        require(moving.sha(path) == expected, "source binding drift: " + path)
    contract = moving.contract(600)
    contract.update(scope="four fixed winners on known roots, no outcome selection or retry",
                    no_gpu=True, physical_replans=0)
    prepare = {"schema": NAME + ".prepare.v1", "plans": plans,
               "contract": contract, "bindings": bindings,
               "sourcepack_sha256": selected.source.SOURCEPACK_SHA,
               "source_prepare_sha256": selected.source.SOURCE_PREPARE_SHA,
               "selection_sha256": SOURCE_SHA, "native_constructor": native,
               "native_constructor_sha256": native_sha,
               "physical_execution": False, "new_neural_calls": 0,
               "historical_environment_equivalence": "unknown", "independent_outcome": False,
               "P3_admission": False}
    return prepare, data


def output(path, prefix):
    p = Path(path)
    require(p.is_absolute() and p.parent == ROOT / "results"
            and p.name.startswith(NAME + prefix) and not os.path.lexists(p), "exclusive output")
    return p


def verify(path, expected):
    require(moving.sha(path) == expected, "external physical prepare SHA")
    saved = moving.read(path)
    current, data = material()
    require(saved == current, "physical prepare drift")
    return saved, data


def run(prepared, expected, out, authority):
    require(authority == AUTH, "explicit CPU physical authority")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
    p, data = verify(prepared, expected)
    out = output(out, "_RUN_")
    store = moving.Store(out)
    store.put("execution_binding.json", {"prepared": str(Path(prepared).resolve()),
                                         "prepare_sha256": expected})
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                        with_target=False, render_action=False)
    return moving.execute(p, data, store, factory, authorization=moving.AUTH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify", "run"))
    parser.add_argument("--output")
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--authority")
    args = parser.parse_args()
    if args.action == "prepare":
        value, _ = material()
        path = output(args.output, "_PREP_")
        print(json.dumps({"prepared": str(path), "sha256": selected.host.host.write_new(path, value)}))
    elif args.action == "verify":
        value, _ = verify(args.prepared, args.sha256)
        print(json.dumps({"status": "prepared_only", "branches": len(value["plans"])}))
    else:
        result = run(args.prepared, args.sha256, args.output, args.authority)
        print(json.dumps({"status": result["status"], "errors": result["errors"],
                          "rows": [{"root": x["root_slot"], "arm": x["arm"],
                                    "status": x["status"], "initial_exact": x["initial_exact"],
                                    "own_returned": x["own_returned"]} for x in result["rows"]]}))


if __name__ == "__main__":
    main()
