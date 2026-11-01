"""Score eight frozen coherent first packets; no physical feedback in selection."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from experiments import checkpoint_compat, gpu_preflight
from . import codex_four_arm_full_cost_20261030 as spec
from . import codex_four_arm_durable_20261030 as durable
from . import codex_four_arm_leaf_suffix_20261030 as leafmod
from . import codex_four_arm_consumer_20261030 as consumer
from . import p1_saved_h4_parfit_20261101 as saved
from . import p1_centered_h4_parfit_20261101_adapter as centered
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import p1_four_arm_extra32_root0_20261031 as extra

NAME = "p1_coherent_same_bank_greedy_20261101_codex"
ROOT = centered.ROOT
REPORT = ROOT / "results/p1_centered_h4_coherent_20261101_launcher_structured_RUN_COHERENTMAIN01/report.json"
REPORT_SHA = "5be4ad99b9140af2f3e0d73389f411fb08007eb71518f1b52d9984a6aeb43a1f"
TRACE_SHA = "645e63280cd27de6feedf6665a0f467793f3545ad9e952f6dde68e6b006eebfa"
PREP_SHA = "ac2e29eeed44c5d48d8fbd050b554435a17394ac369f079685086483add4ad41"
DOC = ROOT.parent / "docs/D2P_P1_COHERENT_SAME_BANK_GREEDY_20261101_CODEX.md"
AUTH = "MAIN_AUTHORIZES_SAME_BANK_GREEDY_HOST_CUDA"
COST = dict.fromkeys(spec.AXES, 0)
COST.update(action_encode_calls=8, predict_calls=8, predict_samples=8,
            replay_calls=8, selection_evaluations=8)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return saved.sha(path)


def frozen_candidates(events):
    def one(kind):
        matches = [row["payload"] for row in events if row["kind"] == kind]
        require(len(matches) == 1, "unique " + kind)
        return matches[0]

    initial = np.asarray(one("initial_candidates")["raw"], np.float32)
    optimized = np.asarray(one("optimized_candidates_before_exchange")["raw"], np.float32)
    require(initial.shape == optimized.shape == (8, 4, 5, 2)
            and np.isfinite(initial).all() and np.isfinite(optimized).all()
            and (np.abs(initial) <= 1).all() and (np.abs(optimized) <= 1).all()
            and np.array_equal(initial[:, 0], optimized[:, 0]),
            "initial bank/optimized prefix identity")
    return initial[:, 0].copy()


def build():
    require(sha(REPORT) == REPORT_SHA, "source report SHA")
    report = json.loads(REPORT.read_bytes())
    require(report["model_before"] == report["model_after"], "source frozen model")
    prepared = Path(report["prepared"])
    require(report["prepare_sha256"] == PREP_SHA and sha(prepared) == PREP_SHA, "source prepare SHA")
    source = json.loads(prepared.read_bytes())
    require(source["schema"] == "p1_reverse_h4_parfit_20261101.prepare.v1"
            and source["protocol"]["initializer_variant"] == "packet_constant_coherent",
            "source coherent H4")
    # Read node identity and trace receipt, never its physical outcomes or later actions.
    node = report["nodes"]["s0_structured"][0]
    require(node["boundary"] == 0 and node["status"] == "sealed", "sealed first node")
    receipt = node["receipt"]
    trace = Path(receipt["trace"])
    require(trace.is_absolute() and trace.resolve().is_relative_to(REPORT.parent)
            and receipt["trace_sha256"] == TRACE_SHA and sha(trace) == TRACE_SHA,
            "frozen trace identity")
    events = saved.chain(trace)
    prefixes = frozen_candidates(events)
    header = [e["payload"] for e in events if e["kind"] == "h4_contract_binding"]
    require(len(header) == 1 and header[0]["boundary"] == 0
            and header[0]["policy"]["horizon"] == 4
            and header[0]["projection"] == "identity_192_exact"
            and header[0]["contract_sha256"] == receipt["contract_sha256"],
            "source H4 policy")
    ledger = receipt["ledger"]
    ledger_path = Path(ledger["journal"])
    require(ledger_path.resolve().is_relative_to(REPORT.parent)
            and sha(ledger_path) == ledger["sha256"], "source node ledger")
    first = json.loads(ledger_path.read_bytes().splitlines()[0])["payload"]["contract"]
    require(spec.digest(first) == receipt["contract_sha256"]
            and first["phase_policy_sha256"] == spec.digest(header[0]["policy"])
            and first["identity"]["optimizer_config"] == spec.digest(header[0]["config"]),
            "source node policy/contract")
    account = consumer.consume(ledger_path, ledger["sha256"], receipt["contract_sha256"],
                               "structured", "node", "structured")
    require(account == receipt["account"] and account["status"] == "closed_unverified",
            "source node ledger receipt")
    report_data, target, metric, _ = centered.load()
    model_hash = json.loads(centered.TARGET.read_bytes())["model_before"]
    require(report["model_before"] == model_hash
            and source["source"]["expected_model_hash"] == model_hash["original_state_dict"],
            "same frozen model")
    task = source["source"]["tasks"][0]
    mean, std = np.asarray(task["action_mean"], np.float32), np.asarray(task["action_std"], np.float32)
    raw10 = np.asarray(report_data["raw_history"], np.float32)
    require(raw10.shape == (10, 2) and mean.shape == std.shape == (2,)
            and np.isfinite(raw10).all() and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std > 0).all(), "raw warmup/normalizer")
    history = ((raw10 - mean) / std).reshape(2, 10).tolist()
    require(history == task["root_context"]["action_history"], "actual normalized root history")
    encodings = [entry for entry in report["rows"][0]["encodings"] if entry["step"] == 0]
    require(len(encodings) == 1, "unique boundary-0 encoding")
    artifact = encodings[0]["latent"]
    root_latent_path = Path(artifact["path"])
    require(root_latent_path.resolve().is_relative_to(REPORT.parent)
            and sha(root_latent_path) == artifact["file_sha256"], "boundary-0 latent source")
    actual_latent = np.load(root_latent_path, allow_pickle=False)
    require(actual_latent.shape == (192,) and actual_latent.dtype == np.float32
            and np.array_equal(actual_latent, np.asarray(target["history_latents"][-1], np.float32)),
            "actual boundary-0 latent equals centered history")
    sources = dict(source["sources"])
    for path in (REPORT, prepared, trace, ledger_path, root_latent_path,
                 centered.TARGET, centered.METRIC, centered.REPORT, DOC,
                 Path(__file__), Path(centered.__file__), Path(spec.__file__), Path(durable.__file__),
                 Path(leafmod.__file__), Path(consumer.__file__), Path(saved.__file__),
                 Path(checkpoint_compat.__file__), Path(gpu_preflight.__file__), Path(locking.__file__)):
        path = path.resolve()
        sources[str(path)] = sha(path)
    require(all(sha(path) == expected for path, expected in sources.items()), "source closure")
    require(target["latent"] == metric["goal_latent"] and len(target["history_latents"]) == 3,
            "goal and RGB history")
    return dict(schema=NAME + ".prepare.v1", prefixes=prefixes.tolist(),
                latent_history=target["history_latents"], action_history=history,
                warmup_raw=raw10.tolist(), mean=mean.tolist(), std=std.tolist(),
                goal=target["latent"], matrix=metric["sqrt_matrix"],
                expected_model_hash=model_hash, model_policy=source["source"]["policy"],
                sources=sources, source_report_sha256=REPORT_SHA, source_trace_sha256=TRACE_SHA,
                source_prepare_sha256=PREP_SHA, goal_metric_sha256=centered.METRIC_SHA,
                expected_cost=COST, tie="first_index", physical_execution=False,
                P1_admission=False, independent_confirmation=False)


def make_contract(value):
    own_sha = sha(Path(__file__).resolve())
    config = dict(horizon=1, candidates=8, metric="centered_goal_Q",
                  tie="first_index", history=3)
    policy = dict(schema=NAME + ".policy.v1", **config)
    identity = dict(root=spec.digest(value["latent_history"]), task=spec.digest([
        value["prefixes"], value["action_history"], value["goal"], value["goal_metric_sha256"]]),
        model=value["expected_model_hash"]["original_state_dict"],
        action_domain=spec.digest(dict(bounds=[-1, 1], packet=[5, 2], normalized=True)),
        history=spec.digest([value["latent_history"], value["action_history"]]),
        goal=spec.digest(value["goal"]), optimizer=own_sha, optimizer_config=spec.digest(config),
        meter=sha(Path(durable.__file__)), hardware=spec.digest({"device_policy": "host_cuda_0"}))
    arms = {arm: dict(initializer_sha256=value["source_trace_sha256"], optimizer_sha256=own_sha,
                      optimizer_config_sha256=identity["optimizer_config"]) for arm in spec.ARMS}
    contract = dict(schema=spec.NAME + ".contract.v1", identity=identity, arms=arms,
                    levels=[dict(id="node", caps=deepcopy(COST)),
                            dict(id="larger", caps={k: v + 1 for k, v in COST.items()})],
                    scope="standalone_per_root_per_arm_including_attributable_setup",
                    phase_policy_sha256=spec.digest(policy))
    spec.preflight(contract, spec.digest(contract))
    return policy, contract, config


def score_bank(leaf, prefixes, states, history, goal, mean, std, matrix):
    """One action_encoder + one predict per prefix, unchanged Q; no gradients."""
    if (not isinstance(prefixes, torch.Tensor) or prefixes.shape != (8, 5, 2)
            or states.shape != (1, 3, 192) or history.shape != (1, 2, 10)
            or goal.shape != (1, 192) or mean.shape != (2,) or std.shape != (2,)
            or matrix.shape != (192, 192) or not bool((std > 0).all())):
        raise ValueError("one-packet input shape")
    scores = []
    with torch.no_grad():
        for prefix in prefixes:
            leaf._dispatch("verification", "replay", {"replay_calls": 1}, lambda: None)
            packet = ((prefix - mean) / std).reshape(1, 1, 10)
            actions = torch.cat((history, packet), dim=1)
            prediction = leaf.predict(states, leaf.action_encode(actions, phase="verification"),
                                      phase="verification")
            require(prediction.shape == states.shape and bool(torch.isfinite(prediction).all()),
                    "finite one-packet prediction")
            def evaluate():
                residual = (prediction[:, -1] - goal).double()
                value = (residual @ matrix.double().T).square().mean()
                require(bool(torch.isfinite(value)), "finite Q")
                return float(value.item())
            scores.append(leaf._dispatch(
                "verification", "selection", {"selection_evaluations": 1}, evaluate))
    return scores, min(range(8), key=lambda i: (scores[i], i))


def exclusive(path):
    path = Path(path)
    require(path.is_absolute() and path.parent == ROOT / "results"
            and path.name.startswith(NAME + "_") and not path.exists() and not path.is_symlink(),
            "exclusive output path")
    return path


def write(path, payload):
    with path.open("xb") as stream:
        raw = spec.canonical(payload) + b"\n"
        require(stream.write(raw) == len(raw), "complete write")
        stream.flush()
        os.fsync(stream.fileno())
    return sha(path)


def run(prepared, expected, output, authorization):
    require(authorization == AUTH, "host CUDA authorization")
    prepared = Path(prepared)
    require(prepared.is_absolute() and prepared.parent == ROOT / "results"
            and not prepared.is_symlink() and spec.is_hash(expected) and sha(prepared) == expected,
            "prepared SHA/path")
    value = json.loads(prepared.read_bytes())
    require(value == build(), "prepared/source drift")
    output = exclusive(output)
    require(shutil.disk_usage(output.parent).free >= 512 * 1024**2, "disk reserve")
    with locking.gpu_lock(extra.gpu.LOCK):
        output.mkdir()
        binding_sha = write(output / "binding.json", value)
        result, error, before, after, host, ledger, account = None, None, None, None, None, None, None
        started = time.monotonic()
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            host = gpu_preflight.require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                model = checkpoint_compat.load_compatible_pretrained(value["model_policy"], device=device).eval().requires_grad_(False)
                before = extra.runner.signal.model_hash(model)
                require(before == value["expected_model_hash"], "frozen model hash")
                policy, contract, config = make_contract(value)
                contract_sha = spec.digest(contract)
                driver = durable.Driver(output / "ledger", contract, contract_sha, "structured", "node", "same_bank")
                try:
                    leaf = leafmod.LeafAdapter(model, driver, contract, contract_sha)
                    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
                    require(time.monotonic() - started < 600
                            and torch.cuda.memory_allocated(device) <= 4 * 1024**3
                            and torch.cuda.memory_reserved(device) <= 6 * 1024**3, "wall/CUDA cap")
                    scores, index = score_bank(leaf, tensor(value["prefixes"]),
                        tensor(value["latent_history"])[None], tensor(value["action_history"])[None],
                        tensor(value["goal"])[None], tensor(value["mean"]), tensor(value["std"]),
                        torch.as_tensor(value["matrix"], dtype=torch.float64, device=device))
                    torch.cuda.synchronize(device)
                finally:
                    ledger = driver.finish()
                account = consumer.consume(ledger["journal"], ledger["sha256"], contract_sha,
                                           "structured", "node", "same_bank")
                require(account["status"] == "closed_unverified"
                        and account["charged_upper_bounds"] == COST, "exact one-packet charges")
                after = extra.runner.signal.model_hash(model)
                require(before == after and all(sha(p) == h for p, h in value["sources"].items()),
                        "model/source drift")
                result = dict(prefixes=value["prefixes"], scores=scores, greedy_index=index,
                              selected_prefix=value["prefixes"][index], policy=policy, config=config,
                              contract_sha256=contract_sha, ledger=ledger, account=account)
        except BaseException as exc:
            error = dict(type=type(exc).__name__, message=str(exc)[:1000])
        report = dict(schema=NAME + ".report.v1", status="complete" if error is None else "partial",
                      error=error, result=result, binding_sha256=binding_sha, prepared=str(prepared),
                      prepare_sha256=expected, host=host, model_before=before, model_after=after,
                      source_report_sha256=REPORT_SHA, source_trace_sha256=TRACE_SHA,
                      physical_execution=False, P1_admission=False, independent_confirmation=False,
                      limitations="known-root development; no independent CUDA-kernel accounting")
        report_sha = write(output / "report.json", report)
        return dict(status=report["status"], output=str(output), report_sha256=report_sha, error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    execute = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        execute.add_argument("--" + key, required=True)
    execute.add_argument("--authorize-host-cuda", required=True, action="store_true")
    args = parser.parse_args()
    if args.command == "dry-prepare":
        value = build()
        _, contract, _ = make_contract(value)
        result = dict(status="dry_prepared_not_written", build_digest=spec.digest(value),
                      prefix_count=len(value["prefixes"]), contract_sha256=spec.digest(contract),
                      cost=COST, cuda_used=False, no_model_executed=True)
    elif args.command == "prepare":
        path = exclusive(args.output)
        result = dict(path=str(path), sha256=write(path, build()))
    else:
        result = run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
