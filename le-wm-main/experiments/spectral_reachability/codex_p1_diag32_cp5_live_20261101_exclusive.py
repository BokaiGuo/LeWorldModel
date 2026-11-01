"""Three fixed prefixes, actual cp5 RGB and own-suffix planning in one episode."""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import codex_p1_diag32_cp5_own_suffix_20261101_exclusive as fixed
from . import codex_p1_diag32_fixed_physical_20261101_exclusive as physical
from . import preflight_pusht_d2p_p0 as p0

ROOT, engine = fixed.ROOT, physical.engine
NAME = "codex_p1_diag32_cp5_live_20261101_exclusive"
PRESEAL = ROOT / "results/codex_p1_diag32_cp5_own_suffix_20261101_exclusive_PREP_CODEXCP5DEV01.json"
PRESEAL_SHA = "b68ed8cb039fb2ff511ffa58ab5ec329742b43c5547fda37bedfab1a21ca8522"
require, sha = engine.require, engine.sha
CAPS = {key: value * 3 for key, value in physical.CAPS.items()}


def build():
    require(sha(PRESEAL) == PRESEAL_SHA, "preseal hash")
    value = json.loads(PRESEAL.read_bytes())
    require(value == fixed.prepare(), "preseal/source drift")
    prepared, _, root, _, _, _, _ = physical.inputs()
    bindings = dict(value["bindings"])
    for path in (PRESEAL, Path(__file__).resolve(), Path(physical.__file__).resolve(),
                 Path(engine.__file__).resolve(), Path(p0.__file__).resolve(), physical.SOURCE_REPORT,
                 Path(root["path"])):
        path = path.resolve()
        h = sha(path)
        require(str(path) not in bindings or bindings[str(path)] == h, "source collision")
        bindings[str(path)] = h
    require(all(sha(p) == h for p, h in bindings.items()), "source closure")
    require(prepared["cp10_identity"]["root_sha256"] == value["root_sha256"], "root binding")
    return {"schema": NAME + ".execution_contract.v1", "preseal": str(PRESEAL), "preseal_sha256": PRESEAL_SHA,
            "physical_caps_total": CAPS, "neural_caps_per_branch": value["budget_per_role"],
            "roster": value["roles"], "same_episode": True, "wall_seconds": 1800,
            "stop": "native termination/truncation or original RGB/canvas guard; preserve partial",
            "physical_future_used_for_selection": False, "P1_admission": False,
            "prior_costs_not_equalized": True, "bindings": bindings}


class NeuralMeter:
    def __init__(self, store, role, check):
        self.store, self.role, self.check = store, role, check
        self.caps = {"cp5_rgb_encode": 1, "action_encoder": 528, "predict": 528, "reverse": 256}
        self.attempts = dict.fromkeys(self.caps, 0)
        self.returns = dict.fromkeys(self.caps, 0)

    def call(self, kind, context, fn):
        self.check()
        require(kind in self.caps and self.attempts[kind] < self.caps[kind], "neural cap")
        self.attempts[kind] += 1
        self.store.emit("neural_attempt", role=self.role, resource=kind, context=context)
        result = fn()
        self.returns[kind] += 1
        self.store.emit("neural_return", role=self.role, resource=kind, context=context)
        return result

    def complete(self):
        require(self.attempts == self.returns == self.caps, "neural budget incomplete")


def plan_suffix(model, device, task, bank, pixels, pixel_sha, meter, store, role):
    import torch

    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    before_pixels = hashlib.sha256(pixels.tobytes()).hexdigest()
    require(before_pixels == pixel_sha and pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8,
            "current cp5 RGB bytes")
    image = p0._image_tensor(pixels, transform=p0.image_transform(), device=device)
    with torch.inference_mode():
        encoded = meter.call("cp5_rgb_encode", {"pixel_sha256": pixel_sha},
                             lambda: model.encode({"pixels": image}))
    latent = encoded["emb"][0, 0].detach().cpu().numpy().astype(np.float32)
    require(latent.shape == (192,) and np.isfinite(latent).all(), "actual RGB latent")
    return latent


def optimize(model, device, task, bank, meter, store, role):
    import torch

    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    states = tensor(task["root_context"]["latent_history"])[None]
    history = tensor(task["root_context"]["action_history"])[None]
    goal, mean, std = tensor(task["goal"])[None], tensor(task["action_mean"]), tensor(task["action_std"])
    require(bool((std > 0).all()), "normalization")

    def evaluate(candidate, iteration, action):
        z, past = states, history
        for packet in range(2):
            loc = {"candidate": candidate, "iteration": iteration, "packet": packet}
            context = torch.cat((past, ((action[packet] - mean) / std).reshape(1, 1, 10)), 1)
            enc = meter.call("action_encoder", loc, lambda: model.action_encoder(context))
            pred = meter.call("predict", loc, lambda: model.predict(z, enc))
            require(pred.shape == z.shape and bool(torch.isfinite(pred).all()), "predict shape")
            z = torch.cat((z, pred[:, -1:]), 1)[:, -3:]
            past = context[:, -2:]
        return (z[:, -1] - goal).square().mean()

    def reverse(candidate, iteration, loss, action):
        return meter.call("reverse", {"candidate": candidate, "iteration": iteration},
                          lambda: torch.autograd.grad(loss, action)[0])

    def record(row):
        store.emit("candidate_score", role=role, **row)

    return fixed.optimize_suffix_bank(tensor(bank), evaluate, reverse, record)


def run(contract_path, contract_sha, output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    require(sha(contract_path) == contract_sha and json.loads(contract_path.read_bytes()) == build(), "execution contract")
    value = json.loads(PRESEAL.read_bytes())
    source_prepared, data, expected_root, _, _, _, _ = physical.inputs()
    base_task, _, _, _ = fixed.blind.load(0)
    engine.NAME, engine.CAPS = NAME, CAPS
    store = engine.Store(output)
    store.put("execution_binding.json", {"contract": str(contract_path), "contract_sha256": contract_sha,
                                        "preseal_sha256": PRESEAL_SHA})
    start = time.monotonic()
    report = {"schema": NAME + ".report.v1", "status": "partial_or_failed", "rows": [], "errors": [],
              "contract_sha256": contract_sha, "preseal_sha256": PRESEAL_SHA,
              "independent_confirmation": False, "P1_admission": False, "fair_total_cost": False}
    def check():
        require(time.monotonic() - start < 1800, "wall cap")
        store.check()
        require(torch.cuda.memory_allocated(device) <= 4 * 1024**3 and
                torch.cuda.memory_reserved(device) <= 6 * 1024**3, "CUDA memory cap")
    meter = engine.Meter(store, check)
    device = torch.device("cuda:0")
    env = None
    try:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        report["host"] = require_cuda_allocation("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(value["model_policy"], device=device).eval().requires_grad_(False)
        before = engine.signal.model_hash(model)
        require(before["original_state_dict"] == value["model_sha256"], "model hash")
        sources = source_prepared["sources"]
        for module in model.modules():
            path = str(Path(inspect.getfile(type(module))).resolve())
            require(path in sources and sha(path) == sources[path], "model module source")
        report["model_before"] = before
        hooks = {"predict": 0, "action_encoder": 0}
        def count(kind):
            def increment(_module, _args):
                hooks[kind] += 1
            return increment
        handles = [model.predictor.register_forward_pre_hook(count("predict")),
                   model.action_encoder.register_forward_pre_hook(count("action_encoder"))]
        store.emit("run_started", roles=value["roles"], caps=CAPS, preseal_sha256=PRESEAL_SHA)
        with torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for role in value["roles"]:
                row = deepcopy(engine.p3.roster()[0])
                row.update(episode_id=role, role=role, status="not_started", transitions=[], observations=[],
                           suffix=None, cp5=None, neural=None)
                report["rows"].append(row)
                meter.episode, meter.region = role, "setup"
                env = None
                try:
                    env = meter.call("constructor", lambda: engine.old.native_factory(meter))
                    body, ref = env.unwrapped, data["references"][0]
                    require(body._live_meter is meter and body._live_constructor_complete and body.relative
                            and body.action_scale == 100 and body.control_hz == 10 and body.dt == .01
                            and body.with_target is False and body.render_action is False and body.render_size == 224
                            and np.array_equal(env.action_space.low, [-1, -1])
                            and np.array_equal(env.action_space.high, [1, 1]), "native action ABI")
                    row["status"] = "running"
                    if not engine.moving.setup(env, ref, data["plan"], row, meter, store):
                        row["status"] = "native_stopped_warmup"
                        continue
                    root = row["last_observation"]
                    require(all(root[k] == expected_root[k] for k in
                                ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")), "root identity")
                    row["observations"].append(deepcopy(root))
                    success, _ = body.eval_state(ref["goal"], np.asarray(root["state"]))
                    require(not success, "root already stopped")
                    prefix = np.asarray(value["prefixes"][role], np.float32)
                    commands = prefix
                    step = 0
                    stopped = False
                    while step < len(commands):
                        command = commands[step]
                        meter.region = "prefix" if step < 5 else "suffix"
                        store.emit("branch_dispatch", role=role, step=step + 1, command=command.tolist())
                        obs, reward, terminated, truncated, info = env.step(command.copy())
                        step += 1
                        row["control_returned"] += 1
                        actual = np.asarray(body.latest_action)
                        require(engine.old.kernel.moving.exact(actual, command) and
                                np.array_equal(info["goal_state"], ref["goal"]) and
                                np.array_equal(info["goal"], ref["goal_rgb"]), "actual command/goal")
                        state = np.asarray(obs["state"])
                        require(state.shape == (7,) and np.isfinite(state).all() and
                                engine.old.kernel.moving.exact(state, np.asarray(body._get_obs()))
                                and type(info.get("n_contacts")) is int, "physical observation")
                        engine.old.native_transition(body, ref["goal"], obs, reward, terminated, truncated, row)
                        event = {"role": role, "step": step, "command": actual.tolist(), "state": state.tolist(),
                                 "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
                                 "n_contacts": info["n_contacts"], "metrics": deepcopy(row["final_metrics"])}
                        row["executed_raw"].append(actual.tolist())
                        row["transitions"].append(event)
                        store.emit("branch_returned", **event)
                        try:
                            pixels, receipt = engine.old.capture(env, obs, row, store, "plan", step)
                        except engine.old.kernel.moving.Rejected as exc:
                            require(str(exc) == "fresh RGB/canvas guard" and engine.guard_origin(exc), "guard")
                            row.update(status="censored_out_of_domain", censor_step=step)
                            stopped = True
                            break
                        row["observations"].append(deepcopy(receipt))
                        if terminated or truncated:
                            row.update(status="native_stopped", stop_step=step)
                            stopped = True
                            break
                        if step == 5:
                            require(np.array_equal(np.asarray(row["executed_raw"], np.float32), prefix), "actual prefix")
                            saved = Path(receipt["path"])
                            require(saved.parent == store.path and sha(saved) == receipt["file_sha256"], "cp5 PNG")
                            with Image.open(saved) as frame:
                                saved_pixels = np.asarray(frame).copy()
                            require(np.array_equal(saved_pixels, pixels), "cp5 saved/fresh RGB")
                            before_state = engine.old.state_seal(body)
                            nm = NeuralMeter(store, role, check)
                            meter.region = "encoding"
                            latent = plan_suffix(model, device, base_task, value["initial_suffix_bank"],
                                                 saved_pixels, receipt["pixel_sha256"], nm, store, role)
                            identity = {"role": role, "boundary": 5, "pixel_sha256": receipt["pixel_sha256"],
                                        "prefix": prefix.tolist(), "preseal_sha256": PRESEAL_SHA}
                            task = fixed.feedback_task(base_task, prefix, np.asarray(row["executed_raw"], np.float32),
                                                       np.asarray(value["raw_warmup10"], np.float32)[5:], latent, identity)
                            require(task["root_context"]["action_history"] == value["cp5_action_histories_expected"][role],
                                    "cp5 action history")
                            row["cp5"] = {"receipt": deepcopy(receipt), "latent": latent.tolist(),
                                          "task_sha256": engine.digest(task)}
                            meter.region = "planning"
                            hooks_before = deepcopy(hooks)
                            winner = optimize(model, device, task, value["initial_suffix_bank"], nm, store, role)
                            torch.cuda.synchronize()
                            nm.complete()
                            require(all(hooks[k] - hooks_before[k] == 528 for k in hooks), "module call hooks")
                            require(engine.old.state_seal(body) == before_state, "planning changed physical state")
                            row["neural"] = {"attempts": nm.attempts, "returns": nm.returns,
                                             "module_hooks": {k: hooks[k] - hooks_before[k] for k in hooks}}
                            row["suffix"] = winner
                            payload = np.asarray(winner["actions"], dtype="<f4")
                            row["suffix_sha256"] = hashlib.sha256(payload.tobytes()).hexdigest()
                            store.put(role + "_suffix_seal.json", {"cp5": row["cp5"], "winner": winner,
                                      "suffix_sha256": row["suffix_sha256"], "preseal_sha256": PRESEAL_SHA})
                            store.emit("suffix_sealed", role=role, suffix_sha256=row["suffix_sha256"])
                            commands = np.concatenate((prefix, payload.reshape(10, 2)))
                    if not stopped:
                        row["status"] = "completed15"
                finally:
                    meter.region = "closing"
                    if env is not None:
                        meter.call("close", env.close)
                        env = None
                    store.put(role + "_row.json", row)
                print(json.dumps({"role": role, "status": row["status"], "steps": row["control_returned"]}), flush=True)
        torch.cuda.synchronize()
        for handle in handles:
            handle.remove()
        report["model_after"] = engine.signal.model_hash(model)
        require(report["model_before"] == report["model_after"] and not meter.pending and not meter.secondary,
                "model/accounting")
        require(all(sha(p) == h for p, h in json.loads(contract_path.read_bytes())["bindings"].items()), "source drift")
        report["status"] = "recorded_development"
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
    finally:
        if env is not None:
            meter.region = "closing"
            try:
                meter.call("close", env.close)
            except BaseException as exc:
                report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
        store.finalizing = True
        report.update(counts=meter.counts, pending=meter.pending, secondary_errors=meter.secondary,
                      wall_seconds=time.monotonic()-start)
        store.emit("run_finished", status=report["status"], counts=meter.counts)
        report["journal_sha256"] = sha(store.path / "journal.jsonl")
        report["journal_chain_tail"] = store.previous
        store.put("report.json", report, final=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prep = ROOT / "results" / (NAME + "_PREP_" + args.tag + ".json")
    output = ROOT / "results" / (NAME + "_run_" + args.tag)
    require(not os.path.lexists(prep) and not os.path.lexists(output), "exclusive output")
    digest = fixed.blind.write_new(prep, build())
    print(json.dumps({"contract": str(prep), "sha256": digest}), flush=True)
    with physical.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        result = run(prep, digest, output)
    print(json.dumps({"status": result["status"], "output": str(output), "errors": result["errors"],
                      "report_sha256": sha(output / "report.json")}), flush=True)
    return int(result["status"] != "recorded_development")


if __name__ == "__main__":
    raise SystemExit(main())
