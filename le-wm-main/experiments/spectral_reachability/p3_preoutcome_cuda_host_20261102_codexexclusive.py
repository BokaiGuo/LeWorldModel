"""Known-root forward-dynamics P3 development host; no physics or training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np

from . import p3_preoutcome_classified_control_20261102_codexexclusive as control
from . import p3_venv_live_host_20261102_codexexclusive as old
from . import p3_preoutcome_saved_consumer_20261102_codexexclusive as consumer

host, ROOT = old.host, old.ROOT
NAME = "p3_preoutcome_cuda_host_20261102_codexexclusive"
AUTHORITY = "KNOWN_ROOT_P3_PREOUTCOME_FORWARD_HOST_CUDA_COORDINATED"
BASE = ROOT / "results/p3_venv_live_host_20261102_codexexclusive_PREP_01.json"
BASE_SHA = "8e0a8010a35750dddfe1d1f9b172b9aedfd474c7206def74767efded6b22f11f"
CAPS = {"observation_encode": 0, "action_encode": 44, "predict_calls": 44,
        "predict_samples": 44, "signal_compute": 2, "candidate_evaluate": 20}


def build():
    previous, tasks, banks = old.verify(BASE, BASE_SHA)
    base = dict(previous["host_plan"])
    inventory = {x["path"]: x for x in base["sources"]}
    for item in host.identity.inventory([Path(__file__), Path(control.__file__), Path(consumer.__file__)]):
        inventory[item["path"]] = item
    base["sources"] = sorted(inventory.values(), key=lambda x: x["path"])
    binding = {"protocol": NAME + ".forward_latent_displacement_h2.v1",
               "model": base["expected_model_hash"],
               "roots": control.digest(previous["task_sha256"]),
               "feedback": "fixed_authenticated_preexisting_latent_context"}
    schedule = control.freeze(20, CAPS, banks, binding)
    plan = {"schema": NAME + ".prepare.v1", "base_sha256": BASE_SHA,
            "host_plan": base, "schedule": schedule, "tasks_sha256": previous["task_sha256"],
            "signal": "mean square difference between zero-action H2 terminal and root latent",
            "candidate_objective": "existing native projected terminal latent cost",
            "physical_release": False, "known_roots": True, "independent_outcome": False,
            "equal_total_compute": False, "P3_admission": False, "C3_admission": False}
    return plan, tasks, banks


def verify(path, sha):
    host.require(host.prior.sha(path) == sha, "external prepare SHA")
    saved = json.loads(Path(path).read_bytes())
    current, tasks, banks = build()
    host.require(saved == current, "prepare or source drift")
    return saved, tasks, banks


def output(path, prefix):
    path = Path(path)
    host.require(path.is_absolute() and path.parent == ROOT / "results"
                 and path.name.startswith(NAME + prefix) and not os.path.lexists(path), "exclusive output path")
    return path


def callbacks(model, device, tasks):
    """Use native root/path logic; route both model APIs through controller."""
    from . import moving_history_budget_scheduler_cuda_20261027_v1 as native
    from . import p1_conditional_direct_screen_20261027_v1 as screen

    class MeteredNative(native.Native):
        def _step(self, task, z, h, raw):
            torch = self.torch
            token = (raw.reshape(10) - self._tensor(np.tile(task["action_mean"], 5))) / self._tensor(
                np.tile(task["action_std"], 5))
            expected = screen.core.normalized_tokens(task, raw.detach().cpu().numpy())[0]
            host.require(np.array_equal(token.detach().cpu().numpy(), expected), "action normalization")
            outgoing = torch.cat((h, token[None, None]), 1)
            m = self._meter
            encoded = m.action_encoder(outgoing, phase=m._phase)
            prediction = m.predict(z, encoded, phase=m._phase)
            host.require(tuple(prediction.shape) == (1, 3, 192)
                         and bool(torch.isfinite(prediction).all()), "native predict output")
            return torch.cat((z[:, 1:], prediction[:, -1:]), 1), outgoing[:, 1:]

    backends = [MeteredNative(model, device, task) for task in tasks]

    class Quiet:
        def emit(self, *args, **kwargs):
            return None

    def rollout(root, commands, meter):
        backend, task = backends[root], tasks[root]
        backend._meter = meter
        try:
            path = backend.rollout(task, commands, Quiet(), "preoutcome")
            native.check_path(task, commands, path)
            return path
        finally:
            del backend._meter

    def signal(_arm, root, meter):
        path = rollout(root, [[0.0, 0.0] for _ in range(10)], meter)
        terminal = np.asarray(path[-1]["latent_history"][-1], dtype=np.float64)
        initial = np.asarray(tasks[root]["root_context"]["latent_history"][-1], dtype=np.float64)
        return float(np.mean((terminal - initial) ** 2))

    def evaluate(_arm, root, candidate, meter):
        path = rollout(root, candidate["commands"], meter)
        return native.cost(tasks[root], path)

    return signal, evaluate


def launch(prepared, sha, destination, authority):
    host.require(authority == AUTHORITY, "explicit coordinated CUDA authority required")
    plan, tasks, banks = verify(prepared, sha)
    out = output(destination, "_RUN_")
    host.require(shutil.disk_usage(out.parent).free >= host.CAP + host.FLOOR, "disk admission")
    with host.lock.gpu_lock(host.LOCK):
        snapshot = host.gpu_snapshot()
        out.mkdir(mode=0o700)
        host.write_new(out / "prepare.json", plan["host_plan"])
        host.write_new(out / "preoutcome_prepare.json", plan)
        pid = os.fork()
        if pid == 0:
            try:
                fd = next(int(x) for x in os.listdir("/proc/self/fd")
                          if x.isdigit() and int(x) > 2 and host._same_lock(int(x)))

                def execute(model, device, values, sink, limits, sync):
                    signal, evaluate = callbacks(model, device, values)
                    def checked_sync():
                        limits()
                        sync()
                        limits()
                    result = control.run_pair(plan["schedule"], banks,
                                              {a: model for a in control.ARMS},
                                              {a: checked_sync for a in control.ARMS},
                                              signal, evaluate, out / "controller.jsonl")
                    sink({"kind": "preoutcome_controller_completed",
                          "controller_sha256": host.prior.sha(out / "controller.jsonl"),
                          "result": result})
                    return result

                host.bridge.execute_host = execute
                report = host.child(plan["host_plan"], tasks, banks, out, fd, snapshot)
                os._exit(0 if report["status"] == "panel_completed_unverified" else 1)
            except BaseException:
                traceback.print_exc()
                os._exit(1)
        _, status = os.waitpid(pid, 0)
    host.require(os.waitstatus_to_exitcode(status) == 0, "child failed; preserve output")
    return {"run": str(out), "report_sha256": host.prior.sha(out / "report.json")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "verify", "run", "consume"))
    p.add_argument("--output")
    p.add_argument("--prepare")
    p.add_argument("--sha256")
    p.add_argument("--run")
    p.add_argument("--report-sha256")
    p.add_argument("--authority")
    a = p.parse_args()
    if a.action == "prepare":
        plan, _, _ = build()
        path = output(a.output, "_PREP_")
        print(json.dumps({"path": str(path), "sha256": host.write_new(path, plan), "cuda_used": False}))
    elif a.action == "verify":
        plan, tasks, banks = verify(a.prepare, a.sha256)
        print(json.dumps({"status": "prepare_verified", "tasks": len(tasks), "banks": len(banks),
                          "execution_authorized": False}))
    elif a.action == "run":
        print(json.dumps(launch(a.prepare, a.sha256, a.output, a.authority)))
    else:
        result = consumer.consume(Path(a.run), a.report_sha256, a.sha256, verify)
        path = output(a.output, "_CONSUME_")
        print(json.dumps({"path": str(path), "sha256": host.write_new(path, result),
                          "status": result["status"]}))


if __name__ == "__main__":
    main()
