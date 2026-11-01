"""Four fixed known baseline branches: CUDA interface and initial-state finite differences."""
import argparse
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_rgb_instrumented_supervision_20261030_codex_v1 as wiring

runner, abi, acc = wiring.runner, wiring.abi, wiring.acc
bridge = abi.bridge
NAME = "p5_rgb_forward_recursive_probe_20261030_codex_v1"
AUTH = "MAIN_THREAD_APPROVED_FORWARD_RECURSIVE_PROBE_CUDA"
WIRING_SHA = "f1f3f1d0bc962b6f153ace297f12da0f6ff63eb990252b96a8a63d62d96ffee7"
INDICES = (0, 41, 82, 123)
EPS = np.float32(0.001)
BASE = {"seq", "prev", "kind", "hash"}
RESULTS = bridge.ROOT / "results"


def need(value, message):
    if not value:
        raise ValueError(message)


def source_sha():
    return bridge.digest(Path(__file__).read_bytes())


def source_preflight():
    need(wiring.source_sha() == WIRING_SHA, "wiring source hash")
    base = wiring.preflight()
    prepared = runner.load_prepared()
    need(prepared["baseline_indices"] == list(INDICES), "fixed baseline indices")
    for i in INDICES:
        row = prepared["rows"][i]
        need((row["index"], row["root_slot"], row["condition"]) ==
             (i, i // 82, ("off", "on")[(i // 41) % 2]), "baseline identity")
        abi.materialize(row)
    return base, prepared


def manifest(prepared):
    return {"schema": NAME + ".contract", "status": "input_only_not_execution_authorization",
            "source_sha256": source_sha(), "wiring_sha256": WIRING_SHA, "prepared_sha256": runner.PREP_SHA,
            "checkpoint_sha256": bridge.FILES["checkpoint"][1], "model_source_sha256": bridge.FILES["model_source"][1],
            "baseline_indices": list(INDICES), "samples": [
                {k: prepared["rows"][i][k] for k in ("index", "root_slot", "condition", "task_id",
                                                       "selection_digest", "previous_rgb", "current_rgb",
                                                       "actions", "actions_sha256")} for i in INDICES],
            "epsilon_float32": float(EPS), "coordinates": list(range(6)),
            "trajectory_order": ["base"] + [f"{c}:{s}" for c in range(6) for s in ("minus", "plus")],
            "horizons": 5, "calls": {"forward": 4, "encode_state": 4, "predict_next": 264, "reset_state": 52},
            "neural_cuda_calls_made": 0, "P5_admission": False}


def perturb(values, coordinate, sign):
    arr = np.asarray(values, dtype="<f4").copy()
    need(arr.shape == (6,) and np.isfinite(arr).all(), "initial state")
    if coordinate is not None:
        need(type(coordinate) is int and 0 <= coordinate < 6 and sign in (-1, 1), "perturbation")
        arr[coordinate] = np.float32(arr[coordinate] + np.float32(sign) * EPS)
        need(arr[coordinate] != np.float32(values[coordinate]), "float32 perturbation vanished")
    return arr


class CudaBackend(runner.CudaBackend):
    def output_tensor(self, tensor):
        t = self.torch
        need(tensor.device == self.device and tensor.dtype == t.float32 and tuple(tensor.shape) == (1, 6),
             "CUDA float32 vector")
        t.cuda.synchronize(self.device)
        return tensor.detach().cpu().numpy()

    def forward(self, inputs):
        t = self.torch
        with t.inference_mode():
            decoded, predicted = self.model(t.from_numpy(inputs["image"]).to(self.device),
                t.from_numpy(inputs["actions"][:, 0, :]).to(self.device),
                t.from_numpy(inputs["previous_image"]).to(self.device))
        return self.output_tensor(decoded), self.output_tensor(predicted)

    def encode(self, inputs):
        result = super().encode(inputs)
        self.initial = self.state.clone()
        return result

    def reset(self, coord, sign):
        t = self.torch
        with t.inference_mode():
            self.state = self.initial.clone()
            if coord is not None:
                self.state[:, coord] += float(np.float32(sign) * EPS)
        return self.output()

    def capture(self):
        return acc.capture(self.model, self.torch)

    def current(self):
        return self.output()


def schedule():
    """Deterministic op schedule; coordinates and action indices are not selected from outcomes."""
    for i in INDICES:
        yield {"branch": i, "op": "forward", "trajectory": "interface", "horizon": 1}
        yield {"branch": i, "op": "encode_state", "trajectory": "interface", "horizon": 0}
        yield {"branch": i, "op": "predict_next", "trajectory": "interface", "horizon": 1}
        for traj in ["base"] + [f"{c}:{s}" for c in range(6) for s in ("minus", "plus")]:
            yield {"branch": i, "op": "reset_state", "trajectory": traj, "horizon": 0}
            for h in range(1, 6):
                yield {"branch": i, "op": "predict_next", "trajectory": traj, "horizon": h}


def traj_fields(traj):
    if traj == "base":
        return None, 0
    coord, sign = traj.split(":")
    return int(coord), {"minus": -1, "plus": 1}[sign]


def execute(prepared, contract_sha, backend, journal, max_seconds, base, clock=time.monotonic, loader=abi.materialize):
    start = clock()
    journal.emit("start", schema=NAME + ".journal", contract_sha256=contract_sha,
                 prepared_sha256=runner.PREP_SHA, source_sha256=source_sha(), backend=backend.metadata(),
                 max_seconds=max_seconds)
    try:
        before = backend.capture()
        acc.check_snapshots(base, before, before)
        journal.emit("snapshot", phase="before", state=before)
        current, values = None, None
        for event in schedule():
            need(clock() - start < max_seconds, "between-call timeout")
            index, op = event["branch"], event["op"]
            row = prepared["rows"][index]
            if current != index:
                values = loader(row)
                current = index
            identity = {**event, "condition": row["condition"], "root_slot": row["root_slot"],
                        "task_id": row["task_id"], "selection_digest": row["selection_digest"],
                        "previous_sha256": row["previous_rgb"]["model_input_chw_float32_sha256"],
                        "current_sha256": row["current_rgb"]["model_input_chw_float32_sha256"],
                        "actions_sha256": row["actions_sha256"]}
            state_in = runner.vector(backend.current()) if op == "predict_next" else None
            if state_in is not None:
                identity["state_in"] = state_in
            journal.emit("attempt", **identity)
            try:
                if op == "forward":
                    decoded, predicted = backend.forward(values)
                    output = {"decoded": runner.vector(decoded), "predicted": runner.vector(predicted)}
                elif op == "encode_state":
                    output = runner.vector(backend.encode(values))
                elif op == "reset_state":
                    coord, sign = traj_fields(event["trajectory"])
                    output = runner.vector(backend.reset(coord, sign))
                else:
                    output = runner.vector(backend.step(values, event["horizon"] - 1))
            except Exception as exc:
                journal.emit("failed", **identity, error_type=type(exc).__name__, error=str(exc)[:500])
                raise
            journal.emit("returned", **identity, output=output)
        after = backend.capture()
        acc.check_snapshots(base, before, after)
        journal.emit("snapshot", phase="after", state=after)
        journal.emit("end", status="complete", elapsed_seconds=clock() - start)
        return "complete"
    except Exception as exc:
        journal.emit("end", status="partial", elapsed_seconds=clock() - start,
                     error_type=type(exc).__name__, error=str(exc)[:500])
        return "partial"


def read_vector(record):
    need(isinstance(record, dict) and set(record) == {"values", "sha256"}, "vector record")
    value = np.asarray(record["values"], dtype="<f4")
    need(value.shape == (6,) and np.isfinite(value).all() and bridge.digest(value.tobytes()) == record["sha256"],
         "vector SHA/finite")
    return value


def review(raw, expected_sha, prepared, contract_sha, base):
    need(bridge.digest(raw) == expected_sha and len(raw) <= 8 * 1024 * 1024 and raw.endswith(b"\n"), "journal SHA")
    events = []
    prev = "0" * 64
    for seq, line in enumerate(raw.splitlines()):
        event = bridge.read_json(line)
        need(bridge.canonical(event).rstrip(b"\n") == line and event.get("seq") == seq
             and event.get("prev") == prev, "journal chain")
        body = dict(event)
        claimed = body.pop("hash", None)
        need(bridge.digest(bridge.canonical(body)) == claimed, "event hash")
        prev = claimed
        events.append(event)
    slots = list(schedule())
    need(len(events) == 4 + 2 * len(slots), "fixed call count")
    start = events[0]
    need(set(start) == BASE | {"schema", "contract_sha256", "prepared_sha256", "source_sha256", "backend", "max_seconds"}
         and start["kind"] == "start" and start["schema"] == NAME + ".journal"
         and start["contract_sha256"] == contract_sha and start["prepared_sha256"] == runner.PREP_SHA
         and start["source_sha256"] == source_sha() and type(start["max_seconds"]) is int
         and 1 <= start["max_seconds"] <= 3600, "start")
    metadata = start["backend"]
    need(set(metadata) == wiring.gate.BACKEND and metadata["device"] == "cuda:0" and
         metadata["checkpoint_sha256"] == bridge.FILES["checkpoint"][1] and
         all(metadata[k] is True for k in ("strict_load", "frozen", "eval")) and
         all(isinstance(metadata[k], str) and metadata[k] for k in ("device_name", "torch_version", "cuda_version")),
         "backend identity")
    before, after = events[1], events[-2]
    for event, phase in ((before, "before"), (after, "after")):
        need(set(event) == BASE | {"phase", "state"} and event["kind"] == "snapshot" and event["phase"] == phase,
             "snapshot envelope")
    acc.check_snapshots(base, before["state"], after["state"])
    end = events[-1]
    need(set(end) == BASE | {"status", "elapsed_seconds"} and end["kind"] == "end" and
         end["status"] == "complete" and type(end["elapsed_seconds"]) in (int, float)
         and math.isfinite(end["elapsed_seconds"]) and end["elapsed_seconds"] >= 0, "complete end")
    records = {str(i): {"trajectories": {}} for i in INDICES}
    for pos, slot in enumerate(slots):
        a, r = events[2 + 2 * pos:4 + 2 * pos]
        row = prepared["rows"][slot["branch"]]
        identity = {**slot, "condition": row["condition"], "root_slot": row["root_slot"],
                    "task_id": row["task_id"], "selection_digest": row["selection_digest"],
                    "previous_sha256": row["previous_rgb"]["model_input_chw_float32_sha256"],
                    "current_sha256": row["current_rgb"]["model_input_chw_float32_sha256"],
                    "actions_sha256": row["actions_sha256"]}
        if slot["op"] == "predict_next":
            state_in = read_vector(a["state_in"])
            group = records[str(slot["branch"])]
            traj = slot["trajectory"]
            if traj == "interface":
                expected_in = np.asarray(group["encode"], dtype="<f4")
            elif slot["horizon"] == 1:
                expected_in = np.asarray(group["trajectories"][traj]["initial"], dtype="<f4")
            else:
                expected_in = np.asarray(group["trajectories"][traj]["outputs"][-1], dtype="<f4")
            need(state_in.tobytes() == expected_in.tobytes(), "recursive state input")
            identity["state_in"] = a["state_in"]
        keys = set(identity)
        need(set(a) == BASE | keys and set(r) == BASE | keys | {"output"} and a["kind"] == "attempt"
             and r["kind"] == "returned" and all(a[k] == r[k] == v for k, v in identity.items()), "call identity")
        group = records[str(slot["branch"])]
        output, op, traj = r["output"], slot["op"], slot["trajectory"]
        if op == "forward":
            need(set(output) == {"decoded", "predicted"}, "forward output")
            group["forward_decoded"], group["forward_predicted"] = (read_vector(output[x]).tolist() for x in ("decoded", "predicted"))
        elif op == "encode_state":
            group["encode"] = read_vector(output).tolist()
        elif traj == "interface":
            group["split_prediction"] = read_vector(output).tolist()
        elif op == "reset_state":
            coord, sign = traj_fields(traj)
            actual = read_vector(output)
            expected = perturb(group["encode"], coord, sign)
            need(actual.tobytes() == expected.tobytes(), "perturbation input")
            group["trajectories"][traj] = {"initial": actual.tolist(), "outputs": []}
        else:
            group["trajectories"][traj]["outputs"].append(read_vector(output).tolist())
    for row in records.values():
        row["interface_decoded_difference"] = (np.asarray(row["forward_decoded"]) - row["encode"]).tolist()
        row["interface_predicted_difference"] = (np.asarray(row["forward_predicted"]) - row["split_prediction"]).tolist()
        matrices = []
        for h in range(5):
            columns = []
            for coord in range(6):
                minus, plus = (row["trajectories"][f"{coord}:{s}"] for s in ("minus", "plus"))
                spacing = float(plus["initial"][coord]) - float(minus["initial"][coord])
                need(spacing > 0 and math.isfinite(spacing), "realized float32 spacing")
                columns.append(((np.asarray(plus["outputs"][h]) - minus["outputs"][h]) / spacing).tolist())
            matrices.append({"horizon": h + 1, "columns_by_input_coordinate": columns})
        row["finite_difference"] = matrices
    return {"schema": NAME + ".review", "status": "saved_self_report_consistency_not_scientific",
            "journal_sha256": expected_sha, "contract_sha256": contract_sha, "branches": records,
            "P5_admission": False, "boundary": "finite-step empirical sensitivity is not causal or fresh confirmation"}


def target(name, suffix):
    need(re.fullmatch(re.escape(NAME + "_") + r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}" + re.escape(suffix), name)
         and RESULTS.resolve() == RESULTS and not RESULTS.is_symlink(), "exclusive name")
    return RESULTS / name


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--name", required=True)
    p = sub.add_parser("run")
    for key in ("authorization", "contract-name", "contract-sha", "output-name"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    p = sub.add_parser("review")
    for key in ("contract-name", "contract-sha", "run-name", "journal-sha", "output-name"):
        p.add_argument("--" + key, required=True)
    args = ap.parse_args()
    if args.command == "run":
        need(args.authorization == AUTH and 1 <= args.max_seconds <= 3600, "explicit CUDA authorization")
    base, prepared = source_preflight()
    expected_contract = bridge.canonical(manifest(prepared))
    if args.command == "prepare":
        print(wiring.gate.write_new(target(args.name, ".json"), manifest(prepared)))
        return 0
    contract = target(args.contract_name, ".json")
    need(re.fullmatch(r"[0-9a-f]{64}", args.contract_sha) and
         contract.is_file() and not contract.is_symlink() and contract.stat().st_size <= 1024 * 1024 and
         bridge.digest(contract.read_bytes()) == args.contract_sha and contract.read_bytes() == expected_contract,
         "contract identity")
    if args.command == "run":
        directory = target(args.output_name, "")
        need(not directory.exists() and not directory.is_symlink(), "existing run")
        checkpoint, _ = runner.verify_sources()
        backend = CudaBackend(checkpoint, prepared, 0)
        journal = runner.Journal(directory)
        try:
            status = execute(prepared, args.contract_sha, backend, journal, args.max_seconds, base)
        finally:
            journal.close()
        print(status)
        return 0 if status == "complete" else 2
    directory = target(args.run_name, "")
    path = directory / "events.jsonl"
    need(directory.is_dir() and not directory.is_symlink() and path.is_file() and not path.is_symlink()
         and path.stat().st_size <= 8 * 1024 * 1024 and re.fullmatch(r"[0-9a-f]{64}", args.journal_sha),
         "journal path")
    result = review(path.read_bytes(), args.journal_sha, prepared, args.contract_sha, base)
    print(wiring.gate.write_new(target(args.output_name, ".json"), result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
