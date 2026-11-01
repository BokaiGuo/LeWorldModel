"""Same-RGB/action CUDA contrast of the erroneous extra live-action /100.

Uses all16 predeclared source windows from the physical packet audit. Public
stats are an explicit unverified convention, not recovered checkpoint truth.
No action selection, fitting, outcome cohort, readout, or P0/P1 admission.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from stable_worldmodel.wm.lewm import lewm as released

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
from experiments.spectral_reachability import pusht_d2p_packet_contract as contract
from experiments.spectral_reachability import audit_pusht_d2p_packet_contract as physical
from experiments.spectral_reachability import reacher_d2p_masked_action_chart as recurrence
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

VARIANTS = ("extra_div100", "live_units")
OFFICIAL_SHA = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"


def model_packets(raw, variant, *, mean, std):
    if variant not in VARIANTS:
        raise ValueError("explicit fixed normalization variant required")
    return contract.normalize_live_packet(raw/100 if variant == "extra_div100" else raw, mean=mean, std=std)


def summarize(rows):
    if len(rows) != 16 or sorted(r["episode"] for r in rows) != list(range(16)):
        raise ValueError("all16 predeclared source episodes required")
    values = {v: [] for v in VARIANTS}
    for r in sorted(rows, key=lambda r: r["episode"]):
        actual = np.asarray(r["observed_vector"], np.float64)
        if actual.shape != (192,) or not np.isfinite(actual).all() or set(r["predicted_vectors"]) != set(VARIANTS):
            raise ValueError("full finite actual and both predicted192-coordinate vectors required")
        for v in VARIANTS:
            p = np.asarray(r["predicted_vectors"][v], np.float64)
            if p.shape != (192,) or not np.isfinite(p).all():
                raise ValueError("full finite forecast required")
            values[v].append(float(np.linalg.norm(p-actual)))
    delta = np.asarray(values["live_units"])-values["extra_div100"]
    return {"mean_prediction_l2": {v: float(np.mean(x)) for v, x in values.items()},
        "per_episode_l2": values, "live_minus_extra_l2_mean": float(delta.mean()),
        "improved": int(np.sum(delta < 0)), "worsened": int(np.sum(delta > 0)), "tied": int(np.sum(delta == 0)),
        "p1_admitted": False, "training_stats_provenance_certified": False,
        "scope": "Known source-window diagnosis, no action policy comparison or P1. This contrast fixes action units under the same explicit public-stat convention; it does not certify that convention as the checkpoint's exact training statistics."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = contract.validate_receipt(json.loads(args.contract_report.read_text()))
    base = args.contract_report.parent
    if (digest(base/"protocol.json") != receipt["protocol_sha256"]
            or digest(base/"implementation.json") != receipt["implementation_sha256"]
            or receipt.get("simulator") != "official"):
        raise ValueError("strict official physical contract seals required")
    prior = json.loads((base/"protocol.json").read_text())
    if prior["episode_indices"] != list(range(16)) or digest(args.policy/"weights.pt") != OFFICIAL_SHA:
        raise ValueError("all16 exact source episodes and original official frozen checkpoint required")
    records, source = physical.select_rows(Path(prior["data"]["source"]["path"]), tuple(range(16)))
    if source != prior["data"]:
        raise ValueError("selected source data changed")
    paths = {k: args.output.with_suffix(s) for k, s in (("protocol", ".protocol.json"), ("stages", ".stages"), ("implementation", ".implementation"))}
    if any(p.exists() for p in (args.output, *paths.values())):
        raise FileExistsError("new immutable output required")
    modules = {"auditor": Path(__file__), "contract": Path(contract.__file__), "physical": Path(physical.__file__),
        "recurrence": Path(recurrence.__file__), "p0": Path(p0.__file__), "released": Path(released.__file__),
        "image_transform": Path(p0.image_transform.__code__.co_filename), "loader": Path(load_compatible_pretrained.__code__.co_filename)}
    protocol = {"episodes": list(range(16)), "variants": list(VARIANTS), "contract_report_path": str(args.contract_report.resolve()),
        "contract_report_sha256": digest(args.contract_report), "weights_sha256": OFFICIAL_SHA, "source": source,
        "normalization": {**physical.PUBLIC_CONVENTION, "used_for_neural_experiment": True},
        "physical_steps_per_episode": 20, "history_boundary_steps": [5, 10, 15], "forecast_boundary_step": 20,
        "expected_predict_samples": 64, "predictions_sealed_before_each_successor": True,
        "checkpoint_training_statistics_certified": False, "physical_actions_unchanged_between_normalizations": True,
        "implementation_sha256": {k: digest(p) for k, p in modules.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["stages"].mkdir()
    paths["implementation"].mkdir()
    for k, p in modules.items():
        shutil.copyfile(p, paths["implementation"]/(k+".py"))
    print("HOST CUDA: all16 fixed PushT source windows; SAME observed RGB and actual5-command packet, extra/100 versus live units. No learned stats, action selection, outcome cohort or P1 admission.", flush=True)
    host = require_cuda_allocation("cuda")
    model = load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    params, buffers = p0._parameter_hash(model), {k: v.clone() for k, v in model.named_buffers()}
    original, samples = model.predict, 0
    def counted(emb, actions):
        nonlocal samples
        if emb.device.type != "cuda" or actions.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += len(emb)
        return original(emb, actions)
    model.predict = counted
    transform = p0.image_transform()
    rows = []
    try:
        for record in records:
            with physical._reset(physical.official_env_factory, record) as (env, _, _):
                history = contract.BoundaryHistory(env, root_state=record["states"][0], goal=physical.DIAGNOSTIC_GOAL)
                for packet in record["actions"][:15].reshape(3, 5, 2):
                    history.execute(packet)
                frames = list(history.frames[-3:])
                pixels = p0._frames_tensor(frames, transform=transform, device=torch.device("cuda"))
                past = torch.tensor(history.packets.copy(), dtype=torch.float32, device="cuda")[None]
                future = torch.tensor(record["actions"][15:20].copy(), dtype=torch.float32, device="cuda")[None, None]
                forecasts = {}
                with torch.no_grad():
                    emb = model.encode({"pixels": pixels})["emb"]
                    for variant in VARIANTS:
                        kwargs = {k: protocol["normalization"][k] for k in ("mean", "std")}
                        encoded_past, encoded_future = (model_packets(v, variant, **kwargs) for v in (past, future))
                        forecast = recurrence.terminal_normalized(model, emb, encoded_past, encoded_future)[0]
                        native = model.rollout({"pixels": pixels[:, None]}, torch.cat((encoded_past[:, 1:], encoded_future), dim=1)[:, None], history_size=3)["predicted_emb"][0, 0, -1]
                        if not torch.equal(forecast, native):
                            raise RuntimeError("released and explicit observed-history forecast differ")
                        forecasts[variant] = forecast.tolist()
                stage = {"episode": record["episode"], "protocol_sha256": digest(paths["protocol"]),
                    "input_boundary_rgb_sha256": [physical._array_hash(f) for f in frames],
                    "raw_history_packets": history.packets.tolist(), "raw_future_packet": record["actions"][15:20].tolist(),
                    "predicted_vectors": forecasts, "future_observations_used": False}
                sp = paths["stages"]/f"episode_{record['episode']}.predictions.json"
                write_new(sp, stage)
                history.execute(record["actions"][15:20])
                with torch.no_grad():
                    actual_pixels = p0._image_tensor(history.frames[-1], transform=transform, device=torch.device("cuda"))
                    observed = model.encode({"pixels": actual_pixels})["emb"][0, 0]
                rows.append({"episode": record["episode"], "source_start_row": record["offset"],
                    "prediction_stage_path": str(sp.resolve()), "prediction_stage_sha256": digest(sp),
                    "predicted_vectors": forecasts, "observed_vector": observed.tolist(),
                    "successor_rgb_sha256": physical._array_hash(history.frames[-1]),
                    "successor_state": contract.state_of(env).tolist()})
                print(json.dumps({"episode": record["episode"], "prediction_l2": {v: float(np.linalg.norm(np.array(p)-observed.detach().cpu().numpy())) for v, p in forecasts.items()}}), flush=True)
    finally:
        model.predict = original
    gates = {"cuda_only": True, "all16_predeclared_episodes": [r["episode"] for r in rows] == list(range(16)),
        "released_and_explicit_predictions_exact": True, "predict_accounting": samples == 64,
        "parameters_unchanged": p0._parameter_hash(model) == params,
        "buffers_unchanged": all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()),
        "all_modules_eval": not any(m.training for m in model.modules()),
        "contract_report_unchanged": digest(args.contract_report) == protocol["contract_report_sha256"],
        "weights_file_unchanged": digest(args.policy/"weights.pt") == OFFICIAL_SHA,
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    result = summarize(rows)
    write_new(args.output, {"schema": "pusht_d2p_live_action_units.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "protocol_sha256": digest(paths["protocol"]),
        "actual_predict_samples": samples, "summary": result, "rows": rows})
    print(json.dumps({"gates": gates, "summary": result}), flush=True)


if __name__ == "__main__":
    main()
