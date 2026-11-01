"""Known-root imagined-midpoint replan, paired with the real-RGB 10+10 run.

Phase two still encodes the real frame (and charges it), but discards that
encoding from planning and substitutes the frozen model's prediction of the
same executed first packet. One additional action encoder and predictor call
per second phase is recorded separately, not hidden inside the stage ledger.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from . import codex_p1_feedback_10x2_20261030_main as real

NAME = "codex_p1_predicted_midpoint_10x2_20261030_main"
launcher = real.launcher
stage = launcher.stage
bridge = launcher.episode.feedback
ORIGINAL_STAGE = stage.run_stage


def predict_midpoint(row, cp5, packet, model):
    states, history, _, mean, std = bridge.common.context(row, cp5, device=cp5.device)
    bridge.packet_bytes(packet)
    command = torch.as_tensor(packet.copy(), dtype=states.dtype, device=states.device)
    token = ((command - mean)/std).reshape(1, 1, 10)
    with torch.inference_mode():
        predicted = model.predict(states, model.action_encoder(torch.cat((history, token), dim=1)))
    bridge.need(predicted.shape == states.shape and bool(torch.isfinite(predicted).all().item()),
                "predicted midpoint ABI")
    return predicted[0, -1].detach().clone()


def configure():
    real.configure()
    original_build = launcher.build

    def build():
        value = original_build()
        value["sources"][str(Path(__file__).resolve())] = launcher.sha(Path(__file__).resolve())
        value["variant"] = NAME
        value["comparison"] = ("same first phase and second optimizer as real feedback; "
                               "second-stage midpoint is model-predicted, not real RGB")
        value["extra_prediction_calls_per_second_stage"] = {"action_encoder": 1, "predict": 1}
        return value

    def run_stage(folder, row, number, png, previous_latent, executed_packet,
                  model, transform, device, sources):
        if number == 1:
            return ORIGINAL_STAGE(folder, row, number, png, previous_latent, executed_packet,
                                  model, transform, device, sources)
        original_encode = bridge.encode_fresh
        evidence = {}

        def substitute(pixels, net, budget, tx, dev):
            actual = original_encode(pixels, net, budget, tx, dev)
            predicted = predict_midpoint(row, previous_latent, executed_packet, net)
            evidence.update(actual_latent=actual.detach().cpu().tolist(),
                            predicted_latent=predicted.detach().cpu().tolist(),
                            actual_sha256=stage.tensor_hash(actual),
                            predicted_sha256=stage.tensor_hash(predicted),
                            packet_sha256=hashlib.sha256(bridge.packet_bytes(executed_packet)).hexdigest(),
                            extra_calls={"action_encoder": 1, "predict": 1})
            return predicted
        bridge.encode_fresh = substitute
        try:
            latent, result = ORIGINAL_STAGE(folder, row, number, png, previous_latent, executed_packet,
                                            model, transform, device, sources)
        finally:
            bridge.encode_fresh = original_encode
        bridge.need(evidence and stage.tensor_hash(latent) == evidence["predicted_sha256"], "midpoint binding")
        launcher.base.write(Path(folder)/"predicted_midpoint.json", evidence)
        return latent, result

    launcher.build = build
    stage.run_stage = run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--directory")
    parser.add_argument("--report-sha")
    args = parser.parse_args()
    configure()
    print("Known-root 10+10 imaginary midpoint control; no training; costs are not equal.", flush=True)
    if args.command == "prepare":
        print(json.dumps({"sha256": launcher.prepare(args.output)}))
    elif args.command == "run":
        launcher.need(args.prepared and args.prepare_sha, "prepare required")
        return launcher.run(args.prepared, args.prepare_sha, args.output)
    else:
        out = Path(args.output)
        launcher.need(args.prepared and args.prepare_sha and args.directory and args.report_sha
                      and out.is_absolute() and out.parent == launcher.ROOT/"results"
                      and out.name.startswith(NAME+"_saved_") and not out.exists() and not out.is_symlink(),
                      "exclusive saved input/output")
        value = real.saved.consume(args.directory, args.report_sha, args.prepared, args.prepare_sha)
        for row in value["rows"]:
            if len(row["stage_resources"]) == 2:
                folder = Path(args.directory)/f"row_{row['index']:02d}"/"stage_2"
                extra = json.loads((folder/"predicted_midpoint.json").read_bytes())
                stage_report = json.loads((folder/"report.json").read_bytes())
                launcher.need(len(extra["actual_latent"]) == len(extra["predicted_latent"]) == 192
                              and extra["predicted_sha256"] == stage_report["result"]["latent_sha256"]
                              and extra["packet_sha256"] == stage_report["binding"]["executed_packet_sha256"]
                              and extra["extra_calls"] == {"action_encoder": 1, "predict": 1}, "prediction binding")
                row["midpoint_sidecar_sha256"] = launcher.sha(folder/"predicted_midpoint.json")
        value["variant"] = NAME
        value["extra_prediction_calls_per_completed_second_stage"] = {"action_encoder": 1, "predict": 1}
        value["warning"] = "saved verifier checks declared bindings, not independent numerical prediction replay"
        print(json.dumps({"sha256": launcher.h10._write_new(out, value), "rows": len(value["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
