"""Recompute H3 chart on host CUDA, no neural model; saved actions are new."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np

from experiments.spectral_reachability import codex_p2_h3_physical_preflight_20261030_v1 as prior
from experiments.spectral_reachability import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as lock

NAME = "codex_p2_h3_rechart_cuda_20261030_v1"
LOCK = prior.root/"results/e3_p2_host_cuda_cooperative.lock"
need = prior.need


def check_packets(row, raw, normalized):
    raw = np.asarray(raw)
    normalized = np.asarray(normalized)
    need(raw.shape == (3,5,2) and raw.dtype == np.float32 and np.isfinite(raw).all()
         and np.all(np.abs(raw)<=1), "bounded raw32")
    need(normalized.shape == (3,10) and normalized.dtype == np.float32 and np.isfinite(normalized).all(),
         "normalized packet")
    expected = np.asarray([c["action_history"][-1] for c in row["winner"]["contexts"]], np.float32)
    need(np.array_equal(normalized, expected), "chart normalized tokens differ from saved context")
    return dict(root=row["root"], arm=row["arm"], logits=row["winner"]["logits"],
                raw_float32_hex=raw.astype("<f4", copy=False).tobytes().hex(),
                commands=raw.reshape(15,2).tolist(), normalized=normalized.tolist(),
                original_raw_bits_proved=False, action_provenance="new_cuda_chart_reconstruction")


def execute(output):
    output = Path(output)
    need(output.is_absolute() and output.parent.resolve() == output.parent and
         output.name.startswith(NAME+"_") and not os.path.lexists(output), "exclusive output")
    evidence = prior.inspect()
    need(evidence["status"] == "blocked_original_raw_float32_not_saved", "original missing status")
    report = prior.checks.run.bridge.host.frontier.read(prior.RUN/"report.json")
    output.mkdir(mode=0o700)
    result = dict(schema=NAME+".report", status="partial_or_failed", slots=[],
                  source_report_sha256=prior.REPORT_SHA, source_prepare_sha256=prior.PREP_SHA,
                  action_provenance="new_cuda_chart_reconstruction", original_raw_bits_proved=False,
                  neural_calls=0, P1_admission=False, P2_admission=False)
    start = time.monotonic()
    try:
        if True:  # main holds the shared lock while this child owns CUDA.
            import torch
            from experiments.spectral_reachability.pusht_d2p_packet_contract import normalize_live_packet
            need(torch.cuda.is_available(), "HOST CUDA required")
            device = torch.device("cuda:0")
            mean = prior.physical.NORMALIZATION["mean"]
            std = prior.physical.NORMALIZATION["std"]
            result["device"] = dict(name=torch.cuda.get_device_name(device), torch=torch.__version__,
                                    cuda=torch.version.cuda)
            for row in report["rows"]:
                logits = torch.tensor(row["winner"]["logits"], device=device, dtype=torch.float32)
                with torch.no_grad():
                    raw = torch.tanh(logits).reshape(3,5,2)
                    norm = normalize_live_packet(raw, mean=mean, std=std)
                torch.cuda.synchronize(device)
                record = check_packets(row, raw.cpu().numpy(), norm.cpu().numpy())
                result["slots"].append(record)
                prior.checks.run.bridge.host.bank._write(output/f'chart_{row["root"]}_{row["arm"]}.json', record)
            torch.cuda.synchronize(device)
            del raw, norm, logits
            result["status"] = "eight_recomputed_charts_saved"
    except BaseException as exc:
        result["error"] = dict(type=type(exc).__name__, message=str(exc)[:1000])
    finally:
        result["elapsed_seconds"] = time.monotonic()-start
        prior.checks.run.bridge.host.bank._write(output/"report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    # Parent never initializes CUDA; the advisory lock outlives the child CUDA context.
    with lock.gpu_lock(LOCK):
        pid = os.fork()
        if pid == 0:
            try:
                execute(a.output)
                os._exit(0)
            except BaseException:
                os._exit(1)
        _, status = os.waitpid(pid, 0)
        need(os.waitstatus_to_exitcode(status) == 0, "chart child failed")
    report = prior.checks.run.bridge.host.frontier.read(Path(a.output)/"report.json")
    print(json.dumps(dict(status=report["status"], slots=len(report["slots"]),
                          error=report.get("error"),
                          report_sha256=prior.checks.run.digest_file(Path(a.output)/"report.json"))))


if __name__ == "__main__":
    main()
