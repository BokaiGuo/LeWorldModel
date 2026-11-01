# P3 UUID-aware host driver V2 handoff

This is a new run contract; it does not overwrite, resume or retry MAIN01.
The fixed six-slot roster, resource ledger, known roots, P1 gate and failure
semantics of the original driver remain unchanged. Operator authorization and
resource coordination are still required; this document does not grant them.

The V2 plan is
`results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP02.json`,
file SHA256 `3e7a8f639b3abc1ba548cb7471df4a244808a76217f2f77f5a60665a66b6afe4`,
canonical SHA256 `bf54ad72631bd437bb88d90fe71b7065b803901fc25100fa3d93d3fc7b8bed07`.
Its distinct output is
`results/p3_decision_host_cuda_driver_20261030_codex_v2_RUN_MAIN02`.
The plan pins the V2 driver, UUID bridge, old driver, GPU preflight and
checkpoint loader; it binds the exact prior resource plan and its existing
adapter, journal, transport, source contract and PREP01 closure. The V2 source
pins are checked before running and after adapter completion. The closure does
not attest installed Python, shared libraries, OS, GPU driver or hardware.

The V2 runtime queries nvidia-smi again before CUDA allocation, checks GPU
UUID, mode and memory, then checks the exact `_CUuuid` type and all 16 bytes
against the admitted full UUID and passes the checked torch view to the
original adapter. Before model loading it writes `uuid_binding_v2.json`;
`v2_binding.json` includes the full V2 plan and hash. UUID read on each guard
is byte-checked again. No global torch monkeypatch is used.

Resource values inherit PREP01: native cap 2 GiB, decision cap 1 GiB, outer
cap 8 MiB, disk floor 2 GiB, allocated/reserved caps 3/4 GiB, native store
wall 3600 s. V2 requires nvidia-smi free/total >= 6144/8192 MiB; systemd must
enforce 3900 s, including initialization. Limits are safety policy, not
measured V2 costs. The driver does not certify that systemd enforcement was
configured or authenticate the coordination reference. The GPU and filesystem
checks are snapshots, not reservations; keep the device and output storage
coordinated until exit. Native log inspection and decision consumption may
require substantially more RAM than file size. Systemd kill or disk failure can
leave partial output without a failure record.

## Main-thread-only preflight

Before execution, explicitly approve this new plan and authorize the GPU,
simulation and rendering/display resources for a window covering at least
3900 seconds plus cleanup. Confirm host user systemd, device visibility,
checkpoint, renderer/dependencies, sufficient RAM, unused output and unit, and
no conflicting GPU allocation. Record the actual coordination reference
separately; the plan's coordination text is not an approval. A failed or
changed precondition stops the launch. No automatic retry, fallback or cleanup.

Run this on the **host**, not in a sandbox that hides NVIDIA devices:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
set -e
set -o noclobber
printf '%s  %s\n' 3e7a8f639b3abc1ba548cb7471df4a244808a76217f2f77f5a60665a66b6afe4 results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP02.json | sha256sum -c
systemctl --user is-system-running
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p3_decision_host_cuda_driver_20261030_codex_v2 \
  --plan /home/gbk/code/LeWorldModel/le-wm-main/results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP02.json \
  --plan-sha256 bf54ad72631bd437bb88d90fe71b7065b803901fc25100fa3d93d3fc7b8bed07 \
  --authorization PREFLIGHT_ONLY --preflight-only \
  > results/p3_decision_host_cuda_driver_20261030_codex_v2_PREFLIGHT_MAIN02.json
```

Preflight is read-only except for the exclusive redirected report; it does not
import torch or allocate CUDA. Review its output and fresh free disk. If its
report already exists, do not overwrite it; choose and explicitly record a
new preflight report name, not a new run output.

After review and explicit authorization, start exactly once:

```bash
systemd-run --user --unit=p3-decision-host-v2-20261030-codex-main02 --wait \
  --property=WorkingDirectory=/home/gbk/code/LeWorldModel/le-wm-main \
  --property=RuntimeMaxSec=3900s --property=TimeoutStopSec=10s \
  --property=KillMode=control-group --property=SendSIGKILL=yes --property=Restart=no \
  --setenv=CUDA_VISIBLE_DEVICES=GPU-4e8973ff-0278-53f3-2bae-2615d461e441 \
  --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 \
  /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.p3_decision_host_cuda_driver_20261030_codex_v2 \
  --plan /home/gbk/code/LeWorldModel/le-wm-main/results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP02.json \
  --plan-sha256 bf54ad72631bd437bb88d90fe71b7065b803901fc25100fa3d93d3fc7b8bed07 \
  --authorization MAIN_THREAD_APPROVED_P3_DECISION_HOST_CUDA_V2
```

From a second terminal during execution inspect the user service's actual
`ControlGroup`, `RuntimeMaxUSec`, `KillMode`, `SendSIGKILL`, `Restart`,
`MainPID` and that process's `/proc/<MainPID>/cgroup`. Preserve systemd result,
exit status and `journalctl --user -u p3-decision-host-v2-20261030-codex-main02.service`.
The driver does not self-verify these supervisor properties. Stop and preserve
partial evidence if they do not match; do not relaunch automatically.

## Independent decision consumer

If a terminal end and decision journal exist, use a separate CPU process:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
set -e
printf '%s  %s\n' 686329f2d0957a932d7459d8ffdda6b3f3494cda83729a884c6d6c244b9d492a experiments/spectral_reachability/p3_decision_saved_consumer_20261030_codex_v1.py | sha256sum -c
JOURNAL_SHA=$(jq -er '.decision_sha256' results/p3_decision_host_cuda_driver_20261030_codex_v2_RUN_MAIN02/end.json)
test "$(sha256sum results/p3_decision_host_cuda_driver_20261030_codex_v2_RUN_MAIN02/decision/journal.jsonl | cut -d' ' -f1)" = "$JOURNAL_SHA"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p3_decision_saved_consumer_20261030_codex_v1 \
  --journal /home/gbk/code/LeWorldModel/le-wm-main/results/p3_decision_host_cuda_driver_20261030_codex_v2_RUN_MAIN02/decision/journal.jsonl \
  --journal-sha "$JOURNAL_SHA" --max-bytes 1073741824 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_decision_host_cuda_driver_20261030_codex_v2_CONSUME_MAIN02.json
```

The journal hash read from the producer is an integrity check, not an
independent trusted timestamp. This consumer checks the decision ledger, not
native IO, cross-journal authentication, GPU attestation, fresh roots or P1.
Even a complete receipt does not establish a scientific mechanism.
