# P3 decision host resource plan PREP01

This is a fixed engineering budget and an execution handoff, not an authorization
or evidence of scientific success. The six-slot known-root protocol and P1 gate
are unchanged. No GPU allocation, model forward, or physics was performed to
prepare this plan.

## Evidence and budget

The opened prior native run
`results/p3_matched_order_feedback_host_v2_codex_20261029_RUN_MAIN01/report.json`
has SHA256 `02b5f2094de56faf807bbb552b43492b05021dc7caac79f7966ce4734da0c8cd`.
It reports four completed callback slots, 51.016670075 s, CUDA allocated peak
143617024 B, reserved peak 148897792 B; category bytes numerical events
105966748, physical events 1500486, physical PNG 435201, encoding arrays
24885248, encoding events 46594 (other categories zero). These are historical
self-reports for a different protocol, not measurements of the present run.
The six-slot protocol pays both horizon panels at each of four decisions, so
neither time nor disk nor CUDA peaks can be extrapolated as an empirical bound.

The fixed plan is
`results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP01.json`;
its file SHA256 is `0fa0ef09f27b7f29558c10014e64cc3d940cd208a2cc0819d56163786953e541`,
and its canonical JSON SHA256 (the driver argument) is
`4eae543e09ea63aec94814f394205ffd3ff4ddc71025d0eb544bd39a5722719f`.
The source contract and four driver files are pinned within it. The actual
source-closure rebuild succeeded offline with two known roots, without importing
torch. The category budgets sum to 1577058304 B; native cap is 2 GiB including
256 MiB reserve, decision cap 1 GiB including 64 MiB terminal reserve, outer
cap 8 MiB, and disk floor 2 GiB. PNG category 64 MiB exceeds the factory's
192 * 256 KiB worst-case frame-byte admission. Each decision journal record
also has an independent 32 MiB hard ceiling; the terminal roster may hit it.

Allocated/reserved peak caps are 3/4 GiB. Require at least 6144 MiB free and
at least 8192 MiB total on the pinned physical UUID immediately before start.
The native store wall budget is 3600 s from store creation; an independent
systemd unit applies a 3900 s job limit with control-group termination. These
are chosen safety caps, not demonstrated sufficiency, a reservation, or a
guarantee of completion. Disk floor is not a filesystem reservation and the
decision journal does not recheck it on every append. The consumer and native
inspection read large JSON journals into memory; host RAM demand is unknown.

Read-only observation during preparation: GPU UUID
`GPU-4e8973ff-0278-53f3-2bae-2615d461e441`, RTX 5060 Ti, total 16311 MiB,
free 15763 MiB, compute mode Default, MIG `[N/A]`; results filesystem available
30712799232 B. These are transient, not launch admission or exclusivity.
User-level `systemctl --user is-system-running` returned `running`; device
visibility inside a future service has not been verified.

Full static work: six slots, 24 decisions, 1056 predictor calls, 9216 VJP
directions, 432 replay calls, 24 packet and 24 encode callbacks, 120 selected
control steps and 60 warm-up steps, excluding native substeps. Counters overlap
and must not be summed as FLOPs. A failed slot may cause later slots to be
recorded setup-failed rather than executed; do not retry or replace slots.

## Main-thread prerequisites

Obtain explicit, traceable approval for this exact plan, GPU, native PushT
simulation, rendering/display and window; the plan's coordination string is
only a placeholder description, not proof of approval. Ensure device is
exclusive or authorized for shared use, output names and service unit are
unused, sufficient host RAM and disk exist, and the user service sees the
required device nodes, checkpoint and render environment. Verify the target
user systemd manager and service cgroup; avoid launching if any item is
uncertain. The explicit UUID mapping is checked again inside the driver; a
missing PyTorch UUID property fails closed. No CUDA allocation or physical
render probe is authorized merely by preparing this document.

The commands below are for the main thread **only after approval**. The
preflight records a snapshot but does not lock GPU or disk: concurrent work
can invalidate it. If any check fails, stop; do not overwrite or retry.

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
set -o noclobber
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import json,hashlib,shutil; from pathlib import Path; from experiments.spectral_reachability import p3_decision_host_cuda_driver_20261030_codex_v1 as d; from experiments.spectral_reachability.p3_feedback_supervised_launch_20261030_codex_v1 import gpu_snapshot,admit; p=Path("results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP01.json"); x=json.loads(p.read_text()); assert d.file_sha(p)=="0fa0ef09f27b7f29558c10014e64cc3d940cd208a2cc0819d56163786953e541"; assert hashlib.sha256(d.control.core.canonical(x)).hexdigest()=="4eae543e09ea63aec94814f394205ffd3ff4ddc71025d0eb544bd39a5722719f"; d.validate(x); out=Path("/home/gbk/code/LeWorldModel/le-wm-main/results/p3_decision_host_cuda_20261030_codex_RUN_MAIN01"); assert not out.exists() and not out.is_symlink(); free=shutil.disk_usage(out.parent).free; assert free>=x["native_cap"]+x["decision_cap"]+x["outer_cap"]+x["floor"]; g=gpu_snapshot(x["gpu_uuid"]); admit(g,6144,8192); print(json.dumps({"gpu":g,"disk_free_bytes":free,"plan_sha256":"4eae543e09ea63aec94814f394205ffd3ff4ddc71025d0eb544bd39a5722719f"},sort_keys=True))' > results/p3_decision_host_preflight_20261030_codex_MAIN01.json
```

On success, immediately review the snapshot and launch once:

```bash
systemd-run --user --unit=p3-decision-host-20261030-codex-main01 --wait \
  --property=WorkingDirectory=/home/gbk/code/LeWorldModel/le-wm-main \
  --property=RuntimeMaxSec=3900s --property=TimeoutStopSec=10s \
  --property=KillMode=control-group --property=SendSIGKILL=yes --property=Restart=no \
  --setenv=CUDA_VISIBLE_DEVICES=GPU-4e8973ff-0278-53f3-2bae-2615d461e441 \
  --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=OPENBLAS_NUM_THREADS=1 \
  /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.p3_decision_host_cuda_driver_20261030_codex_v1 \
  --plan /home/gbk/code/LeWorldModel/le-wm-main/results/p3_decision_host_cuda_resource_plan_20261030_codex_PREP01.json \
  --plan-sha256 4eae543e09ea63aec94814f394205ffd3ff4ddc71025d0eb544bd39a5722719f \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_decision_host_cuda_20261030_codex_RUN_MAIN01 \
  --authorization MAIN_THREAD_APPROVED_P3_DECISION_HOST_CUDA
```

Independently inspect `systemctl --user show p3-decision-host-20261030-codex-main01.service`
for actual control group, runtime limit, kill mode, result and exit status;
preserve `journalctl --user -u p3-decision-host-20261030-codex-main01.service`.
The service setup is not a signed hardware attestation. A hard kill can leave
partial artifacts with no failure or terminal record. Preserve all partials.

## Independent saved decision-journal consumption

Only if `end.json` and a complete journal exist, compare its stored journal
SHA with `sha256sum` and retain both hashes independently. Then run in a
separate CPU process (replace the shell variable only with the verified SHA):

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
set -e
printf '%s  %s\n' 686329f2d0957a932d7459d8ffdda6b3f3494cda83729a884c6d6c244b9d492a experiments/spectral_reachability/p3_decision_saved_consumer_20261030_codex_v1.py | sha256sum -c
JOURNAL_SHA=$(jq -er '.decision_sha256' results/p3_decision_host_cuda_20261030_codex_RUN_MAIN01/end.json)
test "$(sha256sum results/p3_decision_host_cuda_20261030_codex_RUN_MAIN01/decision/journal.jsonl | cut -d' ' -f1)" = "$JOURNAL_SHA"
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p3_decision_saved_consumer_20261030_codex_v1 \
  --journal /home/gbk/code/LeWorldModel/le-wm-main/results/p3_decision_host_cuda_20261030_codex_RUN_MAIN01/decision/journal.jsonl \
  --journal-sha "$JOURNAL_SHA" --max-bytes 1073741824 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_decision_host_cuda_20261030_codex_CONSUME_MAIN01.json
```

Consumer source SHA256 is
`686329f2d0957a932d7459d8ffdda6b3f3494cda83729a884c6d6c244b9d492a`;
verify it before consumption. The consumer validates callback/decision ledger
and fixed denominator, **not** native IO, cross-journal anchoring, GPU identity
or physics. The old six-slot native consumer uses a different roster/schema
and must not be presented as a valid strict consumer for this driver. Independent
native-and-cross-log certification is still missing, including after a
candidate-complete run. No fresh roots, equal FLOPs, P1 passage, or scientific
mechanism claim follows from this plan.
