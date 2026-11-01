# P5 RGB production launch: review-only runbook

Nothing in this runbook is an authorization or an instruction to run now. The
main thread must separately choose and approve the GPU, thresholds, watchdog
service, disk floor and time limits. The current sandbox cannot access the
systemd bus and does not expose the three required NVIDIA device nodes.

## Fixed data and resources

The independent CPU-only census checks the saved, SHA-pinned 164 branches and
80 pairs, including every raw float32 suffix and every pair's 10-coordinate
schedule. The existing input bridge checks the two bound history PNGs for all
164 branches; future labels and the physical receipt are not reopened.
The two rendering conditions remain separate: 82 branches per condition, 410
two-dimensional actions per condition. Corresponding off/on preprocessed
history hashes differ for all 82 previous frames and all 82 current frames.
This establishes a visual difference, not distribution equivalence or accuracy.

Fixed interface maxima are 164 encodes and 820 five-step transitions, 984 calls
total, without retries. The checkpoint contains 93,842 float32 parameters:
375,368 logical bytes. Two 64×64 RGB float32 inputs consume 98,304 logical
bytes per branch, five two-component float32 actions 40 bytes, and six
six-component float32 states 144 bytes. These values are **not GPU memory
requirements**: activations, allocator, libraries, workspaces, contexts and
copies are unmeasured. There is no measured GPU memory upper bound or safe
free-memory threshold.

The journal consumer rejects journals over 16 MiB and each snapshot over 1 MiB,
but these are reader limits, not writer quotas. stdout/stderr, filesystem
metadata/journaling and process memory are not hard-bounded. The 1024 MiB floor
in the saved preparation report is a diagnostic input, not an approved budget.
Choose and enforce resource limits based on reviewed host capacity; no wall
time or GPU compute estimate is established.

## Main-thread steps after explicit authorization

All `<...>` tokens below are placeholders, not runnable values. Use a fresh
unique report name, run suffix and service unit; never reuse prior names.
Generate the host preflight as the same user, interpreter, filesystem and
device-visible environment in which the service will run:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p5_rgb_host_preflight_20261030_codex_v1 --min-disk-free-mib <reviewed-disk-floor-mib> --output-name p5_rgb_host_preflight_20261030_codex_v1_<unique>.json
sha256sum results/p5_rgb_host_preflight_20261030_codex_v1_<unique>.json
```

The preflight intentionally exits **2 even when static checks are good**; inspect
the newly created JSON, its digest, timestamp and blockers rather than treating
exit 2 as launch permission. Static blockers other than the two explicitly
pending GPU/runtime and authorization items cause the gate to reject.

After separately approving launch, submit a bounded system service before the
freshness limit expires, using the externally calculated SHA and a reviewed
physical GPU UUID. The service manager may require host administrator approval:

```bash
systemd-run --unit=<unique-unit> --uid=gbk --working-directory=/home/gbk/code/LeWorldModel/le-wm-main --property=Type=exec --property=RuntimeMaxSec=<reviewed-service-limit>s --property=TimeoutStopSec=10s --property=KillMode=control-group --property=SendSIGKILL=yes --setenv=PYTHONDONTWRITEBYTECODE=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.p5_rgb_launch_preflight_gate_20261030_codex_v1 --authorization MAIN_THREAD_APPROVED_PREFLIGHT_BOUND_RGB_CUDA --gpu-uuid <GPU-uuid> --run-suffix <unique> --preflight-name p5_rgb_host_preflight_20261030_codex_v1_<unique>.json --preflight-sha <sha256> --min-free-mib <reviewed-free-mib> --min-total-mib <reviewed-total-mib> --hard-seconds <reviewed-child-deadline> --runner-seconds <reviewed-between-call-limit> --max-age-seconds <1-to-300> --watchdog-unit <unique-unit>.service --watchdog-max-seconds <reviewed-service-limit>
```

The wrapper requires its own process to be in the service's exact cgroup and
checks service properties twice. Its duration parser accepts only single-unit
systemd renderings (e.g. `120s` or `2min`), so inspect `systemctl show` output;
compound forms and unavailable system buses fail closed. The service limit
covers service startup as well as the child; it must be compatible with the
reviewed child limit and the manager's stop grace. The inner supervisor checks
actual device UUID, MIG mode and a free-memory snapshot at launch. These checks
do not reserve memory or establish scientific validity.

After execution has stopped, separately audit the binding, supervisor log,
strict prediction journal and before/after snapshots using the instrumented
acceptance command, with independently calculated SHA values. Missing/truncated
or partial output stays partial; do not retry under the same suffix. Binding
alone is not proof of dispatch. Existing direct launcher paths can bypass this
opt-in gate; trusted orchestration must enforce the selected entry point.
