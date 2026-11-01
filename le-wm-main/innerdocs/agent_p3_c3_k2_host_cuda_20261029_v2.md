# K2 six-slot compact HOST adapter

K2 is an **exploratory protocol chosen after the K1 queue refusal**. It is
fixed before this new run, not an independent preregistration and not a change
to K1, P1, P3, or any existing scientific gate. The denominator is two known
roots × three fixed modes, 6 slots × 144 request positions. Every slot uses
the frozen K2 27-axis vector. The original atomic queue, verification
reserve, R8+zero, eligibility and no-partial-plan rules are preserved. A
resource refusal is a settled result, not a successful plan or a scientific
negative. No roots, modes, queues or thresholds are selected from outcomes.
The inherited RAM-only K2 protocol retains `HOST_dispatch_authorized=false`;
that flag describes the frozen RAM artifact, not a production receipt. This
separate adapter is reviewed and explicitly authorized by MAIN independently.

## Authority and execution

`prepare` reconstructs the frozen upstream task/goal/root/history and policy
authority, binds the K2 protocol triple by literal SHA, hashes the K2 engine,
online transport, integer-bound source and this source/test/doc triple, and
includes all inherited source and checkpoint files. The active profile and
rules contain only K2 caps and the six-slot roster; the old protocol SHA is
explicitly labelled inherited. `verify` rechecks the
external prepare SHA and reconstructs the whole authority. This must be run by
MAIN after source review. `run` requires explicit MAIN authorization, exact
prepare verification, exclusive output, 4 GiB plus a 200 MiB free-space floor,
and `cuda:0`; there is no CPU fallback. It uses the inherited compatible
loader and records model state and buffers, installed module-source identities,
precision and actual host state before/after. File hashes and resulting model
state do **not** certify that the loader hashed the exact buffer passed to
`torch.load`; no same-buffer loaded-byte receipt is claimed. MATH attention,
TF32 off, no autocast, eval and frozen parameters are checked. No training,
encoder, physics, or gate modification is authorized.

The new online transport routes each event before durable append. An auditable
typed CAS, native transcripts, virtual capsule views, hash-chained journal,
complete slot results and explicit six-row failure record are retained.
Failures are not retried or converted to successful receipts. The reserved
failure write is itself subject to disk, cap and I/O failures; in that case
the already written prefix remains, and no complete failure receipt is claimed.
The reader checks manifest and completion external SHAs, full files and CAS
reachability, then replays the fixed mathematics from saved native calls. It
does not independently rederive the producer's mathematics and does not run
NN or physics. A RAM fixture cannot be relabeled as HOST.

## Integer storage and time boundary

The typed base's 12-slot upper bound is 2,930,867,648 bytes. The K2 conditional
componentwise calculation is 471,946,928 bytes and is no larger on any of
the 27 aggregate resource axes or any original typed component. The runner
conservatively uses the **old larger** bound, not observed compression. The
frozen routing bounds are parent 242,544 ≤ 524,288 bytes, candidate 28,200 ≤
65,536, and slot 215,424 ≤ 1,048,576. Actual routing enforces those buckets,
exact catalog references, finite shapes, source static metadata ≤32 MiB and
the same compact transport headers/event limits. Six-slot scope, row and
resource checks and 144 request identities per slot are enforced separately.

Original typed/transport bound plus full header, duplicate static JSON,
manifest and completion allowances is 3,070,758,388 bytes. Charging another
16 MiB prepare, 8 MiB report and 16 MiB external receipt gives live upper
3,112,701,428 bytes, below 3,758,096,384 bytes. The 512 MiB failure reserve
brings the total upper to 3,649,572,340 bytes, below 4 GiB. The output guard
shares the live allowance across CAS/index/journal/metadata and reserves the
two external maxima; failure publication checks physical bytes plus those
external maxima against the total cap. These are format/count bounds, not a
prediction of actual disk use or a claim every I/O operation must succeed.

Wall cap is 7200 s whole and 900 s per slot; allocated/reserved GPU caps are
3/4 GiB. Completion records the measured time through report/manifest and
directory fsync. Its own write/fsync and process shutdown are excluded from
that durable timestamp; a subsequent live guard checks the former, but the
saved completion alone cannot prove the wall bound through its own fsync.

## MAIN commands after review

From `/home/gbk/code/LeWorldModel/le-wm-main`, with a new output namespace:

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_k2_host_cuda_20261029_v2 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_k2_host_cuda_20261029_v2_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_k2_host_cuda_20261029_v2 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_k2_host_cuda_20261029_v2_prepare_MAIN01.json --prepare-sha PREPARE_SHA256
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_k2_host_cuda_20261029_v2 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_k2_host_cuda_20261029_v2_prepare_MAIN01.json --prepare-sha PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_k2_host_cuda_20261029_v2_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_k2_host_cuda_20261029_v2 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_k2_host_cuda_20261029_v2_prepare_MAIN01.json --prepare-sha PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_k2_host_cuda_20261029_v2_cuda_MAIN01 --manifest-sha MANIFEST_SHA256 --completion-sha COMPLETION_SHA256 --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_c3_k2_host_cuda_20261029_v2_consume_MAIN01.json
```

The placeholders must be replaced by the actual outputs. A successful saved
receipt establishes the engineering transcript only; equal caps do not imply
equal actual costs or FLOPs, and no model or physical result is prefilled.
