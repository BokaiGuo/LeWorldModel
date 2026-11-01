# Blind cp5 RGB to fixed-H2 CUDA planner handoff

Exploratory code only. No GPU, neural inference, simulator, or physics was run
for this handoff. This runner does not execute the resulting commands and is
not physical authorization, independent confirmation or P1 admission.

## Intended computation

The only experiment data input is a SHA-verified blind bundle; model input
files are local `weights.pt` and `config.json`, checked against the bundle.
For each of 16 existing selected/zero rows, the runner decodes only that row's
cp5 RGB pixels, checks their hash, runs the existing native image transform
and encoder/projector on CUDA, saves the float32 192-vector and its byte
digest, and passes it directly to the fixed-candidate H2 kernel.

Each packet runs the existing H1 model compiler with CUDA predictor and
action-encoder module guards, OneAD and exact ledger checks. Encoding is
separately guarded against predictor and AD operations; planning guards
against visual encoder/projector calls. The native model is evaluated frozen,
using float32, no autocast, TF32 disabled and MATH attention. The output
records per-packet selection, replay, contexts, operations and counts. The
second packet uses the first predicted winner context, not cp10. There is no
CPU neural fallback. A full successful pass expects 16 encoder and 16 projector
calls and per row 10 predictor, 10 action encoder, 2 Jacobian and 2 SVD calls;
this excludes model loading and does not establish total-budget parity.

Each new output directory is exclusive; a partial error leaves its attempt and
normally a `partial_or_failed` report with a 16-row denominator. Events and
report have a combined 128 MiB cap, 2 MiB final reserve, 200 MiB disk floor,
and 3600 second event-checked wall limit. Abrupt process termination, disk
failure, or an oversized final report can leave only the attempt directory
and partial events, without a report. There is no resume or automatic retry.
Exclusive directory creation is not cross-entry deduplication.

The runtime records model-class and selected core-code hashes after loading and
checks them for drift. This is **not** a preapproved complete dependency census
and does not prove that dynamically loaded dependencies were sealed before
execution. Bundle and model files are rechecked at the end. Saved tensor and
buffer hashes are checked against the bundle. A successful return remains
`exploratory_plans_only`, `physical_authorized=false`.

## Required before any host run

- A main-thread review of the code, fixed-candidate method and resource
  counting; the current tests use an analytic fake backend and do not
  demonstrate the real CUDA model, hooks, or autograd execution.
- An independently enforced filesystem/process boundary with only the blind
  bundle, approved offline model files, required code/runtime, output directory
  and coordinated CUDA device accessible. Test denied access to old result,
  future-frame and physical-state paths. The CLI acknowledgment is an operator
  assertion, not an isolation test or GPU exclusivity mechanism. Set offline
  network policy outside this process as well.
- Preseal and audit the complete runtime dependency census, GPU/memory/time
  budget and attempt identity; independently verify model and bundle hashes
  and ensure the model directory contains the expected offline files.
- Source and check eight greedy identities and their own cp5 observations
  before any complete three-arm claim. The current bundle has 16 known rows
  with 8 greedy conditions unknown; no results are imputed.
- Review total compute parity including prefix generation and visual encoding,
  and preserve all scientific gates. The fixed candidate method is a new
  exploratory method and was not preregistered before historical results.
- Separately implement and review physical-side root/warmup/prefix replay
  guards, safe exclusive physical claim, literal suffix execution and a strict
  saved-result consumer. This runner supplies none of them.

After the above prerequisites and an explicit host authorization, the CLI
shape is shown below; replace every placeholder with independently approved
isolated paths and a new attempt name. Do not interpret this example as
authorization to run:

```
CUDA_VISIBLE_DEVICES=<coordinated-device> PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /isolated/venv/bin/python -m experiments.spectral_reachability.codex_p1_blind_h2_host_cuda_20261030_v1 --bundle /isolated/input/blind.json --sha256 <trusted-bundle-sha256> --model-dir /isolated/model --output /isolated/output/codex_p1_blind_h2_host_cuda_20261030_v1_attempt_unique --acknowledge ISOLATED_HOST_CUDA_COORDINATED
```

## Prior contamination

The preceding work turn ran an older adapter-test fixture that loaded the
complete historical physical report, including outcome and later-checkpoint
fields, despite the then-current prohibition. Those fields were not supplied
to the new fixed-candidate kernel, but the session as a whole is contaminated:
there is no guarantee that this rule was chosen without prior knowledge, and
no preregistration or prospective-blinding claim is warranted. This fact is
also saved in every runner binding/report. Independently review and freeze the
method in a clean environment before considering future physical work.

For this handoff, tests were restricted to the existing self-contained blind
bundle and synthetic callbacks; the old adapter test and physical report were
not read.
