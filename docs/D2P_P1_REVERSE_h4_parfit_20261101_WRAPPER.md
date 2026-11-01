# Reverse-task H4 wrapper

`p1_reverse_h4_parfit_20261101` is an independent process-local wiring of the
existing reverse-task setup, RGB episode and durable hooked bridge to the
`p1_node_h4_parfit_20261101` planning node. It replaces the inherited H3
episode and node hooks within that process; it does not modify prior modules.
Use a fresh process for each arm and stage. Do not call another setup first.

In `le-wm-main`, a read-only CPU dry-prepare is:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -m experiments.spectral_reachability.p1_reverse_h4_parfit_20261101 \
  --arm structured dry-prepare
```

The dry operation builds and hashes the existing sealed source closure and
validates all four H4 contract caps using placeholder model/hardware hashes.
It writes no prepare artifact, loads no model and does not test physical
execution. The same command accepts `native`, `random`, `immediate_progress`.
Each node's policy horizon and optimizer config bind the active packet horizon,
four-packet allocated width and decision boundary through their contract hashes.
The wrapper checks the Q metric and requires the task projection to be exactly
the 192-dimensional identity before running the node.

To create an exclusive prepare receipt for a later, separately authorized host
run, use an unused absolute output name under `le-wm-main/results/` beginning
`p1_reverse_h4_parfit_20261101_<arm>_`, replacing `<ARM>`, `<TAG>` and
`<ABS_PROJECT>`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  -m experiments.spectral_reachability.p1_reverse_h4_parfit_20261101 \
  --arm <ARM> prepare \
  --output <ABS_PROJECT>/results/p1_reverse_h4_parfit_20261101_<ARM>_PREP_<TAG>.json
```

For future host execution, use the returned SHA and another unused matching
output name. This command was **not** run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  -m experiments.spectral_reachability.p1_reverse_h4_parfit_20261101 \
  --arm <ARM> run \
  --prepared <ABS_PROJECT>/results/p1_reverse_h4_parfit_20261101_<ARM>_PREP_<TAG>.json \
  --sha256 <PREPARE_SHA256> \
  --output <ABS_PROJECT>/results/p1_reverse_h4_parfit_20261101_<ARM>_RUN_<TAG> \
  --authorization MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS \
  --authorize-host-cuda
```

Each node reserves its complete vector, dispatches individual operations via
the existing durable leaf and requires its observed charged delta to equal the
H4 schedule before returning a plan. The existing bridge also consumes the
node journal and compares charges and hooks. This is a cooperative outer-call
meter, not proof of kernel counts or complete runtime cost. RGB encoding,
physical execution and metric construction remain separate scopes.
At feedback boundaries structured initialization retains four selected
response-bank slots and adds four boundary-seeded Gaussian H4 plans, matching
the H3 mix rule with the wider action program.
Candidate seeds are unchanged from the H3 base: native 20261030, random
20261031, immediate-progress 20261101; the structured feedback seed is
20261031 plus boundary. Because the generated tensor has four rather than
three packets, the overall candidate banks are not bitwise identical across
H3 and H4 despite identical seeds.

CPU tests cover all four arms and boundaries with arithmetic tensor fixtures,
real durable ledger consumption for two nodes, and four-boundary wrapper carry
wiring. They do not validate the CUDA model, true physical event chain,
independent hook completeness, saved-result consumer compatibility, time or
resource ceilings, or a full prepare/run/replay. The prepare and run commands
must not be interpreted as having passed those checks.

The supplied-witness offline diagnostic is evidence about the frozen model's
responses to a privileged action sequence, not a planner input. Its H3
open-loop score already beats zero, while 20-step prediction error remains;
therefore H4 search is a testable extension, not an established root cause or
proof of P1.
