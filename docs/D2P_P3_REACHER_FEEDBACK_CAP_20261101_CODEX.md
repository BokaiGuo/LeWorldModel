# Reacher matched-horizon feedback, common-cap development protocol

Known-root exploratory comparison, fixed before execution: both arms have the
same 15-step candidate physical horizon, 5-step packets, and independently
replan with CEM 8 samples / 2 iterations / 2 elites whenever their selected
chunk ends. Fixed executes 5 steps per decision; dynamic chooses 5/10/15 using
the existing fixed thresholds (0.01 improvement, 1.0 disagreement). It may
only use its remaining physical horizon. Both update the RGB/action context
after each actually returned packet; model predictions do not substitute for
observations. Early native termination or truncation is reported without
padding and breaks the equal realized-horizon comparison.

Before any model call, two seeded environments must independently reproduce
the same integration state, MuJoCo model hash, target and marker, goal image,
root RGB, full history RGB and actual warmup actions; otherwise neither arm is
executed. Physical states are evaluator-only, never planner inputs. The check
does not certify every hidden wrapper state or independence of seeds.

Both arms receive the same prospective per-arm cap: 48 predictor calls / 192
predictor samples, 48 visual encoder calls / 192 samples, 64 action encoder
calls / 256 samples; each call is charged at module entry before computation,
including failed attempts. It includes context/goal encoding, all CEM
planning, and dynamic scheduling calls. At most three plans are possible.
Caps are shared per arm across decisions and never refunded. An extension
meter additionally guards each existing chunk adapter. GPU allocation,
model-loading, image preprocessing, Python cost, physical rendering, native
physics, FLOPs, energy and third-party kernels are outside these axes. Equal
caps do not imply equal actual compute.

The result is one known-root exploratory feedback pair, not P3 admission,
independent confirmation, guaranteed equal realized steps, or evidence P1
passed. No witness action, true physical outcome or future state enters
planning or scheduling.

The CLI holds the existing project advisory lock
`results/e3_p2_host_cuda_cooperative.lock` before entering the experiment,
including output creation, environment reset/warmup, model loading, both arms,
environment shutdown and report writing. The lock is released on normal return
or exceptions; unavailable/contended locks prevent experiment entry. The lock
file must already exist, must not be replaced, and only coordinates cooperating
processes. The locked runner SHA-256 is
`0743759ea895f18cf8552b6527965b13e6af274d4cd90e71d7aa7bb15fa482cc`;
the project lock implementation SHA-256 is
`e4ca4fca376b0d111cc53db40a83f51dd8958745ca3eee5aa9444ad9f0b9c748`.
Both implementation files and this protocol are included in the per-run source
hashes.
