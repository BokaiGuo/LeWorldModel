# C3 K2 common-cap prospective contrast: RAM implementation

The existing C3 common-cap runner already executed K1 and K17 with fixedH1,
staticH3-progress and adaptive on both known roots. Its protocol defined
`B(K)=K*(signal+R3)+final_verification` with schedule cap 17. This NEW K2
slice uses the next integer in that existing formula, the same two roots and
all three existing policies. K2 was proposed *after observing that K1 did not
deliver*; it is therefore an explicitly outcome-informed follow-up budget,
not an independently preregistered replication of the old experiment. K2 and
all three policies are fixed before any new outcomes. No policy, root, gate,
bank, threshold or observed-quality selection was changed. K2 is not proven
to deliver a full plan; capacity refusal remains a valid fixed-denominator
result and no post hoc budget escalation is implied.

Per slot the same original 27-axis envelope is fixed: predict API/sample,
action encoder, predictor and pred_proj calls/samples/rows = 142; corresponding
context token or projection rows = 426; Jacobian/signal = 22; VJP/grad/raw
backward callbacks = 4,224; array SVD = 24; solve = 18; program compile = 20;
replay = 38; metric = 36; schedule = 17; encoder, projector and model SVD = 0.
`proof` prints all 27 named axes and the 2×3 fixed roster. This is a shared
resource *ceiling*, not equal actual work, total FLOPs or equal wall time.

The new own-schema `run_tree` is a mechanically copied original queue loop,
with only own K2 cap and explicit RAM-only guard. It invokes the same frozen
policy/decision functions, both H1/H3 signals, progress bank, R8+paid-zero,
original candidate eligibility, beam, minimum-plus-verification reserve and
atomic whole queue admission. Each admitted or refused attempt is retained;
there is no retry, padding, partial plan release or result-based policy choice.
`validate_tree` independently replays the saved RAM transcripts using the
same original mathematical functions; this is not independent mathematics.
`run_panel_ram` fixes the six identities and preserves blocked rows after a
runtime failure. The source/protocol closure is checked against existing pins.

Scope and production audit: this is executable *RAM fixture* code, not a HOST
CUDA adapter. It intentionally refuses HOST, uses no model or physics, and
does not generate a production receipt. Before any HOST execution a separate
audited adapter must bind actual root/task/goal/history and model loaded bytes,
freeze the six-slot K2 manifest before dispatch, implement complete durable
intent/return and partial failure storage, extend the compact native transport
to the new schema and six slots without monkeypatching old globals, and supply
full saved consume plus integer output/wall/memory bounds. MAIN must audit and
authorize that adapter. The K1 and K17 receipts, budgets and gates remain
untouched. Equal caps do not certify equivalent compute or control benefit.

RAM-only commands from `/home/gbk/code/LeWorldModel/le-wm-main`:

```sh
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_k2_common_cap_20261029_v1 proof
.venv/bin/python -m pytest -q tests/test_agent_p3_c3_k2_common_cap_20261029_v1.py
```
