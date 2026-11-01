# First-feedback carry results

## Completed real runs

Both fixed known roots and all four arms completed host CUDA planning and physical
execution under protocol `D2P_P1_FIRST_FEEDBACK_CARRY_PROTOCOL_20261101_MAIN.md`.
No model training and no neural CPU fallback.

| Root | Structured | Native | Random | Immediate progress |
|---|---|---|---|---|
| 0 | success at15 | no success20 | no success20 | success at15 |
| 1 | success at14 | no success20 | success at7 | success at20 |

Source directories under `le-wm-main/results/`:
`p1_first_feedback_carry_20261101_main_root{0,1}_RUN_MAIN02`.
Report SHA256 root0:
`7aaa901a28974c09bd0cb12fb4b7a7efe939a533dbe2649327d9c33802fb5d9d`;
root1:
`4d4f4f074a1b3cdc0b9999c09c71414d78cd56825c067c139e1446f73e19aab2`.

Saved validator checked 14 and 13 nodes respectively: trace hashes/chains, original
candidate/refinement provenance, old-tail provenance, selected full plans, actual
issued actions, current physical checkpoint RGB hash, exact incremental selection
cost, full node ledger and module-hook counts.
It does not independently replay the full physical event chain.
Checks: `p1_first_feedback_carry_20261101_main_root{0,1}_CHECK_MAIN02.json`.

## Interpretation

The original carry implementation did not retain/compare the original tail at cp5.
The new version starts at cp5 and maintains equal effective horizons H2/H1;
after exhaustion at cp15 it scores new H1 plans. Root1 structured success differs
from the earlier cp10/cp15-only carry run, which failed to succeed by20.
This intervention changes multiple linked horizon/replacement decisions; it does
not isolate a single cause or show that every old failure was a coding error.

Structured is successful on both known roots but has no demonstrated superiority
over the immediate-progress arm (also 2/2). Random succeeds earlier on root1.
Initial structured plans were unchanged: previous selected==greedy and immediate
physical progress remain unresolved. Thus P1 non-greedy mechanism is NOT passed.
Two development roots are not independent confirmation.

## Engineering failures retained

MAIN01 prepares were rejected BEFORE model and physical execution because horizon
tuples become lists after JSON serialization. Fixed to JSON-stable lists; regression
test added. MAIN02 uses new files and leaves rejected prepares intact.

The separate response-diverse MAIN01 prepare obtained no GPU execution: cooperative
GPU lock acquisition was nonblocking and failed while this batch held the lock.
It is a scheduling failure, not a negative scientific result. Response-diverse
MAIN02 was launched only after this batch completed.

## Next fixed experiment

Test response-diverse initialization (zero, DLS, +/- three leading singular
directions) under the same first-feedback carry. Preserve all four arms and both
roots. Do not choose directions or suffixes from physical outcomes.
