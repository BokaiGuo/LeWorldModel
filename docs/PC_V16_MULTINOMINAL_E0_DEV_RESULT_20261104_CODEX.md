# Multi-nominal response generation: exact-Wall development result

The four-task, three-arm panel completed and saved prediction batches were
replayed using the alternate same-environment geometry. Shared global banks,
task-independent nominal generation, legal actions, scores, token ledgers,
and best-scored full-chain selection passed the audit.

| Root | v13 | v16 multi-nominal | Shared-bank full-action CEM |
| --- | ---: | ---: | ---: |
| 0 | 0.500260 | 0.504283 | 0.501239 |
| 1 | 0.013569 | 0.084355 | 0.018664 |
| 2 | 0.500157 | 0.501003 | 0.500202 |
| 3 | 0.500809 | 0.500554 | 0.500215 |

At the fixed `<0.05` threshold, v13 and CEM each succeeded on 1/4;
v16 succeeded on 0/4. v16 does not support the proposed improvement on
this panel. Successful v13/CEM chains witness feasibility on root 1 only.
The other roots' feasibility remains unknown.

In this configuration v16 incurred 54 SVD calls versus 18 for v13, and
the resulting remaining joint-refinement samples were 208–230 versus
936–940. Most arms used 4,095 tokens; v16 used 4,093 on two roots due to
a remainder smaller than one full-chain evaluation. Thus the experiment
compares whole methods at a token cap; it does not isolate nominal quality
from the reduced refinement opportunity. Equal tokens are not equal total
cost, and there is no independent statistical confirmation.

The four-task stop has been reached; do not rerun, replace roots or tune
this panel. The alternate implementation is an internal consistency
check, not an independent environment. No final-chain mechanism
intervention or visual control was performed; PC2 and PC3 are unchanged.

Protocol: `PC_V16_MULTINOMINAL_E0_DEV_PROTOCOL_20261104_CODEX.md`.
Manifest, raw prediction traces, plans and audit:
`le-wm-main/results/pc_v16_multinominal_e0_dev_20261104_codex/`.
