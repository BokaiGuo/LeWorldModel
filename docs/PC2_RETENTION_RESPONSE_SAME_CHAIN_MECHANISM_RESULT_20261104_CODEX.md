# PC2 same-chain mechanism intervention result

Date: 2026-11-04

This report evaluates the actual selected chains from the new equal-slot
`response` retention panel. It is a same-environment development result, not
PC2 confirmation, PC3, RGB closed-loop, or independent replication.

## Audit

- 16/16 selected chains completed the exact-Wall intervention.
- Each root used the actual selected 15-step chain from the sealed response
  plan.
- Controls were selected, zero, greedy, lookahead, and deterministic random.
- Each control received the same common-tail set and three 480-sample CEM
  suffix replans.
- Full-tail float32 response measurements were completed at scales 0.01 and
  0.05 for all five controls.
- Journal and source-plan audits passed for all 16 roots.
- Variable-length branch traces caused by exact-Wall termination were retained;
  they were not padded or treated as fixed-length smooth traces.

## Mechanism result

Only 1/16 roots satisfied every predeclared smooth-mechanism condition:

- non-immediate first-prefix progress;
- nonzero selected prefix and remaining tail;
- response matrix and goal-gradient differences against all four controls;
- selected-chain improvement under both common-tail and repeated suffix-replan
  comparisons;
- branch stability for selected and all controls at both scales.

The qualifying root was root 8. Its response differences were materially above
the numerical tolerance: matrix gaps ranged from 5.156 to 10.545 and
goal-gradient gaps from 1.396 to 2.794 across controls and scales.

The other 15 roots failed at least one substantive condition. In particular,
response differences were present for all four controls on every root, but
branch stability was not; therefore a changed response matrix alone was not
counted as a smooth mechanism. Several roots also lacked non-immediate
progress or failed the common-tail/replanning improvement condition.

## Interpretation

This is the first same-version, actual-selected-chain mechanism witness for the
new retention panel, but its denominator is 16 and its strict intersection is
1/16. It does not establish a repeatable mechanism across the development
task family. The result also does not show that response retention caused the
witness: the retention comparison found no stable response-policy advantage,
and the intervention controls are mechanism controls rather than a full
PC3 baseline suite.

The audit script required one post-run repair to accept variable-length
exact-Wall branch traces. The original generation manifest was preserved and
the audit records both sealed and current script hashes.

PC2 remains open. The next required step is to freeze the smooth/non-smooth
scope and test the mechanism on a source-qualified confirmation panel, while
keeping the failed roots and branch crossings in the denominator. PC3 still
requires independent paired tasks, tuned strong baselines, complete cost
policy, and statistical confirmation.
