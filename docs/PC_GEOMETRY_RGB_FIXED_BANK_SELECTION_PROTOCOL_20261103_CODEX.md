# Fixed-bank RGB model selection (exposed development panel)

This is a new development experiment after physical outcomes were already
inspected. It cannot support a blinded or independent-confirmation claim.
The implementation must not read physical outcomes, states or velocity
intervention results. It may read a separate action-only prefix manifest
created by extracting the existing 12 actions per root without filtering.
The direct prefix in that manifest was produced by a privileged physical
one-step planner; therefore this fixed bank is not a fully RGB-generated
candidate bank.

For each of the four fixed geometry setup states, obtain a real RGB reset
and a second real RGB frame after executing one zero action via the 2.1.2
shadow bridge. That action is environment history, not an optimized
candidate. A separate physical check must verify its effect before
comparing with historical zero-velocity reset outcomes; otherwise report
the comparison as unmatched. The planner receives only the RGB frame pair,
goal RGB, fixed camera calibration, normalized action history and the
action-only prefix manifest.

Use the previously frozen checkpoint and projected RGB-derived task cost,
with parameters unchanged. For each of the 12 five-action prefixes, score
its predicted endpoint and optimize a separate ten-action suffix over two
five-action packets, using the identical two-start CEM workload: three
generations of 16 samples, four elites, then evaluate zero and the final
mean per start: 100 full-chain predictions per prefix. Use bounded raw
actions and common Gaussian draws per root; seeds are fixed as 2026110320
plus root index. Score the nonlinear full 15-action rollout. Record all
candidates, predicted costs, suffixes, selected index, equal-budget counts
and model hashes. Select the global minimum predicted terminal cost (stable
bank-order tie break). Independently record reconfiguration qualification:
selected prefix predicted cost is no lower than the initial cost and
selected terminal cost strictly beats both zero and direct prefixes;
otherwise abstain from a reconfiguration claim. Do not force selection of
a non-progress prefix. No physical rank or outcome may enter selection.

Only after the decision receipt is sealed, execute selected action chains
in the formal 2.1.2 environment and compare with already-exposed controls;
retain model/physical score disagreement and all failures. This protocol
does not freeze a final method, establish equality of total planner costs,
or substitute for independent roots and strong closed-loop controls.
