# PointMaze V9: opportunity-gated prospective routing

## Motivation

The completed V5 mechanism audit found that a planning-shaped scorer can only
change a decision where it selects a candidate different from the cost scorer.
Most roots were not such opportunities. Fitting a router to every root mixes
unrepairable zero-difference cases into the supervised signal and can make an
otherwise useful conditional effect look like a null average.

## Frozen V9 intervention

The original V9 range (`11320..11831`) was found, before any V9 model fit, to
overlap 384 roots in the independently completed counterfactual-contrast V9
panel. Its static preflight correctly rejected that configuration. A first r2
fresh panel then exposed a builder schema omission: it materialized a train
bank while the V9 ledger required the held-exact shared bank, and was rejected
before model fitting or outcome access. The r3 V9 panel instead uses the fresh
range (`14264..14775`) and the exact in-support ordered
20-by-2 candidate-action interface. It keeps the V7 scorer and advantage
router architecture, but applies this predeclared rule:

\[
  \text{routing-eligible}(x)=
  \mathbb{1}[\arg\min_a \hat c_{\mathrm{cost}}(x,a) \ne
  \arg\min_a \hat c_{\mathrm{planning}}(x,a)].
\]

Only eligible cross-fit train roots fit the advantage ridge; only eligible
calibration roots choose and certify the release threshold; ineligible roots
always retain the cost-arm action. The router consumes a frozen 15-dimensional
compact map of scorer margins, visual-history summaries, and signed/temporal
differences between the two selected actions; the candidate scorers themselves
still consume the full ordered 20-by-2 sequence. The predeclared split is
160 train / 160 calibration / 64 outcome roots: it preserves the 384-root
evaluation budget while giving an approximately 22%-opportunity process enough
expected train opportunities to satisfy the capacity test. The run fails closed unless
there are at least 31 eligible cross-fit train roots (15 dimensions plus the
declared 16-root surplus) and at least eight eligible roots in each calibration
role. Those counts are feasibility gates, not target outcomes.

The r3 V9 launcher appends its state pool deterministically after CUDA
preflight from the newest state-only pool. It consumes no fitted local-repair
model, calibration label, or outcome artifact from that ancestry; the new
range is separately excluded against every discovered valid prior context.

## Claim boundary and success criterion

This is not a PlanWAM replication, GDD-WM validation, online planning result,
or universal world-model claim. It tests one narrow prospective proposition:
after removing roots at which routing cannot alter the selected action, does a
calibration-frozen router reduce held-out fixed-bank candidate regret compared
with its legally selected fixed-arm fallback?

A report is positive only when the independent calibration certificate
authorizes routing and the untouched outcome panel improves under that frozen
policy. A failure to reach the opportunity-count gate is a valid
identifiability result, not permission to lower the threshold or use CPU.

## r3 host-GPU execution (2026-10-22)

The r3 panel completed on the host RTX 5060 Ti after recurrent CUDA preflight
and used the declared fresh `14264..14775` root range. Static identity and
support checks passed: 160/160/64 root-disjoint train/calibration/outcome
roles and an exact held-candidate scorer bank. The opportunity gates were not
vacuous: the cross-fit training role contained 96 scorer-disagreement roots;
the two calibration roles contained 40 and 43 respectively.

Calibration selected a frozen threshold of `-0.00283734` against the fixed
cost fallback and authorized the routed policy. On the disjoint certification
role, its all-root mean routed-minus-cost regret was `-0.004578`, with a
bootstrap upper 95% bound of `-0.001272`. The untouched 64-root outcome then
gave mean regret `0.048025` for cost, `0.047897` for routed, and `0.041211`
for planning. Thus the authorized router improved its declared fallback by
`0.000128`, but its release precision was only `0.48` and the unconditional
planning arm was materially better on this panel.

This is a lawful, small held-out improvement over the fail-closed baseline;
it is not stable evidence that the router solves repairability or that it is
the preferred controller. The mechanism audit also records 44/64 outcome
disagreement roots, 25 router releases, and 12 effective releases. The
remaining problem is not candidate availability or opportunity detection: it
is discriminating beneficial from harmful planning-arm switches within the
eligible stratum. Any successor experiment must use a new root range and a
predeclared feature or repair intervention; it must not tune V9 from these
outcomes.
