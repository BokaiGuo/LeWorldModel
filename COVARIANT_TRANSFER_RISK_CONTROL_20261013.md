# Covariant Transfer-Risk Control

## Research object

We define a control interface on a declared task metric (M\succ0), a local
linearization

\[
x_{t+1}=Ax_t+Bu_t+w_t,\qquad y_t=Dx_t,
\]

and its finite-horizon input-output maps

\[
G_h^\Phi=D A^{h-1}B.
\]

The coordinate change (x'=Tx) is accompanied by

\[
A'=TAT^{-1},\quad B'=TB,\quad D'=DT^{-1},\quad
M'=T^{-T}MT^{-1},
\]

so (G_h) and metric norms are invariant.  The covariance/predictor
commutator is only one diagnostic used to estimate geometry; it is not the
certificate itself.

## Conditional control theorem

Let (A_M=M^{1/2}AM^{-1/2}) and (B_M=M^{1/2}B). Assume that the nonlinear
predictor remains in a declared local tube and that its local remainder and
all exogenous/model errors are bounded by (epsilon) per step. If

\[
\alpha=\|A_M\|_2<1,\qquad \beta=\|B_M\|_2,
\]

and (|u_t|_2\leq U), then the metric ball

\[
\mathcal T_r=\{x:\|x\|_M\leq r\},\qquad
r=\frac{\beta U+\epsilon}{1-\alpha}
\]

is robust positively invariant, provided (r) is no larger than the declared
local tube radius. For a task readout (D), the state-induced task error is at
most (|D M^{-1/2}|_2r), plus the independent readout error.

For a candidate sequence (u_{0:h-1}), the controller minimizes

\[
J(u)=\|D\hat x_h-y_g\|_M^2+\lambda \bar E_h(u),
\]

where (ar E_h(u)) is the finite-horizon bound with the separately measured
closure, planner-reachable off-manifold, action-effect, transient, and readout
terms. The action is released only if:

1. an independent action-balanced intervention set identifies the action
   effect;
2. an independent real/predicted transition gate passes;
3. every required error envelope is available; and
4. (ar E_h(u)\leq\tau) and the local tube certificate passes.

Otherwise the implementation returns the declared fallback action. This is a
new control contract around (G_h^\Phi), not an unconditional guarantee for
arbitrary learned representations.

## Why the earlier no-go results remain necessary

The theorem uses the external metric (M), local operator norm, (G_h), and
independent residual envelopes. Spectrum, Euclidean commutator, or alignment
alone do not determine these quantities. The same-spectrum/same-commutator
counterexample therefore remains valid, while the new controller makes the
missing information operational: it refuses to plan when that information is
not identified.

## Learned nonlinear applicability

For a learned successor (F_\theta), take (A=J_F(z_0)) and include the
second-order Taylor remainder in (epsilon). The theorem is applicable only
inside a verified local tube, with a task metric fixed before held-out
planning, independent action interventions, and independent planner-reachable
transitions. It does not prove physical causality or global planning.

## Executable evidence

- implementation:
  `le-wm-main/experiments/theory_spectral_transfer/risk_aware_control.py`
- synthetic audit:
  `le-wm-main/experiments/theory_spectral_transfer/run_covariant_transfer_risk_control_audit.py`
- tests:
  `le-wm-main/tests/test_risk_aware_control.py`

The audit is intentionally labeled synthetic. The learned-model gate remains
closed until (G_h^\Phi), all residual envelopes, and an independent
cross-environment consequence are exported from real checkpoints.
