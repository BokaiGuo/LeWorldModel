# Finite-horizon transfer-risk theorem

This document is the formal theory contract for the spectral-transfer audit.
It is deliberately narrower than a universal world-model theorem: it gives a
finite-horizon error bound for a declared lifted realization and states the
conditions under which a learned nonlinear model may be locally tested against
that bound.

## 1. Setting and assumptions

Let the true lifted state, learned lifted state, action, and task readout be
`x_t`, `xhat_t`, `u_t`, and `y_t = D x_t`, respectively.  On a declared
planner-reachable tube, assume the one-step error can be written as

\[
 e_{t+1}=x_{t+1}-\hat x_{t+1}
      = A e_t+w_t,
 \qquad
 w_t=c_t+o_t+a_t,
 \tag{1}
\]

where `c_t`, `o_t`, and `a_t` are closure, planner-off-manifold, and
action-effect residuals.  For `t=0,...,h-1`, assume calibration supplies

\[
 \|c_t\|\le c_t^+,\qquad
 \|o_t\|\le o_t^+,\qquad
 \|a_t\|\le a_t^+ .
 \tag{2}
\]

The task readout used by the learned model may have an additional error
`r_y`; assume `||r_y|| <= eps_D R_y`, where `R_y` is an independently
declared state/readout range.  The initial mismatch obeys
`||e_0|| <= e_0^+`.

The theorem is conditional on (1)--(2), the declared norm, and the tube being
valid for every state/action pair used by the planner.  In particular, a
training-set one-step MSE is not by itself a bound on `o_t`, `a_t`, or the
nonlinear remainder.

## 2. The theorem

Define the finite-horizon task gains and spectral proxy gains

\[
 g_j=\|D A^j\|_2,\qquad
 s_j=\|D\|_2\rho(A)^j,\qquad
 q_j=(g_j-s_j)_+ .
 \tag{3}
\]

For a terminal task error `e_task_h = D e_h + r_y`, define

\[
\begin{aligned}
 E_{\mathrm{closure}}&=\sum_{t=0}^{h-1}s_{h-1-t}c_t^+,\\
 E_{\mathrm{off}}&=\sum_{t=0}^{h-1}s_{h-1-t}o_t^+,\\
 E_{\mathrm{action}}&=\sum_{t=0}^{h-1}s_{h-1-t}a_t^+,\\
 E_{\mathrm{transient}}&=\sum_{t=0}^{h-1}q_{h-1-t}(c_t^++o_t^++a_t^+),\\
 E_{\mathrm{readout}}&=\epsilon_D R_y,\\
 E_{\mathrm{initial}}&=\|D A^h\|_2 e_0^+ .
\end{aligned}
\tag{4}
\]

### Theorem 1 (finite-horizon transfer-risk bound)

Under assumptions (1)--(2), for every admissible action sequence of length
`h`,

\[
 \boxed{
 \|e^{\mathrm{task}}_h\|_2
 \le E_{\mathrm{initial}}+E_{\mathrm{closure}}+E_{\mathrm{off}}
 +E_{\mathrm{action}}+E_{\mathrm{transient}}+E_{\mathrm{readout}} .}
 \tag{5}
\]

The same statement for the sequence `1:h` holds by applying (5) at every
prefix `k <= h` and taking the declared sequence norm (for example, the
maximum over prefix task errors, or the Euclidean norm of the stacked prefix
bounds).  The theorem is therefore finite-horizon and task-metric specific;
it does not say that a spectrum, alignment score, or commutator is a complete
certificate.

### Proof

Unrolling (1) gives

\[
 e_h=A^h e_0+\sum_{t=0}^{h-1}A^{h-1-t}w_t .
\]

Taking the induced norm and using the triangle inequality yields

\[
\|D e_h\|
\le \|D A^h\|e_0^+
 +\sum_t g_{h-1-t}(c_t^++o_t^++a_t^+).
\]

By definition `g_j <= s_j+q_j`.  Substituting this inequality and separating
the three residual arrays gives the first five terms in (4).  The readout
remainder contributes at most `eps_D R_y`.  This proves (5). `\square`

## 3. Tightness and failure counterexamples

The scalar positive system `A=[0.8]`, `D=[1]`, with nonnegative aligned
closure disturbances attains the triangle inequality exactly.  It is the
tightness example emitted by
`experiments/theory_spectral_transfer/finite_horizon_transfer.py`.

The adversarial family emitted by
`adversarial_matched_transfer_counterexample()` fixes

\[
\text{covariance, spectrum, commutator, prediction MSE, and action-inverse risk}
\]

but changes `B` and `D`.  Its two systems have different
`G_h = D A^(h-1) B` and different task loss for the same action.  Thus those
matched scalar diagnostics cannot identify transfer.  The example is
synthetic and is a failure boundary, not learned evidence.

## 4. What is estimable and from which split?

| term / condition | admissible source | not sufficient |
|---|---|---|
| `E_initial` | independent initial-state calibration | training MSE alone |
| `E_closure` | train/calibration transitions inside the declared tube | arbitrary held-out planning outcomes |
| `E_off` | independent planner-reachable transition collection | in-distribution random transitions |
| `E_action` | action-balanced independent interventions and inverse-risk calibration | passive action logs |
| `E_transient` | estimated local operators/readout plus finite-horizon gains | eigenvalues alone; commutator alone |
| `E_readout` | independent task-space labels/readout calibration | latent reconstruction error alone |
| positive planning consequence | fresh, episode-disjoint planning outcomes | the same data used to choose metrics or weights |

Every upper bound must be frozen before reading the planning outcome it is
intended to explain.  If a required split is absent, the tool must return
`rejected_missing_independent_transition` (or the more specific support/tube
failure), never a numerical permission.

## 5. Learned nonlinear applicability proposition

Let `f` and `fhat` be `C^1` on a tube of radius `r` around a reference
trajectory.  Let `J_t` be a local Jacobian estimate and suppose the Jacobian
variation is bounded by `L_t` on that tube.  Taylor's theorem gives a local
remainder bounded by

\[
 R_t^{\mathrm{nl}}\le \tfrac12 L_t r^2 .
 \tag{6}
\]

If (i) every runtime action is inside the calibrated support, (ii) an
independent planner-reachable transition split verifies the residual envelope,
(iii) the accumulated error tube stays within the radius used in (6), and
(iv) an independent task-readout calibration is available, then Theorem 1
applies locally after adding `R_t^{nl}` to the closure envelope.  The
checker in
`experiments/theory_spectral_transfer/nonlinear_transfer_certificate.py`
implements exactly these sufficient conditions.

This proposition is local, not global.  A model with an unbounded Hessian, a
support shift, or no independent successor transitions is rejected.  A local
Jacobian can explain a finite rollout only while its remainder and support
conditions remain certified.

## 6. Explicit nonlinearity failure boundary

Consider the scalar pair `f_+(x,u)=0.9x+u+Kx^2` and
`f_-(x,u)=0.9x+u-Kx^2` on `|x|<=r`.  They have the same local Jacobian at
`x=0`, the same action map, and can have identical one-step errors on the
calibration slice `x=0`, but their finite-horizon task transfers diverge for
any reachable nonzero state when `K r^2` is non-negligible.  A Jacobian-only
certificate therefore cannot distinguish them.  The remainder test must
reject whenever `K r^2/2` exceeds the frozen residual budget.

## 7. Claim boundary

The theorem establishes a conditional finite-horizon bound and an estimability
protocol.  It does not establish that LeWM latents satisfy the assumptions,
that a commutator is a universal transfer certificate, or that any learned
adapter has stable pose closure.  Those are empirical gates downstream of this
theorem.
