# Finite-horizon action-to-task transfer-risk bound

This document upgrades the existing operator-norm recurrence into a bounded,
metric-specified theorem. It is a conditional theorem about a lifted local
realization; it is not a universal learned-world-model or planning theorem.

## Setup

Let \(\eta\) be a lifted latent state and let \(u_t\) be an action. On a
declared local tube \(\mathcal U\), write the true and learned realizations as

\[
\begin{aligned}
\eta_{t+1}&=A\eta_t+B u_t+r^{\mathrm{cl}}_t+r^{\mathrm{off}}_t,\\
\widehat\eta_{t+1}&=\widehat A\widehat\eta_t+\widehat B u_t,\\
y_t&=D\eta_t,\qquad \widehat y_t=\widehat D\widehat\eta_t.
\end{aligned}
\]

The closure term \(r^{\mathrm{cl}}_t\) is the residual of the declared
finite-dimensional lifted model on its calibration support. The off-manifold
term \(r^{\mathrm{off}}_t\) is the additional residual when a planner reaches
states outside that support. Define norm-bounded disturbances

\[
c_t\geq\|(A-\widehat A)\widehat\eta_t+r^{\mathrm{cl}}_t\|,
\quad o_t\geq\|r^{\mathrm{off}}_t\|,
\quad a_t\geq\|(B-\widehat B)u_t\|,
\quad d\geq\|D-\widehat D\|.
\]

Let \(X\geq\|\widehat\eta_{t+h}\|\). All norms below are induced Euclidean
norms. A positive-definite task metric can be inserted by replacing every
operator \(Q\) with \(M_y^{1/2}QM_\eta^{-1/2}\); the metric must be declared
independently of the held-out task outcome.

## Theorem: decomposed finite-horizon transfer-risk bound

Let \(h\geq1\), assume both trajectories remain in \(\mathcal U\) for times
\(t,\ldots,t+h\), and let \(\rho\) be any upper bound on the spectral radius
of \(A\). Define

\[
g_j=\|DA^j\|,\qquad s_j=\|D\|\rho^j,\qquad q_j=(g_j-s_j)_+.
\]

Then

\[
\boxed{\|y_{t+h}-\widehat y_{t+h}\|
\leq E_{\mathrm{initial}}+E_{\mathrm{closure}}+E_{\mathrm{off}}
+E_{\mathrm{action}}+E_{\mathrm{transient}}+E_{\mathrm{readout}}.}
\]

The components are

\[
\begin{aligned}
E_{\mathrm{initial}}&=\|DA^h\|\,\|\eta_t-\widehat\eta_t\|,\\
E_{\mathrm{closure}}&=\sum_{k=0}^{h-1}s_{h-1-k}c_k,\\
E_{\mathrm{off}}&=\sum_{k=0}^{h-1}s_{h-1-k}o_k,\\
E_{\mathrm{action}}&=\sum_{k=0}^{h-1}s_{h-1-k}a_k,\\
E_{\mathrm{transient}}&=\sum_{k=0}^{h-1}q_{h-1-k}(c_k+o_k+a_k),\\
E_{\mathrm{readout}}&=dX.
\end{aligned}
\]

The first three disturbance terms use a spectrum-only gain. The transient
term is the excess finite-horizon amplification not explained by spectral
radius alone. For a normal \(A\) in the declared metric and tight
\(\rho=\rho(A)\), \(g_j\leq s_j\), hence \(q_j=0\) and the transient term
vanishes. Equality need not hold for a task readout \(D\): with
\(A=\operatorname{diag}(0.9,0.1)\) and \(D=[0\;1]\), \(g_1=0.1\) while
\(s_1=0.9\). For a non-normal \(A\), \(q_j\) can be positive even when every
eigenvalue is stable.

### Proof

Let \(e_k=\eta_{t+k}-\widehat\eta_{t+k}\). Subtracting the two transitions
gives

\[
e_{k+1}=Ae_k+v_k,
\qquad \|v_k\|\leq c_k+o_k+a_k.
\]

Unrolling gives

\[
e_h=A^he_0+\sum_{k=0}^{h-1}A^{h-1-k}v_k.
\]

Because \(y_{t+h}-\widehat y_{t+h}=De_h+(D-\widehat D)\widehat\eta_{t+h}\),
the triangle inequality yields

\[
\|y_{t+h}-\widehat y_{t+h}\|
\leq \|DA^h\|\|e_0\|
+\sum_{k=0}^{h-1}g_{h-1-k}(c_k+o_k+a_k)+dX.
\]

Since \(g_j\leq s_j+q_j\) by the definition of \(q_j=(g_j-s_j)_+\),
separating the three residual classes proves the displayed decomposition.
\(\square\)

## What is and is not identifiable

| term | meaning | minimum data source | spectrum/alignment/commutator alone? |
|---|---|---|---|
| \(E_{\mathrm{closure}}\) | local lifted-model and operator closure residual | train or calibration transitions | no |
| \(E_{\mathrm{off}}\) | extra error on planner-reachable states | independent planner-reachable transitions | no |
| \(E_{\mathrm{action}}\) | action-effect operator mismatch | independent action-balanced interventions | no |
| \(E_{\mathrm{transient}}\) | excess amplification over spectral-radius gain | local \(A,D\)/Jacobian plus external metric | no for non-normal systems |
| \(E_{\mathrm{readout}}\) | task-space decoder/readout mismatch | independent task labels or decoder calibration | no |

The theorem does not claim that these terms are all estimable from training
covariance. In particular, off-manifold, action-effect, and readout terms
require data beyond a state-prediction loss. A numerical bound is valid only
after independently justified upper bounds are supplied.

## Tightness and counterexample

The bound is tight for a scalar positive system \(A=[a]\), \(D=[1]\), with
nonnegative collinear residuals: every triangle inequality is attained. The
executable audit uses \(a=0.8\), horizon 4, and constant closure residual
\(0.2\).

Spectrum, covariance alignment, and the covariance-predictor commutator do not
identify action-to-task transfer. Let

\[
A=\operatorname{diag}(0.9,0.4),\qquad \Sigma=I.
\]

For pair one use \(B_1=(1,0)^\top,D_1=(1,0)\); for pair two use
\(B_2=(0,1)^\top,D_2=(0,1)\). Both pairs have the same predictor spectrum,
the same covariance, and zero covariance-predictor commutator, but

\[
D_1AB_1=0.9\neq0.4=D_2AB_2.
\]

Thus \(B\) and \(D\) are part of the decision-relevant transfer object.

## Nonlinear learned-model applicability

For a nonlinear predictor \(F\in C^2(\mathcal U)\), choose a reference point
\(\eta\in\mathcal U\) and set \(A=J_F(\eta)\). If the Hessian operator norm
is bounded by \(L\) throughout a convex tube of radius \(r\), then

\[
\|F(\eta+\delta)-F(\eta)-A\delta\|
\leq \tfrac12L\|\delta\|^2,\qquad \|\delta\|\leq r.
\]

This term enters \(c_t\). The linear theorem applies only while both
trajectories remain in the tube and the accumulated bound remains within its
radius. Otherwise it is a local diagnostic, not a global learned-model
theorem.

## Executable audit

The algebra, tightness example, counterexample, non-normal comparison, and
local nonlinear remainder are generated by
`experiments/theory_spectral_transfer/run_finite_horizon_transfer_risk_decomposition.py`.
The output is synthetic theorem evidence only; it does not establish
learned transfer-risk prediction or planning benefit.
