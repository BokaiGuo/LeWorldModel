# Finite-Horizon Transfer-Risk Theorem

This note states the strongest theory claim currently justified by the
LeWorldModel program. It is a local, finite-horizon bound for a fixed action
sequence or a predeclared candidate bank. It is not a universal safety theorem,
and it does not certify adaptively generated CEM candidates.

## Setting

Let \(x\in\mathcal X\) be the physical state, \(u\in\mathcal U\) an executed
action, \(F\) the true transition map, and
\(\phi:\mathcal X\to\mathbb R^d\) the frozen encoder. Define
\(z_k=\phi(x_k)\), where
\[
x_{k+1}=F(x_k,u_k).
\]

Let \(\widehat F\) be the learned latent predictor and let
\[
\widehat z_{k+1}
  =\widehat F(\widehat z_k,\widehat u_k),
\qquad \widehat z_0=z_0 .
\]
A task readout \(q:\mathcal X\to\mathbb R^p\) and a learned readout
\(\widehat q:\mathbb R^d\to\mathbb R^p\) define
\[
e_h^{\mathrm{task}}=q(x_h)-\widehat q(\widehat z_h).
\]

All quantities are evaluated only on a declared planner-reachable tube
\(\mathcal T_h\) and a declared action-support set \(\mathcal S\). This support
restriction is essential: without it, calibration data cannot imply a finite
out-of-distribution bound.

## Assumptions

For \(k=0,\ldots,h-1\), assume:

1. **Local predictor Lipschitzness.** On
   \(\mathcal T_h\times\mathcal S\),
   \[
   \|\widehat F(z,u)-\widehat F(z',u)\|
   \le L_k\|z-z'\|.
   \]

2. **Closure error.** The one-step representation/predictor mismatch is
   \[
   c_k=\|\phi(F(x_k,u_k))-\widehat F(\phi(x_k),u_k)\|.
   \]
   An independent calibration bound \(\bar c_k\ge c_k\) is required for
   certification.

3. **Off-support deviation.** If the executed pair is outside calibration
   support, its additional predictor deviation is bounded by \(o_k\). For an
   in-support execution, \(o_k=0\). This term is not identifiable from
   in-support prediction MSE alone.

4. **Action mismatch.** The executed action differs from the planned action by
   \(\eta_k=\|\widehat u_k-u_k\|\), and \(\widehat F\) is action-Lipschitz:
   \[
   \|\widehat F(z,u)-\widehat F(z,u')\|
   \le M_k\|u-u'\|.
   \]

5. **Task readout regularity.** The local physical-to-latent task readout is
   \(L_q\)-Lipschitz on the reachable tube, and the learned readout residual is
   \[
   r_h=\|q(x_h)-\widehat q(z_h)\|.
   \]
   The inverse notation is only a local chart; no global inverse is assumed.

6. **Local nonlinear remainder.** On the verified local ball around the
   reference trajectory, a Jacobian envelope is accompanied by a certified
   remainder \(\rho_k\ge 0\):
   \[
   \|\widehat F(z_k+\xi,u_k)-\widehat F(z_k,u_k)-J_k\xi\|
   \le \rho_k\quad\text{for all verified }\|\xi\|\le r_k.
   \]
   If this condition cannot be checked on the whole tube, the certificate
   abstains.

## Theorem (finite-horizon transfer-risk decomposition)

Let \(A_k\) be a valid local Jacobian or incremental linearization of
\(\widehat F\). Define the exact local amplification
\[
\Gamma_{h,j}=\|A_{h-1}\cdots A_{j+1}\|,
\qquad
\Gamma_{h,h-1}=1.
\]
Let \(\overline\Gamma^{\mathrm{ref}}_{h,j}\) be any independently verified
nonnegative reference envelope satisfying
\[
\Gamma_{h,j}\le
\overline\Gamma^{\mathrm{ref}}_{h,j}+\tau_{h,j},
\qquad \tau_{h,j}\ge0.
\]
The reference envelope may be a normal/spectral envelope, a certified
operator-norm envelope, or another declared baseline. For any fixed supported
action sequence of length \(h\),
\[
\|e_h^{\mathrm{task}}\|
\le
E_{\mathrm{readout}}
+L_q\left(
E_{\mathrm{closure}}+
E_{\mathrm{off}}+
E_{\mathrm{action}}+
E_{\mathrm{transient}}+
E_{\mathrm{remainder}}
\right),
\]
where
\[
\begin{aligned}
E_{\mathrm{readout}}&=r_h,\\
E_{\mathrm{closure}}&=
  \sum_{j=0}^{h-1}\overline\Gamma^{\mathrm{ref}}_{h,j}\,\bar c_j,\\
E_{\mathrm{off}}&=
  \sum_{j=0}^{h-1}\overline\Gamma^{\mathrm{ref}}_{h,j}\,o_j,\\
E_{\mathrm{action}}&=
  \sum_{j=0}^{h-1}\overline\Gamma^{\mathrm{ref}}_{h,j}\,M_j\eta_j,\\
E_{\mathrm{transient}}&=
  \sum_{j=0}^{h-1}\tau_{h,j}
  \left(\bar c_j+o_j+M_j\eta_j\right).
\\
E_{\mathrm{remainder}}&=
  \sum_{j=0}^{h-1}
  \left(\overline\Gamma^{\mathrm{ref}}_{h,j}+\tau_{h,j}\right)\rho_j.
\end{aligned}
\]
Thus non-normal or otherwise excess finite-horizon amplification is an
explicit risk term, while the base terms are not double-counted.

### Proof

Let \(\delta_k=\|z_k-\widehat z_k\|\). Insert and subtract
\(\widehat F(z_k,u_k)\) and \(\widehat F(z_k,\widehat u_k)\):
\[
\begin{aligned}
\delta_{k+1}
&\le
\underbrace{\|\phi(F(x_k,u_k))-\widehat F(z_k,u_k)\|}_{c_k}
+\underbrace{\|\widehat F(z_k,u_k)-\widehat F(z_k,\widehat u_k)\|}
_{\le M_k\eta_k}\\
&\quad+
\underbrace{\|\widehat F(z_k,\widehat u_k)
-\widehat F(\widehat z_k,\widehat u_k)\|}
_{\le L_k\delta_k}
+\rho_k+o_k.
\end{aligned}
\]
Unrolling from \(\delta_0=0\), or using the corresponding local linearization,
gives
\[
\delta_h\le
\sum_{j=0}^{h-1}\Gamma_{h,j}
\left(c_j+o_j+M_j\eta_j+\rho_j\right).
\]
The envelope assumption bounds each \(\Gamma_{h,j}\) by
\(\overline\Gamma^{\mathrm{ref}}_{h,j}+\tau_{h,j}\). Upper-bound \(c_j\) by
\(\bar c_j\), and use
\[
\|q(x_h)-\widehat q(\widehat z_h)\|
\le r_h+L_q\|z_h-\widehat z_h\|.
\]
This yields the stated decomposition. \(\square\)
## Tightness and failure examples

### Tightness

For the scalar system
\[
z_{k+1}=Lz_k+c,\qquad
\widehat z_{k+1}=L\widehat z_k,
\]
with \(z_0=\widehat z_0=0\), \(q(z)=L_qz\), and aligned nonnegative errors,
\[
|e_h^{\mathrm{task}}|
=L_qc\sum_{j=0}^{h-1}L^{h-1-j}.
\]
The geometric amplification factors therefore cannot be removed without
additional structure.

### Spectrum is insufficient

The matrices
\[
A_1=\begin{bmatrix}\rho&K\\0&\rho\end{bmatrix},
\qquad
A_2=\begin{bmatrix}\rho&0\\0&\rho\end{bmatrix}
\]
have the same spectrum \(\{\rho,\rho\}\), but
\(\|A_1^m\|\) contains a term proportional to
\(|K m\rho^{m-1}|\), whereas \(A_2^m=\rho^mI\). Equal eigenvalues and
equal one-step error do not identify finite-horizon transfer.

### Covariance-only no-go

Under an invertible latent coordinate change, covariance-only statistics can be
made identical under the chosen whitening convention while action and readout
operators transform. Two models can therefore share covariance and spectrum but
have different
\[
G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi.
\]
No covariance-only certificate can be nontrivial without an action/readout
metric or an explicit gauge choice.

## What is estimable

| Term | Required data | Estimable from ordinary prediction data? |
|---|---|---|
| \(E_{\mathrm{closure}}\) | independent real/predicted paired transitions | only with held-out transitions |
| \(E_{\mathrm{off}}\) | support audit and extrapolation set | no, not from in-support MSE |
| \(E_{\mathrm{action}}\) | executed/planned actions and action-Lipschitz calibration | requires intervention logs |
| \(E_{\mathrm{transient}}\) | local Jacobian, Schur, or resolvent on the reachable tube | not from spectrum alone |
| \(E_{\mathrm{readout}}\) | independent task/readout labels | no, unless task labels exist |
| \(E_{\mathrm{remainder}}\) | local Jacobian/Hessian or finite-difference tube audit | only on verified local balls |

The theorem therefore supports three gates: (i) real/predicted closure,
(ii) support and action-identifiability, and (iii) finite-horizon
amplification/readout. Passing one gate does not authorize the others.

## Learned nonlinear applicability condition

For a differentiable predictor, on a ball \(B(z_k,r_k)\), it is sufficient to
verify
\[
\|J_{\widehat F}(z)\|\le L_k,\qquad
\|\widehat F(z)-\widehat F(z_k)-J_{\widehat F}(z_k)(z-z_k)\|
\le \tfrac12H_kr_k^2.
\]
The nonlinear remainder contributes
\[
E_{\mathrm{remainder}}
=\sum_{j=0}^{h-1}
\Gamma_{h,j}\tfrac12H_jr_j^2,
\]
and must be added to the bound. If the trajectory leaves the verified balls or
support set, the certificate must abstain rather than silently continue the
local argument.

## Claim boundary

This theorem gives a finite-horizon local transfer-risk bound for declared
support and fixed candidate sequences. It does not prove universal planning,
causal physical semantics of the latent, adaptive-CEM coverage, or success on
unseen environments. Those claims require independent interventions and
held-out closed-loop consequences.

