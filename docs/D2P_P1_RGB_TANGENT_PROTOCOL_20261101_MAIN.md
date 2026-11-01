# Goal-RGB geometric tangent scoring experiment

Freeze before neural/control outcomes:

- Use the existing root0 push-and-return target (receipt SHA
  `5bd5f5bbe82fe085cf26ceef12dc8ff6e0bb2fbc784e241ae4f5f8e4f693ed8a`).
  Do not use the clipped reverse V3 target or rejected V4.
- Extract renderer-specific object/pusher masks only from target RGB.
  Require an8-pixel border margin, report this as a screen rather than a
  completeness/occlusion certificate.
- Five geometric axes: object x/y translation, object rotation about its
  RGB-mask centroid, pusher x/y translation. Full units are4 image pixels
  or5 degrees. Encode signed full and half units plus original:21 forwards.
- Central finite differences at half units define a192x5 tangent. Report
  full-versus-half derivative consistency, without treating it as physical
  calibration. Require finite full rank, condition number below1e6.
- For SVD U,S,V, ridge=.01*s_max^2; Q=U diag((s/(s²+ridge))²) U.T
  +(.01/s_max²)*(I-UU.T), normalized to trace192. No fitted weights,
  physical outcomes, simulator states, policy/critic/adapter training.

Control panel: original Q versus geometric Q, each structured/native/random/
immediate_progress, one known development task. Identical H4/H3/H2/H1 planner,
six .05 normalized optimizer updates, prefix protection and RGB feedback.
Change only the goal metric and its attributable construction costs. Keep
all eight declared arms, stopping rules and native task evaluator unchanged.
This is not an independent confirmation or a license to choose by outcome.
Do not select favorable axes, scales or roots after results.

Local geometric directions do not guarantee global physical task alignment.
The experiment must report actual task outcomes and non-greedy/capability
conditions separately. P1 remains closed without its original full evidence.
