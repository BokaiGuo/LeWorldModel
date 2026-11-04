# P3 two-point allocation curve: model-only development

The 20-candidate point reuses the previously saved known-root outcome. The
30-candidate point was newly run on host CUDA under its sealed schedule and
the saved consumer verified both points. The two-point contract itself was
created after the 20-point outcome was opened, and both roots were already
exposed. This is not independent or fully prospective confirmation.

| Candidates per arm | Signal quotas (root 0 / 1) | Reverse quotas (root 0 / 1) | Signal best model costs | Reverse best model costs |
| ---: | --- | --- | --- | --- |
| 20 | 5 / 15 | 15 / 5 | 1.008468 / 0.229728 | 0.972351 / 0.229728 |
| 30 | 8 / 22 | 22 / 8 | 1.008468 / 0.229728 | 0.877305 / 0.229728 |

Lower is better. At both points the signal allocation is worse at root 0 and
ties at root 1; the root-1 winner is the same action. This narrow result does
not establish a useful allocation advantage. At point 30 each arm incurred
64 metered predictor calls/samples, 64 action encodes, two signal calls and
30 candidate evaluations. Model internals, FLOPs, energy, complete wall time
and inherited encoding are not equated, and no physical outcome was measured
for the newly chosen 30-point action.

- Contract SHA-256:
  `d9afb2c9c695b6e4bf70bc866279116c602c6cb8df7e8143e89b736314a52557`.
- Point-30 report SHA-256:
  `08a0ff7d065682918c688365f7f40fae163c908dcf42f4e6e26bdae42802025e`.
- Curve consumer SHA-256:
  `1b65dcafa104b04b6d1f8a054c35a68746f781b757aaca53fa5e26059636e968`.
