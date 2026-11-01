# H25 saved model and physical paired description

This post-hoc descriptive analysis joins the sealed eight-branch CUDA
report to the four-job physical saved receipt. It checks the exact winner
suffix bytes, winner index and program IDs, common prefix bytes and cp15
state, and the observed cp25 endpoint. Smaller model Q, object position error
and wrapped object angle error are compared separately; their units and
meanings are not interchangeable. No model or simulator was run here.

| Base | Model Q preferred | Final position preferred | Final angle preferred | Position difference, library − joint | Angle difference, library − joint (rad) |
|---:|---|---|---|---:|---:|
| 0 | library | tie | tie | 0.000 | 0.000 |
| 1 | joint | joint | joint | +0.700 | +0.0149 |
| 2 | library | library | joint | −28.670 | +0.1514 |
| 3 | library | library | joint | −28.670 | +0.1514 |

Base 0 had identical physical object pose at cp15 and cp25 in both arms,
despite distinct suffix actions. Base 1 improved both object errors from
cp15 for both arms, with joint slightly lower at cp25. Bases 2 and 3 had
matching context and outcomes: library had lower endpoint position error,
while joint had lower wrapped angle error. This is a tradeoff between two
observed endpoint metrics, not evidence of a general ranking or mechanism.
All eight native successes were false; object-only success was unavailable.

Each arm's saved planning resources and actual physical costs remain in their
separate receipts and the machine-readable analysis. In particular, nested
model call counts must not be added, and equal counts do not establish equal
total compute. Two known roots and duplicate contexts do not support
independent-sample p-values or confidence intervals. No P1/P2 gate changed.

Machine-readable result:
`results/codex_p2_program_h25_paired_analysis_20261030_v1_RESULT_01.json`,
SHA256 `ae312a290f779da03c21045a32922e8c157b067e23c1908e84686dd46fdf7708`.
