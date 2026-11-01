# All-eight actual-RGB own-suffix result

All eight fixed indices were attempted. Six completed20 steps; candidates2 and5
hit the unchanged canvas guard at steps4 and5, before valid cp5 feedback.
Native success0/8, retaining both domain failures in the denominator.

Six complete shards passed24 saved decision-node checks. All225 fresh physical
image occurrences passed file/pixel hash and journal-chain validation. Partial
shards are NOT full saved-check passes: their failure type, actual prefix actions,
last state/dynamics/RGB and out-of-domain coordinates reproduce the earlier
independent physical intervention exactly.

| Candidate | cp5 jointXY | cp15 jointXY | cp20 jointXY | Status |
| --- | --- | --- | --- | --- |
| 0 | 101.68241 | 78.99277 | 73.43018 | complete20 |
| 1 | 106.82566 | 104.40471 | 90.39416 | complete20 |
| 2 | unavailable | unavailable | unavailable | domain failure at4 |
| 3 | 103.81712 | 82.54265 | 60.67751 | complete20 |
| 4 | 134.58253 | 143.77437 | 118.23997 | complete20 |
| 5 | unavailable | unavailable | unavailable | domain failure at5 |
| 6 | 121.74564 | 118.59002 | 112.75826 | complete20 |
| 7 | 91.01110 | 55.30098 | 31.09926 | complete20 |

All start at jointXY101.67640. Candidate0 is the zero-action prefix; candidate7 is
the original model choice. Every other index is explicitly forced intervention,
NOT autonomous selection or an independent sample.

Candidate3 improves substantially relative to executing its saved open-loop tail:
cp15 jointXY317.26044→82.54265. But shared-tail cp15 was43.23332, and all own-feedback
trials still fail native success. Real feedback helps this trajectory, but does not
establish robust planning capacity or solve P1. Candidate3 also made immediate
object-position and angle progress, so jointXY non-improvement cannot silently
replace a previously fixed object-only non-progress criterion.

The same original model-selected candidate7 remains better under cp20 jointXY.
There is no hidden successful forced prefix to promote into the method.

Evidence:
`le-wm-main/results/p1_candidate_feedback_analysis_20261101_main_MAIN01.json`
and all eight `p1_candidate_feedback_20261101_main_c{index}_RUN_MAIN01` directories.
Future candidate/search changes require a new prospective contract; this panel
is exhausted as a fixed diagnostic, not a reason to weaken the scientific gate.
