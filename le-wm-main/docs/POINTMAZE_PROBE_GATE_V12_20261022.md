# PointMaze V12: calibration-frozen probe and local-release gates

V11 r5 established two facts on fresh roots: locally supported RGB candidate
banks can be constructed and safely consumed for one action, but unconditional
probing is harmful.  V12 therefore treats probing as a decision, not as a
default controller phase.

For each predeclared pair of thresholds, V12 first uses 32 fresh calibration
roots to evaluate:

1. an information-action gate: probe only if the top-two RGB visual-dynamics
   candidate costs have a gap no larger than a frozen ambiguity margin;
2. a release gate: after probe, use the local bank only if its RGB-chart
   predicted best action exceeds the visual fallback by a frozen advantage.

The original calibration attempt on roots `17864..18071` caught a deployment
contract defect: an abstaining gate could still learn a local chart from later
fallback actions, so it was not behaviorally equivalent to RGB fallback.  It
is retained as an invalid artifact and did not open outcome roots.  The r2
state extension contains 208 fresh roots (`18072..18279`), satisfying the pool builder's
144-evaluation-plus-64-independent-design contract; V12 uses 32 for gate
calibration and reserves 64 untouched roots for the possible outcome panel.
The selector authorizes a configuration only if it releases at least four
calibration roots, does not lose total paired success, and has non-positive
mean paired terminal-distance difference against the visual baseline.  If no
configuration is authorized, the 64 independent outcome roots are not run.
This is a deliberate negative result about prospective repairability, not a
license to tune thresholds on outcome data.

## r2 calibration result and consequence

The corrected r2 host-CUDA calibration ran all 12 declared gate pairs on 32
fresh roots (`18072..18103`). The baseline-equivalence audit passed on every
uncharged abstention (32/32 for the zero-margin policy; 29/32 for the widest
margin, with the other three roots carrying an explicit local-bank decision).

No configuration met the non-empty authorization rule. At margins up to
`0.03`, no root reached local-bank construction/release; at margin `0.10`,
three roots reached a local-bank decision, but two failed support construction
and the one constructed bank predicted a `-0.313` advantage relative to the
visual fallback, so it failed the release gate even at zero required
advantage. Consequently the selector emitted
`rejected_calibration_no_nonempty_noninferior_gate`; the 64 independent
outcome roots (`18104..18167`) remain sealed.

This closes the present **local affine RGB-probe repair class** under the
declared task-safe probes, one-step chart, and visual-dynamics comparator. It
does not establish that every active observation or every dynamics repair is
impossible. The appropriate next repairability branch is a separately
specified representation/dynamics intervention with its own prospective
calibration—not a broader outcome-root threshold sweep.
