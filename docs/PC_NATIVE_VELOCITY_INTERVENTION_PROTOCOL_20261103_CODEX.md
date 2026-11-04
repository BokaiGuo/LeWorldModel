# Fixed-bank velocity intervention (development only)

Use all 48 already-exposed native fixed-bank prefixes and the six already
sealed common ten-action suffixes in
`pc_geometry_native_fixed_bank_suffix_20261103_codex.json`. No candidates,
suffixes or roots are filtered using outcomes. The source was already
evaluated; this is a post-outcome mechanistic diagnostic, not a blinded
confirmation or a model decision.

For each prefix, create two fresh official MuJoCo 2.1.2 simulations from
its recorded endpoint position: one with its recorded velocity, one with
velocity set exactly to `(0,0)`. In each state, apply all six identical
suffixes with the established five-integrator-step action clock and no
early stopping. Record endpoints and full states. For each condition report
absolute end distance, passive change (zero suffix end distance minus
prefix-end distance), and active response (each suffix end distance minus
the same condition's zero-suffix end distance). Lower is better. Compute
the controlled velocity contrast in both passive and active quantities.

The fresh-state reconstruction discards hidden solver and warm-start state;
therefore compare every recorded-velocity suffix endpoint against the
original continuous replay's saved outcome, report any discrepancy, and
do not claim the reset is a pure velocity intervention if this control
does not reproduce the original. The zero-velocity arm is not a physical
action or a deployable observation. Velocity is a legitimate dynamical
state: finding velocity dependence is not a disproof of reconfiguration.
Results do not enter the RGB selector and cannot establish independent
generalization or a P1 gate.
