# Native C1 contact geometry: independent saved-data arithmetic

`experiments/spectral_reachability/p5_native_contact_geometry_main_20261027_v1.py`
reconstructs the circle/T-polygon clearance and kinematic PD substeps from the
sealed native three-branch report. It imports no neural model or simulator and
does not perform fresh physics. Its source report SHA is fixed in the module.

The object transform uses the saved body origin, without subtracting center of
gravity again. The agent radius is 15 in native coordinates. The helper rejects
circle centers inside either polygon instead of claiming to implement a general
signed distance to a nonconvex union. Positive sampled clearance is not a
continuous swept-path certificate.

## Actual result

`results/p5_native_c1_outcome_audit_20261027_v2_main_v1/contact_geometry_main_v1.json`
reports `saved_native_geometry_consistent`, with all 60 checks true. Decision
clearance is 8.45073233561492 native units. Minimum reconstructed suffix
clearance is 7.095282559733288 for compiled, 7.404601574716597 for nominal,
and 3.1529822663746288 for factual_reference. All saved position and velocity
values match the PD reconstruction exactly. Suffix object state is stationary
and saved suffix contact counts are zero.

This localizes this execution's lack of object progress to a non-contact suffix;
it does not establish why the frozen model selected it, certify model contact
derivatives, or prove the absence of all implementation errors. Early prefix
contacts remain part of the original report. No actions, task metrics, contact
thresholds, or evaluation roots were changed in this arithmetic audit.

The standalone geometry tests pass (6 tests). The main-thread joint invocation
with the conditional compiler tests passed 29 tests; these are implementation
tests, not P1/P5 admission or neural prediction evidence.
