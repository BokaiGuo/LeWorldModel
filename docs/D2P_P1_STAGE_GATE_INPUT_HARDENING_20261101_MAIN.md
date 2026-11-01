# P1 stage admission input hardening

The scalar P0→P1 and P1→P2 admission functions previously treated any
falsy `calibration_only` value as an explicit non-calibration run. P1→P2
also compared root counts without verifying their type; `NaN` could bypass
both comparisons and infinity or a floating count could be accepted.
Malformed control rosters could raise an exception instead of returning a
closed gate. These are input-validation defects demonstrated with synthetic
fixtures, not proof that any saved experiment supplied malformed inputs.

The new `d2p_stage_admission_hardened_20261101_main` wrapper requires the
literal Boolean `False`, integer root counts, and a list, tuple or set of
nonempty control names. The existing minimum
count, mandatory controls, mechanism checks and strict interval sign are
unchanged. New negative tests were observed failing before the fix; after it,
64 tests across stage admission, P0 accounting and stage-consumer validation
passed. The consumer's historical positive fixture was updated with matched
synthetic P0 predict receipts, so its negative interval tests exercise the
intended checks rather than failing early on missing accounting.

The separate old `test_reacher_d2p_stage_gate_scope.py` AST comparison still
fails against its October 25 P0 snapshot because current P0 already contains
additional source-sealing and planner-instrumentation functions. This turn
did not edit that P0 runner or the snapshot; the failure is not waived or
represented as passing. These tests do not establish a scientific P1 result.

## Historical source correction

The initial in-place fix changed the hash of `d2p_stage_admission.py`, which
is pinned by the centered-witness source receipt, causing its collector to
reject with source drift. The legacy file has been restored byte-for-byte to
the pinned SHA-256 `24e33a213664c5a1bf2942cf90220aa2c20239cd5252dee1ec7e1e4155dd315b`;
the collector again verifies its four images and 5,319 source bindings.
The hardening remains in the separate wrapper. **Existing runners still use
the legacy module and do not automatically receive this fix**; a future
runner must explicitly import and bind the wrapper in a new protocol.
The legacy direct-call input-validation defect therefore remains an integration
hazard, not a fixed production path. Historical reports were not altered.
