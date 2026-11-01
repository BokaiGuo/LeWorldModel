# cell02 unified gate: proposal and current limits

The delivered executable is a read-only audit that always exits nonzero; it
cannot launch production or create a real claim. Its test-only claim protocol
accepts only a canonical `/tmp` subdirectory. Test success is not production
authorization or evidence of a successful scientific run.

## Proposed integration, not yet authorized or implemented

1. An independent operator establishes an exhaustive cell02 attempt history,
   including arbitrary output paths, existing claims, remote hosts and running
   jobs. Any unresolved possible attempt blocks. Existing claims are retained;
   no automatic deletion, retry, or inferred recovery is allowed.
2. Administrative controls fence all old and alternative invocation paths for
   the entire launch and execution window. A directory scan, source hash, or
   process snapshot cannot establish this. The observed source list is not
   exhaustive; direct function imports and other machines can bypass it.
3. Review and seal the unified gate, its runtime dependency census, launcher
   relationship and externally approved final-environment manifest. Verify
   the unchanged scientific contract and complete resource/GPU gates.
4. Under those controls, acquire a single fixed cell02-wide persistent marker
   on a reviewed filesystem, independent of entrypoint, manifest or output.
   Check old claims and history again after acquisition. Any ambiguity, partial
   write, fsync failure, collision or crash blocks and retains the marker.
   Bind the same gate to the actual launcher and verify the inherited
   environment and source identities at the final dispatch boundary.
5. Run only after separate explicit authorization. Retain the existing
   candidate's own claim semantics; decide and review how the new outer marker
   and that inner claim interact before any actual integration. Completion
   requires the existing saved-output acceptance, not just a gate result.

The existing candidate creates its own fixed claim; three older feedback
launchers accept caller-selected output paths, and other host entrypoints also
exist. None is forced through this proposed new gate. The actual audit checks
only the fixed candidate claim/output and names explicitly spelling cell02 in
one results directory, without opening cell03. Its source census hashes six
known entrypoint files, not all possible callers. Therefore a clear audit
cannot prove historical uniqueness or future mutual exclusion. The marker
experiment additionally assumes a stable, single controlled directory and
cooperating participants; it does not demonstrate cross-host or network
filesystem guarantees.

Any remediation of old claims or policy to fence old entrypoints requires an
explicit authority decision. Fixed 16-cell denominator, original authority,
scientific gates and old artifacts are unchanged.
