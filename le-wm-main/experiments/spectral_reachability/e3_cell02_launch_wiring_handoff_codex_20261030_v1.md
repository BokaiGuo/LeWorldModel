# cell02 unified launch wiring: simulation and production boundary

The accompanying module implements an explicitly **synthetic** state sequence:
fixture prerequisites -> permanent temporary scope claim -> recheck -> fixed
inert child -> strict report and postcheck. Only the candidate is accepted as a
proposed handoff. Failure after the marker leaves it in place; a second attempt
cannot reuse it. The test hooks, flags and temporary-directory inventory are
not evidence of production history, fencing, source sealing, resource
reservation, or authorization. No production command is dispatched.

The separate production dispatch function unconditionally fails, including
when callers provide affirmative flags or a command. The command-line entry
always reports blocked and exits nonzero. No current implementation in this
module launches the existing candidate, acquires a production claim, or
modifies old artifacts.

## Proposed real sequence requiring independent design and review

1. Inventory and adjudicate the complete relevant history, including
   arbitrary paths, claims, other hosts and outstanding attempts; unresolved
   possibilities block. Existing claims are not removed or retried.
2. Establish an independently verified, durable administrative fence over
   *all* competing invocation routes for the entire execution interval.
   Filename scans, a source census, cooperative locks and asserted booleans
   do not constitute that fence. Scope must include direct function imports,
   arbitrary output paths, and any other host sharing the relevant authority.
3. Independently seal and approve the new parent/child code and sources,
   final-environment manifest, claim layout and launch handoff. Revalidate the
   existing candidate manifest in its isolated interpreter with the unchanged
   inherited environment. Review resource and GPU coordination separately.
4. Acquire a fixed, persistent cell02-wide outer claim on a validated shared
   filesystem, then recheck the fence, history, manifests and resources.
   Retain the candidate's existing inner claim: it must be created by the
   original candidate, not pre-created by the outer gate. Any error or partial
   outcome leaves claims in place and returns nonzero; a later recovery needs
   an explicit independent authority decision.
5. Only after separate production authorization, invoke the original candidate
   without changing its environment, fixed output, scientific thresholds or
   parent/child census assumptions. Report completion only after its existing
   saved-output consumer passes. A parent-to-child source sealing and TOCTOU
   analysis is still required for this step.

The simulation deliberately includes a bypass during the inert child: it is
detected afterward but was not prevented. A real physical dispatch would
already have happened by then, so such detection cannot replace fencing.
The existing legacy launcher source confirms an independent lifecycle call;
the tests inspect its source but never invoke it. No proof is currently
available that all legacy paths are fenced. Consequently production remains
blocked; the 16-cell denominator and scientific admission remain unchanged.
