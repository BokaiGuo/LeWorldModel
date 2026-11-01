# P2 four-anchor full20 baseline against prefix recompile

This fixed exploratory contract was written after prior H25 model and physical
results were known and before its own new runs. It is not preregistration,
independent confirmation, or a scientific gate. The fixed denominator is four
known cp15 contexts by two arms; bases 2 and 3 may duplicate context.

Each arm independently computes the root 192x20 Jacobian/SVD and obtains the
same four fixed first-packet spectral anchors (two modes, both signs). The
prefix-chain runs each first packet, then independently computes a 192x10
next-packet Jacobian/SVD in each resulting context and evaluates two modes,
both signs, as four continuations. The no-composition joint baseline uses the
same four initial first-packet logit anchors with zero suffix; for each anchor
it independently computes a full-horizon 192x20 Jacobian/SVD at that anchor
(converted to model float32), adds plus/minus two full20 singular directions
at radius CAP/sqrt(2), and projects only if the full20 logit norm exceeds
CAP=.75*sqrt(10). The full20 baseline can alter both halves and has no
program composition. Both arms evaluate 16 complete ten-step candidates,
select minimum terminal latent MSE with index tie-break, and replay the
winner from the original cp15 context. No outcome-dependent choices.

For each arm, caps are 48 packet and each nested model-entry call, 5
Jacobians, 5 SVDs, 17 scores, and 450 seconds. Upper bounds are not targets:
the joint baseline uses 44 packet calls and the chain 28, under the same
ceilings; the four anchored full20 versus four local10 Jacobians are distinct
cost axes, even though both have 192 mathematical VJP rows. Nested entries
are not additive, and equal FLOPs, energy, model-loading cost, full dynamic
source closure and inherited acquisition are not asserted.

The two candidate banks have the same cardinality, anchor allocation and
full20 norm domain but are not pointwise identical; local recompile versus
joint optimization remain different algorithms. Results do not isolate
composition. All candidates, action bytes, context witnesses, attempts,
returns, journal failures and fixed slots must be retained. No CPU neural
inference, training, P1/P2 admission, substitution, or retry is permitted.
Physical replay, if its independent binding and stop contract validate, is
on all fixed winners, not a selected subset.
