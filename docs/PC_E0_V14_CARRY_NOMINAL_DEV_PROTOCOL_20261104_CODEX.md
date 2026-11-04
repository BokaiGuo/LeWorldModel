# v14 carry-conditioned nominal: fixed E0 development protocol

## Hypothesis and comparison

The stopped v13 traces show that its global candidate affects the complete
incumbent, but not recursive response-layer nominals. Test whether replacing
the second fixed nominal at each depth by the corresponding segment of the
best-scoring global chain improves automatic discovery. The first fixed
nominal remains. For symmetric finite differences, the replacement nominal
is clipped to `[-1+fd_step, 1-fd_step]`; the original complete chain is
also re-scored and preserved unmodified as an incumbent. This change is not
guaranteed to help and does not itself demonstrate a mechanism.

Compare v14, v13, v12, matched global-bank full-action CEM, and three-start
direct CEM. The v14/v13/global-CEM bank has exactly the same 65 full-chain
actions and seed. Use the existing 1920 five-step prediction-token cap for
each arm/root, and record other computation separately. Candidate and
derivative counts may differ due to response rank; the cap includes them.
Global proposals, carry selection and response nominals use only model
outputs and legal actions, not an injected witness.

## Fixed tasks and stopping

The eight roots are the Cartesian product of start `[-.5,y,theta]`, with
`y` in `{-0.31, 0.33}` and `theta` in `{-1.0, 1.35}`, and goal `[.5,gy]`
with `gy` in `{-0.28, 0.24}`. Check exact overlap against explicitly
listed prior development panels. Broader historical independence is unknown.
Run each arm/root once for 15 steps. Success remains terminal distance
`<0.05`; failed executions remain in the denominator. Stop after 40 attempts,
without outcome-driven retries or extensions.

Save every scored action and output, complete results, costs, seeds and
dependency hashes. Independently re-evaluate all actions using the other
same-environment geometry implementation. Audit the three initial banks,
action bounds, token accounting and complete-chain incumbents. Compare
v14 and v13 response nominal/candidate records as a wiring check. The
experiment is development only: it does not establish PC2 mechanism,
PC3 independent advantage, RGB transfer, or spectral increment.
