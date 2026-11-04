# Predicted-successor CEM: preserve suffix witnesses

The development selector used to compute a CEM suffix for each prefix and
then discard the returned tensor, retaining only the model cost. It now
retains every candidate's suffix parameters and explicitly exposes the
selected suffix in its receipt. Each suffix must have the expected horizon
and action dimension and contain only finite values. No ranking rule, CEM
seed, model call count or non-greedy criterion changed.

The saved values are unconstrained optimizer parameters, **not environment
actions**. The existing bounded normalized-action transform and checkpoint
action normalizer are still required before execution. This repair supplies
the missing upstream plan for a later observed-successor rescore; it does not
itself connect that rescore to an executor or account for it in a runtime
budget. Older sealed runs do not retroactively contain suffix witnesses.

The predicted-successor, spectral-candidate and tail-carryover tests passed
(15 tests). These are code/fixture checks, not new physical outcomes or P1
evidence. The two related P2 and P3 tasks are being handled separately.
