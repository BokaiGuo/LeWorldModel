# P3 resolved integer allocation: prospective engineering exploration

Fix both existing known roots, banks and points 20/30 before execution, with
uniform, signal and reverse per point in that order. Root identities and
policies do not depend on physical outcomes. Each slot is a fresh process under
the project cooperative GPU lock; no training. A 30-second process deadline
includes lock wait, CUDA and model load, planning and journal, but excludes
the subsequent independent saved consumer.

Uniform does not acquire signal and searches 24 or 34 candidates. Signal and
reverse independently compute both root signals, including nominal forward,
Jacobian, 384 VJP/gradient/hook calls and two SVDs, then search 20 or 30
candidates. Per-axis caps are identical within a point; nested axes must not
be added as FLOPs. Each arm pays one independently journaled scheduler
attempt/return, including the meter's synchronization and limits checks.

When signals are exactly equal, allocation is equal. Otherwise allocate at
least one candidate to each root, round the ideal proportional allocation away
from the equal split so the higher signal gets at least one extra candidate,
and cap its quota at point minus one. Reverse swaps the resulting counts.
The one-candidate resolution deliberately magnifies an arbitrarily small
nonzero difference; it is a design intervention, not evidence the signal
difference is meaningful. No root, threshold or strategy was selected by
physical outcome. The rule was designed after observing the old rounding
collision: it is not an independent preregistration or scientific admission.

Common caps and deadline do not prove equal actual total cost, hidden work
absence, energy or FLOPs. Saved consumer validates recorded operations and
candidate identities, not independent neural recomputation. OS source
closure and noncooperating GPU processes remain unknown. No physical replay
is authorized merely by this document.
