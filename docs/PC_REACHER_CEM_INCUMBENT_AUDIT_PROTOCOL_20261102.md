# CEM incumbent-retention diagnostic protocol

On the two previously observed Reacher development roots, replay the full
three-packet CEM with the exact same model, inputs, seed, 16 samples, 12
iterations and 4 elites as the archived full-CEM comparison. Compare the
archived final-iteration selected plan with the lowest model-cost sample seen
at any iteration. Keep sampling and distribution updates unchanged and use
no additional model calls. Record both plans, their iteration and predicted
cost, per-iteration best cost, RGB hashes and checkpoint binding.

This model-only audit was designed after the physical outcomes were visible.
Model cost is not a physical task metric. A changed plan requires a separate
physical execution before drawing any control conclusion. The old result is
retained unchanged and is not an independent or equal-compute confirmation.
