# Scored-incumbent output contract: development regression

The historical solver's selected candidate may be an initial sample, an
intermediate sample, or a final branch mean. Its `mean` and `var` fields do not
in general describe a distribution associated with the selected action.
Preserve that historical implementation and provide a versioned adapter that
exposes the selected action and its evaluated cost only. Explicitly report
that selected-action distribution statistics are unavailable; do not substitute
zeros, a different branch's statistics, or a posterior uncertainty estimate.

Under identical seeds and inputs, compare the old solver and adapter on CPU
and, when available, GPU: all scored candidate tensors, selected actions and
costs must match exactly. Exercise both prefix selection modes and multiple
seeds; verify that source kinds include both sampled and mean candidates if
the test data produce them. Verify that missing `mean`/`var` cannot silently
be consumed. Check the installed official policy's actual output consumption.
This is an interface repair, not a search or control improvement, and does not
change previously reported results or any scientific gate.
