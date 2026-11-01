# Prefix-protected common optimizer

Fixed exploration: both existing roots, all four arms; response-diverse initializer
and first-feedback carry unchanged. At boundary0 only, retain each candidate's
first action packet exactly while optimizing the remaining two. This also applies
to the winner's extra refinement. Later feedback nodes optimize all their newly
planned packets as before.

Original optimizer differentiates and updates all H3 packets. This is legitimate
joint trajectory optimization, but NOT fixed-prefix suffix optimization. Test this
distinction rather than declaring the original scientific failure a code bug.

Zero the first-packet update gradient after the metered backward call and assert
exact prefix identity after each optimization. Same H3 forward and backward calls;
no unused calls added to equalize costs. All arms share the restriction. Bind the
new optimizer implementation into contract identity and every arm's optimizer SHA.

No task-state oracle, outcome-picked candidates or forced uphill selection.
Actual prefix progress must still be reported. This test cannot certify capability,
independent generalization or fair overall advantage on two known development roots.
