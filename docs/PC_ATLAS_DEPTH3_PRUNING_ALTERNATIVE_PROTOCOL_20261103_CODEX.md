# Atlas depth-3 pruning alternative: development comparison protocol

## Scope and controls

This is a pre-run, exposed-development experiment on the existing, unmodified
exact unicycle dynamics, not a visual model, physical experiment, preregistered
generalization test, or formal evidence gate. No saved gradient actions,
oracle actions, injected candidates, or injection audit outputs may enter a
search or selector. The goal coordinates are the original eight analytical
smoke goals `angle = 2*pi*i/64 + .031`, `i=0..7`, with root history zero.
Use search seeds `42+i`, `142+i`, and `242+i`. These give 24 goal/seed pairs,
but only eight distinct, previously exposed goals.

Use the original v2 bank construction and compiler configuration, depth 3,
beam width 2, two random bridges, and per-root-branch search. Root actions
and all subsequent candidates come exclusively from v2's ordinary bank.
Each branch's terminal candidate ranking remains the ordinary nonlinear
terminal model cost with stable ordering. Direct replay of each branch winner
must match its split full-context digest. The best completed branch is selected.

## Intermediate strategies

Apply the following only after expanding two action packets:

1. **v2:** retain the lowest two current model costs.
2. **lookahead:** for each node call the existing one-packet `progress`
   compiler at that node and rank by its returned nonlinear model cost.
3. **lookahead-diverse:** first retain the best lookahead node; for the second,
   among other nodes whose lookahead cost is at most `best + 0.01`, retain the
   node maximizing Euclidean distance from the first node's second action,
   normalized by action range. Break ties by original expansion order. If no
   eligible alternative exists, retain the next lookahead-ranked node.
4. **random:** retain two distinct uniformly sampled candidates, sorted back
   into expansion order; use an independent seeded random generator so random
   selection does not consume bank-generation randomness.

The 0.01 cost tolerance is fixed here, in the model's three-coordinate weighted
mean-squared error units; it is not tuned on these results. Selection only
uses currently generated actions, model costs, and model lookahead. This
diversity rule is an exploratory engineering choice, not a claim that diversity
alone identifies useful plans.

## Budget and parity

For every expanded depth-2 node, **all four arms** execute the same existing
one-packet `progress` lookahead. The v2 and random arms discard those results.
This pads their computation to the same candidate-level trial workload; it
must be disclosed rather than represented as the natural cost of v2/random.
Count compiler attempted and returned rollout/Jacobian calls and the actual
number of action steps in each backend call, separating rollout and Jacobian
steps. Assert exact equality per paired goal/seed across all four arms or
fail closed and do not call the comparison call-matched. Candidate-dependent
computation, allocations, runtime and trace-recording cost need not match.
The added unused lookahead may be disabled in the standalone selector for
ordinary v2/random deployment, but the experiment always enables it.

Independently run the original `AtlasSearch` for each goal/seed and compare
all branch root names, chosen complete action sequences, terminal model costs,
full-context witness hashes and selected index against the padded v2 arm.
Record the original unpadded operation and action-step counts separately;
its parity-check work is additional and excluded from the comparison totals.

## Outputs and checks

Save all goal/seed/arm actions, scores, branch results, actual expanded
candidate intermediate and lookahead costs, retained candidate identities,
selection reasons, direct witness hashes, counts, configuration, source
hashes and NumPy/Python versions. Primary descriptive endpoints are terminal
Euclidean position error and model cost; report per-goal/seed values, means,
and paired wins/ties/losses, without statistical significance claims. Costs
are model values because exact unicycle is the backend itself. Log selection
differences and budget equality. Tests must cover v2 parity, stable ties,
diversity threshold and fallback, bounded candidate actions, deterministic
seeds, witness equality, no injected inputs, count accounting, and budget
exhaustion without a partial result.

Do not overwrite old files. If assumptions or budget equality fail, save no
comparison result and report the limitation instead of silently modifying the
protocol after seeing results.
