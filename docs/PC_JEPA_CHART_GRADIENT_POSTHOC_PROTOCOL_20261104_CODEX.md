# JEPA saved response probes: chart versus full-score gradient

Use only the three sealed response decisions from the existing v12 contact
feedback run. For each decision's two initial five-step nominal probe
batches, verify all 21 recorded actions are exactly the nominal and its
ten `±0.05` coordinate probes, bind the original decision hashes and
context, and reconstruct the saved 2×10 response matrix. Compute central
finite differences of (a) full predicted latent MSE, and (b) squared chart
distance to the goal. Report both norms, cosine when both norms are
nonzero, and coordinate sign agreement using an absolute `1e-6` exclusion
threshold. Also report the candidate modes selected by the original
compiler. Do not extrapolate near-zero gradients, and do not claim that
finite differences at one scale are exact derivatives.

This is a post-hoc analysis of one exposed constructed goal, three feedback
contexts and two nominal actions, not six independent tasks. It does not
run the model or simulator, change any planning decision, prove physical
gradient accuracy, or establish PC1/PC3. Verify source audit and decision
hashes; write one new result without overwriting existing artifacts.
