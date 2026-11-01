# Ratio34 downstream physical replay contract

This new downstream physical protocol is separate from and does not change the
original model-only ratio34 protocol. Fix both known roots, three arms and two
budget points: 12 branches. Execute each branch exactly once in a fresh
environment, including duplicate selected actions. No selection by outcome,
retry, neural call, training, or replanning. For each branch derive the action
only from the independently reconsumed corresponding fixed model receipt.
Bind source hashes, root and task identity, selected candidate, ten raw float32
action bytes, actual model-axis counts and caps before any physical run.

Use the existing warmup/source/RGB guards and native action acknowledgment.
Stop immediately on native termination or truncation; record the returned
prefix and missing steps as censored, never impute or call a partial endpoint
complete. Any other error is retained as failure; completed physical panels
must pass the saved consumer. Preserve actual attempted and returned physical
resource axes; never combine with model axes or sum nested axes. Per point,
the existing vector caps include 6 constructions, 120 public steps and 1224
hooked physics steps, with 600 seconds wall. Two six-branch point runs retain
the complete denominator even if either fails.

The roster and method are fixed after seeing known-root model results.
This is exploratory and not independent confirmation, P3 admission, or
equal-total-compute evidence. Dynamic source closure is not certified.
