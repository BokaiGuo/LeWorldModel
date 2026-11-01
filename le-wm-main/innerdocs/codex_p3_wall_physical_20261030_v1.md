# Physical replay of fixed-wall winners

For each point, execute root-major order (root 0 then 1, each uniform, signal,
reverse) in a fresh environment per slot, all six slots, with the same original
root, goal, reset and ten-step warmup. Execute the saved raw float32 ten-step
winner unchanged; on native stop retain the actual prefix and do not pad or
retry. Use the existing physical resource caps, 600 s outer physical budget,
durable per-operation records, native action ACK and saved consumer. Physical
evaluation adds no neural inference. The earlier 30-second planning wall limit
is not a physical wall limit; report the two costs separately.

Input is the complete, hash-pinned six-slot wall batch, including its original
fixed contract, actual wall measurements, per-slot reports and saved consumer
receipts. No policy or winner is selected based on physical results. Both
points execute all six slots, even if actions coincide. Existing online-budget
physical runs used the same candidate bank and can have identical actions and
roots: this replay is not an independent physical sample or confirmation.
Known roots, unequal total neural computation, cache and dynamic-source
uncertainties, and lack of fresh-root evidence remain explicit. No gate change.
