# Fixed eight-slot prefix-recompile physical replay

After the CUDA batch is sealed, replay all fixed base 0–3 slots in order,
joint then prefix_chain. Each arm executes the original 15 raw prefix steps
followed by its two saved CUDA winner packets. The natural warmup, original
root/goal, exact 106-event cp15 reference and observed context must match;
no zero padding, resampling, substitutions, retries or post-hoc arm choice.
Native termination/truncation or engineering failure remains recorded.

Each two-arm job has caps of 70 public steps, 708 physics steps and 222
explicit renders, with two constructors/closes, four resets, and a 900-second
wall limit. Physical counts and prior CUDA accounting remain separate.
This is an exploratory known-root replay, not a prospective preregistration,
independent confirmation, equal total compute or P1/P2 admission.
