# FD cp5 CEM: first host attempt

The fixed two-root/four-arm development plan was executed on host CUDA under
the cooperative GPU lock. Two earlier attempts stopped before any physical
action: the sandbox lacked NVIDIA device nodes, then the host check compared
PyTorch's bare UUID with the `GPU-`-prefixed UUID. The identity check was
corrected to compare canonical UUIDs, tested on equal/different/invalid
identities, and the changed code was sealed in a new plan.

The host run's saved consumer found a partial result:

| Root | Selected | Parameter-zero |
| --- | --- | --- |
| 20263160 | prefix action mismatch before dispatch | 20 steps returned, final wrapped joint distance 0.546408 |
| 20263161 | prefix action mismatch before dispatch | 20 steps returned, final wrapped joint distance 0.598310 |

For each completed parameter-zero arm the 20 steps include five prefix steps,
an additional five-step parameter-zero control from the restored original
root, and a ten-step CEM suffix from its observed successor. The CEM uses 32
candidate rollouts; other model and physical costs are not fully metered.
Both selected arms failed prior to their first action. Hence these outcomes
cannot be compared as a four-arm experiment and do not show a scientific
failure of the selected prefix. The exact prefix mismatch needs diagnosis
before changing the bridge; no tolerance or frozen actions are changed here.

A subsequent controlled same-host CUDA conversion diagnostic identified the
cause: FD search produced float64 candidate parameters, but the bridge cast
them to float32 before decoding. Relative to the sealed actions, float32
reconversion differed at two components on root 20263160 and three on
20263161 (at most two ULP, absolute difference at most 3.73e-9). Converting
the saved float64 parameters reproduced all ten action components bit for bit
on both roots. Parameter zero matched either way. The original failed trace
did not save the intermediate conversion, so these are controlled
recomputations rather than direct recordings of that intermediate array.
The physical action identity requirement remains exact; the repair must
preserve the original precision instead of relaxing it.

Plan SHA-256:
`3272c6bb45aa494a49880d453478b36962e93313d8313166974b6edf68224395`.
Journal SHA-256:
`f5fe04b5572a544793aa9ff0d5f4f314a9888b65367cd0077d909727a7386f6b`.
The independent consumer output is
`le-wm-main/results/reacher_fd_cp5_host_saved_20261102_codexexclusive_01.json`.
It checks saved trace consistency, not independent live execution or source
closure. This uses exposed roots, not confirmation, and does not pass P1.
