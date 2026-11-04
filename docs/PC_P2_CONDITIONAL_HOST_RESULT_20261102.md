# P2 conditional initialization: host-CUDA development result

The fixed protocol `PC_P2_CONDITIONAL_HOST_PROTOCOL_20261102.md` completed
on one exposed PushT B0 root using the frozen checkpoint. The plan SHA-256 is
`a0d05017806eb70d2b8514cc5179a3a3026d5eb45cdb829e6ff8783a3a88064f`;
the complete report SHA-256 is
`4f7fe258941ac4157671143dd025db51ffa41a70dc75c8bea7b22efc6b48e02a`.
The checkpoint parameter hash was unchanged. The plan was checked again
after execution.

Both arms used all 16 same-bank joint-spectral candidates and the same
eight-update projected-gradient optimizer. For every candidate, the
conditional arm first compiled its suffix at its own predicted intermediate
context; all 16 resulting initializers differed from their control
counterparts. The optimizer could then modify both packets.

| Model-only terminal latent MSE | Conditional initialization | Unmodified joint-spectral initialization |
| --- | ---: | ---: |
| Best of 16 | 0.649799 | 0.776401 |
| Mean of 16 candidate winners | 1.087340 | 1.145802 |

Both best scores used candidate 12, iteration 8. Conditional initialization
had lower model MSE for 16/16 paired candidates; these candidates share one
root and are not independent task samples.

Per arm the common optimization used 144 scores, 128 gradients and 288
packet transitions. Conditional initialization additionally used 16 prefix
transitions, 32 suffix forwards, 16 Jacobian calls requesting 192 output
directions each, and 16 ridge solves. The model hooks recorded 48 additional
action-encoder, predictor, predict and projection calls. These costs were
not given to the control, so **this is not an equal-total-budget win**.
Reported wall time was 15.16 s for the full run, including setup but not
separating arm times. Loading, inherited bank generation, energy, FLOPs and
internal backward kernels remain outside the detailed accounting.

The saved hash-chained ledger verified with no outstanding operations. For
all 32 candidate/arm winners, action bytes, iteration and score matched the
corresponding optimizer return event. Full report and journal are in
`le-wm-main/results/pc_p2_conditional_host_20261102_RUN_v1/`.

This is evidence that the two initialization mechanisms execute differently
on the real frozen model, not evidence of physical improvement, fair
compute, independent generalization or P2 admission. The control is the
joint-spectral bank under the common optimizer, not the separately specified
historical 40-candidate SRA.
