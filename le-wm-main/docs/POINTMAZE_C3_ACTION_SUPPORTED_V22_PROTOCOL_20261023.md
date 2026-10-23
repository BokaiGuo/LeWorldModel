# V22: action-supported post-repair C3

V21's sealed diagnostic showed zero full-bank release. Its decisive protocol
limitation is that a frozen V20 model was evaluated on freshly sampled action
sequences, making the purported post-repair test action-OOD.

V22 corrects only that error: all state roots and all calibration/outcome
labels are fresh, but the 16 declared candidate action hashes are exactly the
frozen V20 held bank. The static ledger proves this equality before rendering.

V22 uses independent 256-root calibration and 256-root outcome splits. It
must still fail closed if winner margins do not exceed the calibration-only
simultaneous fixed-bank contrast bound. It is not a conditional safety,
adaptive-planning, contact-transfer, or universal-world-model claim.
