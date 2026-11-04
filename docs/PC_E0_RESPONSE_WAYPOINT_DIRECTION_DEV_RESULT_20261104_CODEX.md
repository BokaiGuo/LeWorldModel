# E0 response-derived waypoint direction: development result

The fixed eight-root protocol evaluated a new two-dimensional response-derived
waypoint candidate, isotropic random and goal-perpendicular directions with
matched finite-difference probes and paired optimizer seeds, plus v11 and
direct CEM. No plan failed.

| Arm | Mean terminal distance | Success `<0.05` | Prediction tokens |
|---|---:|---:|---:|
| Response waypoint | 0.471851 | 0/8 | 1919 per root |
| Random waypoint | 0.450298 | 0/8 | 1919 per root |
| Goal-perpendicular waypoint | 0.331349 | 2/8 | 1919 per root |
| v11 response | 0.500430 | 0/8 | 1893–1917 |
| Three-start direct CEM | 0.500721 | 0/8 | 1920 per root |

Response waypoint reached 0.260807 on one root but did not match the two
successes of the geometric control. This finite development comparison does
not establish an advantage for the specified response-direction rule, nor
does it rule out other response programs or prove that the theory is false.
The random and geometric arms deliberately also paid for the response probes
and SVD to isolate direction; they are not optimized random or geometric
planners with that cost reclaimed. The waypoint scale and decomposition are
hand-designed and informed by the exposed wall family.

At the declared nominal, the two response singular values were approximately
0.670820 and 0.644607 on these roots. The absolute cosine between the
response-selected and goal-perpendicular directions ranged from 0.738 to
0.992. Similar singular values and direction differences are descriptive;
they do not alone identify the cause of search failure. There is no
high-dimensional spectral-efficiency claim.

A separate post-hoc audit compared the local linear prediction from this
nominal with the actual endpoints of all 48 optimized waypoint prefixes. For
response/random/geometric directions, median endpoint errors were
0.765/0.556/0.941 and median error divided by actual displacement was
0.974/0.705/1.134. Predicted versus actual goal-cost-change signs agreed for
10/16, 12/16 and 12/16 prefixes, respectively. Only 2/16, 1/16 and 0/16
prefixes changed the wall-blocking sequence relative to nominal, so in this
limited data large extrapolation errors are not explained solely by a changed
blocking sequence. This test used actual optimized large actions, not a
certification of the response over all directions or scales.

The main audit recomputed 31,892 trajectories and the response matrix from
saved float32 probes; a separate line-intersection implementation replayed
383,340 action steps. A structural audit rebuilt the mode selection, target
directions and common probes. These use the same abstract E0 wall, not an
independent physical environment. [Raw results and audits](../le-wm-main/results/pc_e0_response_waypoint_direction_dev_20261104_codex/).
The panel is stopped as specified. PC0–PC5 remain unchanged; in particular,
no new same-chain mechanism, independent advantage or RGB transfer has been
established.
