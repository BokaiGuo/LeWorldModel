# Observed-RGB direction CEM: development physical result

All 26 fixed prefixes on the two exposed roots completed the real
prefix–RGB feedback–CEM suffix chain. The saved consumer checks source
hashes, all 26 action sequences, 520 step intents/returns (including 130
zero-control diagnostic steps), goals, RGB histories, no early stop, frozen
model hashes, declared objective and 32 candidate rollouts per suffix. The
existing pairing consumer also matches every cp5 frame, goal and actual
action history to the earlier L2 run. This is saved-record consistency,
not independent simulator replay.

| Root | Direction method better than L2 | Mean direction minus L2 final joint distance | Range |
| --- | ---: | ---: | ---: |
| 20263160 | 0/13 | +1.300547 | +1.208489 to +1.395814 |
| 20263161 | 0/13 | +0.571463 | +0.182755 to +1.430078 |

Lower wrapped joint distance is better. Every optimized suffix differs from
its L2 counterpart. The two roots and their 13 shared-prefix cases are
development data, not 26 independent tests. Different objectives and
candidate rankings cannot establish the underlying cause of the poorer
physical result. No method selection, equal-total-budget advantage, P1
mechanism, independent confirmation, or strong-baseline result is established.

## Objective-scale check

The fixed formula is `mean(e^2) + 0.1 * (e dot u)^2`, with 192 latent
coordinates. Thus along `u`, the quadratic coefficient is **20.2 times**
the coefficient in a perpendicular direction, not a 10% increase over
ordinary isotropic weighting. The implementation matches the written
formula and a fixture checks that ratio. This strong anisotropy is a concrete
design/scale concern, not proof of a coding bug or proof that it caused the
physical outcome. Any alternative weighting must be a new development
protocol; these results remain unchanged.

The 52 extra encoder invocations are declared by the implementation
(two per suffix), not verified with instrumentation. Equal CEM candidate
counts do not prove equal total compute. The first saved pairing receipt
used reused helper field names `q_final` for the new method; it is preserved
as v1, and v2 renames them to `direction_final` to avoid confusing this
objective with the earlier goal-Q experiment.

- Plan SHA-256:
  `ee383a1763d84a40d3023b89b82b0afff8ec062be5a1725c380da01e45f18039`
- Journal SHA-256:
  `13adeb4ba17140ca70de76c6c47d43fae8dc2805d1954faed8bd4c73824d5239`
- Saved paired receipt v2 SHA-256:
  `68744fa686f020dd9ea1538c4b2d11638ad456658dffd0569eba8daaacae6250`
- Targeted tests: 12 passed.
