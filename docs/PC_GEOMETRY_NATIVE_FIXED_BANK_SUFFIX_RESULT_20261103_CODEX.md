# Official-backend fixed-bank suffix diagnostic

The fixed geometry panel was replayed in persistent MuJoCo 2.1.2 using the
official executor XML. For each of four starts, all 12 prefixes received
the same six common suffixes and 100 privileged CEM suffix evaluations.
The direct prefix was freshly generated on this backend. The frozen
[protocol](PC_GEOMETRY_NATIVE_FIXED_BANK_SUFFIX_PROTOCOL_20261103_CODEX.md)
defines distance, action clock, RNG and search budget; every chain runs 15
actions. The complete [receipt](../le-wm-main/results/pc_geometry_native_fixed_bank_suffix_20261103_codex.json)
includes all 4,800 search evaluations, common sequences and prefix states.

| Start | Position non-improving direction | Initial → prefix distance | Own CEM end distance | Zero / direct CEM end distance |
| --- | --- | ---: | ---: | ---: |
| (0.8, 0.8) | none beats both controls | — | — | 0.69875 / 0.36365 |
| (0.8, 1.8) | direction 3 | 1.41421 → 1.45946 | 0.02761 | 0.39714 / 0.05839 |
| (2.8, 0.8) | none beats both controls | — | — | 0.71414 / 0.23157 |
| (2.8, 1.8) | direction 1 | 1.41421 → 1.45946 | 0.01912 | 0.23586 / 0.10821 |

The table is an exhaustive description of this fixed bank, not a
preselected method or an independent outcome. Root-specific random draws
make these finite CEM scores search-dependent. These are position-only
endpoints, not native success, and the finite optimizer does not establish
an optimal control upper bound.

For start (0.8, 1.8), the same common suffix 5 gives final distances
0.0436 / 0.9393 / 0.3387 for direction 3 / zero / direct respectively;
the corresponding changes from each prefix's own zero-suffix endpoint are
-0.6527 / -0.4749 / -0.5211. For start (2.8, 1.8), common suffix 2 gives
0.3489 / 1.2487 / 0.5717 and changes -0.3475 / -0.1655 / -0.2880 for
direction 1 / zero / direct. These examples are descriptive after observing
all six common suffixes; the full panel is in the receipt.

The non-improving prefixes also produce substantial passive movement:
their ten-zero-action endpoints are at distance 0.6964, while their prefix
endpoints are 1.45946. Thus improved endpoint distance may reflect velocity
or passive motion as well as changed action response. Common suffix response
differences alone do not establish a useful reconfiguration mechanism.
This experiment used privileged state and physical simulation both for
one-step direct actions and suffix optimization. No frozen world model
selected either direction, and there is no direct-full-horizon or
equal-total-cost planning comparison.

An independent native evaluator replayed all 288 common-suffix chains and
all 48 best-search chains, reproducing their terminal distances within
`1e-12`. Three tests covering this replay and backend comparisons passed.
The next development step is to examine whether saved prefix velocity and
passive drift account for the effect, and then test whether an RGB-only
model can predict and select a useful prefix without access to physical
outcomes; strong-planner resource comparisons remain required.
