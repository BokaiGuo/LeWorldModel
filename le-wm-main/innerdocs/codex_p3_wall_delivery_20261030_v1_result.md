# Fixed wall delivery: exploratory result

The fixed 30-second per-slot process deadline produced six of six plans, and
all six passed the separate saved semantic consumer. The batch journal chain,
slot order, wall deadlines, and per-slot report and receipt hashes were
independently rechecked. The saved consumer runs after the delivery deadline;
its runtime is not included in the process wall.

| Point / strategy | Allocation | Process wall s | Model load s | Signal s | Search s |
|---|---:|---:|---:|---:|---:|
| 20 uniform | 12,12 | 20.047 | 2.474 | 0 | 0.755 |
| 20 signal | 15,5 | 22.358 | 2.369 | 2.839 | 0.484 |
| 20 reverse | 5,15 | 22.244 | 2.422 | 2.790 | 0.462 |
| 30 uniform | 17,17 | 19.878 | 2.375 | 0 | 0.962 |
| 30 signal | 22,8 | 22.403 | 2.371 | 2.752 | 0.683 |
| 30 reverse | 8,22 | 22.486 | 2.381 | 2.757 | 0.698 |

Process wall includes waiting, initialization and publication, and contains the
other intervals; those intervals overlap and are not additive. Lock waits were
3.4–4.4 microseconds. The four forward axes each count 48 at point 20 and
68 at point 30. Metric calls are 24/20/20 and 34/30/30. Signal and reverse
each additionally execute 2 Jacobians, 2 SVDs and 384 each VJP directions,
gradient API calls and backward hooks; these counts overlap and are not FLOPs.
Uniform uses four additional candidates instead of signal calculation.

This is a new run but uses known roots and a budget selected with prior measured
times. It is not independent confirmation, preregistration, equal total compute,
or scientific admission. Wall includes cooperative lock waiting and varies with
contention; the host lock does not exclude non-cooperating processes. Cache,
dynamic source closure, energy and FLOPs remain unknown or uncontrolled. No
physical validation was performed. Existing results and contracts remain intact.
