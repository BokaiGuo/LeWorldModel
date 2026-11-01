# P3 queued standalone-cost exploratory batch result

The new six-slot batch completed with status `measured_not_scientifically_admitted`;
all six separate saved consumers succeeded. The batch journal chain, fixed slot
order, rows and referenced report/consumer hashes were independently rechecked.
The previous failed batch remains unchanged.

| Point / arm | Root allocation | Process wall s | Child wall s | Load s | Signal s | Search s |
|---|---:|---:|---:|---:|---:|---:|
| 20 uniform | 12, 12 | 19.809 | 7.251 | 2.413 | 0 | 0.718 |
| 20 signal | 15, 5 | 22.185 | 9.749 | 2.382 | 2.770 | 0.456 |
| 20 reverse | 5, 15 | 22.133 | 9.744 | 2.375 | 2.759 | 0.458 |
| 30 uniform | 17, 17 | 19.871 | 7.450 | 2.412 | 0 | 0.927 |
| 30 signal | 22, 8 | 22.461 | 10.090 | 2.381 | 2.850 | 0.742 |
| 30 reverse | 8, 22 | 22.718 | 10.174 | 2.505 | 2.773 | 0.718 |

Process wall includes lock waiting, imports and verification. Lock waits were
3.36–5.90 microseconds. The child, load and phase intervals overlap by design;
signal and search include journal overhead. CUDA preflight was 0.217–0.229 s.
Forward axes each count 48 at point 20 and 68 at point 30. Metric counts are
24/20/20 and 34/30/30 respectively. Signal and reverse each have 2 Jacobians,
2 SVDs and 384 each of VJP directions, gradient API and backward hooks;
uniform has zero of these. These overlapping axes are not additive compute.
Model and device identity and each slot's budget caps were checked. Peak CUDA
allocated/reserved bytes were 106384896/111149056 for uniform and
141664768/144703488 for both signal strategies.

This is a known-root, post-prior-result exploratory rerun; it is neither
independent budget/performance confirmation nor a scientific gate pass. Equal
forward-axis caps do not establish equal total compute. Cache, full dynamic
source closure, interference, FLOPs and energy are not established. The lock
only serializes cooperating local processes. No physical experiment was run.
