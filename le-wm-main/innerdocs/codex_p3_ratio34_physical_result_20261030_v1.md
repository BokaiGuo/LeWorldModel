# Ratio34 downstream physical replay: known-root exploration

Both fixed six-branch point panels completed and passed the existing saved
consumer, with no new neural calls. Every branch returned 10/10 selected
actions; native terminated and truncated were false, so no planned step was
censored. Native success was false in all 12 branches.

Final object-position and wrapped-angle errors by point/root/arm (position
units are the native task coordinates; angle is radians):

| point | root | uniform | signal | reverse |
|---|---|---|---|---|
| 20 | 0 | 71.5224, 0.4998 | 65.0958, 0.4406 | 70.7867, 0.4759 |
| 30 | 0 | 70.7867, 0.4759 | 67.3121, 0.6945 | 70.7867, 0.4759 |
| 20 or 30 | 1 | 76.4027, 0.0822 | 76.4027, 0.0822 | 76.4027, 0.0822 |

For each point, attempted/completed physical counts are construct 6/6, close
6/6, reset 12/12, set-state 24/24, goal-render 12/12, goal-bind 6/6,
public-step 120/120, physics-step 1224/1224, explicit-render 132/132. These
are separately scoped axes; public steps include warmup and physics steps
are nested. Recorded physical execution wall is 6.105 and 6.474 seconds for
points 20 and 30 respectively; source reconstruction is outside this wall.

Model resource counts and caps remain in the prepare plans and producer batch.
At point 20, each forward axis used 48/48, metric uniform 24/24 and signal
and reverse 20/24. At point 30 these are 68/68, 34/34 and 30/34. Jacobian
uniform 0/2, each signal arm 2/2; VJP, grad API and raw backward hook uniform
0/384, each signal arm 384/384; SVD uniform 0/2 and signal arms 2/2; scheduler
all 1/1. No different axes are added.

Physical report SHA-256 P20
`c7d9c6f26e1d5cdff168314a20861bb4536b2fefb9c9b624112495a5b3c6a160`,
P30 `7036ea5f78de9eee2111bbdab448b608a577c412eddbc38f118dd0ebe9136867`.
The physical result is after observed model outcomes on two known roots; it
is not independent confirmation, equal-total-compute evidence, a P3 advantage,
or a gate pass. Saved validation is not independent physical re-execution.
