# CP12 prefix replay gate: blocked

The current RGB bridge could not reproduce the saved fixed-prefix RGB for
any of four sampled branches (`raw`, candidate 2, one per root). In the
prewritten preflight each branch used the stored reset and first 12 stored
raw actions, stepped one command at a time:

| Root | Exact frames 0–12 | Different RGB channels per frame, min–max | Largest channel difference | Largest causal velocity-vector difference at 2/7/12 |
| ---: | ---: | ---: | ---: | ---: |
| 8343 | 0/13 | 3469–3509 | 47 | 0.1303 |
| 8345 | 0/13 | 3407–3473 | 34 | 0.1213 |
| 8349 | 0/13 | 3465–3505 | 53 | 0.0848 |
| 8351 | 0/13 | 3417–3504 | 45 | 0.2083 |

Velocity differences compare two estimates from adjacent RGB frames with
dt=0.05, not simulator qvel or a measurement of physical velocity error.
Pixels differ at frame 0 as well as after actions. Thus exact source binding
fails and the difference propagates to the planned proprio input; its cause
has not been determined. This does not establish a dynamics error, prove the
historical execution was incorrect, or prove that a separately defined
comparison using consistently regenerated RGB is impossible.

Under the preflight's fail-closed rule, no CEM, model prediction, suffix
action, or paired planning result was produced. Do not combine historical
saved RGB with current physical continuation as if their feedback were
equivalent. A possible next investigation is to identify and validate the
original rendering environment or separately predeclare a fully regenerated
real-RGB comparison, including its changed source contract; neither has
been validated here.

The sealed preflight performed 4 resets and 48 prefix commands; two earlier
interactive probes each also replayed one 12-command prefix. None selected
or executed a suffix. Receipt SHA-256
`e076dfb81265614a3c110d288b057e859e781129440cd551ba3728f4d0507b08`.
The focused CPU tests report **3 passed**; they validate gate arithmetic and
receipt binding, not an independent physical replay. No old artifact was
modified. The requested planning protocol and implementation are deliberately
not claimed as completed because this input interface gate failed first.
