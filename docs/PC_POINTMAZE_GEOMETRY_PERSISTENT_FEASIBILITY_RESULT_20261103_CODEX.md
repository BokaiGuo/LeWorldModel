# PointMaze geometry-selected persistent-dynamics feasibility

The four starts were determined solely by the existing static intersection
rule, before executing either controller. They are development contexts,
not certified new training-disjoint or independent roots. Each row below
uses persistent MuJoCo 3.1.6, zero initial velocity and the same 24-action
horizon and distance threshold defined in the accompanying protocol.

| Start | Direct: steps, terminal distance | Waypoint: steps, terminal distance |
| --- | --- | --- |
| (0.8, 0.8) | 22, 0.247238 | 15, 0.236436 |
| (0.8, 1.8) | 21, 0.222887 | 11, 0.187306 |
| (2.8, 0.8) | 22, 0.247238 | 15, 0.236436 |
| (2.8, 1.8) | 21, 0.222887 | 11, 0.187306 |

All eight trajectories reached position distance <= 0.25. The direct
controller uses a privileged one-step physical simulation and has five or
six non-decreasing distance transitions per trajectory; the waypoint
controller's recorded distance decreases at every action. This does not
show that non-progress was necessary or changed subsequent controllability.
The direct oracle's success also rules out treating obstacle intersection
alone as evidence that a sophisticated reconfiguration program is needed.
Neither controller is the RGB-only Planning Compiler.

Candidate simulation reinitializes MuJoCo data from position and velocity;
execution itself retains persistent data. The version is 3.1.6, not the
previously checked 2.1.2/2.2 pair. The contact count is the total number of
simulator contacts, not a classified wall-contact measurement. Results
are feasibility witnesses under this exact protocol, not native success,
fair planning performance, a backend-equivalence claim, or P1 admission.

The complete [receipt](../le-wm-main/results/pc_geometry_persistent_feasibility_20261103_codex.json)
includes actions, scores, states and provenance hashes. Two targeted tests
replayed every persistent action, checked all stop conditions, and checked
the frozen inputs. A stronger next diagnostic is to replay a common fixed
candidate bank and compare no-progress prefixes with zero and direct-greedy
under equal-budget suffix optimization; this remains separate from
selecting an RGB method or claiming that any such prefix exists.
