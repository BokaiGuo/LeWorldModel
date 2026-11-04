# PointMaze prior-frame clock check

For exposed roots 8343, 8345, 8349 and 8351, the offline position backend was
given the saved `states[root,39]` reset and `actions[root,39]` as one raw
archive-level action. Its resulting four-component states did not reproduce
the saved `states[root,40]`: maximum absolute position discrepancies per
root were approximately 0.03859, 0.07609, 0.03451 and 0.02645; maximum
velocity discrepancies were approximately 0.69623, 0.57042, 0.76163 and
0.74338 respectively.

This tests only that particular reset/action/index and stepping hypothesis.
It does not establish that the archive is incorrect; indexing, action
semantics, hidden simulation state or time scale may differ. In particular,
the saved prior frame must not be silently treated as a certified causal RGB
history for the frame-40 reset. A deployable alternative is to acquire two
successive RGB observations through the actual executor before planning,
then charge the observation/action opportunity equally to every comparison.
