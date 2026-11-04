# v22 PointMaze prediction replay diagnostic

After both arms were sealed, replay all saved prediction batches independently
using the established explicit-packet model implementation and the saved RGB
context. Check the same frozen checkpoint, action normalizer and dependencies.
Run both arms, without modifying or choosing any action. Report maximum
position and score discrepancies against the original saved predictions using
the established limits of 2e-5 and 2e-6 respectively. Refuse overwrites and
retain a partial failure. This is implementation consistency, not an
independent checkpoint or physical prediction validation, and it does not
identify the cause of observed batch-dependent numerical score differences.
