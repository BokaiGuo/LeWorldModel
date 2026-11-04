# Post-detection backend/XML factorial diagnostic

The earlier fixed-action comparison found a discrepancy but changed both
engine and XML. Replay the same eight fixed action trajectories in all four
combinations: MuJoCo 2.1.2 and 3.1.6, each using the official executor XML
and the legacy diagnostic XML. Use the same persistent worker and action
clock as the stepwise comparison. No actions, inputs, threshold or roots
change. Compare every step's position, velocity, clock and particle-wall
contact indicators, retaining all four full traces.

This separates these specific engine and XML changes; it cannot identify a
particular contact-solver implementation change without further evidence.
Neither a close nor a divergent result justifies changing the formal
executor or updating historical results.
