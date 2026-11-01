# P2 H25 native replay (exploratory)

Fixed base order 0–3, joint then library_chain; each action sequence is the
original fifteen-step prefix followed by the ten saved CUDA winner steps.
No padding, retry, reselection, training, or scientific admission. The actual
cp15 native/RGB prefix must match before any suffix action. Native terminal
or truncation stops execution; missing observations remain missing.

Per two-arm job caps: 70 public steps, 708 physics steps, 222 explicit renders,
two constructors and closes, four resets, eight state placements, six goal
setters, four goal renders and two goal binds. Wall cap is 900 seconds.
Saved CUDA costs and physical costs are distinct; equal total computation is
not asserted. Known roots and duplicated contexts are not independent samples.
