# Shadow render context switch addendum

Before v3: the v2 host attempt passed all physical tick, state-copy and
geometry gates but matched zero sealed frames. In the installed `mujoco_py`
source, offscreen `render` updates the scene and calls `mjr_render` without
making that simulator's OpenGL context current. Each context has an exposed
`opengl_context.make_context_current()` method. V3 switches to the appropriate
context explicitly immediately before each raw and shadow render. Everything
else, including all predeclared panel and pass/fail gates, is unchanged.
This is a debugging rerun on the same panel, not independent confirmation.
Retain the prior worker, receipt, full arrays and negative result unchanged.
