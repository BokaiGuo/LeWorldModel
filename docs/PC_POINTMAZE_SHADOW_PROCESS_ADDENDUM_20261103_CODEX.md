# Process-isolated shadow renderer addendum

Before v4: explicitly switching the two same-device contexts in v3 still
reproduced 0/1152 legacy images; the v2/v3 arrays are byte-identical. The
installed offscreen context delegates current-context management to a
device-indexed global implementation, so a separate process is now used for
shadow rendering. Main and shadow communicate only copied state through a
private socket; the reference forward remains in the main process and is never
rendered. The same panel and all frozen gates remain, including full-tick
historical main-state and legacy-frame checks. Preserve every earlier failure.
This is further debugging on the same data, not an independent validation.
