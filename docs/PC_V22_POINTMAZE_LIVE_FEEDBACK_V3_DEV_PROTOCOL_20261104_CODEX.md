# v22 PointMaze v3 engineering entry

This entry preserves the v22/CEM planning and fixed 15/10/5-step contract of
the v2 development diagnostic but adds an exclusive per-action journal:
persist and sync each action attempt before dispatch and each returned image
immediately after dispatch. A pending attempt without a returned record is
ambiguous and must never be automatically replayed.

This file specifies an engineering interface, not authorization to reopen the
completed v2 panel. A future experiment must specify its own root, sources,
stopping policy and expected output namespace before use. The default
development setup and exposed goal do not establish source independence.
Normal completion still creates the conventional decision, receipt and image
artifacts. A process or machine crash can leave partial files and directories;
no guarantee of directory-entry durability under power failure is claimed.
