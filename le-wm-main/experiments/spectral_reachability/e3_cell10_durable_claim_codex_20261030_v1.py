"""Cell10 cooperative one-shot marker primitive; this is not a production gate.

Old launchers do not call this module. An administrative cross-entrypoint and
cross-host fence, filesystem review, and a separately sealed launcher are
required before this marker can participate in a real production protocol.
"""
import json
import os
from pathlib import Path

DIRECTORY = Path("/home/gbk/code/LeWorldModel/le-wm-main/results")
MARKER = "e3_cell10_scope_codex_20261030_v1.claim"
AUTHORITY_SHA = "33064f0c5bae40aa06d662acd0673bf0127eb7faad1ff98e4247977ef84cf484"


def claim(closure_sha):
    """Create the fixed marker exactly once; retain it on every later failure.

    Only cooperating callers using this same directory benefit. O_EXCL and
    directory fsync semantics must be independently verified on the actual
    shared filesystem; this function makes no cross-host or fencing claim.
    """
    if (type(closure_sha) is not str or len(closure_sha) != 64
            or any(c not in "0123456789abcdef" for c in closure_sha)):
        raise ValueError("closure sha256 required")
    if not DIRECTORY.is_absolute() or DIRECTORY.resolve() != DIRECTORY:
        raise ValueError("canonical marker directory required")
    record = json.dumps({"schema": "e3_cell10_cooperative_claim_v1",
                         "cell": 10, "authority_sha256": AUTHORITY_SHA,
                         "closure_sha256": closure_sha},
                        sort_keys=True, separators=(",", ":")).encode() + b"\n"
    directory = os.open(DIRECTORY, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        try:
            view = memoryview(record)
            while view:
                n = os.write(fd, view)
                if n <= 0:
                    raise OSError("claim write made no progress")
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"claimed": True, "production_authorized": False, "global_fence": False}
