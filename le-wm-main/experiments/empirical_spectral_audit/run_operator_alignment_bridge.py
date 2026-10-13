"""CLI wrapper for the empirical operator-alignment bridge summary."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.operator_alignment_bridge import main


if __name__ == "__main__":
    main()
