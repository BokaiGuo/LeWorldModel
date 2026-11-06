"""Run the local native evaluator with the existing dm-control compatibility shim.

This is an explicitly non-official development entry point. Arguments following
the script name are passed unchanged to the native Hydra evaluator.
"""
from pathlib import Path
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MUJOCO_GL"] = "egl"

from experiments.dmcontrol_compat import compatibility_status, install_missing_field_compat

print("Development-only dm-control compatibility:", install_missing_field_compat(), flush=True)
try:
    runpy.run_path(str(ROOT / "eval.py"), run_name="__main__")
finally:
    print("Final dm-control compatibility:", compatibility_status(), flush=True)
