"""Small compatibility shim for dm-control named MuJoCo fields.

Some dm-control releases ship a static list of MuJoCo fields that is newer or
older than the installed Python MuJoCo binding.  The failure occurs while
constructing ``physics.named`` and does not affect fields that actually exist
on the model.  We remove only absent fields for the duration of the original
indexer call and restore the global table afterwards.

This is an environment-runtime compatibility repair, not a learned-model
repair.  Callers must record that the shim was active in their artifact.
"""

from __future__ import annotations

from typing import Any


def install_missing_field_compat() -> dict[str, Any]:
    """Install an idempotent, fail-safe shim for dm-control's indexer."""
    try:
        from dm_control.mujoco import index
    except ImportError:
        return {"available": False, "installed": False, "missing_fields": []}

    if getattr(index, "_lewm_missing_field_compat", False):
        return {
            "available": True,
            "installed": True,
            "missing_fields": list(getattr(index, "_lewm_missing_fields", ())),
        }

    original = index.struct_indexer
    observed_missing: set[str] = set()

    def compatible_struct_indexer(
        struct: Any, struct_name: str, size_to_axis_indexer: dict[str, Any]
    ) -> Any:
        normalized = struct_name.lower()
        fields = index.sizes.array_sizes.get(normalized)
        if fields is None:
            return original(struct, struct_name, size_to_axis_indexer)
        missing = tuple(name for name in fields if not hasattr(struct, name))
        if not missing:
            return original(struct, struct_name, size_to_axis_indexer)

        observed_missing.update(f"{normalized}.{name}" for name in missing)
        original_fields = fields
        index.sizes.array_sizes[normalized] = {
            name: spec for name, spec in original_fields.items() if hasattr(struct, name)
        }
        try:
            return original(struct, struct_name, size_to_axis_indexer)
        finally:
            index.sizes.array_sizes[normalized] = original_fields

    index.struct_indexer = compatible_struct_indexer
    index._lewm_missing_field_compat = True
    index._lewm_missing_fields = observed_missing
    return {"available": True, "installed": True, "missing_fields": []}


def compatibility_status() -> dict[str, Any]:
    """Return the current shim state after environment construction."""
    try:
        from dm_control.mujoco import index
    except ImportError:
        return {"available": False, "installed": False, "missing_fields": []}
    return {
        "available": True,
        "installed": bool(getattr(index, "_lewm_missing_field_compat", False)),
        "missing_fields": sorted(getattr(index, "_lewm_missing_fields", ())),
    }
