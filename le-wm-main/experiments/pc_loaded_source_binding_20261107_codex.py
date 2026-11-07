"""Opt-in drift detection for loaded local Python module source files.

This is an inventory of loaded modules, not a complete import closure or a
proof that executing code matches bytes on disk. Create it only after the
runner has imported its dependencies, and call verify at the end of the run.
"""
from pathlib import Path
import sys

from experiments.pc_input_permissions import file_hash
from experiments.pc_checkpoint_bound_rgb_20261107_codex import CheckpointBoundRGB


def loaded_sources(roots, modules=None):
    roots = tuple(Path(root).resolve(strict=True) for root in roots)
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError("source roots must be existing directories")
    modules = sys.modules.copy() if modules is None else dict(modules)
    sources = {}
    for name, module in sorted(modules.items()):
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        # Some dynamic extension namespaces (for example torch.classes)
        # expose a synthetic relative __file__, not a file on disk. Resolving
        # it against cwd would falsely place it inside the research tree.
        if not Path(origin).is_absolute():
            continue
        path = Path(origin).resolve()
        if not any(path.is_relative_to(root) for root in roots):
            continue
        if path.suffix != ".py":
            continue
        if not path.is_file():
            raise ValueError(f"loaded source missing: {name}: {path}")
        sources[name] = {"path": str(path), "sha256": file_hash(path)}
    return sources


class LoadedSourceBinding:
    def __init__(self, roots):
        self.roots = tuple(Path(root).resolve(strict=True) for root in roots)
        self.initial = loaded_sources(self.roots)
        if not self.initial:
            raise ValueError("no loaded Python sources within roots")

    def verify(self):
        current = loaded_sources(self.roots)
        if current != self.initial:
            before, after = set(self.initial), set(current)
            changed = sorted(name for name in before & after
                             if self.initial[name] != current[name])
            raise ValueError(
                f"loaded source inventory changed: added={sorted(after - before)}, "
                f"removed={sorted(before - after)}, changed={changed}"
            )
        return {
            "schema": "pc.loaded_source_binding.v1",
            "roots": [str(root) for root in self.roots],
            "modules": self.initial,
            "scope": "loaded local .py disk bytes at start and finish only",
        }


class SourceBoundCheckpointRGB(CheckpointBoundRGB):
    """Combine checkpoint checks with an opt-in loaded-source inventory."""

    def __init__(self, model, audit, weights, config, source_roots):
        super().__init__(model, audit, weights, config)
        self._source_binding = LoadedSourceBinding(source_roots)

    def finish(self):
        source_receipt = self._source_binding.verify()
        checkpoint_receipt = super().finish()
        source_receipt_after = self._source_binding.verify()
        if source_receipt != source_receipt_after:
            raise ValueError("source inventory changed during finish")
        return {"schema": "pc.source_checkpoint_bound_rgb.v1",
                "checkpoint": checkpoint_receipt, "loaded_sources": source_receipt}
