"""CPU checkpoint/reference audit and non-dispatching runtime identity guard.

The reference is loaded without inference. A matching hash is local evidence,
not remote attestation or proof of the full executable dependency closure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import codex_online_model_action_20261030 as online
from experiments.spectral_reachability import codex_sra_online_run_bank_20261030 as sra

HERE = Path(__file__).resolve()
LOADER = HERE.parents[1] / "checkpoint_compat.py"
LOADER_SHA = "50b01e20864dc929943df7cb3566c271a771e69c06ea0fdbc9b598c59e6e9dff"
WEIGHTS_SHA = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
CONFIG_SHA = "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09"


def need(value, message):
    if not value:
        raise ValueError(message)


def tensor_record(tensor):
    import torch
    need(isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
         and not tensor.is_quantized, "unsupported checkpoint tensor")
    data = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "sha256": hashlib.sha256(data).hexdigest()}


def tensor_map(values):
    return {name: tensor_record(value) for name, value in sorted(values)}


def model_snapshot(model):
    """Hash parameters, persistent/nonpersistent buffers, structure and modes."""
    import torch
    need(isinstance(model, torch.nn.Module), "torch module required")
    modules = list(model.named_modules(remove_duplicate=False))
    parameters = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers(remove_duplicate=False))
    need(parameters and all(not p.requires_grad for _, p in parameters)
         and all(not module.training for _, module in modules), "model must be frozen eval")
    need(all(p.device.type != "meta" for _, p in parameters + buffers), "meta tensor")
    structure = [{"name": name, "class": type(module).__module__ + "." + type(module).__qualname__}
                 for name, module in modules]
    return {"parameters": tensor_map(parameters), "buffers": tensor_map(buffers),
            "persistent_state": tensor_map(model.state_dict().items()), "modules": structure,
            "interpolate_pos_encoding": getattr(model, "interpolate_pos_encoding", None)}


def assets(policy, weights_sha, config_sha):
    policy = Path(policy).resolve()
    weight, config = policy / "weights.pt", policy / "config.json"
    need(sra.file_sha(weight) == weights_sha, "checkpoint weights drift")
    need(sra.file_sha(config) == config_sha, "checkpoint config drift")
    need(sra.file_sha(LOADER) == LOADER_SHA, "checkpoint loader drift")
    return weight, config


def checkpoint_proof(policy, weights_sha=WEIGHTS_SHA, config_sha=CONFIG_SHA, *, fixture_loader=None):
    """Compare safe-loaded, remapped checkpoint with an independently loaded CPU reference."""
    import torch
    weight, config = assets(policy, weights_sha, config_sha)
    from experiments.checkpoint_compat import load_compatible_pretrained, remap_legacy_vit_keys
    original = torch.load(weight, map_location="cpu", weights_only=True)
    need(isinstance(original, dict) and original, "checkpoint tensor dictionary required")
    remapped, changed = remap_legacy_vit_keys(original)
    ref = (fixture_loader if fixture_loader is not None else load_compatible_pretrained)(str(weight.parent), device="cpu")
    snapshot = model_snapshot(ref)
    need(all(p.device.type == "cpu" for p in list(ref.parameters()) + list(ref.buffers())),
         "reference must be CPU only")
    checkpoint = tensor_map(remapped.items())
    need(checkpoint == snapshot["persistent_state"], "checkpoint and loaded reference state differ")
    assets(policy, weights_sha, config_sha)
    return {"schema": "codex_sra_checkpoint_identity_v1",
            "checker_sha256": sra.file_sha(HERE),
            "checkpoint": {"weights_path": str(weight), "weights_sha256": weights_sha,
                           "config_path": str(config), "config_sha256": config_sha,
                           "loader_path": str(LOADER), "loader_sha256": LOADER_SHA},
            "checkpoint_tensor_count": len(checkpoint), "remapped_keys": changed,
            "reference": snapshot, "fixture": fixture_loader is not None,
            "boundaries": {"model_inference": False, "gpu_used": False, "physical_used": False,
                           "independent_execution_attestation": False, "full_dependency_closure": False,
                           "equal_budget": False}}


def save_proof(path, proof):
    return sra._save_exclusive(path, proof)


def verify_proof(path, expected_sha, model, replay, identity, *, allow_fixture=False):
    """Validate the actual model object against the saved CPU reference and assets."""
    need(sra.file_sha(path) == expected_sha, "identity proof hash mismatch")
    proof = sra.read_json(path)
    need(proof["schema"] == "codex_sra_checkpoint_identity_v1" and type(proof["fixture"]) is bool and
         (proof["fixture"] is False or allow_fixture is True) and proof["checker_sha256"] == sra.file_sha(HERE),
         "fixture, schema or checker source forbidden")
    cp = proof["checkpoint"]
    if not proof["fixture"]:
        need(cp["weights_sha256"] == WEIGHTS_SHA and cp["config_sha256"] == CONFIG_SHA,
             "not the fixed checkpoint")
    need(cp["loader_path"] == str(LOADER) and cp["loader_sha256"] == LOADER_SHA,
         "loader identity mismatch")
    need(Path(cp["weights_path"]).parent == Path(cp["config_path"]).parent and
         Path(cp["weights_path"]).name == "weights.pt" and Path(cp["config_path"]).name == "config.json",
         "checkpoint asset paths")
    assets(Path(cp["weights_path"]).parent, cp["weights_sha256"], cp["config_sha256"])
    need(identity["model_sha256"] == cp["weights_sha256"], "journal checkpoint identity mismatch")
    need(getattr(replay, "model", None) is model, "replay model instance mismatch")
    snapshot = model_snapshot(model)
    import torch
    from experiments.checkpoint_compat import remap_legacy_vit_keys
    disk = torch.load(cp["weights_path"], map_location="cpu", weights_only=True)
    remapped, changed = remap_legacy_vit_keys(disk)
    need(changed == proof["remapped_keys"] and len(remapped) == proof["checkpoint_tensor_count"]
         and tensor_map(remapped.items()) == snapshot["persistent_state"],
         "in-memory state does not match checkpoint")
    need(snapshot == proof["reference"], "in-memory model or nonpersistent buffer drift")
    assets(Path(cp["weights_path"]).parent, cp["weights_sha256"], cp["config_sha256"])
    return proof


def guarded_callbacks(*, model, replay, identity, proof_path, proof_sha256, encoder, planner,
                      manifest_path, manifest_sha256, allow_fixture=False):
    """Bind proof and assets in the method manifest; check before/after each callback.

    This guard cannot attest arbitrary callback internals. The caller must use
    the native SRA encoder/planner and keep the model exclusively owned.
    """
    manifest = sra.checked_manifest(manifest_path, manifest_sha256)
    proof = verify_proof(proof_path, proof_sha256, model, replay, identity, allow_fixture=allow_fixture)
    refs = {"checkpoint_identity_proof": (Path(proof_path), proof_sha256),
            "checkpoint_loader": (LOADER, LOADER_SHA),
            "checkpoint_weights": (Path(proof["checkpoint"]["weights_path"]), proof["checkpoint"]["weights_sha256"]),
            "checkpoint_config": (Path(proof["checkpoint"]["config_path"]), proof["checkpoint"]["config_sha256"]),
            "checkpoint_guard": (HERE, sra.file_sha(HERE))}
    for name, (path, digest) in refs.items():
        entry = manifest["files"].get(name)
        need(entry is not None and Path(entry["path"]).resolve() == path.resolve()
             and entry["sha256"] == digest, "manifest missing checkpoint binding: " + name)

    def check():
        sra.checked_manifest(manifest_path, manifest_sha256)
        verify_proof(proof_path, proof_sha256, model, replay, identity, allow_fixture=allow_fixture)

    def checked_encoder(frames, history):
        check()
        value = encoder(frames, history)
        check()
        return value

    def checked_planner(request, context, meter):
        need(request["identity"] == identity and meter.decision_id == online.sha(online.encode(
            {"request": request, "method_sha256": manifest_sha256})), "guarded request binding")
        check()
        value = planner(request, context, meter)
        check()
        return value
    return checked_encoder, checked_planner


def guarded_online_select(*, journal, arm, frames, sources, method_source, model, replay,
                          proof_path, proof_sha256, encoder, planner, receipt_path,
                          allow_fixture=False):
    """Opt-in guarded journal entry; callbacks must themselves use the supplied model."""
    need(arm == "sra_no_composition", "SRA arm required")
    identity = journal.protocol.identity
    checked_encoder, checked_planner = guarded_callbacks(
        model=model, replay=replay, identity=identity, proof_path=proof_path, proof_sha256=proof_sha256,
        encoder=encoder, planner=planner, manifest_path=method_source["path"],
        manifest_sha256=method_source["sha256"], allow_fixture=allow_fixture)
    return online.online_select(journal, arm, frames, sources, method_source,
                                checked_encoder, checked_planner, receipt_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path)
    args = parser.parse_args()
    proof = checkpoint_proof(args.policy)
    print("CPU checkpoint/reference audit only; no inference, GPU or physics.")
    proof_sha = save_proof(args.output, proof)
    print("saved proof SHA256:", proof_sha)
    if args.verification_output is not None:
        from types import SimpleNamespace
        from experiments.checkpoint_compat import load_compatible_pretrained
        model = load_compatible_pretrained(str(args.policy), device="cpu")
        verify_proof(args.output, proof_sha, model, SimpleNamespace(model=model),
                     {"model_sha256": WEIGHTS_SHA})
        receipt = {"schema": "codex_sra_checkpoint_cpu_recheck_v1", "proof_path": str(args.output.resolve()),
                   "proof_sha256": proof_sha, "checker_sha256": sra.file_sha(HERE),
                   "second_cpu_instance_matched": True, "model_inference": False,
                   "gpu_used": False, "physical_used": False, "independent_execution_attestation": False}
        print("saved CPU recheck SHA256:", save_proof(args.verification_output, receipt))


if __name__ == "__main__":
    main()
