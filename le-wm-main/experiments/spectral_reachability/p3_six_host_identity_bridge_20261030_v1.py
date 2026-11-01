"""Live identity guards around the existing producer, not a production driver.

No model construction, device probing or physical execution happens on import.
The caller owns admission, precision contexts, durable outer records and authority.
"""
from pathlib import Path

from experiments.spectral_reachability import p3_six_bound_producer_20261030_v1 as producer
from experiments.spectral_reachability import p3_matched_order_feedback_host_v2_codex_20261029 as host

need = producer.need
NAME = "p3_six_host_identity_bridge_20261030_v1"


def own_sha():
    path = Path(__file__).resolve()
    need(path.is_file() and not path.is_symlink(), "identity bridge source")
    return producer.terminal.file_sha(path)


def snapshot(model, torch, device, prepared, contract):
    """Inspect an already initialized model/device; this is not hardware attestation."""
    need(getattr(device, "type", None) == "cuda" and type(getattr(device, "index", None)) is int
         and device.index >= 0, "explicit CUDA device index")
    modules = list(model.modules())
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    need(modules and tensors and all(not m.training for m in modules), "model eval state")
    need(all(not p.requires_grad for p in model.parameters()), "model frozen")
    need(all(t.device == device for _, t in tensors), "model tensor device")
    props = torch.cuda.get_device_properties(device.index)
    uuid = getattr(props, "uuid", None)
    need(uuid is not None and str(uuid).strip() and str(uuid).lower() != "none", "device UUID unavailable")
    weights = host.source.host.signal.model_hash(model)
    need(weights["original_state_dict"] == contract["expected_model_hash"], "model weight hash")
    sources = host.prior.model_sources(model, prepared)
    precision = host.prior.checked_precision(torch)
    need(precision == contract["precision"], "precision source contract")
    return dict(model_hash=weights, model_sources=sources, precision=precision,
                device=dict(index=device.index, uuid=str(uuid), name=str(props.name)),
                tensor_devices=[dict(name=n, device=str(t.device)) for n, t in tensors])


def execute(store, factory, backend_factory, *, source_sha256, producer_sha256,
            bridge_sha256, permit_ram=False):
    """Wrap producer with live checks; return an unsealed in-memory receipt.

    This entry point deliberately does not grant production authorization. The
    backend factory's use of the guarded model is not independently verified.
    """
    need(own_sha() == bridge_sha256, "identity bridge source pin")
    need(type(factory) is producer.native_factory.Factory and factory.store is store,
         "native factory identity")
    contract, prepared, _ = producer.strict.source_material()
    need(producer.strict.core.digest(contract) == source_sha256, "strict source contract pin")
    need(producer.own_sha() == producer_sha256, "producer source pin")
    model, torch, device = factory.model, factory.torch, factory.device
    hooks = host.hook_snapshot(model)
    before = snapshot(model, torch, device, prepared, contract)
    result = None
    try:
        result = producer.execute(store, factory, backend_factory, source_sha256=source_sha256,
                                  producer_sha256=producer_sha256, permit_ram=permit_ram)
    finally:
        # If producer raised, Python retains its exception as context if this
        # final verification also fails; no successful receipt is returned.
        need(host.hooks_restored(hooks), "model hooks not restored")
        after = snapshot(model, torch, device, prepared, contract)
        need(before == after, "model/device identity drift")
        final_contract, _, _ = producer.strict.source_material()
        need(final_contract == contract and own_sha() == bridge_sha256 and
             producer.own_sha() == producer_sha256, "identity source drift")
    return dict(schema=NAME + ".receipt", before=before, after=after,
                bridge_sha256=bridge_sha256, source_contract_sha256=source_sha256,
                producer_sha256=producer_sha256, terminal=result["terminal"],
                identity_is_attestation=False, full_saved_strict=False,
                physical_certified=False, scientific_pass=False)
