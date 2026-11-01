"""Owned dense macro eval extension. No pickle, foreign classes, or CPU neural fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import agent_hilewm_trusted_loader_20261028_v1 as parent
from experiments.spectral_reachability import agent_hilewm_runtime_attributes_20261028_v1 as literal

ROOT, need = parent.ROOT, parent.need
TAG = "agent_hilewm_trusted_loader_20261028_v2"
ATTR_PATH = ROOT/"results/agent_hilewm_runtime_attributes_20261028_v1_MAIN01.json"
ATTR_SHA = "5415f2a2e97e9bdeb282200e4420b9270c607190bf15a1a482963755b850b3c8"
PINS = {
    "experiments/spectral_reachability/agent_hilewm_trusted_loader_20261028_v1.py": "df313393f1ead36a512dc549f8883a1da85dd3e7ffa057f690d39627a1e7b411",
    "tests/test_agent_hilewm_trusted_loader_20261028_v1.py": "5f217dff654d1609025f87a608f11814ba1be6057b350b9a36c8255ec57bfda7",
    "innerdocs/agent_hilewm_trusted_loader_20261028_v1.md": "bee0f03bb7813bb35f03ce71fe2ae2f64b64575fb5ae0d365cc0b01455002662",
    "experiments/spectral_reachability/agent_hilewm_runtime_attributes_20261028_v1.py": "670d46a6e6b459e07d487007c965db1d1b5f95b83f46434d8be51a0937db5cde",
    "tests/test_agent_hilewm_runtime_attributes_20261028_v1.py": "1c59cd9162adcf4c5ed848d74a36b130868dbab1281ccee4ed4e3b6fce2e5357",
    "innerdocs/agent_hilewm_runtime_attributes_20261028_v1.md": "213e06e5c884d292cd078920c2430ca36578957846f1e265cff6f460c87a9a1a",
}


def closure():
    for name, expected in PINS.items():
        need(parent.sha(parent.bytes_read(ROOT/name, parent.JSON_CAP)) == expected, "frozen dependency SHA: " + name)
    paths = (Path(__file__), ROOT/"tests"/("test_"+TAG+".py"), ROOT/"innerdocs"/(TAG+".md"))
    return {str(p): parent.sha(parent.bytes_read(p, parent.JSON_CAP)) for p in paths}


def macro_contract(saved, graph, extracted):
    """Literal values and every used shape must close; no constructor defaults."""
    need(extracted["ir_sha256"] == parent.IR_SHA and extracted["defaults_applied"] is False,
         "literal extraction authority")
    need(extracted["pickle_executed"] is False and extracted["foreign_code_executed"] is False,
         "inert extraction only")
    modules = {tuple(r["path"]): r for r in graph["modules"]}
    rows = {tuple(r["path"]): r for r in extracted["modules"]}
    need(len(rows) == len(extracted["modules"]) == len(modules), "literal module denominator")
    entries = {tuple(r["module_path"]+[r["name"]]): r for r in saved["nn"]["entries"]}
    evidence = []
    prefix = ("latent_action_encoder",)

    def attr(path, key, expected):
        p = prefix + tuple(path)
        m, r = modules[p], rows[p]
        need(m["node"] == r["node"] and m["class_declaration"] == r["class_declaration"], "literal node/class join")
        field = r["attributes"][key]
        decoded = field["decoded"]
        value = literal.literal_value(decoded)
        # Compare both value and exact builtin type (bool must not become int).
        need(type(value) is type(expected) and value == expected, "unsupported saved macro field: " + ".".join(p+(key,)))
        need(m["attribute_nodes"][key] == field["value_node"] == decoded["origin"]["node"], "literal attribute origin join")
        evidence.append(dict(path=list(p), field=key, origin=decoded["origin"]))
        return value

    def tensor(path, shape):
        p = prefix + tuple(path)
        row = entries[p]
        need(row["metadata"] is not None and row["metadata"]["shape"] == shape
             and row["metadata"]["dtype_declaration"] == "float32", "macro parameter dtype/shape: " + ".".join(p))

    attr((), "max_seq_len", 15)
    for name, inp, out in (("input_proj", 10, 192), ("output_proj", 192, 32)):
        attr((name,), "in_features", inp); attr((name,), "out_features", out)
        tensor((name, "weight"), [out, inp]); tensor((name, "bias"), [out])
    tensor(("cls_token",), [1, 1, 192]); tensor(("pos_embedding",), [1, 16, 192])
    attr(("encoder",), "num_layers", 2)
    attr(("encoder",), "norm", None)
    layer_paths = {p for p in modules if p[:3] == prefix+("encoder", "layers") and len(p) == 4}
    need(layer_paths == {prefix+("encoder", "layers", str(i)) for i in range(2)}, "exact macro layer paths")
    eps = []
    for i in range(2):
        layer = ("encoder", "layers", str(i))
        attr(layer, "norm_first", False)
        attr(layer, "activation_relu_or_gelu", 2)
        decoded = rows[prefix+layer]["attributes"]["activation"]["decoded"]
        need(decoded["kind"] == "global" and decoded["status"] == "opaque"
             and decoded["symbol"] == ["torch._C._nn", "gelu"], "saved GELU symbol (not executed)")
        for key, value in (("embed_dim",192), ("kdim",192), ("vdim",192), ("num_heads",4),
                           ("head_dim",48), ("batch_first",True), ("add_zero_attn",False),
                           ("bias_k",None), ("bias_v",None), ("dropout",0.1), ("_qkv_same_embed_dim",True)):
            attr(layer+("self_attn",), key, value)
        tensor(layer+("self_attn","in_proj_weight"), [576,192])
        tensor(layer+("self_attn","in_proj_bias"), [576])
        tensor(layer+("self_attn","out_proj","weight"), [192,192])
        tensor(layer+("self_attn","out_proj","bias"), [192])
        for name, inp, out in (("linear1",192,768), ("linear2",768,192)):
            attr(layer+(name,), "in_features", inp); attr(layer+(name,), "out_features", out)
            tensor(layer+(name,"weight"), [out,inp]); tensor(layer+(name,"bias"), [out])
        for name in ("dropout", "dropout1", "dropout2"):
            attr(layer+(name,), "p", 0.1); attr(layer+(name,), "inplace", False)
        layer_eps = []
        for name in ("norm1", "norm2"):
            attr(layer+(name,), "normalized_shape", (192,))
            attr(layer+(name,), "elementwise_affine", True)
            layer_eps.append(attr(layer+(name,), "eps", 1e-5))
            tensor(layer+(name,"weight"), [192]); tensor(layer+(name,"bias"), [192])
        eps.append(layer_eps)
    derived = literal.constructor_values(extracted, "latent_action",
        ["input_dim","model_dim","latent_dim","max_seq_len","num_layers","num_heads","mlp_dim","dropout"])
    need(derived == dict(input_dim=10,model_dim=192,latent_dim=32,max_seq_len=15,
                         num_layers=2,num_heads=4,mlp_dim=768,dropout=0.1), "derived/literal config join")
    return dict(**derived, norm_eps=eps, evidence=evidence, dense_unmasked_only=True,
                original_class_equivalence=False, source_transplant_forward_parity=False)


def audit():
    own = closure()
    saved, graph, facts = parent.audit()
    # Bohr's builtin-only extractor, not a second pickle/config parser.
    sources = literal.read_sources()
    expected = literal.extract(saved, sources)
    expected["own_sha256"] = literal.own_closure()
    expected["source_closure"] = [s["pointer"] for s in sources]
    extracted = parent.json_read(ATTR_PATH, ATTR_SHA)
    need(extracted == expected, "actual attribute report must exactly match builtin-only recomputation")
    config = macro_contract(saved, graph, extracted)
    need(closure() == own, "source closure changed during audit")
    return dict(schema=TAG+".audit", macro=config, parent_facts=facts,
                own_sha256=own, dependency_sha256=PINS, ir_sha256=parent.IR_SHA,
                attributes_report_sha256=ATTR_SHA,
                baseline_runnable=False, scientific_admission=False)


def macro_kernel(torch, node, chunks, config):
    """Dense unmasked eval equations. Called only after facade CUDA/freeze gates.

    Test injection is symbolic shape tracing, not a second production backend.
    No original module.forward or class import is used.
    """
    f = torch.nn.functional
    x = f.linear(chunks, node.input_proj.weight, node.input_proj.bias)
    x = torch.cat((node.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + node.pos_embedding[:, :x.shape[1]]
    for i in range(config["num_layers"]):
        layer = node.encoder.layers._modules[str(i)]
        attn = layer.self_attn
        q = x.transpose(0, 1)
        y, _ = f.multi_head_attention_forward(q, q, q, config["model_dim"], config["num_heads"],
            attn.in_proj_weight, attn.in_proj_bias, None, None, False, 0.0,
            attn.out_proj.weight, attn.out_proj.bias, training=False, key_padding_mask=None,
            need_weights=False, attn_mask=None, use_separate_proj_weight=False, is_causal=False)
        x = f.layer_norm(x + y.transpose(0, 1), (config["model_dim"],),
                         layer.norm1.weight, layer.norm1.bias, config["norm_eps"][i][0])
        y = f.linear(f.gelu(f.linear(x, layer.linear1.weight, layer.linear1.bias), approximate="none"),
                     layer.linear2.weight, layer.linear2.bias)
        x = f.layer_norm(x + y, (config["model_dim"],), layer.norm2.weight, layer.norm2.bias,
                         config["norm_eps"][i][1])
    return f.linear(x[:, 0], node.output_proj.weight, node.output_proj.bias)


class OwnedDenseMacro:
    """Explicit new integration API; never monkeypatch the frozen graph/classes."""
    def __init__(self, graph, config, torch):
        self.graph, self.config, self.torch = graph, config, torch
        self.attempts = self.returns = 0

    def encode_macro(self, chunks, *, action_mask=None):
        need(action_mask is None, "masked/nested macro semantics not certified; no silent mask discard")
        t, c = self.torch, self.config
        need(t.is_tensor(chunks) and chunks.device.type == "cuda" and chunks.dtype == t.float32,
             "HOST CUDA float32 grouped-action inputs only")
        need(chunks.ndim == 3 and 1 <= chunks.shape[0] <= 256 and
             1 <= chunks.shape[1] <= c["max_seq_len"] and chunks.shape[2] == c["input_dim"],
             "explicit grouped (B,T,10); raw two-dimensional actions not accepted")
        need(all(not p.requires_grad and p.device == chunks.device for p in self.graph.parameters())
             and all(not m.training for m in self.graph.modules()), "frozen same-device eval graph required")
        need(bool(t.isfinite(chunks).all()), "finite macro inputs required")
        self.attempts += 1  # backend failures are not refunded
        with t.inference_mode():
            output = macro_kernel(t, self.graph.latent_action_encoder, chunks, c)
            t.cuda.synchronize()
            need(output.shape == (chunks.shape[0], c["latent_dim"]) and bool(t.isfinite(output).all()),
                 "complete finite macro return required")
        self.returns += 1
        return output


def construct_cuda(report, report_sha, receipt, receipt_sha, *, main_authorized=False):
    need(main_authorized is True, "MAIN authorization required")
    metadata = audit()
    graph, result = parent.construct_cuda(report, report_sha, receipt, receipt_sha, main_authorized=True)
    import torch
    facade = OwnedDenseMacro(graph, metadata["macro"], torch)
    need(closure() == metadata["own_sha256"], "source changed during construction")
    result = dict(result, schema=TAG+".construction", parent_construction=result,
                  own_sha256=metadata["own_sha256"], dependency_sha256=PINS,
                  macro_contract=metadata["macro"], macro_api="OwnedDenseMacro.encode_macro(B,T,10); dense unmasked only",
                  macro_forward_attempts=0, macro_forward_returns=0,
                  macro_kernel_parity_verified=False, pixel_encoding_allowed=False,
                  baseline_runnable=False, scientific_admission=False)
    return facade, result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("audit", "construct"))
    p.add_argument("--export-report", type=Path); p.add_argument("--report-sha256")
    p.add_argument("--export-receipt", type=Path); p.add_argument("--receipt-sha256")
    p.add_argument("--main-authorized", action="store_true"); p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    need(a.out is None or not a.out.exists(), "exclusive new receipt path required")
    if a.command == "audit":
        result = audit()
    else:
        need(all((a.export_report,a.report_sha256,a.export_receipt,a.receipt_sha256,a.out)),
             "actual report/verify receipt paths and hashes plus new output required")
        _, result = construct_cuda(a.export_report,a.report_sha256,a.export_receipt,a.receipt_sha256,
                                   main_authorized=a.main_authorized)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    if a.out is not None:
        with a.out.open("xb") as f:
            f.write(raw)
    print(raw.decode(), end="")


if __name__ == "__main__":
    main()
