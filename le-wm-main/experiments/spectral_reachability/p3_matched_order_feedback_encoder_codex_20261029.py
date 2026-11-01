"""Matched-order CUDA visual IO capture; no loader, launcher, or authority.

The original E3 encoder fixes boundaries to 10 and 15.  This implementation
uses the same preprocessing and tensor auditor but admits the four boundaries
of a 20-action matched-order episode.  It does not itself certify a source.
"""
import hashlib
import io
import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_e3_encoder_io_20261029_v1 as original

BOUNDARIES = (5, 10, 15, 20)
need = original.u.need
verify_arrays = original.verify_arrays


def encode(model, torch, device, store, frame, boundary, check):
    need(getattr(device, 'type', None) == 'cuda' and type(boundary) is int and boundary in BOUNDARIES,
         'matched-order actual CUDA boundary')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    png = original.u.read_bytes(store.output / frame['file'], 256*1024)
    need(original.u.fsha(png) == frame['sha256'], 'captured PNG SHA')
    with Image.open(io.BytesIO(png)) as im:
        need(im.mode == 'RGB' and im.size == (224,224) and im.format == 'PNG', 'RGB224 PNG')
        image = np.asarray(im).copy()
    need(hashlib.sha256(image.tobytes()).hexdigest() == frame['pixel_sha256'], 'captured decoded pixels')
    refs = {'pixels': store.array('encoding', 'pixels', image)}
    modules, handles, pending = [], [], {}

    def save(role, tensor):
        need(tensor.device == device and tensor.dtype == torch.float32, 'actual CUDA float32')
        check(); torch.cuda.synchronize(device)
        value = tensor.detach().cpu().contiguous().numpy().copy(); check()
        refs[role] = store.array('encoding', role, value)
        return value

    def pre(kind):
        def hook(module, args):
            check()
            need(len(args) == 1 and not torch.is_grad_enabled() and not pending and
                 [x['kind'] for x in modules] == ([] if kind == 'encoder' else ['encoder']), 'module order')
            token = store.index
            store.event('encoding', 'intent', kind, token, {'boundary': boundary})
            pending[module] = (kind, token)
            save('encoder_input' if kind == 'encoder' else 'projector_input', args[0])
        return hook

    def post(kind):
        def hook(module, args, out):
            need(module in pending and pending[module][0] == kind, 'matching module callback')
            if hasattr(out, 'last_hidden_state'):
                out = out.last_hidden_state
            save('encoder_hidden' if kind == 'encoder' else 'projector_output', out)
            _, token = pending.pop(module); modules.append(dict(kind=kind, token=token))
            store.event('encoding', 'return', kind, token, {'boundary': boundary})
        return hook

    def forbidden(module, args):
        raise ValueError('predictor or action module during observation encoding')

    try:
        for kind in ('encoder','projector'):
            module = getattr(model, kind)
            handles.extend((module.register_forward_pre_hook(pre(kind)), module.register_forward_hook(post(kind))))
        for kind in ('action_encoder','predictor','pred_proj'):
            handles.append(getattr(model, kind).register_forward_pre_hook(forbidden))
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
            x = p0._image_tensor(image, transform=p0.image_transform(), device=device)
            original.exact(save('preprocessed', x), original.pixels.preprocess_pixels(image), 'original preprocessing')
            z = model.encode({'pixels': x})['emb']
            need(tuple(z.shape) == (1,1,192), 'native projected shape')
            latent = save('latent', z[0,0]); torch.cuda.synchronize(device)
        need(not pending and [e['kind'] for e in modules] == ['encoder','projector'], 'exact module completion')
        store.event('encoding', 'record', 'encoding_complete', store.index,
                    dict(boundary=boundary, frame=frame, arrays=refs))
        return latent, refs
    finally:
        for handle in reversed(handles):
            handle.remove()
