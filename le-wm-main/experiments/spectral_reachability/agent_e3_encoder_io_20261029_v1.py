"""Original HOST encoder IO recorder and independent saved arithmetic checks.

Caller must bind model/source/physical environment and overall lifecycle.
"""
import io
import hashlib
import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_e3_unified_transport_20261029_v1 as u
from experiments.spectral_reachability import agent_p1_teacher_library40_encode_20261028_v1 as pixels


def exact(a,b,why):
    x=np.asarray(a);y=np.asarray(b)
    u.need(x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),why)


def encode(model,torch,device,store,frame,boundary,check):
    """Capture actual model.encode hook IO. Requires genuine HOST CUDA caller."""
    u.need(getattr(device,'type',None)=='cuda' and type(boundary) is int and boundary in (10,15),'HOST only actual boundary')
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    png=u.read_bytes(store.output/frame['file'],256*1024)
    u.need(u.fsha(png)==frame['sha256'],'actual captured PNG SHA')
    with Image.open(io.BytesIO(png)) as im:
        u.need(im.mode=='RGB' and im.size==(224,224) and im.format=='PNG','original RGB224')
        image=np.asarray(im).copy()
    u.need(hashlib.sha256(image.tobytes()).hexdigest()==frame['pixel_sha256'],'actual PNG decoded pixels')
    refs={'pixels':store.array('encoding','pixels',image)};modules=[];hooks=[];pending={}
    def save(role,tensor):
        u.need(tensor.device==device and tensor.dtype==torch.float32,'actual native CUDA f32')
        check();torch.cuda.synchronize(device)
        value=tensor.detach().cpu().contiguous().numpy().copy();check()
        refs[role]=store.array('encoding',role,value)
        return value
    def pre(kind):
        def hook(module,args):
            check();u.need(len(args)==1 and not torch.is_grad_enabled() and not pending
                and [x['kind'] for x in modules]==([] if kind=='encoder' else ['encoder']), 'encoder/projector sequence')
            token=store.index;store.event('encoding','intent',kind,token,{'boundary':boundary})
            pending[module]=(kind,token)
            save('encoder_input' if kind=='encoder' else 'projector_input',args[0])
        return hook
    def post(kind):
        def hook(module,args,out):
            u.need(module in pending and pending[module][0]==kind,'matching module callback')
            if hasattr(out,'last_hidden_state'):out=out.last_hidden_state
            save('encoder_hidden' if kind=='encoder' else 'projector_output',out)
            _,token=pending.pop(module);modules.append(dict(kind=kind,token=token))
            store.event('encoding','return',kind,token,{'boundary':boundary})
        return hook
    def forbidden(module,args):raise ValueError('predict/AD modules forbidden during actual encoding')
    try:
        for kind in ('encoder','projector'):
            module=getattr(model,kind);hooks.extend((module.register_forward_pre_hook(pre(kind)),module.register_forward_hook(post(kind))))
        for kind in ('action_encoder','predictor','pred_proj'):
            hooks.append(getattr(model,kind).register_forward_pre_hook(forbidden))
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            x=p0._image_tensor(image,transform=p0.image_transform(),device=device)
            prep=save('preprocessed',x);exact(prep,pixels.preprocess_pixels(image),'original preprocessing exact')
            z=model.encode({'pixels':x})['emb'];u.need(tuple(z.shape)==(1,1,192),'original embedding shape')
            latent=save('latent',z[0,0]);torch.cuda.synchronize(device)
        u.need(not pending and [e['kind'] for e in modules]==['encoder','projector'],'exact module return count')
        store.event('encoding','record','encoding_complete',store.index,dict(boundary=boundary,frame=frame,arrays=refs))
        return latent,refs
    finally:
        for handle in reversed(hooks):handle.remove()


def verify_arrays(frame_bytes, arrays):
    """Check persisted full encoder connections; does not rerun neural model."""
    with Image.open(io.BytesIO(frame_bytes)) as im:
        u.need(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'RGB224');image=np.asarray(im).copy()
    u.need(type(arrays) is dict and set(arrays)=={'pixels','preprocessed','encoder_input','encoder_hidden',
        'projector_input','projector_output','latent'},'all seven saved tensors')
    for role,value in arrays.items():
        x=np.asarray(value)
        u.need((role,x.shape,x.dtype.str) in u.ENC_SHAPES and np.isfinite(x).all(),'exact saved tensor shape/dtype/finite')
    exact(arrays['pixels'],image,'frame pixels');exact(arrays['preprocessed'],pixels.preprocess_pixels(image),'preprocessing')
    exact(arrays['encoder_input'],arrays['preprocessed'].reshape(1,3,224,224),'encoder input')
    hidden=np.asarray(arrays['encoder_hidden']);u.need(hidden.dtype==np.float32 and hidden.shape==(1,257,192) and np.isfinite(hidden).all(),'full native hidden')
    exact(arrays['projector_input'],hidden[:,0,:],'CLS to projector')
    exact(arrays['latent'],arrays['projector_output'].reshape(192),'projector to native latent')
    return dict(saved_io_connected=True,neural_parity_recomputed=False,scientific_admission=False)
