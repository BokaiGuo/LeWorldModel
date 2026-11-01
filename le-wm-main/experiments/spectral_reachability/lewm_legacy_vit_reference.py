"""Independent frozen HF4-style ViT algebra using ORIGINAL checkpoint keys.

Official 224/14 tiny config only. Host CUDA only. No current ViT modules,
compatibility remap, extra training, or changes to the production loader.
"""


def cls_reference(pixels,state):
    if getattr(getattr(pixels,'device',None),'type',None)!='cuda':
        raise ValueError('reference neural computation requires host CUDA')
    import torch
    import torch.nn.functional as f
    if pixels.dtype!=torch.float32 or pixels.ndim!=4 or pixels.shape[1:]!=(3,224,224):
        raise ValueError('official float32 [B,3,224,224] required')
    def value(key):
        x=state['encoder.'+key]
        if x.device!=pixels.device or x.dtype!=pixels.dtype:raise ValueError('raw checkpoint must be float32 on same CUDA device')
        return x
    def linear(x,prefix):return f.linear(x,value(prefix+'.weight'),value(prefix+'.bias'))
    def norm(x,prefix):return f.layer_norm(x,(192,),value(prefix+'.weight'),value(prefix+'.bias'),eps=1e-12)
    patches=f.conv2d(pixels,value('embeddings.patch_embeddings.projection.weight'),
                     value('embeddings.patch_embeddings.projection.bias'),stride=14)
    patches=patches.flatten(2).transpose(1,2)
    cls=value('embeddings.cls_token').expand(len(pixels),-1,-1)
    hidden=torch.cat([cls,patches],dim=1)+value('embeddings.position_embeddings')
    if hidden.shape[1:]!=(257,192):raise ValueError('official tiny positional layout required')
    for layer in range(12):
        prefix=f'encoder.layer.{layer}'
        x=norm(hidden,prefix+'.layernorm_before')
        q,k,v=[linear(x,prefix+'.attention.attention.'+name).reshape(len(x),257,3,64).permute(0,2,1,3)
               for name in ('query','key','value')]
        probabilities=torch.softmax((q@k.transpose(-1,-2))/(64**.5),dim=-1)
        attention=(probabilities@v).transpose(1,2).reshape(len(x),257,192)
        hidden=hidden+linear(attention,prefix+'.attention.output.dense')
        x=norm(hidden,prefix+'.layernorm_after')
        x=f.gelu(linear(x,prefix+'.intermediate.dense'),approximate='none')
        hidden=hidden+linear(x,prefix+'.output.dense')
    return norm(hidden,'layernorm')[:,0]


def projected_reference(cls,state,prefix='projector'):
    if getattr(getattr(cls,'device',None),'type',None)!='cuda':
        raise ValueError('reference neural computation requires host CUDA')
    import torch
    import torch.nn.functional as f
    def value(suffix):
        x=state[prefix+'.net.'+suffix]
        if x.device!=cls.device:raise ValueError('CUDA state mismatch')
        return x
    x=f.linear(cls,value('0.weight'),value('0.bias'))
    x=f.batch_norm(x,value('1.running_mean'),value('1.running_var'),value('1.weight'),value('1.bias'),
                   training=False,eps=1e-5)
    x=f.gelu(x,approximate='none')
    return f.linear(x,value('3.weight'),value('3.bias'))
