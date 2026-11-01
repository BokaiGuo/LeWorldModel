"""Complete visible goal-agent sprite translations on white RGB background.

Preserve anti-alias/outline pixels. Refuse non-isolated sprites; do not invent
occluded backgrounds or paste white rectangles across goal/object appearance.
Not a general physical rendering equivalence certificate.
"""
import numpy as np


def build(rgb):
    rgb=np.asarray(rgb)
    if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8:raise ValueError('official goal RGB required')
    pure=np.all(rgb==[65,105,225],axis=-1);ys,xs=np.nonzero(pure)
    if not len(xs):raise ValueError('canonical visible agent missing')
    x0,x1=int(xs.min())-2,int(xs.max())+3;y0,y1=int(ys.min())-2,int(ys.max())+3
    if x0<0 or y0<0 or x1>224 or y1>224:raise ValueError('sprite margin clipped')
    patch=rgb[y0:y1,x0:x1].copy()
    if not (np.all(patch[0]==255) and np.all(patch[-1]==255) and np.all(patch[:,0]==255) and np.all(patch[:,-1]==255)):
        raise ValueError('goal agent not isolated by white margin; background unknown')
    clean=rgb.copy();clean[y0:y1,x0:x1]=255
    h,w=patch.shape[:2];valid=[];invalid=[]
    coords=np.rint(np.linspace(16,208,16)).astype(int)
    for cy in coords:
        for cx in coords:
            left,top=int(cx)-w//2,int(cy)-h//2;right,bottom=left+w,top+h
            identity=f'agent_{cx}_{cy}'
            if (left<0 or top<0 or right>224 or bottom>224 or
                not np.all(clean[top:bottom,left:right]==255)):
                invalid.append(dict(id=identity,reason='not_wholly_visible_white_background'));continue
            image=clean.copy();image[top:bottom,left:right]=patch
            valid.append(dict(id=identity,rgb=image))
    if not valid:raise ValueError('no eligible complete sprite placements')
    return dict(valid=valid,invalid=invalid,sprite=patch,source_bbox=[x0,y0,x1,y1],
        source_nonwhite_pixels=int(np.sum(np.any(patch!=255,axis=-1))),
        old_pure_mask_residual_pixels=int(np.sum(np.any(patch!=255,axis=-1)&~pure[y0:y1,x0:x1])),
        source_removed_rgb=clean)
