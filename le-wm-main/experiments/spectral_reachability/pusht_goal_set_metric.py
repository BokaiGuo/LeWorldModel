"""Finite nuisance-equivalent goal-set distance, no learned parameters.

The image catalog is renderer-specific approximate augmentation, not a certified
physical goal manifold. Nearest-goal distance does not establish task fidelity.
"""
import numpy as np
from experiments.spectral_reachability.pusht_goal_object_tangent_metric import _real


def catalog_images(rgb):
    rgb=np.asarray(rgb)
    if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8:raise ValueError('official goal RGB required')
    actor=np.all(rgb==[65,105,225],axis=-1);ys,xs=np.nonzero(actor)
    if not len(xs):raise ValueError('canonical visible goal agent required')
    cx,cy=int(np.rint(xs.mean())),int(np.rint(ys.mean()))
    positions=np.rint(np.linspace(16,208,16)).astype(int)
    clean=rgb.copy();clean[actor]=255
    for y in positions:
        for x in positions:
            nx,ny=xs-cx+x,ys-cy+y
            if np.any(nx<0) or np.any(nx>=224) or np.any(ny<0) or np.any(ny>=224):
                raise ValueError('goal agent shape cannot fit fixed grid')
            image=clean.copy();image[ny,nx]=[65,105,225]
            yield dict(id=f'agent_{x}_{y}',rgb=image)


def distances(vectors,catalog):
    vectors,catalog=map(_real,(vectors,catalog))
    if (vectors.ndim!=2 or catalog.ndim!=2 or not len(vectors) or not len(catalog) or
        vectors.shape[1]!=catalog.shape[1] or vectors.shape[1]<1):raise ValueError('nonempty aligned vectors and finite goal catalog required')
    costs=np.mean((vectors[:,None]-catalog[None])**2,axis=-1)
    indices=np.argmin(costs,axis=1)
    return dict(costs=costs[np.arange(len(vectors)),indices].tolist(),goal_indices=indices.tolist(),
        objective='min over supplied equivalent-goal embeddings of latent MSE',P1_admission=False)
