"""RGB-only finite goal augmentation metric, not a learned physical readout.

Renderer-specific visible-color edits are approximate image augmentations, NOT
certified task-equivalent simulator states. No geometry/state labels are used.
"""
import numpy as np


def variants(goal_rgb):
    from PIL import Image
    rgb=np.asarray(goal_rgb)
    if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8:
        raise ValueError('official uint8 RGB goal required')
    color=np.array([119,136,153],np.uint8)
    mask=np.all(rgb==color,axis=-1)
    if not mask.any():raise ValueError('canonical visible object color absent')
    agent=np.all(rgb==[65,105,225],axis=-1)
    ys,xs=np.nonzero(mask); center=(float(xs.mean()),float(ys.mean()))
    original=Image.fromarray(mask.astype(np.uint8)*255)
    transformations=[('x_minus4',-4,0,0),('x_plus4',4,0,0),
                     ('y_minus4',0,-4,0),('y_plus4',0,4,0),
                     ('angle_minus5',0,0,-5),('angle_plus5',0,0,5)]
    result=[]
    for name,dx,dy,angle in transformations:
        if angle:
            edited=np.asarray(original.rotate(angle,resample=Image.Resampling.NEAREST,center=center,fillcolor=0))>0
        else:
            # PIL inverse affine translation; no wrapping at the canvas border.
            edited=np.asarray(original.transform((224,224),Image.Transform.AFFINE,(1,0,-dx,0,1,-dy),resample=Image.Resampling.NEAREST,fillcolor=0))>0
        image=rgb.copy();image[mask]=255;image[edited]=color
        image[agent]=rgb[agent]  # Keep known visible agent pixels unchanged.
        result.append(dict(id=name,rgb=image))
    return result


def _real(value):
    a=np.asarray(value)
    if a.dtype.kind not in 'fi' or not np.isfinite(a).all():raise ValueError('finite real embeddings required')
    return a.astype(float)


def _basis(directions):
    u,s,_=np.linalg.svd(directions,full_matrices=False)
    tol=(s[0] if len(s) else 0)*max(directions.shape)*np.finfo(float).eps
    rank=int(np.sum(s>tol))
    return u[:,:rank],s


def compile_metric(goal_vectors,object_vectors):
    goals,objects=map(_real,(goal_vectors,object_vectors))
    if (goals.ndim!=2 or goals.shape[0]<2 or goals.shape[1]<2 or
        objects.shape!=(6,goals.shape[1])):raise ValueError('goal nuisance set and six fixed object variants required')
    nuisance,_=_basis((goals[1:]-goals[0]).T)
    complement=np.eye(goals.shape[1])-nuisance@nuisance.T
    changes=(objects-goals[0]).T
    task,s=_basis(complement@changes)
    if not task.shape[1]:raise ValueError('no object directions survive nuisance projection')
    projection=task@task.T
    return dict(projection=projection.tolist(),task_rank=task.shape[1],nuisance_rank=nuisance.shape[1],
        projected_object_singular_values=s.tolist(),physical_labels_used=False,additional_training=False,
        construction='span of six fixed visible-object goal augmentations orthogonal to supplied nuisance span',
        normalization='orthogonal projector; no outcome-selected scaling',
        image_augmentations_are_physical_states=False,P1_admission=False)
