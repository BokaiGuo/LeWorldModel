"""Current/goal RGB nuisance augmentation; approximate edits, no learned head."""
import numpy as np
from experiments.spectral_reachability.pusht_goal_object_tangent_metric import _basis,_real


POSITIONS=((32,32),(112,32),(192,32),(32,112),(192,112),(32,192),(112,192),(192,192))


def agent_variants(rgb):
    rgb=np.asarray(rgb)
    if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8:raise ValueError('official RGB required')
    mask=np.all(rgb==[65,105,225],axis=-1);ys,xs=np.nonzero(mask)
    if not len(xs):raise ValueError('canonical visible agent required')
    cx,cy=int(np.rint(xs.mean())),int(np.rint(ys.mean()))
    objectmask=np.all(rgb==[119,136,153],axis=-1)
    result=[]
    for x,y in POSITIONS:
        nx,ny=xs-cx+x,ys-cy+y
        valid=(nx>=0)&(nx<224)&(ny>=0)&(ny<224)
        image=rgb.copy();image[mask]=255
        image[ny[valid],nx[valid]]=[65,105,225]
        image[objectmask]=rgb[objectmask]
        result.append(dict(id=f'agent_{x}_{y}',rgb=image))
    return result


def compile_metric(goal_vectors,object_vectors,extra_nuisance_differences):
    goals,objects,extra=map(_real,(goal_vectors,object_vectors,extra_nuisance_differences))
    if (goals.ndim!=2 or goals.shape[0]<2 or objects.shape!=(6,goals.shape[1]) or
        extra.shape!=(16,goals.shape[1])):raise ValueError('six object and sixteen fixed scene nuisances required')
    n=np.concatenate([goals[1:]-goals[0],extra],axis=0).T
    nuisance,ns=_basis(n);complement=np.eye(goals.shape[1])-nuisance@nuisance.T
    task,ts=_basis(complement@(objects-goals[0]).T)
    if not task.shape[1]:raise ValueError('no surviving task span')
    q=task@task.T
    return dict(projection=q.tolist(),task_rank=task.shape[1],nuisance_rank=nuisance.shape[1],
        nuisance_singular_values=ns.tolist(),task_singular_values=ts.tolist(),
        physical_states_used=False,additional_training=False,physical_invariance_certified=False,
        normalization='orthogonal projector; not scaled after seeing outcomes',P1_admission=False)
