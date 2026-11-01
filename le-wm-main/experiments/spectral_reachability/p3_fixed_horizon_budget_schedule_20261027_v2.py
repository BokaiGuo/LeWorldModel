"""Fixed-H3, full-16-bank allocation. Non-neural core; no physical feedback.

Chi is the OLD finite-response residual fraction, not a Jacobian signal:
r=(g0-x0)Q.T; D[i]=(predicted_H1[i]-x0)Q.T, uncentered (includes drift).
Rank uses relative SVD cutoff 1e-6. Goal, Q, x0 and every H1 response must be
bound to the frozen task/root. SVD singular values alone do not define chi.
"""
from copy import deepcopy
import math

import numpy as np

from experiments.spectral_reachability.d2p_kinodynamic_program_search import _bank, _validated_context


def require(ok,message):
    if not ok:
        raise ValueError(message)


def allocate(chi,mode):
    require(type(chi) in (int,float) and math.isfinite(chi) and 0<=chi<=1,'finite chi in [0,1]')
    require(mode in ('uniform','dynamic'),'uniform or dynamic mode required')
    k2=4 if mode=='uniform' else 2+math.floor(4*chi+.5)
    return k2,8-k2


def geometry_inputs(root,goal,projection):
    x=np.asarray(root['latent_history'][-1],dtype=float)
    require(goal is not None,'explicit frozen target goal required')
    g,q=np.asarray(goal,dtype=float),np.asarray(projection,dtype=float)
    require(g.shape==x.shape and q.shape==(len(x),len(x)) and
            all(np.isfinite(v).all() for v in (x,g,q)),'finite aligned goal/projection/latent required')
    require(np.allclose(q,q.T,rtol=0,atol=1e-8) and np.allclose(q@q,q,rtol=0,atol=1e-8),
            'task Q must be a symmetric idempotent projection')
    residual=(g-x)@q.T
    norm2=float(residual@residual)
    require(math.isfinite(norm2) and norm2>0,'nonzero finite goal-projected residual required')
    return x,g,q,residual,norm2


def response_geometry(root,children,goal,projection):
    require(len(children)==16,'complete sixteen H1 responses required, not a success subset')
    x,g,q,residual,norm2=geometry_inputs(root,goal,projection)
    ys=np.asarray([c['latent_history'][-1] for c in children],dtype=float)
    require(ys.shape==(16,len(x)) and np.isfinite(ys).all(),'finite complete H1 latent matrix')
    response=(ys-x)@q.T
    require(np.isfinite(response).all(),'finite projected H1 response required')
    _,singular,vt=np.linalg.svd(response,full_matrices=False)
    rank=int(np.sum(singular>singular[0]*1e-6)) if singular[0]>0 else 0
    require(rank>0,'nonzero finite-response rank required; no uniform fallback')
    parallel=(vt[:rank]@residual)@vt[:rank]
    chi=float(np.clip(np.sum((residual-parallel)**2)/norm2,0,1))
    require(math.isfinite(chi),'finite goal-related chi required')
    return dict(chi=chi,response_rank=rank,singular_values=singular.tolist(),relative_svd_cutoff=1e-6,
        residual=residual.tolist(),residual_squared_norm=norm2,projected_response=response.tolist(),
        space='frozen goal0-projected latent coordinates; r=(g0-x0)Q.T, D=(Y_H1-x0)Q.T',
        centering='uncentered finite displacement, includes autonomous drift',
        scope='original finite-response chi, not action Jacobian, identified controllability or physical geometry')


def run_search(*,root_context,primitive_bank,replay,goal,projection,mode,event_sink=None):
    require(mode in ('uniform','dynamic'),'explicit scheduler mode')
    root=_validated_context(root_context)
    bank=_bank(primitive_bank)
    require(len(bank)==16 and all(a.shape==(5,2) for a in bank),'exact full sixteen five-raw-step bank')
    _,g,q,_,_=geometry_inputs(root,goal,projection)  # No neural work on invalid task geometry.
    result=dict(schema='p3_fixed_horizon_budget_schedule.v2',status='incomplete',mode=mode,
        fixed_horizon_tokens=3,planning_cap=144,verification_reserve=3,
        events=[],layers=[],signal=None,svd_calls=0,allocation=None,best_plan=None,
        planning_edge_attempts=0,failure=None,P3_admission=False)
    frontier=[dict(context=root,actions=[],primitive_indices=[],depth=0)]

    def emit(event):
        if event_sink is not None:
            event_sink(deepcopy(event))

    for depth in (1,2,3):
        children=[]
        layer=dict(depth=depth,parents=[p['primitive_indices'] for p in frontier],
                   scheduled_edges=16*len(frontier),attempted_edges=0,complete=False)
        result['layers'].append(layer)
        for parent in frontier:
            for i,primitive in enumerate(bank):
                require(result['planning_edge_attempts']<144,'planning reserve must not be consumed')
                path=parent['primitive_indices']+[i]
                event=dict(kind='edge',depth=depth,parent_path=parent['primitive_indices'],
                           primitive_index=i,path=path,succeeded=False,replay_completed=False)
                result['events'].append(event)
                result['planning_edge_attempts']+=1
                layer['attempted_edges']+=1
                emit(dict(kind='attempt',depth=depth,path=path))
                try:
                    child=_validated_context(replay(deepcopy(parent['context']),primitive.copy()),root)
                    event['replay_completed']=True
                    residual=(np.asarray(child['latent_history'][-1],dtype=float)-g)@q.T
                    cost=float(np.mean(residual*residual))
                    require(math.isfinite(cost) and cost>=0,'finite goal-projected candidate cost')
                    event.update(succeeded=True,cost=cost,context=deepcopy(child))
                    children.append(dict(context=child,actions=parent['actions']+primitive.tolist(),
                        primitive_indices=path,depth=depth,cost=cost))
                except Exception as exc:
                    event.update(error_type=type(exc).__name__,error=str(exc))
                emit(event)
        layer['complete']=len(children)==layer['scheduled_edges']
        if not layer['complete']:
            result['failure']=dict(stage='layer',depth=depth,message='incomplete full bank; no reschedule or winner')
            return result
        children.sort(key=lambda n:(n['cost'],n['primitive_indices']))
        if depth==1:
            try:
                result['svd_calls']+=1
                # Original bank order, not cost order, is the sealed response row order.
                ordered=sorted(children,key=lambda n:n['primitive_indices'])
                result['signal']=response_geometry(root,[n['context'] for n in ordered],g,q)
                result['allocation']=list(allocate(result['signal']['chi'],mode))
                emit(dict(kind='allocation',signal=result['signal'],allocation=result['allocation']))
            except Exception as exc:
                result.update(status='signal_unavailable',failure=dict(stage='signal',type=type(exc).__name__,message=str(exc)))
                return result
        if depth<3:
            keep=result['allocation'][depth-1]
            require(len(children)>=keep,'complete parent pool required')
            frontier=children[:keep]
        else:
            result['best_plan']=deepcopy(children[0])  # NEVER an H1/H2 incumbent.
    require(result['planning_edge_attempts']==144,'exact 144 planning edges required')
    result['status']='completed'
    return result
