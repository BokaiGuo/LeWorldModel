"""Four explicit, distinct metered model-side optimizers for one observed node.

No model loader, environment, RGB observation, launcher, or fairness claim.
Caller binds actual observations and durably stores every emitted event.
"""
import json
import math

import torch

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_raw_suffix_20261030_main as suffix
from experiments.spectral_reachability import codex_four_arm_spectral8_suffix_20261030 as spectral
from experiments.spectral_reachability import codex_four_arm_raw_candidates_20261030 as raw
from experiments.spectral_reachability import codex_four_arm_final_replay_20261030 as replay

CONFIG = {"steps":2, "learning_rate":0.05, "history_size":3}
SEEDS = {"native":20261030, "random":20261031, "immediate_progress":20261101}
LABELS = {"structured":"spectral8_h3_plus_per_candidate_gradient_not_original_R",
          "native":"two_generation_raw_CEM_variant_not_installed_native_or_strong_baseline",
          "random":"two_pool_random_search",
          "immediate_progress":"one_packet_progress_screen_then_h3_replay"}


def cost(arm):
    spec.require(arm in spec.ARMS, "arm")
    if arm == "structured":
        return spectral.required_caps(CONFIG)
    v = dict.fromkeys(spec.AXES,0)
    calls = 40 if arm == "immediate_progress" else 72
    v.update(action_encode_calls=calls,predict_calls=calls,predict_samples=calls,
             replay_calls=24,selection_evaluations=24)
    return v


def contract_for(task, arm, model_sha, hardware_sha):
    """One per-arm, per-node ledger contract; source hashes are not attestation."""
    spec.require(arm in spec.ARMS and spec.is_hash(model_sha) and spec.is_hash(hardware_sha), "identity")
    config = dict(CONFIG)
    policy = spectral.policy(task["action_mean"],task["action_std"],3)
    optimizer = spectral.file_sha(suffix.__file__ if arm == "structured" else __file__)
    initializer = spectral.file_sha(spectral.__file__ if arm == "structured" else __file__)
    identity = {"root":task["binding"]["root_sha256"],"task":spec.digest(task),
                "model":model_sha,"action_domain":task["binding"]["action_domain_sha256"],
                "history":spec.digest(task["root_context"]),"goal":spec.digest(task["goal"]),
                "optimizer":optimizer,"optimizer_config":spec.digest(config),
                "meter":spectral.file_sha(durable.__file__),"hardware":hardware_sha}
    # Four schema entries are placeholders for one bound arm, not four runs.
    arms = {a:{"initializer_sha256":initializer if a == arm else spec.digest({"unused":a,"for":arm}),
               "optimizer_sha256":optimizer,"optimizer_config_sha256":identity["optimizer_config"]}
            for a in spec.ARMS}
    caps=cost(arm)
    contract={"schema":spec.NAME+".contract.v1","identity":identity,"arms":arms,
              "scope":"standalone_per_root_per_arm_including_attributable_setup",
              "phase_policy_sha256":spec.digest(policy),
              "levels":[{"id":"node","caps":caps},{"id":"unused_larger","caps":{k:v+1 for k,v in caps.items()}}]}
    spec.preflight(contract,spec.digest(contract))
    return policy,contract,config


def plan(leaf, arm, task, config, policy, contract, *, device, emit):
    """Execute one fixed arm on one observed task; record candidates before calls."""
    spec.require(type(leaf) is leafmod.LeafAdapter, "leaf")
    try:
        spec.require(arm in spec.ARMS and callable(emit) and spec.digest(contract)==leaf.contract_sha, "binding")
        p,c,cfg=contract_for(task,arm,contract["identity"]["model"],contract["identity"]["hardware"])
        spec.require(p==policy and c==contract and cfg==config and leaf.config_sha==spec.digest(config), "task binding")
        with leaf.driver.path.open("rb") as stream:
            header=json.loads(stream.readline(1024*1024))["payload"]
        spec.require(header["arm"]==arm, "arm binding")
        expected=cost(arm)
        spec.require(all(leaf.driver.charged[k]+expected[k]<=leaf.driver.caps[k] for k in spec.AXES),
                     "full node reserve unavailable")
        tensor=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
        states=tensor(task["root_context"]["latent_history"])[None]
        history=tensor(task["root_context"]["action_history"])[None]
        goal=tensor(task["goal"])[None]
        mean,std=tensor(task["action_mean"]),tensor(task["action_std"])
        projection=tensor(task["projection"])
        for value in (states,history,goal,mean,std,projection):
            leaf._tensor(value)
            spec.require(leafmod._finite(value), "nonfinite input")
        spec.require(states.shape==(1,3,192) and history.shape==(1,2,10) and goal.shape==(1,192)
                     and mean.shape==std.shape==(2,) and bool((std>0).all().item())
                     and projection.shape==(192,192)
                     and torch.equal(projection,torch.eye(192,device=states.device)), "feedback ABI")
        if arm=="structured":
            result=spectral.run(leaf,states,history,goal,mean,std,projection,config,policy,contract,emit=emit)
            index=result["selected_index"]
            emit("selected",{"arm":arm,"index":index,"raw":result["optimized_raw"][index].cpu().tolist(),
                             "label":LABELS[arm]})
            return dict(result,selected_raw=result["optimized_raw"][index].detach(),label=LABELS[arm])
        gen=torch.Generator(device="cpu").manual_seed(SEEDS[arm])
        def draw(n,center=None,scale=None):
            noise=torch.randn((n,3,5,2),generator=gen,dtype=torch.float32).to(states.device)
            center=torch.zeros((1,3,5,2),device=states.device) if center is None else center
            scale=torch.full_like(center,0.4) if scale is None else scale
            return (center+scale*noise).clamp(-1,1)
        def evaluate(samples, stage):
            emit("candidates",{"stage":stage,"raw":samples.detach().cpu().tolist()})
            r=replay.replay_and_select(leaf,states,history,samples,goal,mean,std,policy,contract)
            emit("scores",{"stage":stage,"scores":r["scores"]})
            return r
        if arm=="native":
            center=torch.zeros((1,3,5,2),device=states.device);scale=torch.full_like(center,0.4)
            for i in range(2):
                samples=draw(8,center,scale);samples[0]=center[0]
                measured=evaluate(samples,f"cem_{i}")
                elite=sorted(range(8),key=lambda j:(measured["scores"][j],j))[:2]
                emit("elite",{"stage":i,"indices":elite})
                center=samples[elite].mean(0,keepdim=True)
                scale=samples[elite].std(0,unbiased=False,keepdim=True).clamp_min(.05)
            final_candidates=draw(8,center,scale);final_candidates[0]=center[0]
        elif arm=="random":
            pools=[];scores=[]
            for i in range(2):
                sample=draw(8);res=evaluate(sample,f"random_{i}")
                pools.append(sample);scores.extend(res["scores"])
            order=sorted(range(16),key=lambda i:(scores[i],i))[:8]
            emit("shortlist",{"indices":order})
            final_candidates=torch.cat(pools)[order]
        else:
            sample=draw(16);emit("screen_candidates",{"raw":sample.cpu().tolist(),"seed":SEEDS[arm]})
            scores=[]
            with torch.no_grad():
                for i in range(16):
                    leaf._dispatch("screening","replay",{"replay_calls":1},lambda:None)
                    prediction=raw._normal(leaf,"screening",states,history,sample[i:i+1,:1],mean,std)
                    def score():
                        value=float((prediction-goal).square().mean().item())
                        spec.require(math.isfinite(value),"nonfinite screening score")
                        return value
                    value=leaf._dispatch("screening","selection",{"selection_evaluations":1},score)
                    scores.append(value);emit("screen_score",{"index":i,"score":value})
            order=sorted(range(16),key=lambda i:(scores[i],i))[:8]
            emit("shortlist",{"indices":order});final_candidates=sample[order]
        final=evaluate(final_candidates,"final")
        selected=final["selected_index"]
        emit("selected",{"arm":arm,"index":selected,"raw":final_candidates[selected].cpu().tolist(),
                         "label":LABELS[arm]})
        return {"selected_raw":final_candidates[selected].detach(),"scores":final["scores"],
                "selected_index":selected,"label":LABELS[arm],"fairness_proven":False}
    except BaseException:
        leaf._abort()
        raise
