"""Fixed four-initializer common-suffix node, with cooperative per-arm accounting.

No loader, physical execution, durable sink implementation, or fairness claim.
Caller must bind actual feedback and persist events before allowing work.
"""
import json
import math

import numpy as np
import torch

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_spectral8_suffix_20261030 as spectral
from experiments.spectral_reachability import codex_four_arm_raw_suffix_20261030_main as suffix
from experiments.spectral_reachability import codex_four_arm_raw_candidates_20261030 as raw
from experiments.spectral_reachability import codex_four_arm_final_replay_20261030 as replay

CONFIG = {"steps": 2, "learning_rate": .05, "history_size": 3}
SEEDS = {"native": 20261030, "random": 20261031, "immediate_progress": 20261101}
LABELS = {"structured": spectral.LABEL,
          "native": "zero_centered_gaussian_native_style_not_installed_native_or_strong_baseline",
          "random": "random_raw_gaussian_bank",
          "immediate_progress": "one_packet_progress_top8_of16"}


def first_packet(result):
    """Extract only the first packet of the selected, actually replayed H3."""
    selected=result["selected_raw"]
    optimized=result["optimized_raw"]
    index=result["selected_index"]
    spec.require(isinstance(selected,torch.Tensor) and isinstance(optimized,torch.Tensor)
                 and selected.shape==(3,5,2) and optimized.shape==(8,3,5,2)
                 and type(index) is int and 0<=index<8
                 and leafmod._finite(optimized) and bool((optimized.abs()<=1).all().item())
                 and torch.equal(selected,optimized[index]), "selected H3 binding")
    return selected[0].detach().clone()


def cost(arm):
    spec.require(arm in spec.ARMS, "arm")
    v = dict.fromkeys(spec.AXES, 0)
    calls = 8 * CONFIG["steps"] * 3 + 8 * 3
    v.update(action_encode_calls=calls, predict_calls=calls, predict_samples=calls,
             backward_calls=8 * CONFIG["steps"], vjp_rows=8 * CONFIG["steps"],
             replay_calls=8, selection_evaluations=8)
    if arm == "structured":
        for key in ("action_encode_calls", "predict_calls", "predict_samples"):
            v[key] += 3
        v["backward_calls"] += 192
        v["vjp_rows"] += 192
        v["jacobian_calls"] = v["svd_calls"] = 1
    if arm == "immediate_progress":
        for key in ("action_encode_calls", "predict_calls", "predict_samples",
                    "replay_calls", "selection_evaluations"):
            v[key] += 16
    return v


def contract_for(task, model_sha, hardware_sha):
    spec.require(spec.is_hash(model_sha) and spec.is_hash(hardware_sha), "identity")
    cfg = dict(CONFIG)
    policy = spectral.policy(task["action_mean"], task["action_std"], cfg["history_size"])
    optimizer = spectral.file_sha(suffix.__file__)
    identity = {"root": task["binding"]["root_sha256"], "task": spec.digest(task),
                "model": model_sha, "action_domain": task["binding"]["action_domain_sha256"],
                "history": spec.digest(task["root_context"]), "goal": spec.digest(task["goal"]),
                "optimizer": optimizer, "optimizer_config": spec.digest(cfg),
                "meter": spectral.file_sha(durable.__file__), "hardware": hardware_sha}
    initializer = spectral.file_sha(__file__)
    arms = {a: {"initializer_sha256": initializer, "optimizer_sha256": optimizer,
                "optimizer_config_sha256": identity["optimizer_config"]} for a in spec.ARMS}
    caps = {k: max(cost(a)[k] for a in spec.ARMS) for k in spec.AXES}
    contract = {"schema": spec.NAME+".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": spec.digest(policy),
                "levels": [{"id": "node", "caps": caps},
                           {"id": "unused_larger", "caps": {k: v+1 for k,v in caps.items()}}]}
    spec.preflight(contract, spec.digest(contract))
    return policy, contract, cfg


def plan(leaf, arm, task, config, policy, contract, *, device, emit):
    spec.require(type(leaf) is leafmod.LeafAdapter, "leaf")
    try:
        spec.require(arm in spec.ARMS and callable(emit) and spec.digest(contract)==leaf.contract_sha,
                     "contract binding")
        p,c,cfg = contract_for(task, contract["identity"]["model"], contract["identity"]["hardware"])
        spec.require((policy,contract,config)==(p,c,cfg) and leaf.config_sha==spec.digest(config),
                     "task/config binding")
        with leaf.driver.path.open("rb") as stream:
            header=json.loads(stream.readline(1024*1024))["payload"]
        spec.require(header["arm"] == arm and header["level"] == "node", "arm/level binding")
        expected=cost(arm)
        spec.require(all(leaf.driver.charged[k]+expected[k]<=leaf.driver.caps[k] for k in spec.AXES),
                     "full node reserve unavailable")
        tensor=lambda x: torch.tensor(x,dtype=torch.float32,device=device)
        states=tensor(task["root_context"]["latent_history"])[None]
        history=tensor(task["root_context"]["action_history"])[None]
        goal=tensor(task["goal"])[None]
        mean,std=tensor(task["action_mean"]),tensor(task["action_std"])
        projection=tensor(task["projection"])
        for x in (states,history,goal,mean,std,projection):
            leaf._tensor(x)
            spec.require(leafmod._finite(x), "nonfinite input")
        spec.require(states.shape==(1,3,192) and history.shape==(1,2,10) and goal.shape==(1,192)
                     and mean.shape==std.shape==(2,) and bool((std>0).all().item())
                     and projection.shape==(192,192)
                     and torch.equal(projection,torch.eye(192,device=states.device)), "feedback ABI")
        emit("node_header", {"arm":arm,"task_sha256":spec.digest(task),
                             "label":LABELS[arm],"config":config})
        if arm=="structured":
            leaf._dispatch("jacobian","jacobian",{"jacobian_calls":1},lambda:None)
            nominal=torch.zeros((1,3,5,2),device=states.device,requires_grad=True)
            pred=raw._normal(leaf,"jacobian",states.detach(),history.detach(),nominal,mean,std)
            rows=[]
            for i in range(192):
                grad=leaf._dispatch("jacobian","backward",{"backward_calls":1,"vjp_rows":1},
                    lambda i=i:torch.autograd.grad(pred[0,i],nominal,retain_graph=i<191)[0])
                spec.require(grad.shape==nominal.shape and leafmod._finite(grad),"invalid Jacobian")
                rows.append(grad.detach().reshape(-1))
            jac=torch.stack(rows).cpu().numpy()
            emit("spectral_signal",{"nominal":pred.detach().cpu().tolist(),"jacobian":jac.tolist(),
                                    "projection":projection.cpu().tolist()})
            bank,singular=leaf._dispatch("initialization","svd",{"svd_calls":1},
                lambda:spectral.spectral_bank(jac,projection.cpu().numpy()))
            initial=torch.from_numpy(bank).to(device=states.device,dtype=states.dtype)
            emit("spectral_values",{"singular_values":singular.tolist()})
        else:
            generator=torch.Generator(device="cpu").manual_seed(SEEDS[arm])
            n=16 if arm=="immediate_progress" else 8
            candidates=(.4*torch.randn((n,3,5,2),generator=generator)).clamp(-1,1).to(states.device)
            if arm=="native":
                candidates[0]=0
            emit("generated_candidates",{"raw":candidates.cpu().tolist(),"seed":SEEDS[arm]})
            if arm=="immediate_progress":
                scores=[]
                with torch.no_grad():
                    for i in range(16):
                        leaf._dispatch("screening","replay",{"replay_calls":1},lambda:None)
                        prediction=raw._normal(leaf,"screening",states,history,candidates[i:i+1,:1],mean,std)
                        def score():
                            value=float((prediction-goal).square().mean().item())
                            spec.require(math.isfinite(value),"nonfinite screen")
                            return value
                        scores.append(leaf._dispatch("screening","selection",{"selection_evaluations":1},score))
                        emit("screen_score",{"index":i,"score":scores[-1]})
                order=sorted(range(16),key=lambda i:(scores[i],i))[:8]
                emit("shortlist",{"indices":order})
                initial=candidates[order]
            else:
                initial=candidates
        emit("initial_candidates",{"raw":initial.detach().cpu().tolist()})
        optimized=torch.cat([suffix.optimize(leaf,states,history,initial[i:i+1],goal,mean,std,config)
                             for i in range(8)])
        emit("optimized_candidates",{"raw":optimized.detach().cpu().tolist()})
        final=replay.replay_and_select(leaf,states,history,optimized,goal,mean,std,policy,contract)
        emit("final_selection",{"scores":final["scores"],"index":final["selected_index"],
                                "id":final["selected_id"]})
        return {"initial_raw":initial.detach(),"optimized_raw":optimized.detach(),
                "selected_raw":optimized[final["selected_index"]].detach(),
                "scores":final["scores"],"selected_index":final["selected_index"],
                "label":LABELS[arm],"fairness_proven":False,"physical_execution_authorized":False}
    except BaseException:
        leaf._abort()
        raise
