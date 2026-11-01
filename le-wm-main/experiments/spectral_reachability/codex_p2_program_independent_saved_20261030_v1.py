"""Saved-only structural and resource validation for independent P2 branches.

Does not authenticate CUDA device or input files on its own. A future outer
consumer must first reconstruct the pinned prepare, journal hash chain and
producer/model/device identity and then call this function for all eight rows.
"""
import numpy as np
from experiments.spectral_reachability import codex_p2_program_independent_budget_20261030_v1 as budget
from experiments.spectral_reachability import codex_p2_program_library_saved_20261030_v1 as saved


def validate_branch(case,branch,events):
    k=budget.kernel
    arm=branch["arm"];base=case["base"]
    k.require(arm in budget.ARMS and branch["schema"]==budget.NAME+".branch" and
              branch["base"]==base and branch["physical"] is False and branch["P2_admission"] is False
              and branch["context_sha256"]==k.digest(case["context"]) and
              branch["goal_sha256"]==k.digest(case["goal"]),"source-bound branch identity")
    jac=np.asarray(branch["jacobian"],dtype=np.float64)
    singular,basis=k.canonical_basis(jac)
    k.require(np.array_equal(singular,np.asarray(branch["singular_values"])) and
              np.array_equal(basis,np.asarray(branch["basis"])),"actual saved Jacobian SVD")
    library,banks=k.candidates(basis)
    k.require(branch["library"]==library and
              branch["unique_action_sequences"]==len({x["raw_float32_hex"] for x in banks[arm]}),
              "fixed library and true unique-action count")
    saved.validate(branch["result"],basis,case["context"],case["goal"])
    k.require(events and all(e.get("base")==base and e.get("arm")==arm for e in events),
              "branch event scope")
    body=[e["event"] for e in events]
    k.require(body[0]["kind"]=="branch_start" and body[-1]["kind"]=="branch_complete" and
              body[-1]["result_sha256"]==k.digest(branch) and
              body[0]["context_sha256"]==branch["context_sha256"] and
              body[0]["goal_sha256"]==branch["goal_sha256"] and
              body[0]["caps"]==budget.CAPS and
              body[-1]["resource_counts"]==branch["resource_counts"],"branch durable bookends")
    attempts=dict.fromkeys(budget.CAPS,0);returns=dict.fromkeys(budget.CAPS,0)
    stack=[]
    for e in body:
        kind=e["kind"]
        if kind=="resource_attempt":
            axis=e["resource"]
            k.require(axis in attempts and e["ordinal"]==attempts[axis]+1 and e["depth"]==len(stack),
                      "actual ordered resource attempt")
            attempts[axis]+=1;stack.append((axis,e["ordinal"]))
        elif kind=="resource_return":
            axis=e["resource"]
            k.require(stack and stack[-1]==(axis,e["ordinal"]) and e["depth"]==len(stack)-1,
                      "nested actual resource return")
            stack.pop();returns[axis]+=1
        else:
            k.require(kind in ("branch_start","branch_complete","packet_intent","packet_return",
                               "candidate_return"),"unexpected or failed event")
    counts=branch["resource_counts"]
    k.require(not stack and counts["attempted"]==attempts and counts["returned"]==returns and
              attempts==returns and counts["failed"] is False and
              all(attempts[x]<=budget.CAPS[x] for x in attempts) and
              0<=counts["wall_seconds"]<=budget.contract()["wall_seconds"],"resource census and caps")
    # A completed real packet goes through every nested model entry. No
    # synthetic fixture or empty counter may be promoted to production.
    k.require(attempts["packet"]>0 and all(attempts[x]==attempts["packet"] for x in
              ("action_encoder","predict","predictor","pred_proj")), "actual instrumented model entry counts")
    rows=branch["result"]["candidates"]+[branch["result"]["replay"]]
    packets=[e for e in body if e["kind"]=="packet_return"]
    candidates=[e for e in body if e["kind"]=="candidate_return"]
    intents=[e for e in body if e["kind"]=="packet_intent"]
    k.require(len(packets)==len(intents)==2*len(rows) and len(candidates)==len(rows)==attempts["score"],
              "all candidate/replay receipts")
    k.require(attempts["jacobian"]==attempts["svd"]==1 and
              attempts["packet"]==len(packets)+2, "actual H2 Jacobian and candidate packet work")
    for i,row in enumerate(rows):
        for j,edge in enumerate(row["edges"]):
            intent=intents[2*i+j];out=packets[2*i+j]
            k.require(out["edge"]==edge and intent["raw_float32_hex"]==edge["raw_float32_hex"] and
                      intent["input_context_sha256"]==edge["input_context_sha256"] and
                      intent["index"]==row["candidate"]["index"] and
                      out["index"]==row["candidate"]["index"] and
                      intent["replay"]==(i==len(rows)-1) and out["replay"]==intent["replay"],
                      "literal candidate packet intents and returns")
        event=candidates[i]
        k.require(event["cost"]==row["cost"] and event["edges"]==row["edges"] and
                  event["terminal_context_sha256"]==k.digest(row["terminal_context"]) and
                  event["index"]==row["candidate"]["index"] and event["replay"]==(i==len(rows)-1),
                  "candidate scored and replayed")
    return dict(status="branch_structure_and_resource_checked_not_cuda_authentication",
                base=base,arm=arm,counts=counts,unique_action_sequences=branch["unique_action_sequences"],
                P2_admission=False)
