"""Saved-data reconstruction for program-chain search, without neural calls."""
import numpy as np

from experiments.spectral_reachability import codex_p2_program_library_pair_20261030_v1 as kernel


def validate(result, basis, initial_context, goal):
    """Verify literal candidates, all connected contexts, scores and winner.

    This checks saved content only; hashes alone do not establish CUDA execution.
    A future runner must separately verify its durable operation journal and
    source/device identities before this can authenticate a production run.
    """
    arm=result["arm"]
    kernel.require(arm in kernel.ARMS and result["schema"]==kernel.NAME+".arm_result" and
                   result["physical"] is False and result["P2_admission"] is False,
                   "saved result scope")
    _,banks=kernel.candidates(basis)
    bank=banks[arm];rows=result["candidates"]
    kernel.require(len(rows)==len(bank)==kernel.COUNT and
                   result["packet_returns"]==34 and result["score_returns"]==17,
                   "complete fixed denominator")
    target=np.asarray(goal,dtype=np.float64)
    kernel.require(target.shape==(192,) and np.isfinite(target).all(),"goal")
    for item,row in zip(bank,rows,strict=True):
        kernel.require(row["candidate"]==item and len(row["edges"])==2, "exact candidate bank")
        raw=bytes.fromhex(item["raw_float32_hex"])
        context=initial_context
        for i,edge in enumerate(row["edges"]):
            expected_id=item["program_ids"][i] if arm=="library_chain" else item["program_ids"][0]
            kernel.require(edge["index"]==i and edge["program_id"]==expected_id and
                edge["raw_float32_hex"]==raw[i*40:(i+1)*40].hex() and
                edge["input_context_sha256"]==kernel.digest(context) and
                edge["output_context_sha256"]==kernel.digest(edge["output_context"]),
                "action and complete context chain")
            context=edge["output_context"]
        z=np.asarray(context["latent_history"][-1],dtype=np.float64)
        kernel.require(z.shape==(192,) and np.isfinite(z).all() and context==row["terminal_context"] and
                       row["cost"]==float(np.mean((z-target)**2)), "terminal score")
    winner=min(rows,key=lambda x:(x["cost"],x["candidate"]["index"]))
    kernel.require(result["winner_index"]==winner["candidate"]["index"] and
                   result["winner"]==winner==result["replay"],"winner and full replay")
    return dict(status="saved_structure_checked_not_cuda_authentication",arm=arm,
                candidates=len(rows),P2_admission=False)
