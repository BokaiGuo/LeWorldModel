"""Train LeWM with directional latent-space regularizers.

The script mirrors train.py and keeps the original LeWM objective intact while
adding optional action/reachability/nuisance alignment terms.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from experiments.empirical_spectral_audit.regularizers import directional_regularizer_losses
from module import SIGReg
from utils import SaveCkptCallback, get_column_normalizer, get_img_preprocessor


def _load_teacher_model(path: str | None, device: torch.device):
    if not path:
        return None
    from experiments.io_utils import load_policy

    teacher = load_policy(path, device=str(device))
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def directional_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, and compute directional losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight
    directional = cfg.loss.get("directional", {})
    top_k = int(directional.get("top_k", 32))
    alignment_mode = str(directional.get("alignment_mode", "none"))
    alignment_conditioning = str(
        directional.get("alignment_conditioning", "magnitude")
    )

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    teacher = getattr(self, "teacher_model", None)
    if teacher is not None:
        with torch.no_grad():
            batch["teacher_emb"] = teacher.encode(dict(batch))["emb"]

    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    output.update(
        directional_regularizer_losses(
            emb=emb,
            pred_emb=pred_emb,
            target_emb=tgt_emb,
            batch=batch,
            top_k=top_k,
            encode_pixels=lambda nuisance_batch: self.model.encode(nuisance_batch)["emb"],
            nuisance_enabled=directional.get("nuisance_weight", 0.0) > 0,
            alignment_mode=alignment_mode,
            alignment_bins=int(directional.get("alignment_bins", 2)),
            alignment_min_samples=int(directional.get("alignment_min_samples", 16)),
            alignment_ridge=float(directional.get("alignment_ridge", 1e-3)),
            alignment_conditioning=alignment_conditioning,
        )
    )

    output["loss"] = (
        output["pred_loss"]
        + lambd * output["sigreg_loss"]
        + directional.get("dyn_weight", 0.0) * output["action_tangent_align_loss"]
        + directional.get("controllability_weight", 0.0) * output["action_controllability_align_loss"]
        + directional.get("reach_weight", 0.0) * output["reach_align_loss"]
        + directional.get("nuisance_weight", 0.0) * output["nuisance_deemphasis_loss"]
        + directional.get("spectral_pred_weight", 0.0) * output["spectral_prediction_loss"]
        + directional.get("residual_band_weight", 0.0) * output["residual_band_regularization_loss"]
        + directional.get("contact_weight", 0.0) * output["contact_event_spectral_auxiliary_loss"]
        + directional.get("teacher_anchor_weight", 0.0) * output["teacher_anchor_loss"]
        + directional.get("alignment_weight", 0.0) * output["operator_alignment_loss"]
    )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    losses_dict[f"{stage}/directional_top_k"] = torch.as_tensor(top_k, device=emb.device)
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="../config/train", config_name="lewm")
def run(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        if "directional" not in cfg.loss:
            cfg.loss.directional = {
                "top_k": 32,
                "dyn_weight": 0.0,
                "controllability_weight": 0.0,
                "reach_weight": 0.0,
                "nuisance_weight": 0.0,
                "spectral_pred_weight": 0.0,
                "residual_band_weight": 0.0,
                "contact_weight": 0.0,
                "teacher_anchor_weight": 0.0,
                "teacher_policy": None,
            }

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    world_model = hydra.utils.instantiate(cfg.model)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }
    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(directional_forward, cfg=cfg),
        optim=optimizers,
    )
    teacher_policy = cfg.loss.directional.get("teacher_policy")
    if teacher_policy:
        world_model.teacher_model = _load_teacher_model(teacher_policy, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    checkpoint_cfg = cfg.get("checkpoint", {})
    checkpoint_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=checkpoint_cfg.get("epoch_interval", 0),
        save_final=checkpoint_cfg.get("save_final", True),
        final_filename=checkpoint_cfg.get("final_filename", "weights.pt"),
        keep_last_n=checkpoint_cfg.get("keep_last_n", 1),
        epoch_dir=checkpoint_cfg.get("epoch_dir", "epoch_snapshots"),
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=checkpoint_cfg.get("enable_lightning_checkpointing", True),
    )
    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == "__main__":
    run()
