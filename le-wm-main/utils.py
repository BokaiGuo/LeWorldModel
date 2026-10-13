from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from stable_pretraining import data as dt


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


class SaveCkptCallback(Callback):
    """Export stable-worldmodel weights with a bounded retention policy."""

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval: int = 0,
        save_final: bool = True,
        final_filename: str = "weights.pt",
        keep_last_n: int = 1,
        epoch_dir: str = "epoch_snapshots",
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = max(int(epoch_interval), 0)
        self.save_final = bool(save_final)
        self.final_filename = final_filename
        self.keep_last_n = int(keep_last_n)
        self.epoch_dir = epoch_dir.strip("/") or "epoch_snapshots"

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if not trainer.is_global_zero:
            return

        epoch = trainer.current_epoch + 1
        max_epochs = getattr(trainer, "max_epochs", None)
        is_final_epoch = (
            max_epochs is not None and max_epochs > 0 and epoch >= max_epochs
        )

        if self.epoch_interval and epoch % self.epoch_interval == 0:
            self._save(pl_module.model, f"{self.epoch_dir}/weights_epoch_{epoch}.pt")
            self._prune_epoch_checkpoints()

        if self.save_final and is_final_epoch:
            self._save(pl_module.model, self.final_filename)

    def _save(self, model, filename):
        from stable_worldmodel.wm.utils import save_pretrained

        checkpoint_path = self._checkpoint_dir() / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=filename,
        )

    def _prune_epoch_checkpoints(self):
        if self.keep_last_n < 0:
            return

        checkpoint_dir = self._checkpoint_dir() / self.epoch_dir
        epoch_files = sorted(
            checkpoint_dir.glob("weights_epoch_*.pt"),
            key=lambda path: self._epoch_from_name(path),
        )
        stale_files = epoch_files[: max(0, len(epoch_files) - self.keep_last_n)]
        for path in stale_files:
            path.unlink(missing_ok=True)

    def _checkpoint_dir(self) -> Path:
        from stable_worldmodel.data import get_cache_dir

        return Path(get_cache_dir(sub_folder="checkpoints"), self.run_name)

    @staticmethod
    def _epoch_from_name(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1
