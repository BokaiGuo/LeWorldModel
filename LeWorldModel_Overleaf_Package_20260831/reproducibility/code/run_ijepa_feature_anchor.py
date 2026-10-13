"""Run a learned-representation spectral-transfer anchor with official I-JEPA.

The experiment treats a frozen I-JEPA image encoder as a latent-state map.  For
paired image views x -> T(x), it measures whether the feature displacement
f(T(x)) - f(x) is concentrated in the high-variance eigenspace of f(x).  This is
a learned-representation anchor for latent state spectral transfer; it is not a
world-model transition or planning evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL import ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, STL10, SVHN
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset


DEFAULT_MODEL = "facebook/ijepa_vith16_1k"
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


@dataclass(frozen=True)
class AnchorConfig:
    model: str
    dataset: str
    split: str
    max_images: int
    batch_size: int
    top_k: int
    seed: int
    view_transform: str
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dataset",
        choices=(
            "cifar10",
            "cifar100",
            "stl10",
            "svhn",
            "tiny-imagenet",
            "food101",
            "beans",
            "sklearn-digits",
        ),
        default="cifar10",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/theory_operator_alignment_iclr_oral_20260829/jepa_anchor"),
    )
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--view-transform",
        choices=("hflip", "color", "crop"),
        default="hflip",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-draws", type=int, default=256)
    return parser.parse_args()


class PairedCifarViews(Dataset):
    def __init__(self, base: Dataset, view_transform: str):
        self.base = base
        self.view_transform = view_transform

    def __len__(self) -> int:
        return len(self.base)

    def _target_view(self, image):
        if self.view_transform == "hflip":
            return ImageOps.mirror(image)
        if self.view_transform == "color":
            return ImageEnhance.Color(image).enhance(0.35)
        if self.view_transform == "crop":
            w, h = image.size
            box = (max(0, int(0.08 * w)), 0, w, max(1, int(0.92 * h)))
            return image.crop(box).resize((w, h))
        raise ValueError(f"unknown transform: {self.view_transform}")

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return image, self._target_view(image), label


class SklearnDigitsImages(Dataset):
    def __init__(self):
        from sklearn.datasets import load_digits

        digits = load_digits()
        self.images = digits.images.astype(np.float32)
        self.labels = digits.target.astype(np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        image = self.images[index]
        image = np.clip(image / 16.0 * 255.0, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image, mode="L").convert("RGB")
        return pil_image, int(self.labels[index])


class HFCifar100Test(Dataset):
    def __init__(self):
        self.dataset = load_dataset("uoft-cs/cifar100", split="test", trust_remote_code=True)
        self.targets = list(self.dataset["fine_label"])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        image = row["img"].convert("RGB")
        return image, int(row["fine_label"])


class HFImageDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, image_key: str, label_key: str):
        self.dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        self.image_key = image_key
        self.label_key = label_key
        self.targets = list(self.dataset[label_key])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        image = row[self.image_key].convert("RGB")
        return image, int(row[self.label_key])


class TinyImageNetVal(Dataset):
    def __init__(self, root: Path):
        self.root = Path(root) / "tiny-imagenet-200"
        self.val_root = self.root / "val"
        self.images_root = self.val_root / "images"
        self.ann_file = self.val_root / "val_annotations.txt"
        if not self.root.exists():
            self._download_and_extract(Path(root))
        self.samples: list[Path] = []
        self.targets: list[int] = []
        class_names = self._read_class_names()
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        image_to_class: dict[str, str] = {}
        for line in self.ann_file.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                image_to_class[parts[0]] = parts[1]
        for image_path in sorted(self.images_root.glob("*.JPEG")):
            cls_name = image_to_class.get(image_path.name)
            if cls_name is None:
                continue
            self.samples.append(image_path)
            self.targets.append(class_to_idx[cls_name])

    def _download_and_extract(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "tiny-imagenet-200.zip"
        if not archive.exists():
            print(f"Downloading Tiny ImageNet from {TINY_IMAGENET_URL}")
            urllib.request.urlretrieve(TINY_IMAGENET_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)

    def _read_class_names(self) -> list[str]:
        wnids = self.root / "wnids.txt"
        return [line.strip() for line in wnids.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image = Image.open(self.samples[index]).convert("RGB")
        return image, int(self.targets[index])


def collate_views(batch):
    source, target, labels = zip(*batch)
    return list(source), list(target), torch.tensor(labels, dtype=torch.long)


def balanced_subset(dataset: Dataset, max_images: int, seed: int) -> Subset:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    if hasattr(dataset, "targets"):
        targets = list(getattr(dataset, "targets"))
        for idx, label in enumerate(targets):
            by_class.setdefault(int(label), []).append(idx)
    else:
        for idx, (_, label) in enumerate(dataset):
            by_class.setdefault(int(label), []).append(idx)
    classes = sorted(by_class)
    per_class = max(1, max_images // len(classes))
    selected: list[int] = []
    for label in classes:
        indices = by_class[label][:]
        rng.shuffle(indices)
        selected.extend(indices[:per_class])
    selected = selected[:max_images]
    rng.shuffle(selected)
    return Subset(dataset, selected)


def choose_device(request: str) -> torch.device:
    if request != "auto":
        return torch.device(request)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def embed_images(
    model,
    processor,
    images: Iterable,
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(images=list(images), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        features = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
        if hidden.ndim == 3 and hidden.shape[1] > 1:
            features = hidden[:, 1:, :].mean(dim=1)
        else:
            features = hidden.reshape(hidden.shape[0], -1)
    else:
        raise RuntimeError("model output does not expose pooler_output or last_hidden_state")
    return features.detach().cpu().float()


def load_image_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset == "cifar10":
        return CIFAR10(root=str(args.data_root / "cifar10"), train=False, download=True)
    if args.dataset == "cifar100":
        try:
            return CIFAR100(root=str(args.data_root / "cifar100"), train=False, download=False)
        except Exception:
            return HFCifar100Test()
    if args.dataset == "stl10":
        return STL10(root=str(args.data_root / "stl10"), split="test", download=True)
    if args.dataset == "svhn":
        return SVHN(root=str(args.data_root / "svhn"), split="test", download=True)
    if args.dataset == "tiny-imagenet":
        return TinyImageNetVal(args.data_root)
    if args.dataset == "food101":
        return HFImageDataset("food101", "validation", "image", "label")
    if args.dataset == "beans":
        return HFImageDataset("beans", "test", "image", "labels")
    if args.dataset == "sklearn-digits":
        return SklearnDigitsImages()
    raise ValueError(f"unknown dataset: {args.dataset}")


def dataset_split_name(dataset: str) -> str:
    if dataset in {"cifar10", "cifar100", "stl10", "svhn", "beans"}:
        return "test"
    if dataset in {"food101", "tiny-imagenet"}:
        return "validation"
    return "full"


def extract_features(args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = load_image_dataset(args)
    subset = balanced_subset(dataset, args.max_images, args.seed)
    paired = PairedCifarViews(subset, args.view_transform)
    loader = DataLoader(
        paired,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_views,
    )

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval().to(device)

    source_features: list[torch.Tensor] = []
    target_features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for source, target, batch_labels in loader:
        source_features.append(embed_images(model, processor, source, device))
        target_features.append(embed_images(model, processor, target, device))
        labels.append(batch_labels)

    return (
        torch.cat(source_features).numpy(),
        torch.cat(target_features).numpy(),
        torch.cat(labels).numpy(),
    )


def pca_basis(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigvals = (singular_values**2) / max(1, centered.shape[0] - 1)
    return centered, eigvals, vt.T


def top_lift(displacement: np.ndarray, basis: np.ndarray, top_k: int) -> float:
    top_k = min(top_k, basis.shape[1])
    centered = displacement - displacement.mean(axis=0, keepdims=True)
    total = float(np.sum(centered**2))
    if total <= 0:
        return 0.0
    projected = centered @ basis[:, :top_k]
    top_energy = float(np.sum(projected**2))
    isotropic_fraction = top_k / basis.shape[0]
    return (top_energy / total) / isotropic_fraction


def random_orthonormal(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dim, k))
    q, _ = np.linalg.qr(matrix)
    return q[:, :k]


def norm_matched_random(displacement: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    norms = np.linalg.norm(displacement, axis=1, keepdims=True)
    random_vectors = rng.normal(size=displacement.shape)
    random_norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    random_norms[random_norms == 0] = 1.0
    return random_vectors / random_norms * norms


def class_preserving_permutation(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(labels.shape[0])
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        perm[idx] = shuffled
    return perm


def normalized_subspace_overlap(left: np.ndarray, right: np.ndarray, k: int) -> float:
    k = min(k, left.shape[1], right.shape[1])
    return float(np.linalg.norm(left[:, :k].T @ right[:, :k], ord="fro") ** 2 / k)


def run_anchor(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(
        "Running I-JEPA feature anchor: "
        f"dataset={args.dataset}, view={args.view_transform}, "
        f"max_images={args.max_images}, random_draws={args.random_draws}"
    )
    source, target, labels = extract_features(args, device)
    print(f"Extracted source/target features with shape {source.shape}")

    source_centered, eigvals, source_basis = pca_basis(source)
    displacement = target - source
    _, delta_eigvals, delta_basis = pca_basis(displacement)
    dim = source.shape[1]
    top_k = min(args.top_k, dim, source.shape[0] - 1)

    observed_lift = top_lift(displacement, source_basis, top_k)
    class_perm = class_preserving_permutation(labels, rng)
    pair_shuffle_lift = top_lift(target[class_perm] - source, source_basis, top_k)
    random_lifts = [
        top_lift(norm_matched_random(displacement, rng), source_basis, top_k)
        for _ in range(args.random_draws)
    ]
    print("Computed norm-matched random-direction control")

    observed_overlap = normalized_subspace_overlap(source_basis, delta_basis, top_k)
    random_overlaps = [
        normalized_subspace_overlap(source_basis, random_orthonormal(dim, top_k, rng), top_k)
        for _ in range(args.random_draws)
    ]
    print("Computed random-subspace overlap control")

    stable_rank_den = float(np.sum(eigvals**2))
    delta_stable_rank_den = float(np.sum(delta_eigvals**2))
    stable_rank = float((eigvals.sum() ** 2) / stable_rank_den) if stable_rank_den > 0 else 0.0
    delta_stable_rank = (
        float((delta_eigvals.sum() ** 2) / delta_stable_rank_den)
        if delta_stable_rank_den > 0
        else 0.0
    )

    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": asdict(
            AnchorConfig(
                model=args.model,
                dataset=args.dataset,
                split=dataset_split_name(args.dataset),
                max_images=int(source.shape[0]),
                batch_size=args.batch_size,
                top_k=top_k,
                seed=args.seed,
                view_transform=args.view_transform,
                device=str(device),
            )
        ),
        "feature_shape": list(source.shape),
        "source_stable_rank": stable_rank,
        "displacement_stable_rank": delta_stable_rank,
        "top_lift": {
            "observed_view_pair": observed_lift,
            "class_preserving_pair_shuffle": pair_shuffle_lift,
            "norm_matched_random_mean": float(np.mean(random_lifts)),
            "norm_matched_random_p05": float(np.percentile(random_lifts, 5)),
            "norm_matched_random_p95": float(np.percentile(random_lifts, 95)),
        },
        "top_subspace_overlap": {
            "observed_source_vs_displacement": observed_overlap,
            "random_subspace_mean": float(np.mean(random_overlaps)),
            "random_subspace_p05": float(np.percentile(random_overlaps, 5)),
            "random_subspace_p95": float(np.percentile(random_overlaps, 95)),
            "isotropic_expectation": top_k / dim,
        },
        "leading_eigenvalues": {
            "source": [float(x) for x in eigvals[:8]],
            "displacement": [float(x) for x in delta_eigvals[:8]],
        },
        "evidence_boundary": (
            "Frozen I-JEPA image feature anchor for covariance-relative "
            "view-transfer readouts; not a world-model transition, planning, or "
            "finite-dimensional theorem result."
        ),
    }


def make_figure(result: dict[str, object], output_dir: Path) -> None:
    metrics = result["top_lift"]
    overlaps = result["top_subspace_overlap"]
    eigvals = result["leading_eigenvalues"]
    cfg = result["config"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), constrained_layout=True)

    lift_names = ["I-JEPA pair", "Class shuffle", "Random"]
    lift_values = [
        metrics["observed_view_pair"],
        metrics["class_preserving_pair_shuffle"],
        metrics["norm_matched_random_mean"],
    ]
    axes[0].bar(lift_names, lift_values, color=["#1b4f72", "#7f8c8d", "#b7950b"])
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("(a) Top-eigenspace lift")
    axes[0].set_ylabel("lift over isotropic")
    axes[0].tick_params(axis="x", rotation=18)

    overlap_names = ["I-JEPA", "Random"]
    overlap_values = [
        overlaps["observed_source_vs_displacement"],
        overlaps["random_subspace_mean"],
    ]
    lower = overlaps["random_subspace_mean"] - overlaps["random_subspace_p05"]
    upper = overlaps["random_subspace_p95"] - overlaps["random_subspace_mean"]
    axes[1].bar(overlap_names, overlap_values, color=["#1b4f72", "#b7950b"])
    axes[1].errorbar([1], [overlap_values[1]], yerr=[[lower], [upper]], fmt="none", color="black", capsize=3)
    axes[1].axhline(overlaps["isotropic_expectation"], color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("(b) Top-subspace overlap")
    axes[1].set_ylabel("normalized overlap")

    source_eigs = np.array(eigvals["source"], dtype=float)
    delta_eigs = np.array(eigvals["displacement"], dtype=float)
    x = np.arange(1, len(source_eigs) + 1)
    axes[2].plot(x, source_eigs / max(source_eigs[0], 1e-12), marker="o", linewidth=1.5, label="source")
    axes[2].plot(x, delta_eigs / max(delta_eigs[0], 1e-12), marker="s", linewidth=1.5, label="view displacement")
    axes[2].set_title("(c) Leading spectra")
    axes[2].set_xlabel("rank")
    axes[2].set_ylabel("relative eigenvalue")
    axes[2].set_yscale("log")
    axes[2].legend(frameon=False)

    fig.suptitle(f"Frozen I-JEPA features on {cfg['dataset']} {cfg['view_transform']} pairs", y=1.05, fontsize=9)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(output_dir / f"ijepa_feature_anchor.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_anchor(args)
    path = args.output_dir / "ijepa_feature_anchor.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    make_figure(result, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
