# 原始数据与模型来源清单

本清单说明结果级数据包之外的原始输入从哪里来。大文件不复制进 Overleaf 包，但每个文件都给出工作区路径和 SHA-256，方便合作者核对。

| 输入 | 工作区路径 | 用途 | SHA-256 |
|---|---|---|---|
| DINO-WM PointMaze archive | `data/dino_wm_osf/point_maze.zip` | 2,000 episode PointMaze 轨迹、状态、动作和观测 | `6c48ccf22c90b9af8dcf0e2cd70849aec8dd8e214ac5f1f09552bf8bc9494acc` |
| DINO-WM outputs archive | `data/dino_wm_osf/outputs.zip` | DINO-WM 发布 checkpoint 和配置 | `425b0d8c2a4194d3ec996b553575f69409aa5714176b063d8401789db9179482` |
| DINO-WM PushT archive | `data/jepa_wm_osf/pusht_noise.zip` | 18,685 train / 21 validation PushT trajectories | `442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08` |
| JEPA-WM PointMaze checkpoint | `data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar` | epoch-50 PointMaze predictor | `a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc` |
| JEPA-WM PushT checkpoint | `data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar` | epoch-50 PushT predictor | `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb` |
| I-JEPA ViT-H/16 checkpoint | `external/ijepa_vith16_1k_hf/model.safetensors` | 冻结 image encoder | `66512006bee119d7705941a51ab9afb28f993b12bb8d7fcc47f24a34c632cbd9` |

## 文件大小

- PointMaze archive：718,363,945 bytes。
- DINO-WM outputs archive：953,204,628 bytes。
- PushT archive：2,785,304,515 bytes。
- 每个 JEPA-WM checkpoint：211,639,615 bytes。
- I-JEPA ViT-H/16 checkpoint：2,526,733,944 bytes。

## 官方代码与 checkpoint 来源

- JEPA-WM 代码：`https://github.com/facebookresearch/jepa-wms.git`，本次记录的 commit 为 `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`。
- JEPA-WM PushT checkpoint：Hugging Face `facebook/jepa-wms`。
- I-JEPA ViT-H/16 checkpoint：Hugging Face `facebook/ijepa_vith16_1k`。

## split 说明

- DINO-WM PushT：发布目录直接分成 `train/` 和 `val/`，因此验证集是显式目录 split。
- DINO-WM PointMaze：发布 archive 没有单独 membership 文件；使用发布 loader 的 `split_ratio=0.9`、`torch.randperm`、seed 42 和原始 episode 顺序重建 1,800/200 membership。重建结果见 `results/temporal_world_models/dino_wm_pointmaze_split_manifest_reconstructed.json`。
- JEPA-WM PointMaze：使用同一 public archive 和 seed-42 重建 membership，以便做 matched-environment comparison。

## 下载与许可

下载大文件时请遵守原始项目的许可证和数据集条款。推荐把原始数据放在工作区的 `data/` 和 `external/` 路径，再运行 `reproducibility/code/` 中的评估脚本；不要把受限模型权重重新上传到 Overleaf 项目。
