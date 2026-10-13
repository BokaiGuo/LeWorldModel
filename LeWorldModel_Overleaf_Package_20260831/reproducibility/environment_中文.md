# 运行环境说明

复现分成两个难度层级。

## A. 理论结果和已有 JSON 检查

只查看结果 JSON 不需要安装额外软件。重新运行理论矩阵实验通常需要：

- Python 3.10 或更高版本
- NumPy
- SciPy
- Matplotlib

理论脚本计算量小，不需要 GPU。

## B. I-JEPA / JEPA-WM / DINO-WM 测量

这部分还需要：

- PyTorch、TorchVision
- Transformers、Datasets
- scikit-learn
- Pillow、PyYAML
- 官方 `facebookresearch/jepa-wms` 代码
- DINOv2 encoder 依赖
- 对应 checkpoint 和 trajectory archive

本次新增的 pixel-to-latent 合成基准在 CPU 上运行。复现实验环境中的
PyTorch `2.12.0+cu130` 无法初始化 CUDA/NVML：
`torch.cuda.is_available()` 为 `False`，可见设备数为 0。因此该基准的 JSON
明确记录 `device: cpu`；本包没有把它表述为 GPU 性能或真实环境训练结果。

I-JEPA ViT-H/16 编码和 world-model rollout 的内存、磁盘与运行时间明显高于理论实验。建议先阅读已有 JSON，只有需要重新生成数字时再准备模型与数据。

## 路径约定

复制出的 `reproducibility/code/` 是论文快照。脚本中的默认路径仍按原始仓库布局：

```text
experiments/theory_spectral_transfer/
results/
data/
external/
```

因此，最稳妥的重跑方式是把这些脚本放回原 LeWorldModel checkout 的相同位置，或在命令行中显式传入 `--dataset-root`、`--checkpoint` 和 `--output`。
