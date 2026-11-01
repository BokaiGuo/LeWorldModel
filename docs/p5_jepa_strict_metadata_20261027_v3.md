# P5 JEPA restricted meta 检查 v3（2026-10-27）

最终报告：`le-wm-main/results/p5_jepa_strict_metadata_20261027_v3/report_native_interfaces.json`。
三份本地 SHA 绑定 checkpoint 均成功通过显式
`torch.load(weights_only=True, map_location='meta', mmap=True)`，运行于 `FakeTensorMode`。
未添加任何 safe globals，未切换 weights_only=False，未读取 dataset、构造原生模型或执行神经前向。
文件哈希和内存映射不是零文件 I/O；没有实际张量 storage 分配，所有返回张量均检查为 meta。
报告逐层保存真实 nested keys（含 optimizer/scaler/stats）、每个张量 shape/dtype/stride/device，
每份共 376 个张量记录；这已不只是 opcode 字面量线索。

## 精确匹配结果与具体跨任务错配

| 任务 | predictor | proprio encoder | missing / unexpected / shape / dtype |
| --- | --- | --- | --- |
| PushT | 92 键，float32 | 2 键，weight [16,4,1] | 原生任务配置下全部为空 |
| PointMaze | 92 键，float32 | 2 键，weight [16,4,1] | 原生任务配置下全部为空 |
| Wall | 92 键，float32 | 2 键，weight [16,2,1] | 原生任务配置下全部为空 |

仅去掉一次开头 `module.`，拒绝名称碰撞，不进行其他重命名或宽松装载。
源码推导使用 AdaLN_vit.py、modules.py、prop_embedding.py、utils.py 及三份对应 YAML；
原生 proprio_dim 来自三个既有 temporal evaluator：PushT/PointMaze=4、Wall=2。
这些路径及当前 SHA 在最终报告中保存，不把 commit ID 当当前修改源码的完整身份。

确切形状：视觉输入投影 [384,384]，动作 encoder [400,10]；
block norm [400]、qkv [1200,400]、attention projection [400,400]、
MLP [1600,400] / [400,1600]、AdaLN modulation [2400,400]；
最终 norm [400]，输出视觉投影 [384,384]，六层 block。
384+16=400 是 block 工作宽度，不能误把所有矩阵都当 384 维。

首次共同使用 PushT 的 proprio_dim=4 时，Wall 精确报告
`proprio_encoder.patch_embed.weight: actual=[16,2,1], expected=[16,4,1]`。
首次结果原样保留于同目录 `report.json`，不覆盖；最终报告按封存 Wall evaluator
的原生二维接口核对后精确匹配，同时保留 `cross_task_pusht_proprio_comparison` 错配字段。
这不是 Wall 权重损坏，也不是放宽匹配门，而是明确禁止跨任务硬套四维 proprio。

## 给主线程的 CUDA-only strict native dynamics 装载命令（未运行）

在 `/home/gbk/code/LeWorldModel/le-wm-main` 执行：

```bash
.venv/bin/python experiments/spectral_reachability/p5_jepa_strict_metadata_20261027_v3.py --validate-cuda --task pusht --metadata-report results/p5_jepa_strict_metadata_20261027_v3/report_native_interfaces.json --output results/p5_jepa_strict_metadata_20261027_v3/cuda_strict_pusht.json
```

可分别将 `--task` 改为 `pointmaze` 或 `wall`，使用各自全新输出路径。
该分支先验证 inspector/原生结构源码/配置/evaluator 字节 seals 和 checkpoint SHA，
CUDA 不可用则拒绝；只在 CUDA 构造原生 AdaLN predictor 和 proprio encoder，
显式 restricted CUDA checkpoint load，再逐键检查 shape/dtype 并执行 `strict=True`。
没有 encoder、dataset、forward 或 CPU 神经 fallback；不调用原有宽松 `init_module`。
输出 exclusive-create，拒绝覆盖。此命令是已实现但尚未运行的严格装载检查，不是预测验证。
CUDA 构造依赖、初始化或导入失败仍可能出现；不将源码推导匹配当作已执行 native module
state_dict 验证。主线程运行后才可升级为实际严格装载成功。

本次不能认证权重数值有限性、动作敏感性、HF encoder 等价性、RGB/proprio/动作
预处理、完整 native runtime 或 Full-D2P；`P5_admission=false` 与 `D2P_ready=false`。
checkpoint 的 opt/scaler/stats 也只检查元数据，不恢复训练。

## 实际非神经验证

```bash
.venv/bin/python -m pytest -q tests/test_p5_jepa_strict_metadata_20261027_v3.py
.venv/bin/python experiments/spectral_reachability/p5_jepa_strict_metadata_20261027_v3.py --output results/p5_jepa_strict_metadata_20261027_v3/report_native_interfaces.json
```

11 passed（1.26 秒）。覆盖显式 restricted/meta/mmap 参数、meta storage pointer=0、
SHA 错配加载前拒绝、真实 numpy global 的 weights_only 拒绝不降级、准确缺失/意外/形状/
dtype 差异、前缀碰撞、非 meta 拒绝、三任务源码形状清单及无不安全 allowlist/forward 调用。
真实三份权重均未触发 numpy rejection；测试中的 numpy fixture 实际触发并保留拒绝。
未运行 CUDA 模式。experiment-code 技能仅用于记录和可复算边界；未训练或生成效果图。
所有本轮之前的封存源码、测试、文档、报告保持不变。
