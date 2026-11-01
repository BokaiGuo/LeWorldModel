# P5 PushT 真实数据原生 forward 交接 v4

仅新增 `p5_jepa_real_forward_20261027_v4` 命名域模块、测试、文档和准备清单；
不修改 v3 或 external。experiment-code 技能用于可复算记录与结论边界，不训练或作效果图。
本 worker 没有运行 CUDA 或任何模型 forward。

## 实际非神经准备

核验权威 v3 `report_frozen.json`、PushT 主线程实际 strict-CUDA 收据及其源文件 SHA。
三份 native dynamics 均已由主线程装载，但本 v4 只处理固定 PushT episode 0，不扩任务。
原生源代码、离线 HF shim、缓存 encoder 配置/权重、checkpoint、实际数据数组和视频
被写入 `le-wm-main/results/p5_jepa_real_forward_20261027_v4/manifest_frozen.json` 的字节绑定。
`manifest.json` 是开发期首次准备现场，保留不覆盖；运行命令必须用最终 frozen 清单。

原始数组使用显式 `torch.load(weights_only=True,map_location='cpu')`，受固定 v2 SHA 和
2 MiB 上限约束；只加载数据数组，不执行 CPU 神经计算。numpy-global 拒绝不降级，
不添加 unsafe allowlist。完全不打开 seq_lengths.pkl/shapes.pkl，不导入 native dataset。
通过 AST 读取原生 source 中四组 literal normalization constants，不执行 dataset 顶层代码。

实际 episode_000.mp4 有104帧，decord 原生优先解码器选择 RGB 帧0、5、10、15，
shape=[4,224,224,3]，uint8；视频和数组边界均检查，未读取填充区或选择替代样本。
解码像素含 shape/dtype 的 SHA256 为
`a49c9b4734efe1dc746db1cbc472f9d3965ac38d10c1c97f38440591f2c5fe0f`。

三帧 context 为0、5、10；后继真实编码 target 为15。三段 outgoing action 分别为
0:5、5:10、10:15，逐 time-major 展平为三份10维 packet，不将历史动作补零。
使用 native training normalization `(raw_relative_action/100 - ACTION_MEAN)/ACTION_STD`；
这与旧 temporal diagnostic 只除 STD 的口径不同，清单保存全部原始和归一化值。
四维 proprio 为 agent x/y 加实际 velocity，并用原生 PROPRIO_MEAN/STD 归一化；
这是明确使用特权 proprio 的工程验证，不是 RGB-only 部署。

CPU 非神经测试已在这四张真实 RGB 上配对检查原生 deterministic eval transform
和显式 full-frame bilinear resize + ImageNet normalization；max-abs 门为1e-6。
这是预处理验证，不是 encoder 神经验证。

## 主线程 CUDA 命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python experiments/spectral_reachability/p5_jepa_real_forward_20261027_v4.py --validate-cuda --manifest results/p5_jepa_real_forward_20261027_v4/manifest_frozen.json --output results/p5_jepa_real_forward_20261027_v4/pusht_real_forward_cuda_main_v1.json
```

输出 exclusive-create，已有文件拒绝覆盖；没有 CUDA 则硬拒绝，不启动 CPU 神经 fallback。
固定 SHA 绑定的 checkpoint 显式 weights_only CUDA 装载；实际原生 predictor/proprio
state_dict 先逐键、shape、dtype 比较，再 strict=True。缓存 DINOv2 权重 SHA 固定，
HF config 检查 ViT-S/14 的384维、12层、6 heads，且对实际 HF module 与 safetensors
header 做完整 key/shape 比较。强制本地 shim 路径与缓存环境、HF offline/local-files-only。

执行内容严格限于：真实四帧 encoder 两次配对调用、三帧 proprio encoder 一次、
predictor 三次（observed、完全相同输入重放、只把最后一个动作包置为归一化零）。
前两历史动作包保持不变，没有随机 hidden states、优化、反向或模拟器动作执行。
成功/失败均由固定 gates 决定，不更换 episode 或阈值：

- CUDA 预处理配对与 native/source shim patch-token 接口 max-abs ≤1e-6。
- 三路 native output 均 finite/CUDA，visual=[1,3,256,384]、proprio=[1,3,256,16]。
- 最后 context 输出对应帧15；重复调用 max-abs ≤1e-6。
- 只改变新动作包后的后继 visual max-abs >1e-7；恒同干预拒绝，不另挑样本。
- predictor、proprio、encoder 所有 state tensors 有限/CUDA，参数冻结且所有子模块 eval，
  前后完整 state 字节 SHA 一致；source/weights/data seals 前后相同。

target patch MSE 只作单条真实数据的描述性记录，不作预测效能门或 P5 结论。
若接口、形状或源码检查异常则直接失败，不降级；数值门不通过则保存 validation_failed。

## encoder 边界必须保留

本地 native `DinoEncoder` 的源码路由实际是 `external/local_dinov2/hubconf.py` 的 HF shim。
CUDA paired check 比较 native encoder.forward 的 patch tokens 与同一 HF model 的
`last_hidden_state[:,1:]`；这可以核实源码包装/shape 接口，不是两种独立官方 encoder。
记录 actual imported source 路径/SHA 与 state receipts，HF/torch 等第三方依赖不声称
已获得全源码可复现证明。尚无原始官方 DINOv2 implementation + 原始权重对照，
所以即便全部门通过，也仅为 **shim-backed real-data native predictor forward**。
`official_dinov2_equivalence_proven=false`、`P5_admission=false`、`D2P_ready=false` 始终保留。

## 非神经测试

```bash
.venv/bin/python -m pytest -q tests/test_p5_jepa_real_forward_20261027_v4.py tests/test_p5_jepa_strict_metadata_20261027_v3.py
```

首轮联合25 passed（14项v4、11项v3）；包括真实安全读取、实际 RGB 预处理配对、
动作/历史/单位对齐、坏SHA拒绝、nonfinite拒绝、AST constants不执行、CUDA缺失硬拒绝、
seal drift拒绝及无 unsafe pickle/allowlist/random hidden states/backward。
测试不调用 CUDA 验证入口或任何模型 forward。最终准备清单与联合回归完成后冻结。
