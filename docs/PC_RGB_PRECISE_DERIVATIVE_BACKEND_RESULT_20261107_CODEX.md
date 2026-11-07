# RGB 动作导数后端修复与回归

## 固定样本定位

针对上次未通过伴随检查的曝光样本（任务 9000、25 步、方向 0），使用相同模型、动作和投影，在 math SDPA 下进行精度对照：

| 模式 | JVP/VJP 伴随绝对差 |
| --- | ---: |
| 原 float32，cuDNN TF32 开启 | 1.92523e-4 |
| float32，关闭 cuDNN TF32 | 2.98023e-8 |
| float64 诊断副本 | 3.05312e-16 |

前两种模式下该样本的名义表示和分数逐元素相同；仅关闭 cuDNN TF32 即改变反向结果。该对照表明 cuDNN TF32 设置是此样本数值差异的关键因素，不代表所有差异均由此引起。float64 的名义表示最大元素差为 7.17e-6，分数差约 0.000993；它不是部署协议。

首次双精度运行发现上游动作编码器强制 `x.float()`，与双精度权重不兼容；该次未写出结果。诊断重试仅在内存中的双精度模型副本上，将此转换改为匹配权重 dtype；float32 下适配器与原动作编码器输出逐位一致。原安装包和权重文件未修改。三种模式各重复三次，原始输出见[精度对照](../le-wm-main/results/pc_tworoom_latent_adjoint_precision_20261107_codex.json)。

## 后端策略回归

新增局部 [导数后端上下文](../le-wm-main/experiments/pc_rgb_derivative_backend_20261107_codex.py)：使用 math SDPA，禁用 cuDNN 和 matmul TF32，并在退出或异常时恢复调用前设置。TF32 设置属于进程全局状态，不得与需要不同设置的并发计算混用；该工具尚未自动接入历史或正式规划器。

对原来 24 个已曝光方向事后重放：

- JVP/VJP 伴随检查 **24/24** 达到原开发判据，最大绝对差 **4.77e-7**；原结果的 23/24 和失败样本仍然保留。
- 三个尺度的 JVP/有限差分分别 **24/24** 达到原开发判据；最大 L2 误差分别约 **0.003075、0.001141、0.000764**。
- 与先前默认后端目标分数最大差约 **3.17e-4**。这不是对排序或端到端规划行为的完整等价认证。
- 上下文异常恢复、嵌套恢复、解析函数和动作编码器 dtype 检查等 **7 项测试通过**。

[回归原始结果](../le-wm-main/results/pc_tworoom_precise_derivative_backend_20261107_codex.json) · [精度脚本](../le-wm-main/experiments/pc_tworoom_latent_adjoint_precision_20261107_codex.py) · [回归脚本](../le-wm-main/experiments/pc_tworoom_precise_derivative_backend_20261107_codex.py)

以上是事后、已曝光样本上的数值修复与回归，不证明真实物理后果、描述增量、端到端控制优势或独立泛化；不改变 PC0–PC5 判定。
