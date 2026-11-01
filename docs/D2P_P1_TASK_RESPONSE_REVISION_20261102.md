# P1：后缀覆盖与任务评分的开发对照

## 本轮结论

已从“新谱前缀没有测到有用物体响应”推进到一个已知开发根上可重复的
有用响应差异。仍未通过 P1：前缀具有即时 native-position 进展；所有
分支原生成功为 false；没有独立根、完整计算成本认证或强基线确认。

此次使用 code-debugging 流程复算动作和评价链，保留旧封存源码及旧结果，
仅新增开发协议。技能附带 debug-patterns 参考文件不存在，采用项目已有
源码与原始轨迹直接复核，没有将缺失参考当成停止实验的理由。

## 具体问题与修改

1. 复算上轮 cp5 后缀发现，latent-selected 确实推动物体，但位置误差从
   236.6382 增至 253.4010；goal-tangent 后缀几乎没有物体位移。
   因而不能把全部失败解释为“没有接触”。
2. 原 CEM 从零均值逐步独立扰动出发，没有显式覆盖相关的连续动作。
   新分支在原 64 个候选名额内加入 13 个固定种子：zero，以及两种幅度、
   六种第二包转向的 approach/turn/reverse 序列。不增加模型调用数。
3. 另开物体／推杆坐标分离的目标切空间评分对照，权重固定为 1/.05。
   这是探索性目标设计，不是物理姿态估计，也不改变正式任务成功指标。
   其结果未稳定改善，因此不升级为正式方法。

新模块位于 `le-wm-main/experiments/spectral_reachability/`：

- `p1_task_response_revision.py`：候选和评分修改。
- `p1_task_response_experiment.py`：prepare/plan/physics 16 臂实验。
- `p1_task_response_saved.py`：独立保存结果复算。
- `p1_task_response_intervention.py`：4 前缀 × 5 共同后缀干预。

`tests/test_p1_task_response_revision.py` 包含旧 V13 精确复现、相同评分调用
数、坐标分离、输入异常和指标分栏测试；联合回归为 43 passed。

## 16 臂 factorial

四前缀 selected/zero/greedy/random × 原/相关种子 × latent/物体切空间评分。
每臂 5 轮、64 候选、5 扰动、两包预测；hook 验证每臂 10 次 predictor
和 10 次 action_encoder，各 3200 batch samples。总预测样本 51200。
这是相同记录模型预算，不是完整计算等价认证。

协议在本轮规划和物理执行前封存：
`le-wm-main/results/p1_task_response_factorial_20261102_01/protocol.json`

SHA256：`849e57a6a74b8c83cd368c6761b23395c61df5fae522fee457734bd6eb6ab3c7`

真实模拟器全部完成 5 步前缀和 10 步后缀，16 臂均无原生成功。
保存消费器验证 240 张图片、动作轨迹、原生 reward 与保存状态一致。

### 原 latent 评分下的终态

| 前缀 | 后缀搜索 | 物体位置误差 | wrapped 角度误差 | native reward |
|---|---|---:|---:|---:|
| selected | 原 CEM | 253.401 | 3.003 | -303.280 |
| selected | 相关种子 | 215.255 | 1.604 | -304.861 |
| zero | 相关种子 | 236.638 | 3.051 | -366.865 |
| greedy | 原 CEM | 246.341 | 2.783 | -291.535 |
| greedy | 相关种子 | 238.856 | 2.333 | -309.395 |
| random | 相关种子 | 245.870 | 3.132 | -447.129 |

selected 相关种子改善物体误差，并优于同配置 greedy，但 native reward
仍不如原 CEM greedy。不能删掉这个较强的旧对照。

## 共同后缀干预

在看到 factorial 开发结果后另封存，仍属开发性机制诊断：
四个 latent-coherent 各自选出的后缀，加 zero-tail，全部接到四前缀后。
未按结果挑某条尾段替代正式 planner 输出。

协议 SHA256：
`736a79ee9f6527aa5fb085afe8e94e8696c262f3029ab39382fa79c2b73bec20`

报告：
`le-wm-main/results/p1_task_response_intervention_20261102_01/physics/report.json`

报告 SHA256：
`b7a68716984e988eb7426f43707052a0f3e66982303a9390ab0fc4eda11ee1f6`

20 条轨迹、300 张 RGB 文件及 decoded pixels、动作和 relative-zero 数值
已复核；均执行满尾段 10 步，无原生成功。

同一个 `01_latent_coherent` 后缀的绝对终态：

| 前缀 | 物体位置误差 | wrapped 角度误差 |
|---|---:|---:|
| selected | 215.255 | 1.604 |
| zero | 236.638 | 3.051 |
| greedy | 234.497 | 2.238 |
| random | 236.638 | 3.051 |

相对 selected 自己的 zero-tail：位置改善 21.383，角度改善 1.447。
同一后缀在 zero 后没有有效物体变化。这支持有限后缀族下有用响应改变，
不支持“对所有动作的可规划能力更强”或独立泛化。

greedy 自己生成的另一后缀接在 selected 后，物体位置误差达到 177.982；
这只是交叉干预诊断，不得事后指定为 selected 的正式动作。它提示后缀
覆盖／选择仍有改进空间，也不能把其他反事实分支的真实观测输入部署规划器。

## 尚未解决的 P1 条件

- selected 前缀使 native-position 误差下降约 88.048、推杆位置误差下降
  135.413。物体几乎不动不能证明整个任务没有即时进展。
- 零后缀的被动运动已单列，但全阶段非贪心机制合取尚未满足。
- 本轮所有分支均未达到原生成功。
- 来源独立性、训练数据归属、完整成本与强规划基线确认尚未完成。

下一次方法修改应利用相关动作搜索的有效证据，处理前缀生成和多步真实
反馈中的任务进展／重配置区分；不采用失败的物体加权评分，不改 gate，
不把这一个开发根升格为确认样本。
