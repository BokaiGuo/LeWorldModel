# Q-consistent conditional H2：独立 saved outcome 算术审计 v1

本模块只读已完成的真实主机运行，不导入 compiler/consumer runtime，不载入模型或模拟器，不执行 SVD、神经前向、物理步或网络请求。仅新建独占审计目录；不能覆盖已有输出。

## 固定输入与接口

- 物理报告：`results/p1_goal_consistent_conditional_physical_20261027_v1_main_v1/report.json`，SHA256 `445ffe9c231bb4f132049389b9e689c01c3e6f8572eebfe2845c6445f143ebed`。
- 物理 bound prepare SHA256：`1950e191ab4dad3125b81d7380c1b23d8c9002c291dec54b0e32c0443901eb0b`。
- 继承 compiler report SHA256：`64fd87f48f73674d98b8eb2ddce7415507bf1e6d457df8e78ad91da1c15c4fe1`。
- 检查 prepare/protocol、live/archive source seal、compiler 输出哈希、物理 durable 分支与最终报告一致性、逐步 journal/RGB 哈希。所有读取有大小和路径边界，结束重新核对输入哈希。
- 全部四个 calibration root、两个 Q 臂、一次 repetition：八分支，非八个独立根。新增独立根为零。

## 独立重算

直接由每个保存的七维 state 和该分支保存的 goal 重算 pose 与官方指标，检查逐步 reward 是负的 full-state distance。pose 为 object XY 距离加 20 倍 wrapped angle；官方 position 涵盖 agent/object 前四维，distance 涵盖七维（包括速度）。官方角差保持原 evaluator 的 `min(abs(diff), 2*pi-abs(diff))`，不暗中替换为改进 wrap；两类指标不能混称。

从 PNG 独立计数 canonical agent 像素，逐步验证 guard、dispatch/return/observation 顺序、未终止且新鲜可见才 dispatch、原始动作内容及次序，禁止 clip/reselect。验证三帧 zero warmup 身份和 float32 tiled normalizer 的 token 内容，不仅检查形状。终态必须来自最后一次已返回 observation，不引用旧终点。

每个分支唯一保存的 192 维 endpoint latent 同时乘两套保存的 Q，与同一独立 goal latent 比较，重算 `mean((Q @ (z-goal))**2)`；检查 Q 对称、幂等、非零及哈希。只在同 Q 下比较 old/new 编译出的两个端点，不比较不同度量的数值大小。

完整配对要求同 root、同初始 context、两臂各完整 H2/10 steps、新鲜终点；partial 保持在 census 中且不进入完整配对均值。当前执行输入全部成功；本入口有意要求该完整成功报告及全部编码，不宣称是任意失败运行的通用恢复器。partial 分类另有反例测试。

## 计算预算

独立解析 intent/completion 事件，不只相信报告汇总：80 selected commands +120 warmup commands，另列 root collection 的 32 resets 和分支 warmup 的 8 resets；八次 encoder 加八次 existing projector，新增 predictor/action encoder 为零。

继承 planner 一次：528 predictor +528 action encoder、88 Jacobian、96 SVD。每臂 66 predictor，其中 AD forward 共 88、独立 nonlinear replay 共 192 已包含在 528 中，不能额外相加。检查每臂 10 candidate replay + winner whole/两段 replay。16896 是数学 VJP 方向数，backward engine API 次数未知，不能冒充测得调用数。历史模型、goal、context 取得成本未知，不宣称全 pipeline 等成本。

## 结果解释与缺口

固定四根的 new-minus-old pose cost 均小于零；官方 full-state distance 仅一根下降、三根上升。因此不能从 object-pose 改善推出官方任务整体改善，更不能作 held-out/P1/P3 准入。精确逐根数值与同 Q 端点差异写入新报告，不用于后验改阈值或调公式。

该审计能证明保存数组与 journal 的算术及绑定一致，不能独立证明神经输出真实性或物理可重放性：不重跑 encoder/physics，不重算模型 tensor hash，不重证训练 normalization 来源或 Jacobian/SVD 科学正确性。输出缺少原始 pre-warmup root state 与 warmup RGB，故 root/goal/frame 依赖已封存来源指纹；goal state 仅为 saved context 中的值，不能由 goal RGB 反演证明。两臂 goal/context 必须逐值一致，但不声称独立再生成 root/goal。

## 非神经运行

在 `le-wm-main` 下：

```bash
.venv/bin/python -m pytest -q tests/test_p1_goal_consistent_conditional_outcome_audit_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p1_goal_consistent_conditional_outcome_audit_20261027_v1 --output results/p1_goal_consistent_conditional_outcome_audit_20261027_v1
```

结果仅保存紧凑 JSON、此次新 source seal 和输入哈希，不复制原始帧或模型。旧 compiler/consumer/实际结果不修改。
