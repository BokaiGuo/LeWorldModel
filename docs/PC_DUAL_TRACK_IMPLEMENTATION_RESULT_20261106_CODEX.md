# 双轨输入权限：首轮实现与 GPU 验证

## 实现范围

新增 `le-wm-main/experiments/pc_input_permissions.py`：

- 显式 `primary_rgb` 与 `official_state_assisted`；缺失或未知轨道拒绝。
- LeWM RGB 入口仅允许 pixels、goal、action；PointMaze JEPA 入口必须提供 visual、proprio，不自动填充。
- PointMaze 字段为 x,y,vx,vy；检查加载模型的 proprio 路径，绑定本地 checkpoint、配置、normalizer 来源、预处理和时钟说明。
- 模型须 eval 且参数全部冻结；封装不禁止动作梯度，单元测试验证动作反传。
- 逐调用收据记录轨道、输入审计摘要、字段、形状、dtype、张量哈希。调用不向原始输入字典添加预测字段。
- 缓存键包含轨道、算法版本和输入审计。

新增 `pc_track_registry.py`：证据入口核对轨道、算法、协议、任务与实际输入收据；拒绝失败/部分运行、跨轨道引用和接口结果自行声称科学通过。建立两份独立 PC0–PC5 未评估记录，不改动历史 gate。

## 真实 GPU 结果

| 轨道 | 模型 / 条件 | 封装调用数 | 与直接调用最大差 | 权重前后 |
|---|---|---:|---:|---|
| primary_rgb | LeWM PushT，根0左侧白背景，三条预定链 | 3 | 0 | 指纹一致 |
| official_state_assisted | PointMaze JEPA，既有第一个状态，三动作×两相机 | 6 | 0 | 指纹一致 |

RGB 三个代价分别为 171.654694、189.430450、144.605682，仅为接口一致性记录，不是控制成绩。
JEPA 预测张量为 `[2,1,1,16,16,384]`（上下文及一个预测步），均有限。
两轨由独立 Python 进程实际使用 CUDA 执行，结果分目录保存。没有把 State-assisted 改称 RGB-only。

## 排错记录

1. 沙箱内 CUDA 不可访问；申请沙箱外 GPU 权限后正常运行，没有 CPU 回退。
2. 初版 State-assisted 入口仅接受 DINO 仓库名，而现有兼容加载器已传入本地路径，白名单拒绝。新增独立的 `pc_state_track_smoke_20261106_codex.py`，只允许该仓库名或解析到预定本地目录的路径；原 RGB 封存入口保持不变。
3. 原 `pc_dual_track_smoke_20261106_codex --track official_state_assisted` 保留为失败历史入口；实际通过的是新 State-assisted 入口，不能继续使用旧命令重现成功。

## 验证

权限、证据登记、接触几何、相机审计与动作时钟共 **34 passed**。
新增测试覆盖未知轨道、RGB 注入额外字段、缺失历史/proprio、错误模型轨道、未冻结权重、动作梯度、调用字典不变、依赖漂移、缓存分隔、跨轨道/任务误用、未完成结果及伪造科学通过。
单元测试包含合成模型和日志，仅为工程验证。

实际产物另由 `pc_dual_track_receipt_audit_20261106_codex` 重新读取并验证，不只依赖测试夹具。审计输出在同一结果目录的 `receipt_audit.json`。

## 复现入口

工作目录：`/home/gbk/code/LeWorldModel/le-wm-main`。

```bash
env SDL_VIDEODRIVER=dummy .venv/bin/python -m experiments.pc_dual_track_smoke_20261106_codex --track primary_rgb
env SDL_VIDEODRIVER=dummy .venv/bin/python -m experiments.pc_state_track_smoke_20261106_codex
.venv/bin/python -m experiments.pc_dual_track_receipt_audit_20261106_codex
```

已存在结果时拒绝覆盖；复跑须新建版本化运行入口/目录，不能删除历史来重复执行上述命令。

产物根：`le-wm-main/results/pc_dual_track_smoke_20261106_codex/`，各轨道分别包含 `result.json`。

## 未完成边界

这是新入口的 API 权限封装、依赖绑定与 GPU 数值保持，不是所有历史规划器的迁移，更不是操作系统级不可绕过沙箱。完整评价进程隔离、字段来源/时间逐步认证、真实动作历史闭环，以及共享算法在双轨中的描述—保留—组合消融仍未完成。

RGB smoke 显式沿用初始空历史占位，没有新增 proprio；但它不证明多帧真实动作历史正确。PointMaze 使用本地 DINO/JEPA 兼容路径，加载权重不变不等于官方实现等价认证。State-assisted 权限也不等于可在规划阶段查询完美模拟器。

原始源码/配置及大数据源哈希计算带来额外 I/O；本轮未测质量—成本优势。没有证明 PC1 行动描述增量、PC2 机制或 PC3–PC5 的性能、闭环和泛化通过。
