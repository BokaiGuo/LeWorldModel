# P2 程序组合：同协议、同计量资源 CUDA 对照

日期：2026-11-01  
范围：P2 开发性已知根；不修改 P1；不训练冻结模型；不使用 CPU 神经替代。

## 结论

本轮补上了此前 P2 对照最具体的工程缺口：组合臂与无组合 joint 谱规划现在
使用同一输入、同一目标、同一候选数、同一选择器，并在八个已计量资源轴上
逐项相等。四个开发 slot 的主机 CUDA 运行和 saved-only 复算均完成。

组合臂在四个 slot 的模型终态代价都低于 joint 臂。新 winner 动作与
2026-10-30 已封存物理实验中的动作逐字节相同，因此复用了八条既有物理轨迹，
没有重复消耗物理预算。四个 slot 上，组合臂的物体位置误差、角度误差和
native distance 都低于 joint；但八条轨迹原生成功仍为 0，四个 slot 实际只
来自两个已知根，其中 base 2 与 base 3 是相同任务、相同动作和相同结果。

因此：

- 已建立“同协议、同计量资源、saved-only 可复算”的模型侧对照；
- 已建立新等资源 planner 到既有真实轨迹的逐字节动作 lineage；
- 这是组合机制的正向开发证据；
- `P1_admission=false`、`P2_admission=false`、`scientific_admission=false`
  保持不变。

## 为什么旧对照不够

此前 `joint_multistart` 与 `prefix_chain` 都有 16 个候选、5 次 Jacobian、
5 次 SVD 和 17 次评分，但模型 packet 数分别是 44 和 28。原因是组合臂把
每个前缀的预测结果缓存后，四个后缀候选共享该前缀；joint 臂则必须对每个
完整十步候选执行两个 packet。

本轮没有用空调用补齐预算。组合臂改为：

1. 每个前缀执行一次，用于计算该前缀后的局部 Jacobian；
2. 每个候选再次从原始上下文执行完整前缀；
3. 再执行该候选的后缀；
4. 对 winner 从原始上下文完整重放。

因此每个组合候选也是一个完整十步程序评估，与 joint 候选的执行语义一致。

## 固定资源向量

每个 case、每个 arm 的实际 attempted 与 returned 均严格等于：

| 资源轴 | joint | composition |
|---|---:|---:|
| packet | 44 | 44 |
| action encoder | 44 | 44 |
| predict | 44 | 44 |
| predictor | 44 | 44 |
| prediction projection | 44 | 44 |
| Jacobian | 5 | 5 |
| SVD | 5 | 5 |
| score | 17 | 17 |
| 候选数 | 16 | 16 |

这证明上述计量轴相同，不证明总 FLOPs、能耗、I/O、模型加载成本或全部计算
完全相等。实际 wall time 也不被强制相等；两臂接受同一 450 秒上限并报告
真实消耗。

## 主机 CUDA 结果

设备：NVIDIA GeForce RTX 5060 Ti  
GPU UUID：`GPU-4e8973ff-0278-53f3-2bae-2615d461e441`  
完整 CUDA 作业 wall time：12.1180 秒。

| base | joint 模型代价 | composition 模型代价 | 模型偏好 |
|---:|---:|---:|---|
| 0 | 0.870782 | 0.788026 | composition |
| 1 | 0.672910 | 0.614643 | composition |
| 2 | 0.523988 | 0.508096 | composition |
| 3 | 0.523988 | 0.508096 | composition |

base 2 与 base 3 的来源、动作和结果重复，不能作为两个独立样本。

第一次启动写入
`results/p2_matched_program_host_20261101_parallel_RUN_01`，在模型初始化前因
另一个 P3 作业持有 cooperative lock 而 fail-closed。该失败目录保留且未复用。
锁释放后使用新的 `RUN_02` 完成执行。

## 物理结果复用

新等资源 planner 的八条 winner 动作，与
`codex_p2_multistart_joint_host_20261030_v1_RUN_01` 的八条 winner 完全相同。
saved consumer 进一步绑定了旧 planner receipt、四份物理 job receipt 与 joined
physical receipt。没有新物理调用。

负数表示 composition 优于 joint：

| base | 物体位置误差差值 | 角度误差差值 | native distance 差值 | 原生成功 |
|---:|---:|---:|---:|---:|
| 0 | -65.8275 | -0.8206 | -12.4907 | 两臂均否 |
| 1 | -40.9519 | -0.3493 | -46.0679 | 两臂均否 |
| 2 | -19.1759 | -1.1311 | -64.8624 | 两臂均否 |
| 3 | -19.1759 | -1.1311 | -64.8624 | 两臂均否 |

这些是已知开发根上的描述性结果。0/8 原生成功意味着不能宣称组合方法已解决
任务，也不能仅凭连续误差更低打开 P2。

## 新增文件

- `le-wm-main/experiments/spectral_reachability/p2_matched_program_kernel_20261101_parallel.py`
- `le-wm-main/experiments/spectral_reachability/p2_matched_program_host_20261101_parallel.py`
- `le-wm-main/experiments/spectral_reachability/p2_matched_program_saved_20261101_parallel.py`
- `le-wm-main/experiments/spectral_reachability/p2_matched_program_physical_reuse_20261101_parallel.py`
- `le-wm-main/tests/test_p2_matched_program_20261101_parallel.py`
- `le-wm-main/tests/test_p2_matched_program_physical_reuse_20261101_parallel.py`

## 封存证据

| artifact | SHA-256 |
|---|---|
| prepare | `d7a1a93d6fdf208b1a965c9632dfabf081f3b0035676c0232d31766eb711adbe` |
| CUDA report | `3db63a0eadd898283222774431157676bea698f7754ca22f0329663fb013b749` |
| CUDA child journal | `6aac50313d4ebf662f5506fc84fdba91b484d993bd9b54d12d2ce7caffdb9f3a` |
| CUDA supervisor journal | `52134e3f65e13c3b29c983ca2a8a08556c8ea429ed2b27aab11079713f8c1d35` |
| saved CUDA receipt | `a364af8c5f632c8da564ef6c4389dc51764d76078124f788c881acb76323c6cd` |
| saved physical-reuse receipt | `39ad4cc27e51ea42db49e940924166a3bd8e0d9be9820ea61375c8b0c9015ca4` |

## 验证命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main

.venv/bin/pytest -q -p no:cacheprovider \
  tests/test_p2_matched_program_20261101_parallel.py \
  tests/test_p2_matched_program_physical_reuse_20261101_parallel.py \
  tests/test_codex_p2_multistart_joint_kernel_20261030_v1.py \
  tests/test_codex_p2_multistart_joint_host_20261030_v1.py
```

CUDA prepare、run 与 saved consumer 的参数和 SHA 已由上述 artifacts 固定。
所有输出使用新路径，存在即失败，不覆盖任何 sealed artifact。

## 仍缺前置

1. P1 尚未通过，按阶段合同不能把本结果晋升为 P2 科学通过。
2. 当前只有两个已知开发根，且四个 slot 不是四个独立任务。
3. 八条真实轨迹均未达到原生成功。
4. 总 FLOPs、能耗、I/O 与全生命周期总计算仍未测得，故只声明计量轴匹配。
5. 需要在方法冻结后，使用来源可审计、未参与开发的新根执行同一合同。
6. 独立确认必须包含完整物理执行、成功率与预先固定统计判据；不能按结果挑根
   或追加到显著。
