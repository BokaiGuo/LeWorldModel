# PushT CEM / random shooting 实际运行交接

仅新增 runner 与测试，复用已有 frozen checkpoint、四根 calibration compilation、
NumPy sampling evaluator/random shooting、PushT packet normalization、独立 CUDA
packet replay、predict instrumentation、source seal 和物理 reset/execute。
没有修改既有 seal 源码。本轮未启动 GPU、未训练、未执行新物理未来。

## 固定协议

- 4 个已知 calibration 根，顺序、goal0、Q 与 emb context 全部继承已封存 compilation。
- root-anchored 三帧 latent；只取 incoming[1:] 两个 between-state action；
  每新 token append 一次，恰好两次 predict，无额外 bootstrap。
- 两 token、每 token 同一二维 live command 重复五步。参数为 [-3,3]，
  float32 tanh 后严格合法；这是既有 repeated-command family 的有界参数子域，
  不是所有可能五步动作的穷尽搜索。模型计分与执行使用同一 float32 参数映射。
- 固定种子 202610270/271/272/273，两个方法同根同 seed。
- random shooting：零候选 + 256 个参数盒均匀样本；不是 live-space 均匀。
- CEM：零候选 + 4×64 clipped diagonal Gaussian，elite=8，初始 mean=0/std=1，
  每轮含 mean，population std(ddof=0)、floor=.1，无平滑，best-ever，稳定首个平局。
- 每方法每根 257 候选，microbatch=16，实测计数应为 34 calls / 514 samples。
  额外独立 selected-plan replay 每根每方法 2 calls / 2 samples。
  全部合计 planning 272 calls / 4112 samples；verification 16 / 16。
- 目标仅为 saved goal0 的 Q-projected terminal MSE，无额外 control cost。
  不使用物理未来筛选、调整超参或选择根。

Q 当前已知可在物体静止、agent 移动时产生能力假象。这是接口/成本校准，
不是 task efficacy 修复、P1/P2/P3 准入或优势证明。不是官方 MPPI/GRASP；
相同候选/预测样本数不等于相同总算力。与旧 fixed-bank compiler 的 stage+own-V
选择目标也不同，不可把本轮 terminal-Q 结果当成等目标强基线优胜证据。

## 命令

Python cwd 必须为 `/home/gbk/code/LeWorldModel/le-wm-main`。所有输出目录必须全新。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main

# CPU-only: 完整输入验证与源码/协议/种子预封存；不分配 CUDA。
.venv/bin/python -m experiments.spectral_reachability.run_pusht_sampling_comparison_20261027 prepare \
  --output results/pusht_sampling_comparison_20261027_v1

# 主线程执行：唯一 GPU 阶段。若上一步已完成，直接执行此条。
.venv/bin/python -m experiments.spectral_reachability.run_pusht_sampling_comparison_20261027 plan \
  --run results/pusht_sampling_comparison_20261027_v1

# 仅全部 8 份 plan 封存且完整性通过后执行；物理模拟，不加载模型、不分配 CUDA。
.venv/bin/python -m experiments.spectral_reachability.run_pusht_sampling_comparison_20261027 execute \
  --run results/pusht_sampling_comparison_20261027_v1 \
  --output results/pusht_sampling_comparison_20261027_physical_v1

# 普通 CPU 逻辑测试；无神经推断。
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_pusht_sampling_comparison_20261027.py
```

若目录已存在或执行失败，不能覆盖/清理旧目录；另取 v2 新目录重新 prepare。
prepare/plan/execute 都会重查输入与已 seal 源码；多人并行有任何相关字节变化即拒绝，
不会更新旧 seal 或绕过验证。prepare 成功后请保存打印出的 prepared.json 中 seal hash。

## 成本、失败与资源

每候选 JSONL 保存 float32 参数、成本、状态、失败、所属 microbatch 实际 predict
样本数和耗时；microbatch 数字按成员重复展示，不能逐候选相加。异常批全员标记失败，
此前已完成成本保留；未提交的后续批不算已评估。任何方法失败即停止规划，不执行部分计划。

外层独立 predict wrapper 记录 planning 与 verification 的 actual calls/samples/
failed calls；action_encoder hook 另记 calls/samples。没有 encoder RGB 工作：本阶段
复用原始封存 context/goal，旧编译/encoder 成本不是免费，而是明确不在本轮新增工作内。
无 backward。完整 optimizer wall time 包含参数初始化、采样、转换、搬运、
action encoder、predict、CPU Q 评分、elite 更新和逐候选日志 I/O；同步 CUDA，
单独记录 load/hash setup、verification 之外的整体 wall time 与 peak allocated/reserved。
不将 predict samples 冒充 FLOPs 或总算力。耗时包含首轮冷启动，不做隐藏 warmup。

plan 在任何物理执行前保存全部 8 份动作见证、候选日志 hash、protocol hash 与
根/goal/context 输入绑定；plans_report 再绑定 plan hash。
execute 校验这些绑定，重建原 root/goal/实际 RGB history、比较两方法 reset context，
每物理步保留 state/reward/termination，每 token 保存 pose/RGB hash；失败分支不丢弃。
物理 pose/reward 只是校准描述，绝不回流到模型动作选择。

RTX 5060 Ti 16GB、microbatch16、frozen no-grad predictor-only：粗估显存 1–4 GiB
（含 CUDA allocator，未经本 runner 实测），应能容纳 16GB 单卡。
粗估规划 1–10 分钟，CPU prepare/hash 10–60 秒，物理 8 分支几十秒至几分钟；
没有主机 GPU 实测，不承诺吞吐。CUDA 阶段不占用 ViT 图像编码显存。
新增结果预计 <30 MiB（不复制 weights/11MiB compilation，不写视频或图像堆），
适合当前 2.5GB 空余；有实际 peak 与 timing 后以报告替换估计。

CPU tests 仅验证优化器确定性/257 候选/失败账、root recurrence 算术、既有四根
Q 成本一致性、非法参数与 CPU neural 拒绝、exclusive writer；不是 CUDA 或效果证据。
