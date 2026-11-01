# PushT sampling comparison v2：整 stack 身份与 live RGB guard

v1 的逐帧 hash 列表与上游整 stack 单 SHA 不相等，是实质执行错误；且 v1
没有 live RGB guard。**v1 不应作为可运行物理结论使用。** v1 源码、文档、
prepared 目录保持封存原样，无需也不得修改；本轮未启动 v1 或任何 GPU 推断。

v2 仅复用 v1 的优化器、CUDA scorer 和 recurrence；不复用 v1 physical execute。
独立 v2 prepare/plan/execute 与测试/文档全部进入新源码 seal；物理 executor、
guard、labels 也在 GPU 启动前封存，实际 installed eval_state 源码在物理前另 seal。
任何当前源文件变化仍会 fail closed，不刷新旧 seal。

## 修复和固定范围

- 用 `sha256(np.stack(frames).tobytes())` 比较真实 `history_rgb_sha256` 单字符串，
  同时验证三帧 uint8 RGB 尺寸、实际 goal hash；不接受逐帧 SHA 列表。
- 与主线程 P2 共用 `pusht_visible_command_executor.execute`，每 primitive 观察
  RoyalBlue agent，消失立即中止；不能防止第一次出界，不会盲走余下动作。
- 8 分支全部保留，包括 initial-invisible、途中消失、environment termination、
  execution failure；无 retry/padding、无重选择。recorded 不等于全部执行完毕。
  报告明确列 full-horizon completed 与 aborted/failed 数。
- 每观测步与终态保存官方 labels、object pose、state、RGB hash；每步另存 command、
  reward、terminated/truncated；官方 success/distance 与 installed eval_state 独立比对。
  初态也保存 labels/pose。标签只作 evaluator bookkeeping，不进入 guard/规划。
- V2 物理阶段无 neural load/encode，不做 P2 的 terminal latent 重编码；因此只共用
  相同执行/中止 protocol，不冒充与 P2 相同的全套成本或观测输出。保存 RGB hash，
  不存视频或图像堆。执行收据的计数必须匹配实际 callback trace 与封存动作前缀，
  不能用“completed”字符串或空 trace 伪通过。

规划固定协议沿用 v1：4 已知 calibration roots、goal0/Q、emb3 + between2，
无 bootstrap，2 token 各重复5步，参数 [-3,3] 经 float32 tanh。
CEM=零候选+4×64，elite8，std floor .1，ddof0；random=零候选+256 uniform parameter
box；seed 202610270–273。每方法每根 257 候选；规划共 272 calls/4112 samples，
独立 selected-plan replay 另16 calls/samples。逐候选成本/失败、独立 predict 计数、
同步完整 optimizer wall time（含采样/搬运/Q/elite/log）与 peak CUDA memory 均保留。
不训练，不放宽门槛。Q 的 agent-motion/static-object 假象尚未解决，仍只是接口/
成本校准；P1/P2/P3 不准入，不宣称 task efficacy、官方 MPPI/GRASP 或等总算力。

## 主线程命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，必须使用 `.venv/bin/python`。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main

# CPU-only prepare；若新 v2 目录已由本次任务生成，不要重复执行。
.venv/bin/python -m experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 prepare \
  --output results/pusht_sampling_comparison_20261027_v2_prepared

# 唯一需要主机 CUDA 的步骤；由主线程启动。
.venv/bin/python -m experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 plan \
  --run results/pusht_sampling_comparison_20261027_v2_prepared

# 全部8份 v2 plan 封存成功后：CPU physics + shared liveRGBguard，无模型分配。
.venv/bin/python -m experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 execute \
  --run results/pusht_sampling_comparison_20261027_v2_prepared \
  --output results/pusht_sampling_comparison_20261027_v2_physical

.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_pusht_sampling_comparison_20261027_v2.py \
  tests/test_pusht_sampling_comparison_20261027.py \
  tests/test_pusht_visible_command_executor.py
```

旧输出不可覆盖。若 seal 被多人并行改动导致拒绝，不能篡改 seal 或继续旧目录，
须重新审核并在全新带 v2 tag 的目录 prepare。

资源仍为未实测粗估：RTX5060Ti16GB、microbatch16、frozen no-grad，规划显存
1–4GiB、1–10分钟；物理几十秒至几分钟，无 CUDA。新增文件/结果预计 <30MiB，
不复制 checkpoint/大 compilation。CPU 测试不是神经实验或效果证据。
