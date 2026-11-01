# 真 simulator goal-agent catalog：额外目标 RGB 协议

仅新增同名模块、测试、本文，不改任何已封存源码/报告。不训练，不学 readout，
不调 rank/阈值，不选择有利根。正式 P1 未开放。

## reset 可行性与资格

已读 installed `stable_worldmodel/envs/pusht/env.py`：reset 先 `_set_state(goal_state)`，
再 `_set_goal_state`，再 `self._goal=self.render()`，最后恢复 `state`。
`_set_state` 包含 `space.step(self.dt)`，因此 info.goal_state 只是请求值，
不能保证图像实际物体位姿等于请求值；reset 返回的当前 state 也不是 goal-render 状态。

生成器只包裹官方 `render`，在 reset 的那次 goal render 前后捕获 `_get_obs()`，
保留真实 agent/body pose、穿透接触、RGB digest。未直接写 body 或调用私有 setstate。
渲染必须不改变状态，缓存 goal RGB 必须逐像素匹配。实际 object xy/angle 与
task goal 固定相等（浮点比较 atol=1e-9，角度圆周差）；agent xy/velocity 同样检查。
若穿透物体/墙或 agent 几何越出画布，则资格无效但保存图像与所有原因。
资格不靠蓝色像素多少、不靠未来状态、不靠分数，也不补采。

四个已知校准根由原 seed 重建并精确核对完整 root fingerprint；不重新筛根。
先核对原 goal RGB hash，再每根完整遍历16×16：image coordinates 为
round(linspace(16,208,16))，world xy=image xy×512/224。原 task object xy/angle
及零 agent velocity 不变。全部1024格保存，不合格项不编码，根不能删除。
这是固定网格碰撞审查，不声称这些 reset 目标从原起点动力学可达。

实际无神经先导探查：四根重建身份全部匹配；agent 请求(256,256)时，首个
goal-render 的状态仍可能逐项相等，但 Pymunk 明确报告 agent/block 穿透。
因此仅检查 pose 不够，必须并行检查几何；不会将这类图称作物理合格目标。

## 权限和诊断边界

生成器读取原始 task spec 中 goal object pose 和 simulator geometry，是 privileged
diagnostic。规划可获得的只是这些**额外目标 RGB**及其冻结 embedding，不是
single-RGB 部署。目录是 object-only 目标等价类，不是官方包含 agent 位置的 success。
资格/图像在读取 probe/control outcomes 前全部保存封存；结果仅离线诊断保存向量。
神经部分只允许主机 CUDA、官方 checkpoint、冻结 eval、encode，不准 predict。

目标距离是所有有效 simulator grid goal embedding 的 minMSE；不混入旧五目标，
不做 Q/SVD/rank截断。逐根保留全部五探针及全部 control branches，包括 guard。
保存物体静止标记、cost range、每个 probe 成本/最近goal索引、control pose cost，
不生成新动作或收益结论。与旧 clean/partial sprite 目录范围不同，不能将原始
cost-range 差异全归因于 augmentation artifact；如比较须先按相同 grid ID 取交集，
保留两侧完整资格清单，不按诊断分数筛身份。当前报告不会自动宣称该假设已解决。
runner 已提供 common-ID 两侧 minMSE 诊断，排除 clean 原五个目标，并另列
simulator-only/clean-only/common 全部 ID 及双方完整目录分数；不据此宣称修复。

## 主线程命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，仅主线程调度 CUDA：

```bash
SDL_VIDEODRIVER=dummy PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.spectral_reachability.pusht_simulator_goal_catalog_20261027 \
  --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --probes results/pusht_spectral_response_physical4_v1_20261026_cuda/report.json \
  --calibration results/pusht_task_spectral_bank4_v1_20261026_cuda/report.json \
  --clean-report results/pusht_clean_goal_catalog4_v1_20261027_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/pusht_simulator_goal_catalog_20261027_cuda_v1
```

加 `--prepare-only` 并使用不同全新 output 目录，可只做真实 reset/render，无
神经工作。此 prepared 目录不可重用或原地续跑；CUDA命令用全新目录从头核对。
先验证旧 compilation/source seals/checkpoint，再封存新源码与protocol；结束检查
inputs/图像/sources/params/buffers/eval。所有输出独占新目录，旧目录不覆盖。
所有图像由官方 renderer 生成 PNG，最多1024张224×224；预计磁盘<100MB，
无权重复制。编码 batch16、最多1024样本/64调用；预留单GPU4GiB/10分钟为
保守调度估计，不是实测承诺。本任务未运行 GPU，不把 prepare-only 当 CUDA结果。

非神经测试：

```bash
SDL_VIDEODRIVER=dummy PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -q -p no:cacheprovider tests/test_pusht_simulator_goal_catalog_20261027.py
```

主机 prepare-only 已实际确认四根原目标兼容、每根256格全部记录：239合格、
17几何穿透不合格；未放宽任何资格、未丢根。准备过程只运行官方物理/reset/render，
没有模型加载或神经推理。先前版本准备目录保留，不复用为最终源码封存。
旧五variant的20原goal已由主线程独立审计：goal hash全匹配、实际object geometry
改变0/20；本模块不将其失败归咎于未观察到的物体漂移。
