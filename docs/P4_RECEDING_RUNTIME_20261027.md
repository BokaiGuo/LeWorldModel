# PushT 两轮 receding runtime：仅工程校准

原研究 P4 要求同 checkpoint、匹配优化预算、根级配对统计、环境 success/cost，
PushT 另需接近/接触/推送/闭合分解；前置 P1/P2/P3 不能省略。已读取原研究计划
P4 合同与 `P45_PREREQUISITE_SNAPSHOT_20261027.md`：**P1 未通过，Full D2P P4
保持 blocked_upstream**。本新增 runner 不修改中央计划、已有任何 seal/source/test/report。
未训练/拟合 readout、未选新 outcome 根、未改阶段门、未自行启动 CUDA 或物理实验。
适用路径内未发现 AGENTS.md；使用 experiment-code 技能进行独立实现与回归测试。

## 可执行最小合同与不公平之处

- 只用既有4个 PushT calibration 根，官方冻结同 checkpoint/config、原 goal0/Q、
  P2 每根16个已封存 prefix packet bank；不学习新 bank，不用物理 labels 搜索。
- 两次决策，每次只执行首个五步 packet，共最多10步。初始真实 F1,F2,F3 对应
  between A1,A2；完成真实 B 后用 F2,F3,F4 和 A2,B 重新编码。不得把 incoming3
  当 between3，不 bootstrap，不用预测 latent 冒充新观测，不在两轮中间 reset。
- root seed 与所有根身份从原绑定读取。pair paths 用固定 seed202610274+root_index
  排列16×16共256有序对，取前80，无替换、无物理结果选择。协议预封存所有路径。
- receding 第一轮搜索前40个两-token程序，执行所选第一个；第二轮以新的真实三帧
  context 搜索后40个程序，再执行新所选第一个。每候选用现有正确 CudaReplay 两次
  predict，固定 Q terminal MSE，cost/失败逐候选落盘。
- fixed 在物理前一次搜索全部80个 pair，锁定同一两-token计划；分两次执行它的
  第一、第二 token。第二轮仍编码真实3帧并保存 context，但不以它改变动作。
- 两臂都无中止时 planning=160 predict calls/samples；receding 各轮 whole replay2，
  fixed 初次 whole2 + segmented2，verification 均4。RGB encoder 都2 calls/6 frames，
  action_encoder 等于 predictor实际调用；无 backward、无 optimizer、复用 saved goal0
  因此本轮无 goal encode，原编译/goal-set成本不能当免费。

**仅完整分支的预测/图像调用预算匹配。**fixed 在初始分配80而 receding 各轮40，
候选数与预算时间分配不同，因此不是只改变反馈的纯因果消融；完整资源不声称相等。
没有用无意义 shadow search 烧掉预算。早停时两臂真实预算可能不等，保留实际账，
不补算、不垫步、不筛掉失败。即使两个臂两轮完整，也不是公平同策略/总算力/墙钟上限
匹配；所有吞吐、encoder、Q评分、日志等开销单独保存，不把predict数冒充FLOPs。

Q 仍有 agent-motion/static-object 假象，extra goal-image-set 的旧来源也不变；不能
声称新任务度量、task efficacy修复、谱机制生成或 P4成立。这是通过有限 bank 把真实
观测→冻结CUDA预测→动作见证→guarded执行→新观测回灌的接口打通。

## 见证、观测和资源

每轮先独立重放核验（fixed第二轮复用已封存计划），exclusive写 decision JSON、
hash并关闭后才执行。见证含当前observed RGB/stack hashes、raw between packets、
encoded context、实际选择所用context SHA、选中full program和terminal、bank index、
即将执行的packet、root/goal/protocol/source seals和前一decision hash。

执行复用 shared visible_command_executor，每 primitive RoyalBlue RGB guard。
缺初始RGB、途中消失、env终止均原样保留并停止本分支；另一个方法仍完整保留。
每步 state/reward/termination/RGB hash、official labels（对 installed evaluator核验）
与 object pose，只作 evaluator，不进入动作评分。失败不重试，无CPU neural fallback。
部分packet不能成为下一轮 context。分支/每轮JSON和候选JSONL逐步落盘，不存视频。

独立 PredictInstrumentation 现场统计 phase calls/samples/failed；action encoder hook
统计 calls/samples；图像编码统计 calls/frames、同步CUDA耗时。完整分支 wall time
包含reset/warmup、变换/编码、CPU候选循环、CUDA预测、Q、verification、seal/log、
physics等；每轮 encode/search/replay wall time、每候选wall、peak allocated/reserved
另保存。无FLOP/Joule或严格wall cap。report逐root并列两臂，不做4根显著优势宣称。

finalize使用Path适配的hash，source/hash/model检查异常先写report；不覆盖旧输出。
神经加载前封存已解析实验/模型源码、独立runner/tests/doc和协议，物理前另封存
installed evaluator；前后校验所有输入和当前source字节、模型parameter/buffer哈希。
磁盘不可写等I/O故障不能保证finalize落盘，保留已有partial artifacts。

完整长程任务至终止、接触/接近/推送/闭合机制分解、独立有效任务度量、held-out
roots/显著配对统计和完整强基线都未交付，不许由工程检查自动解锁P4。

## 主线程 CUDA 命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p4_receding_runtime_20261027_v1 \
  --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/p4_receding_runtime_20261027_v1_cuda
```

必须主机CUDA统一调度，输出目录必须全新，不自动重启覆盖。预计无中止总 planning
1280 predict calls/samples，verification32，总1312；RGB48帧/16 calls；8分支最多80
新物理步，另有每分支15步标准root warmup与root collection成本。
粗估RTX5060Ti16GB显存1–4GiB，时间1–10分钟，结果<30MiB，均未经本runner CUDA实测；
不复制权重/大报告，无下载。当前磁盘约2.4GB足够小型日志，仍以实际空间为准。

CPU 非神经测试：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_p4_receding_runtime_20261027_v1.py
```

测试覆盖真实bank逐值一致、固定调度、实际观测回灌和between移位、固定计划不改、
预算算术、见证先于执行、partial拒绝、错误候选保留且不得部署、CPU神经拒绝、
str-path/sourceverify失败的final report；这些均非CUDA或物理效果证据。

本次实跑新增测试18项；连同现有v2 sampling/visible executor/predict instrumentation
回归共 **66 passed**。另通过非神经 `validate_inputs` 复核原 compilation、4根、
checkpoint/config bindings及既有source seal当前字节；未加载神经模型或启动GPU。
没有预建CUDA输出目录，主线程可直接运行上述命令。代码/tests/doc交付后冻结，
任何后续修正须新版本，不改变已运行seal绑定字节。
