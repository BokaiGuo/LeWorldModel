# P4 等轮次预算反馈对照 v2：仅工程校准

本版本新增独立 runner/tests/doc，不编辑已有 source/test/doc/result、中央进度文档或 seal。
已阅读 v1 runner、原 report、说明和中央进度文档。v1 fixed 初始80对、
receding 每轮40对混杂了预算分配；本版不改写其历史结果。
P1 未通过，P4 科学状态仍为 blocked_upstream。不存在本版 CUDA/物理效能结果。
使用 experiment-code 技能组织独立实现、候选日志和非神经回归；
遵守本任务不训练、不调参、不画假设性效能图、由主线程统一调度 CUDA 的限制。

## 冻结对照

两臂 imagined / actual 都执行实际 PushT physics，均两轮，每轮只执行所选
第一个五步 packet，总计划10步。每轮搜索 horizon 均为2个 packet，
并非第二轮将 horizon 缩短为1。全部未来动作仅由冻结模型 + 原 goal0/Q 选取；
真实 state/reward/official success/object pose 不进入候选评分或路径选择。

只使用原4个已知校准根，runner 明确锁定全部指纹和顺序：

- 9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333
- 23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050
- 4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca
- cf1b9bb913bf7815c9184d5de649b77bf09cc268dd80905a0feca9bd2a2edbf2

沿用 seed=202610274+root_index，对原16×16有序 packet 对排列取前80，
第0轮用前40，第1轮用后40；两臂逐轮候选路径精确相同，无物理 outcome 选根。
raw action 表示固定为原16个合法五步二维 bank packet，不重建/拟合 bank。
沿用 sealed compilation 的 checkpoint/config、goal0 和 Q，不重新编码目标、
不拟合 readout、不改阈值、不改阶段门。该 Q 现有任务排序局限不由本版修复。

两臂第一轮 context 均为真实 F1,F2,F3 / A1,A2。执行首包 B 后，两臂都保存
并编码实际 F2,F3,F4 / A2,B，图像调用也按轮匹配。

- actual：第二轮用新实际三帧及执行动作历史编码的 context 选策。
- imagined：第二轮忽略该实际编码，只使用原 context 经模型推进 B 的
  三 latent / 两 normalized action context；不冻结旧计划，仍搜索同40对。

因此被改变的是第二轮反馈来源，而非候选预算时间分配、horizon、种子、动作表示、
目标或优化规则。imagined 仍使用真实 RGB guard，guard 仅允许/阻止执行，不选
替代动作；两臂均不 reset/rebootstrap/recover，部分 packet 不能进入下轮。
这是已知根上的短程 feedback contrast，不是完整长程 Full D2P/P4 实证。

## 预算与记录

每臂每轮40对×2 = 80 planning predict calls/samples。
独立 full2 + segmented1+1 验证共4 evaluation predict calls/samples；
得到的 mid context 也是 imagined 臂下一轮的输入。两臂每轮都执行相同验证工作。
每轮1次实际 RGB encode / 3帧。完整两轮每臂160+8=168 predict，
8分支全部完整时合计1344 predict、16 encode calls / 48帧，最多80个选中动作
物理步。root collection、标准 reset/warmup 不计入这80步；walltime包含这些开销。
没有本 runner 的 CUDA 实测吞吐、显存或效能估计。

PredictInstrumentation 每个真实调用 scope 标注 round，独立统计 calls/samples/
failed/context_lengths/horizon；action_encoder hook 单独计数。
final gate 按每个实际执行轮验证40个 horizon2 planning scope、[2,1,1] verification
scope、CUDA predict calls/samples、context3、无失败，以及实际 encoder/action hook
总账；不是仅检查160的分支合计。8分支全部完整才可标记所有配对逐轮预算匹配。
中止分支保留实际花费，绝不补算凑预算、padding、重试、丢弃或替换根。
不声称 FLOPs、总算力、wall-clock cap、能耗匹配或已验证 task efficacy。

输出必须是 results 下含本版本 tag 的全新目录，existing 目录拒绝。
加载模型前封存已解析源码、模型实现、本版 runner/tests/doc 和协议；
物理前再封存 installed evaluator，结束复核输入/源码 seal 与模型参数/buffer。
require_cuda_allocation 在模型加载前执行，encode 和 CudaReplay 都拒绝 CPU；
不存在神经 CPU fallback 参数。
run 入口在模型加载/规划前显式设置 torch.backends.cuda.matmul.allow_tf32=False，
协议和最终 resources 都记录该值。

每轮输入保存 actual RGB stack 和 raw between .npy。候选含路径/cost/失败，
逐条写 JSONL 并 flush/fsync；任一候选失败保留完整调度日志且拒绝部署有限子集。
decision exclusive JSON 写入并 fsync 后才执行，含：
root/goal/protocol/source manifest binding、observed/planning context及hash、
完整选中动作及terminal、model mid context、执行packet、上轮decision hash。
两臂每根先后顺序交替。

执行沿用 RoyalBlue per-primitive live RGB guard。逐步 state/command/reward/
termination/RGB hash 先独立落盘，之后写完整 execution receipt（含全部guard状态）
和 terminal .npy，再核验轨迹长度/动作前缀。即使零步/中途/第五步guard中止，
也保留本轮收据、终帧和已完成轨迹，不在汇总/标签异常后吞掉已执行分支。
轻量 env.step 代理在每次调用底层 physics 之前先 exclusive/fsync 写 attempt JSON，
保存授权命令和步序；底层 step 抛错时也保留 attempted-but-not-completed 记录。
此记录证明调用前授权/持久化，不能证明异常调用内部发生了多少物理积分；
完成态仍以原 executor receipt 与 observation trace 区分。
最终 object pose/official 标签只用于描述，另存 branch JSON，不参与未来选择。
运行异常分支另保留已有 decision/execution 文件路径；final report 不认证异常为成功。
模型/CUDA加载前即写全4根×2臂 branch_roster。环境setup/根验证失败时，
未到达的分支另写 blocked_setup_failure JSON并保留进report，不伪造执行或重试；
模型/CUDA加载失败则保留pending_setup roster，不伪造最终report。
磁盘/进程硬故障不能保证尚未完成的写入；不可冒用现场目录重启。

## 主线程可直接审查/启动

从 /home/gbk/code/LeWorldModel/le-wm-main 运行，主线程统一主机 CUDA 调度：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_matched_20261027_v2 \
  --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/p4_feedback_matched_20261027_v2_cuda_main_v1
```

没有预建此 CUDA 输出目录，没有启动神经、GPU 或物理作业。
若任一完整性门失败，runner 先写 report 再抛错；
模型加载之前的 CUDA/来源检查失败直接停止，保留已有封存现场，不伪造分支报告。

非神经测试：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_p4_feedback_matched_20261027_v2.py
```

已实跑新增56项测试；连同旧v1、visible executor、RGB guard、predict instrumentation
五文件联合114 passed（0.71s）。全部非神经 algebra/metadata/executor fakes，
及保存 bank/goal/Q 的离线算术，不是神经 CPU fallback。
测试覆盖逐轮同候选/horizon/budget、真实历史移位、imagined不读实际反馈、
两臂实际dispatch接口、见证先于执行、独立重放拒绝、guard各中止点持久化、
坏账本/假匹配拒绝、CUDA failfast、保留失败和exclusive写入。
主线程review后的增量覆盖step抛错前attempt已落盘、setup失败全8臂保留且不覆盖已执行分支。
另只读 validate_inputs 通过原 compilation、4根、6个输入绑定和已有source seal，
没有模型实例化/神经执行。源码稳定交付后任何修正须新版本，不修改已经运行的seal。
