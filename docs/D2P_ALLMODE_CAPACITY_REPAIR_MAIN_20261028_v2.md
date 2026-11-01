# 全方向消融：实际输出容量错误与 v2 工程修复

MAIN 实际主机 CUDA v1 已终止，退出 1。保存第四 row 时出现 `output cap/final reserve/floor`，full report 再次因容量拒绝，未发布完整报告或动作。旧目录和源码均保留。

主线程只读解压实际 gzip journal：eof=true，共 14,112 事件、89,132,918 解压字节；四项各 258 forward attempt/return、43 AD attempt/return，即已支付 1,032 次预测、172 J。前三完整 row 解压分别为 21,537,759 /21,501,646 /21,375,599 bytes；原目录文件实际 62,523,524 bytes，磁盘空闲 63,400,988,672 bytes。

根因是 128MiB 总cap/64MiB常规区和终态reserve低估了 dense float JSON 的重复存储，而不是磁盘不足、GPU失效、模型排序负结果或理论失败。稀疏 RAM fixture 没覆盖真实尺寸；本次实际执行捕获了工程缺口。全部数值计算已返回也不能替代缺失的 final model/source/report 验收，不能拼接前三根当全分母成功。

## v2 最小工程更改

仅新建 `agent_allmode_prefix_20261028_v2` 的source/tests/innerdoc，原版不改。

- 输出cap512MiB、final reserve256MiB；disk floor200MiB、1800s wall不变。
- full report失败时尝试写有限failure.json，保留4项/168候选分母和首错/追加错，不释放部分动作。底层IO失败仍不能保证落盘。
- 原模型、任务、Q、normalizer、动作raw authority、42候选顺序、目标与beta、1032预测/172J合同均不变；无精度压缩或删日志。
- 新增真实完整dense row保守JSON尺寸界、合同仅工程容量差异、常规区拒绝后可写终态诊断回归。

MAIN实际41 tests通过（23.69s）。新prepare679,560 bytes，SHA `c11543eaa251903ff6d0429d2447f03c7427e1fe65be3d0aabfb0d9c4ef79d0d`；新主机CUDA输出为 `le-wm-main/results/agent_allmode_prefix_20261028_v2_cuda_MAIN01/`，已启动，尚待终态与strict consume。不能把已启动算完成，也不能抹去v1已支付计算成本。

两known roots×off/on仍不是独立四根或fresh确认。新增方向消融不能穷尽连续prefix空间；任何零改善都只能针对固定候选合同解释。P1、C2和全研究闭环未通过。

## 终态更新：v2 已完成

上述“已启动，尚待终态”为启动时记录，现已完成。实际 report 为 `le-wm-main/results/agent_allmode_prefix_20261028_v2_cuda_MAIN01/report.json`，SHA256 `c3e38899a12cd163d75918da79ef029adb14b2479f74f15bdfac0b6ffc57242d`；errors=[]，source_verified/model_unchanged=true。1,032 次预测、172 次 J 均完成，严格 consume 退出 0；状态仍为 not_admitted，不是 P1 通过。

MAIN 独立数值复算见 `le-wm-main/results/analyze_allmode_MAIN_20261028_v2_MAIN01.json`。四项原十候选的数值与所选 raw10 全部复现；增加其余谱方向后，四项选择均不变且仍等于 stage-greedy。各项预测非贪心候选数为20/27/18/18，但这些候选均不在(stage_cost, own_reach_value)的 Pareto 前沿。没有发现 stage/reach 的 sum-vs-mean 接线错误，两者均按192维归一。

结论限于固定候选库和两个已知根：容量错误已修复；增加基方向未解决 P1，不能提升为理论不可能。旧失败计算仍计账；不因动作相同无条件重复物理执行，复用必须通过完整来源和真实执行消费者核验。
