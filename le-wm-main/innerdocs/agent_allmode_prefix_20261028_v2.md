# MAIN allmode v2：仅修复输出容量与终态诊断

固定算法、动作、Q、四条件、42候选、1032 predict、172 J、176成功helper内SVD与1800秒限制完全沿用v1。不根据物理结果调参、不改变科学门槛。

v1真实主机CUDA已完成全部1032 forward与172 AD返回，但保存第四row触发输出常规区64MiB限制，最终full report又超过128MiB总cap。原目录保持不变，未释放动作。实际文件62,523,524 bytes，前三sealed row解压21,537,759/21,501,646/21,375,599 bytes；完整gzip journal89,132,918 bytes、14,112事件，eof=true。磁盘仍63,400,988,672 bytes空闲。这是密集浮点JSON重复造成的工程容量估算错误，非CUDA不可用或理论负结果。旧1032调用的成本仍保留，不能被新实验账抹掉。

v2仅新建source/tests/doc：输出总cap512MiB，final reserve256MiB，disk floor200MiB不变。依据前三完整实际row估计完整四row约86MiB而非RAM稀疏fixture的小尺寸；增加容量不修改任何数值精度或删日志。失败full report额外尝试写独占、有限的failure.json，包含四分母census及错误，永不放行部分动作；若磁盘本身IO失败仍无法保证写出。

原完整三文件说明见v1 innerdoc；v2改变名称并封存自己的三文件，父report/prepare与模型完全相同。无旧module/global替换。所有新的prepare/run必须以v2新目录命名；不覆盖v1，不借新source声称旧run已通过final model/source验收。再次实际CUDA执行是新版本独立执行，不是丢弃失败候选或隐式重试。consumer仍须全部数组/258×4 forward/模型source/全部候选与日志齐全。

MAIN命令模块：experiments.spectral_reachability.agent_allmode_prefix_20261028_v2；四子命令prepare/verify/run/consume与v1参数一致。神经只用主机CUDA；固定两known roots不是fresh，P1/P3 false。v2新增回归核验真实密集row的保守JSON尺寸界，而不将RAM测试当科学证据。
