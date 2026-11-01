# 固定动作库搜索已完成主机 CUDA 接入

这是本地 bounded beam/tree search 的实际模型接入，不是官方 kinodynamic search 复现、等预算对比、P1/P2/P3 通过或真实控制收益。

## 完成证据

- 代码：`le-wm-main/experiments/spectral_reachability/run_pusht_program_search_cuda.py`。
- 完成产物：`le-wm-main/results/pusht_program_search4_v2_20261026_cuda/report.json`，`integration_completed`，退出码 0，8 项完整性门全 true。
- 使用同一四个既有校准根、冻结官方 checkpoint、16 个已封存前缀原始命令、目标及 Q。每个命令重复五步成为 primitive；深度上限 2，beam width 4。没有真实状态、成功标签或物理结果进入搜索。
- 每根先扩展 16 个节点，再从保留的四个父节点扩展 64 个子节点；实际模型预测规划样本共 320。完整最佳动作串独立重放额外 7 样本，合计 327。每根实际总预测 cap 82；失败不退款。编码、反向、FLOPs 和 walltime 并未据此宣称等预算。
- 每个返回计划的完整动作重放与逐边预测 context 精确一致；参数、buffer、eval、原输入与源码封存核验通过。

| 根 | 最佳 primitive 索引串 | 深度 | Q 终态预测代价 | 实际预测样本（含重放） |
|---|---|---:|---:|---:|
| 0 | [10, 2] | 2 | 1.174314 | 82 |
| 1 | [2, 14] | 2 | 0.659088 | 82 |
| 2 | [4] | 1 | 1.045510 | 81 |
| 3 | [3, 3] | 2 | 1.468976 | 82 |

搜索返回所有已访问非空节点中的最好值，不强制深度 2，根 2 的结果不可偷偷补零变成等长对照。beam 剪枝导致四根均为 partial coverage，不声称全局最优。预测代价不是物理任务代价，也不同于原 D2P stage + beta * capability 选择目标，不能直接相减作为优劣。

## 代码失败和修复留痕

v1 目录保留：`le-wm-main/results/pusht_program_search4_v1_20261026_cuda`。首次 CUDA 调用在第一根规划后因 instrumentation 不支持 `verification` 阶段名而报错，未生成完成报告；不是算法负结果。修正为已有的 `evaluation` 阶段。

同时代码审阅纠正目标度量接线：扰动目标只用于构造 nuisance projector，实际任务使用 goal[0]，平方残差按 latent_dim 归一化；不能将扰动目标当多任务平均，也不能把 SSE 冒充 MSE。新增回归逐一复算原四根所有 64 个保存预测，与既有 scorer 一致。该评分错误在本轮新 runner，不能外推为原研究历史失败的原因。

本轮主线程执行搜索和新 runner 测试 43 项通过。单测与上述实际 CUDA 证据分列。所有 P1/P2/P3 admission 继续 false。尚缺强基线正式协议统一、固定长度/停止规则、独立新根及真实执行对照。
