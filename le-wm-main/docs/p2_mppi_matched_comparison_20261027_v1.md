# MPPI-inspired / full20 CEM / random：只读配对比较（v1）

只新增同名模块、测试、本文和全新results输出，不改任何sealed源码或原结果。
使用experiment-code和data-analysis技能约束来源核验、稳定身份配对、描述性不确定性
及回归测试；用户范围优先，不运行训练、GPU、模型forward、真实physics或新候选评估。
两层及其下属目录本次未发现AGENTS.md。

## 固定输入与只读核验

- MPPI物理：`results/p2_full20_mppi_physical_20261027_v1_cuda_main_v1/report.json`。
- CEM/random物理：`results/p2_full20_physical_20261027_v2_cuda_v1/report.json`。
- 跟随两份既有protocol的显式planning/prepared/source-seal/input bindings，固定原
  compilation SHA；不扫描结果目录、不打开未来heldout、不以观察结果选择方法。

逐份核验历史manifest/args/archive/current source SHA和checkpoint/config/input
绑定；MPPI execution→physical prepare→planning的protocol/selection SHA链，及
CEM/random原physical计划→planning见证链。复用原保存内容validator，不调用runner。
MPPI kernel仅用保存cost重建原proposal收据，不新调用模型或调整超参。

按`(root_sha256, arm)`join全部四根、三臂、12分支；原数组反转不改变配对结果。
full root/goal RGB、完整latent/action context、normalizer、raw action domain、
metric/goal vectors、物理reset初态/goal/history、预算单位须跨臂一致。
root_index只用于已保存文件名，不能用于实验间join。

每臂每根40完整H2候选、80planning+4replay predict，84 action-encoder调用、
252 processed context tokens、10 selected-path物理步、1 terminal encode均复核。
两个full20 baselines共672历史predict，MPPI共336；比较器新增predict/physics为0。
**80 predict不等于80完整候选**。同20维raw box不表示proposal分布相同，更不表示
总FLOPs、上游编码/建库、共同wall-time预算相等；这些明确保留unmeasured/unmatched。

复核durable physical_report、逐branch JSON、31条initial/attempt/returned/observation
journal、所有逐步PNG SHA与terminal PNG linkage、encoding sidecar和最终report一致。
再检查raw action逐步前缀、step顺序、最终state、guard身份/计数、encoding root/arm。
任何缺失、重复、错绑、篡改、非有限值或非完整H2都拒绝，不静默删根/换终点。

## 配对指标和统计边界

由保存七维state与goal重新计算原pose和official labels，不调用env.eval_state：
pose为object XY距离+20×wrapped angle距离；official_state_distance保持原七维raw
L2（含agent位置/速度及raw angle），official angle沿用原evaluator公式，不偷偷换
圆周成功定义。逐字段核对原记录，保留object-only与official的区别。

保存terminal encoder向量用原goal_vectors[0]/Q重算real encoder Q-MSE，预测Q-MSE
也与原selected terminal witness核对。无新encoder，不把模型Q排名当任务成功。
每根报告三臂pose、XY/angle、official distance/position/angle/success、预测及真实Q，
以及七个state坐标的absolute error；state raw-angle误差与wrapped angle单独标识。

配对差统一为MPPI-inspired−baseline，成本负值更好，success正值更好。对CEM和
random分别报告逐根差、均值、中位数、sample SD、范围及胜/平/负；七维state误差
也逐坐标汇总。不报告这四个已用校准根的前瞻CI/p值，不声明总体优越、标准MPPI
复现或组合增量，不据此调温度/改seed/选方法。MPPI-inspired仍无importance-density
修正或自动control交叉项，不能称官方MPPI。`P1_admission=P2_admission=false`。

## 运行

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，输出必须不存在。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_mppi_matched_comparison_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_mppi_matched_comparison_20261027_v1 --output results/p2_mppi_matched_comparison_20261027_v1_audit01
```

输出report.json、report.md和本次source_seal。完整输入/源码SHA清单结束时再核验。
exit0只表示保存内容审计/比较完成，绝非P2准入；exit1表示非法/缺失证据并保留
invalid报告。hash是字节一致性，不是实际执行、独立outcome或前瞻封存真实性证明。

## 本次验证与观察

新增专属39项非神经测试；与既有producer/consumer/full20/kernel/witness/contract/
admission共9文件联合356 passed（23.99秒），仅第三方pygame弃用警告。
包括错根/缺根/重复、输入乱序、预算总计与root错绑、动作/goal/metric篡改、durable
branch/encoding/journal不一致拒绝，以及新独占输出与完整来源封存。

四根均保留，三臂官方成功均0/4。MPPI-inspired−CEM的平均pose差为+38.688985，
胜/平/负1/0/3；对random为+13.457195、2/0/2。正差表示成本更高。这只是已用
校准根的描述，不据此认定总体算法优劣、改方法，或排除faithful MPPI；完整official/
state/Q差值及sample SD见生成report，不以pose替代官方任务结论。P2保持false。
