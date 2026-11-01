# SRA 与 CEM/random/MPPI-inspired 的只读配对算术审计

本模块仅消费四个已用 calibration root 的真实封存记录，不训练、不启动模型或
physics、不读取新的 heldout，不选择后续方法、不修改旧 sealed 文件。
使用 experiment-code 技能的验证/日志要求；本任务为已有结果算术审计，不生成
训练流水线或效果曲线。独占新增 module、tests、本 innerdoc 及同名前缀结果目录。

## 固定输入与验证链

明确读取以下目录的 report/protocol，不扫描其它结果选择输入：

- `results/p2_no_composition_sra_physical_20261027_v2_cuda_main_v1`
- `results/p2_full20_physical_20261027_v2_cuda_v1`（CEM/random）
- `results/p2_full20_mppi_physical_20261027_v1_cuda_main_v1`

复用旧 arithmetic comparison 的 baseline 检查，不修改它：源 seal 的当前/归档
字节与 manifest、准备协议、规划预算与 winner witness、完整 H2、逐步动作绑定、
durable physical_report、每分支 .physical/.encoding 文件、journal attempt/return/
observation 顺序及各 RGB 文件哈希均需通过。SRA 额外验证 physical prepare/run/
planning 源链及原已封存 exporter；不追加 exporter，不重封存追认旧 report。
SRA planning 每 root 独立 JSON、bank 和 compilation operation journal 亦绑定。
物理编码数必须是每分支一个、四个 SRA 总数4，不允许 guard/失败终点替代固定H2。

四臂按 `(root_sha256, arm)` 唯一映射 join，拒绝重复、缺根、缺编码和 wrong-arm。
完整 root predictor context、root/history RGB、goal RGB/向量、normalization、Q metric、
完整 reset history/state/goal 必须相同；不得按数组位置配对。
SRA 40 candidates 与 bank 动作、label、logits、原稳定 first-tie winner、receipt/provenance
再做只读检查。检查已封存 winner 不是重新选择；无新预测或 proposals。

## 指标和统计边界

从保存的7维 terminal state 与同一个 goal 重算原 official evaluator 算术和 pose：
pose = object XY distance + 20 × circular angle distance。
保持原 official angle/success 定义和旧验证容差，不修改科学门槛。
从实际保存的192维 CUDA terminal encoding 重算原 goal0/Q MSE，与规划预测Q MSE分列。
保存每root四臂原始状态、逐坐标绝对误差、pose/XY/angle、official state/position/
agent/object/angle error、两种 success、real encoder Q 与 predicted Q。
逐坐标 raw angle 误差不是 circular angle，各坐标混合单位，不能合成新的指标。

比较方向固定为 SRA minus CEM/random/MPPI-inspired；cost 负值有利 SRA，success
正值有利 SRA。输出4对原始差值、mean/median/sample SD/range及精确W/T/L；
只有描述性不确定性，不给CI/p-value，不作独立总体推断或多比较显著性声明。
四根全部保留，不以观察结果删根、换方法或调目标。

## 资源开销不能抹平

| 每root已记录单位 | SRA | CEM/random/MPPI-inspired |
| --- | ---: | ---: |
| 完整H2 candidate | 40 | 40 |
| candidate predict | 80 | 80 |
| replay predict | 4 | 4 |
| extra compile predict | 4 | 0（仅该ledger内） |
| 总 predict / action encoder calls | 88 / 88 | 84 / 84 |
| processed-context-token samples | 264 | 252 |
| selected-path env steps | 10 | 10 |
| terminal CUDA encode samples | 1 | 1 |

四根SRA为352predict，对应每个参考方法336；多16predict（88/84≈1.047619），
以及4个Joint Jacobian、8 SVD、768数学 scalar-output VJP directions。
VJP方向不是实测 backward API/kernel 次数。参考的0 extra compile predicts只说明
已记录 sampling ledger 内的分解，不代表上游编译/建bank/编码耗费为0；参考
Jacobian/SVD/VJP记为unknown/null，不填0。SRA数组bank复核每次physical.inputs
还需8 SVD/4 bounded solves，单列verification工作，不当成新的神经调用或免费操作。
保存的 SRA root wall time仅保留原记录，不宣称共同计时口径或由此推出速度提升。
FLOPs、上游RGB/建bank、backward内部开销及共同wall-time cap未测量。

因此 shared candidate/replay/physical/encode units 不等于 total-budget matched。
MPPI 仍标注 MPPI-inspired shooting heuristic，不能称官方 MPPI。
这里没有组合臂或隔离 composition 的配对消融，不能识别组合收益：
composition_admission/P1_admission/P2_admission/fair_total_budget_comparison/
prediction_budget_matched/physical_efficacy_verified 均为 false，即使完整性审计通过。

## 独立运行

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_sra_matched_comparison_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_sra_matched_comparison_20261027_v1 --output results/p2_sra_matched_comparison_20261027_v1_audit01
```

输出必须不存在；生成 report.json、report.md、自身源码/测试/doc seal 及输入源哈希
manifest。缺证据/哈希变化/不一致则输出 invalid_evidence 并exit1，不降级放行。
哈希只证明审计时所见字节与已有归档一致，不独立证明真实执行、执行时间或prospective
independence；真实执行属于主线程已提交的 CUDA/physical 证据，本模块不重跑证明。
