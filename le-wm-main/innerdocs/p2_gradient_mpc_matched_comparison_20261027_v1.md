# Gradient MPC 对 CEM/random/MPPI-inspired/SRA 的封存证据算术比较

独立新 module/tests/innerdoc/输出，不改旧 sealed 文件，不运行神经或physics。
遵循 experiment-code 的日志/验证要求，此任务不训练或生成效果图。
输入全部为四个已知calibration roots，禁止结果后选择方法、删除困难root、调整阈值、
读取未来heldout或把完整性通过解释为P2通过。

## 固定输入与证据链

明确读取以下目录，并沿其protocol引用校验prepared/planning/input/source链：

- gradient planning：`results/p2_full20_gradient_mpc_20261027_v1_prepared01`
- gradient physical：`results/p2_gradient_mpc_physical_20261027_v1_cuda_main_v1`
- CEM/random physical：`results/p2_full20_physical_20261027_v2_cuda_v1`
- MPPI-inspired physical：`results/p2_full20_mppi_physical_20261027_v1_cuda_main_v1`
- SRA physical：`results/p2_no_composition_sra_physical_20261027_v2_cuda_main_v1`

复用旧SRA/MPPI arithmetic验证器和gradient physical/planner的只读验证，不调用cuda/run。
seal当前源码/归档/manifest/args绑定、准备protocol、实际模型budget binding、40个H2
candidate、原goal0/Q、稳定first argmin与single-edge winner replay witness均检查。
gradient逐root JSON与report必须一致；238条planning journal事件逐条校验：前39个
candidate为packet0/packet1/cost/ADrequest/ADcompleted/candidate，末candidate无AD两条。
检查80个forward packet dispatch、40个cost/terminal context、39个AD request/completion
和39个已记录raw-gradient/logit-gradient/update轨迹；不重新求神经梯度。

physical的durable physical_report、逐分支physical/encoding JSON、journal初始状态与
10组attempt/return/observation、所有RGB文件hash、trace/terminal/action绑定均校验。
仅4条完整10-command H2和4个实际CUDA endpoint encodings可进入完整配对，guard截断、
缺root或失败分支明确拒绝，不能偷偷过滤后配对。reset/root reconstruction不算作
selected-path10步，不声称物理总步数仅40。

## 配对、指标与不确定性

按 `(root_sha256,arm)` 唯一索引join，拒绝重复/缺失/wrong-arm，完全不依赖report数组序。
五臂完整root predictor context、root RGB、goal RGB/向量、Q、动作domain/normalization、
reset state/history/goal必须一致。动作序列绑定各自已选witness，不要求不同方法动作相同。

由保存的7维state重算原official evaluator与pose=objectXY+20*circularAngle；不改门槛。
由保存的192维CUDA terminal vector重算原goal0/Q MSE，与predicted Q MSE分别报告。
保存每root五臂pose/XY/angle、official state/position/agent/object/angle、两种success、
real encoder Q、predicted Q、原state与逐坐标绝对误差。raw angle coordinate error不等于
circular angle，混合单位不汇总为新指标。

差值固定gradient减参考，cost负值有利gradient，success正值有利gradient。
四组配对输出原差值、mean/median/sample SD/range与精确W/T/L，无CI/p-value或总体推断。
所有root/方法全部保留，描述结果不构成后续方法推荐或因果composition结论。

## AD开销与backend差异不能抹平

所有方法每root有40个完整H2候选=80candidate predict，及4winner replay；selected
physical为10steps/1encode。gradient、CEM/random/MPPI总predict各84，SRA为88。
四根gradient336predict、156次autograd.grad、156个scalar-objective VJP directions、
156次动作logit更新，模型训练更新0。SRA352predict、4Jacobian、8SVD、768数学
scalar-output VJP directions。两类VJP的任务不同，均不等于实测backward kernel/FLOPs，
不能用156/768比例宣称计算收益；参考未单独测量AD的项保留null而非填零。
SRA只读bank复核每次physical.inputs还需额外8SVD/4bounded solves，单列核验成本。

gradient源与protocol在CUDA前固定search及replay均为CUDA SDPA MATH、严格deterministic、
TF32 off与冻结模型，search/segmented/full replay context精确相等才生成有效见证。
其它参考protocol未显式固定/归档MATH runtime dispatch，实际backend未知，不能倒推
为flash或声称相同；报告same_backend_established=false。因此跨方法的预测数匹配
不证明bitwise预测相同或同backend计时公平。保存wall time仅作为原记录，不推出速度。
FLOPs、backward内部、上游RGB/建bank和共同wall-time cap均未测量。
gradient/SRA的tanh联合logit球是full raw box真子集，不同proposal/support不能忽略。

gradient标注internal projected-gradient open-loop H2 MPC，非官方论文实现；MPPI仍为
MPPI-inspired shooting而非官方MPPI。没有组合臂或隔离组合因素的matched ablation。
P1/P2/composition_admission、physical_efficacy_verified、fair_total_budget_comparison、
prediction_budget_matched均为false，不将calibration完整性当科学准入。

## 运行与独占输出

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_gradient_mpc_matched_comparison_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_gradient_mpc_matched_comparison_20261027_v1 --output results/p2_gradient_mpc_matched_comparison_20261027_v1_audit01
```

output必须不存在；创建report.json/report.md、自身source/test/doc seal与完整输入SHA
manifest。缺证据、哈希变化、预算/动作/目标/backend不一致均记录invalid_evidence并exit1，
不降级放行。新测试含真实保存证据验证与明确的metadata/tamper检查，不执行模型。
哈希及数组重建只说明审计所见字节一致，不能独立证明真实执行、attention dispatch、
时间先后或prospective独立性。真实CUDA/physics证据来自main已执行的保存记录。
