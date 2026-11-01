# P2 封存预测与 native 指标差距：四根八臂 saved 分析

## 范围与完成条件

独占 source/tests/innerdoc 同名三文件。只读原四个 known roots、两臂各十步及原 CUDA 数组；没有新神经、物理、训练、拟合、选择、阈值或 MAIN 文档写入。个人交付完成条件为可运行 saved 分析、RAM/变异回归、实际 saved 运行、三文件 SHA 与冻结；不是研究准入完成。

输入固定为 MAIN 指定的 physical report、strict consume、独立 native audit，以及原 prepare。各自完整外部 SHA 写在源码 `INPUTS`，不接受替换某根或只报赢家。分析核实被消费 metadata 及所有原 physical artifact 的文件 SHA；不重新跑旧严格 consumer，也不加载/重新核验权重内容。对安装 native evaluator 和原 Q/native labels 源码检查继承 SHA，不导入环境。

额外发现并复用旧真实 CUDA 资产，而非重复请求终态编码：

- `results/p2_composition_predict_frontier_physical_20261027_v1_main01/report.json`，SHA `9144c3f25f2880af62d94dd88257918fa508b5ddbea51d98e9ccbc3d3b7532e2`。
- 对应 frozen physical protocol SHA `7d1131e9dce9ed152ce53e5ae35f22cb4e7c328f8de50381d4363a52feadf2ef`。

旧、新 export 的 selection hash **不同**，不能冒称相同 receipt。分别将旧编码接旧 plan、新物理接新 handoff，再逐项比对 raw10、三 latent/two token 上下文、终态预测上下文、goal/Q、模型绑定 metadata；八臂实际完整 trace、初态/目标、终态七维及终态 PNG SHA 完全一致。旧报告自带全部八个 terminal_vector；无需侧载未封存数组。旧报告声明 HOST CUDA、模型未变、八次 encoder/projector 返回，仅作为已有证据复用，不称独立复现。

## 原目标与状态语义

原 compiled objective 是 `mean(((z_pred - goal0) @ Q.T)**2)`，192 维 frozen latent，原五个 goal variants 的 nuisance-span projector（记录的 nuisance rank=4），不是 identity Q，也不是物理坐标距离。代码用原 float64 saved 数组复算，未重做 SVD/编译/预测/选计划。teacher Q 使用已编码的同一终态 PNG、同一 goal/Q；未引入 learned readout。

native state 七维依次是 agent XY、object XY、object angle、**agent velocity XY**。原 native7 是未加权混单位欧氏距离，reward 为负该距离：

`native7² = agent_xy_sq + object_xy_sq + raw_angle_sq + agent_velocity_sq`。

jointXY² 只含前两个位置平方项；objectXY 只含 object 的二维位置。位置是原环境坐标（渲染像素坐标），角度为弧度，速度是原环境 agent 速度单位，不能将这些单位的 native7 解释成纯位置误差。success 保留原 `jointXY < 20` 且原 circular angle `< pi/9`；这不是新增阈值。raw angle 参与 native7；wrapped angle 单列诊断，绝不替代原 evaluator 的角度计算或终止标签。速度也不是 object velocity。

## 实际 saved 复算

全四根两臂各十步，80 个已封存真实返回；八条完整轨迹每步原 reward 与 native 距离对账，原 success 和 MAIN independent audit 全部一致。下表均越小越好，不同量纲不互相相减。J=joint、C=composition；数值显示取六位，输出保留完整精度。

|root|臂|预测 Q|真实图片 teacher Q|native7|jointXY|objectXY|agentXY|agent速度误差|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|0|J|1.391151|1.680956|481.790052|447.978130|229.382014|384.796435|177.236613|
|0|C|1.322498|1.744179|390.356234|347.391395|159.539975|308.589983|177.967248|
|1|J|0.775976|0.895922|290.201147|142.147573|10.551904|141.755387|252.970542|
|1|C|0.894781|0.907032|234.267881|99.533563|53.058027|84.212682|212.032962|
|2|J|1.209249|1.227599|209.552369|197.917091|35.160421|194.768888|68.837047|
|2|C|1.076584|1.247039|334.914498|187.396828|51.446584|180.196616|277.574760|
|3|J|0.895831|1.717167|344.809133|251.592741|53.801781|245.772813|235.783738|
|3|C|1.029175|1.344770|435.383602|277.075473|84.046116|264.020962|335.835819|

composition 在预测 Q/teacher Q/native7/jointXY/objectXY 的距离胜出数依次为 **2/1/2/3/1（各分母四）**，不是 success 数。两臂 native success 均为 **0/4**。0/4 仅描述这些 known roots 的封存程序，既不证明理论不可能，也不是“composition 从未改善任何距离”。

以下为 composition-minus-joint 的 native7 **平方项**贡献，不是因果干预效应，也不跨单位解释占比：

|root|agent位置平方差|object位置平方差|raw angle平方差|agent速度平方差|
|---|---:|---:|---:|---:|
|0|-52840.518818|-27163.104515|0.434464|259.524447|
|1|-13002.813822|2703.811545|-0.145578|-19036.117961|
|2|-5464.099535|1410.495760|-0.078483|72309.208200|
|3|9302.792331|4169.117927|1.906314|57191.726236|

- root0：agent/object 都改善，native7/jointXY/object 同向。但预测偏好 C、teacher Q 偏好 J，且 teacher Q 与物理距离排序不一致。
- root1：object-only 会偏好 J；agent 改善更大，所以 jointXY 偏好 C；速度改善进一步降低 C 的 native7。不能因 object-only 一项就判完整 PushT 任务失败。两臂原 native success 仍 false。
- root2：C 的 agent 改善超过 object 变差，jointXY 略好；agent 速度平方增加约 72309，使 native7 反向。角度项不是本例的大项。
- root3：位置与速度都偏好 J；teacher Q 却偏好 C。不能以该 mixed-unit 数值分解宣称哪种机制因果失效。

预测/teacher 的成对 Q 差（C-J）分别为 `(-.068653,+.063222)`、`(+.118805,+.011110)`、`(-.132665,+.019440)`、`(+.133344,-.372397)`。因此 root0/2/3 已可见预测→真实图片 Q 排序偏差；teacher Q→native7 在 root0/1/3 也不一致。输出另存完整 latent L2 误差、每臂 predicted-minus-teacher Q，以及成对恒等式 `ΔpredQ = ΔteacherQ + Δ(predQ-teacherQ)`。这将两种失配分栏，但不证明二者的因果来源，也不能将 nuisance-Q 当 native success 校准器。

## 下一轮最小可检验缺口（提案，不是新门或已执行预注册）

**无需重跑终态 encoder 或 physics。** 最小增量是在 MAIN 审批事前固定后，仅编码同四根八臂已保存的 step5 RGB，继续原 frozen model/normalizer/原 goal/Q；完整八样本，八次 encoder+八次原 projector，零新 predict/AD/physics，旧终态编码与历史费用另列而不是计为零。先核同一 transform/ABI 及 saved midpoint prediction 对应真实第5步；若映射不闭合，报告缺失，不拿终态或 root latent 代替。

预先固定分析：每根两臂 step5 和 step10 的 predicted/teacher Q、原 agent/object/速度/角度误差全列；不选 winner、不拟合权重、不改变门槛、不设事后成功标准。问题是“失配在前包后已出现还是后包扩大”，不是再验证已知终态排序。这里仅定位 rollout 与目标对齐的时间结构；不等价于 child-local mechanism、不解锁 P1/P2、也不是 full C3。若之后检验修复能否改善 native success，仍需另行事前固定动作生成规则、独立样本与真实执行；本分析没有提供那些证据。

当前明确缺口：未消费 step5 teacher 编码及其精确预测时间对齐；无独立 fresh cohort、无 P1 admission；四 known roots、单次 deterministic 路径，不能一般化或做因果声称；没有公平全资源优势结论；未把旧报告模型 metadata 声明伪装成本次独立神经复算。未知值留 null；缺必要输入直接报错，不补零。

## MAIN 精确命令与已运行验证

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_native_composition_gap_20261028_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_native_composition_gap_20261028_v1 --output results/agent_p2_native_composition_gap_20261028_v1_MAIN01.json
```

`--output` 只接受不存在的新文件（xb），不覆盖封存结果；不传则完整 JSON 到 stdout。本 worker 已实际调用 `load_saved()`、测试中运行完整 CLI stdout，复核 124 个 consumed SHA（含原 physics PNG/journal、相关语义源码及旧编码 report/protocol）。未在 workspace 另造第四个结果文件；MAIN 可用上述命令独占持久化。首次开发运行发现历史 native prepared/protocol 是 canonical JSON digest 而非 raw file SHA，已分别核对 canonical ID 和继承 raw SHA 后回归通过，未改变历史资产。

单文件回归 **45 passed in 12.33s**：RAM 七维分解、agent/object/velocity 排序反转、raw/wrapped 分离、missing/null/bool/NaN 拒绝、四根完整分母、saved 状态/动作/Q/selection/metric/audit 变异、teacher image/state/context/goal/Q/model/vector/selection 变异、外部 SHA 拒绝，以及独立子进程禁止导入 torch/gym/physics 后完成 saved 分析。子进程不是子智能体，不运行任何模型。测试通过不是科学通过。

联合回归可追加原已冻结的两套测试，不改其文件：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_native_composition_gap_20261028_v1.py tests/agent_p2_composition_execution_20261028_v1_test.py tests/test_agent_p2_execution_assets_20261028_v1.py
```

冻结通知以最终回复的 source/tests/innerdoc SHA 为准；之后任何修改必须新版本。本任务无 CUDA/physics run 入口，MAIN 的下一条可执行命令是上面的纯 saved CLI，不是实际实验命令。
