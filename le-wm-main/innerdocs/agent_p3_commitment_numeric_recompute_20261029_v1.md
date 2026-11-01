# P3 六臂独立数值/付费计数重算（saved only）

## 与原 strict 的边界

独占新 source/tests/doc，不修改 commitment controls v1/v2 或 MAIN 输出。
source **不 import producer、consumer 或任何项目计算函数**；仅标准库与 NumPy。
此消费者不是第二个模型/RGB full strict，也不代替原 producer strict。它对 MAIN
随后提供的 actual report、prepare、完整 strict receipt 三个外部 SHA 联合绑定，
独立重算原生数值、全部同根分支配对和实际付费 journal。没有结果时只做合成fixture。
使用 experiment-code 技能组织预声明结构与回归，不训练、不按outcome调参。

已知 producer full strict receipt 本身没有 direct report SHA 字段。本工具明确输出
`inherited_strict_direct_report_SHA_field=false`，不伪造该字段；同时核实际 report 的
prepared身份、terminal journal、所有 native journal、原strict中的counts/
per_episode_neural/native_metrics 全数组与新重算结果的交叉一致性。本工具新输出
明确绑定所有三个外部 SHA。不能把输入receipt status=True当作独立数学验证。
原 RGB/encoder/模型参数/recipe 完整验证仍是 MAIN 提供的原 full strict 的职责，
这里不声称重新运行那个消费者或重新做神经推理。

## 独立数值合同

固定 2 known roots × fixedH1/fixedH3/adaptive，保留6臂/18节点；不是6独立样本，
不是新根或fresh确认。每个实际 control 返回均用该臂 bind 后 plan0 的 native goal7
与实际 returned state7 重算。warmup旧goal不混入 taskgoal收益。

原公式完整保留：native distance = norm(state7-goal7)，native reward = -distance；
native success = norm(delta[:4])<20 且 min(abs(delta_angle),2π-abs(delta_angle))<π/9。
不“修复”超过2π时原 native angle term 为负的行为。单列 jointXY、agentXY、objectXY、
velocity误差、真正wrapped angle；object-only不是完整任务success。
核每步 raw float32bits、goal不变、dispatch→observation→return顺序、reward/done、
末state/last_observation/final_metrics。无可用终态时写null，不填零。

固定全部6对（每根三策略全部三对），不按成功、结果或模式筛选。
有双方真实step5 prefix时，核完整 state/snapshot/goal/goalRGB/goalpose/pixel身份，
再输出 right-left 的全部数值差；缺prefix或metric则该配对仍保留而差值null。
实际控制步数是否相同单列，不能因此声称匹配计算成本或单因素因果识别。
所有 `budget_match=notmatched`、equal_FLOPs=false、P1/P3/scientific/freshness=false。

## 独立付费计数

物理链是 sorted spaced JSON digest、单LF；native链是 sorted compact JSON digest、
单LF。独立实现两个reader，逐字节canonical格式、索引、previous与digest全核；
双LF、坏非空行、未知事件拒绝，不用filter empty。全部日志通过外部report中的
实际SHA绑定。真实partial/failed原件不删除、不重写。

物理/视觉：从实际 operation_attempted/completed 重算所有12资源，检查嵌套返回、
连续ID/units、分臂/总计及pending。clean时独立核每臂constructor1/reset2/set_state4/
close1、public=warm+control、physics=2*reset+10*public。

模型：对每个实际节点 native journal，根据 actual shape/resource/directions 独立
实现API/sample/context/row/J/grad/VJP/SVD/solve/replay/metric/schedule轴的delta。
不直接求和保存的delta：先按独立单位规则核它，再从durable resource attempts与
returns/failures重计。逐项核receipt.events的全生命周期、attempted/completed/caps，
分开 signal/program，汇总每臂、每节点与report aggregate对账。失败尝试不退款。
FLOPs/CUDA_ns写null；没有任何等FLOPs或实际wall匹配结论。

若producer为partial，保留六臂及六对null和可严格重计的付费前缀，complete=false。
若native journal自身未闭合或损坏则fail closed，不把损坏前缀认证为完整计数。
此工具不验证每个模型输出的科学正确性，也不能检测未记录的隐藏调用；原fullstrict
和真实producer instrumentation依然是必要前置。

## CLI（只读取真实完成产物）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```sh
.venv/bin/python -m pytest -q tests/test_agent_p3_commitment_numeric_recompute_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commitment_numeric_recompute_20261029_v1 --output results/agent_p3_commitment_controls_20261029_v2_MAIN01 --report-sha ACTUAL_REPORT_SHA --prepared results/agent_p3_commitment_controls_20261029_v2_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --strict-receipt results/agent_p3_commitment_controls_20261029_v2_consume_MAIN01.json --strict-sha ACTUAL_STRICT_SHA --receipt results/agent_p3_commitment_numeric_recompute_20261029_v1_MAIN01.json
```

输出采用独占新stem、flush/fsync新文件及目录；不能覆盖输入或原结果。
MAIN尚未提供本轮actual report/strict SHA，因此worker不运行实际analyze，不读取
正在生成的partial文件当作终态。只提供独立合成日志/手算数值回归，不伪造执行。
