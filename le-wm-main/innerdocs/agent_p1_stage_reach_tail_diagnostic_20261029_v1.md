# P1 stage+reach 断点与 P3 统一尾段算术

本件实现 P1 实际已封存 selector derivative 的独立 saved 算术，以及未来十支
settling 结果的 RAM 投影算术。它不是原生物理 driver、尾段 receipt 或生产授权。
不修改原 gate、已封存文件、模型、候选或训练；不执行 NN/physics/upstream fullstrict。

## P1 实际输入与可识别范围

唯一实际输入 `results/agent_p1_terminal_witness_selector_20261028_v1_MAIN01.json`，
SHA `a3a317b557da8bb6eb7a38d582f7df4d230b1b849cced00eca1add85403e63d9`。
这是已封存 saved derivative，不是新 source receipt；本件只认证其字节和内部完整
8 panels、4×42+4×222=1056行、原排序和保存成本，不重新执行或声称重建上游 fullstrict。
四 root-condition 只来自两 known roots；两个库首42重叠按原分母保留。

对于每个候选，相对于本 panel **原 stage+reach native float winner** 输出：stage差、reach差、
精确和、原float加法保存的score差、两者的精确舍入差、非线性 model terminal差，以及相对
initial/zero/stage-greedy三种不同 uphill 标识。Fraction 将保存 float 精确表示；原 winner
始终按原 native `(original_objective,index)`，不能用精确和重选。model terminal 更好不等于真实
收益；`real_benefit=null`。不读物理结果、不改 selector/beta、不按结果改候选。

该原 P1 selector 的 `stage+reach` 与 P3 原 `I>I_zero && R<R_zero` 后按 `(I,index)` 的资格
和排序不同；这里 P1 的分解不能直接解释 P3 immediate false，更不能证明唯一根因。

## 十支统一尾段：当前仅 RAM 数值实现

独立 `tail_ram(rows)` 固定 endpoint `(r0C,r0Z,r0P,r1Z,r1P)` 各 U/Z2，接收完整10行
`index/endpoint/root/role/kind/status/goal7/cp15/cp20`，计算两个时刻原native7距离、reward、
success、joint/agent/object XY、agent velocity error、raw及wrapped angle，输出五对 Z2−U，
tie/null分开。合法早停cp20=null，保留10支/5对及原4C/8比较/3 no_witness分母；不能补零。
同对goal7包括速度和signed-zero位型必须一致。`goal7[-2:]` 为原非零目标 agent velocity，
误差是与目标的差而不是速度幅值；native success不含速度。原angle历史公式保留，wrapped仅诊断。
输入只是 RAM projection，不认证原轨迹、CP15 guard、物理或 receipt，不开放正式 saved-tail CLI。

新 tail 应按冻结 `agent_p3_nongreedy_counterfactual_contract_20261029_v1`：十支全为原生
相对 raw(+0,+0) 5步＝0.5秒，完整 fresh env replay，从原 control15继续到20；原cp15 state、
snapshot、PNG、history、goal和动作一致且 guard 持久化后才能dispatch第16步。零动作通过PD阻尼
不保证位姿静止，接触可推物体，更静止不保证更接近非零速度目标。不得改原cp15结果、gate或预算。
全十若完成，300 public、3040 physics、960 render；提前停止不续跑、missing不补、费用不退款。

**仍未实现**：真实原生十环境 tail driver、source-bound cp15 guard、完整 ledger/PNG预算和
saved consumer。不能以本件 RAM 投影或旧十支 receipt 作为它们的替代；后续需 MAIN 审批与独立审读。
固定目标图不直接编码目标 agent velocity 是表征边界，不是根因唯一证明。MAIN 的独立速度语义
audit `results/main_p3_velocity_semantics_audit_20261029_v1_MAIN01.json` SHA
`8526bc6f1a914958858add0a39ea5190201da744455cea25f692a4fee7ec6927`；其零目标速度重算
仅固定轨迹后验敏感性，不是新gate或方法收益。

## MAIN 命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；输出用 canonical 绝对路径且不可覆盖。
此命令只产生 P1 saved 算术，不产生尾段结果。MAIN负责实际运行和整份verify。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_stage_reach_tail_diagnostic_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_stage_reach_tail_diagnostic_20261029_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_stage_reach_tail_diagnostic_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_stage_reach_tail_diagnostic_20261029_v1 verify --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_stage_reach_tail_diagnostic_20261029_v1_MAIN01.json
```

测试只使用 RAM fixture/临时磁盘，非真实结果。`experiment-code` 技能仅用于固定设计及失败
记录；未采用其训练、调参、绘图或择优建议。
