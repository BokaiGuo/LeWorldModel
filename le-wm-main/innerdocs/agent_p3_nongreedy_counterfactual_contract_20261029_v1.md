# P3 非贪心断点：固定支持集反事实与统一 settling tail

仅新协议与可执行 RAM 合同。已知本批结果之后提出，**不是原实验盲预注册**。
新机制实验仍须 MAIN 审批、真实 source-bound 接线及逐次付费；本件不运行或导入
模型、torch、环境、physics、upstream fullstrict，不拟合/部署 selector，不改科学门。

## 1. 先核已有工作，明确不是同一个断点

读取外层 `docs/AGENT_P1_THEORY_GATE_ANALYSIS_20261028_v1.md` §4–7/10、
`innerdocs/p1_constructive_mechanism_prerequisites_20261027_v1.md` 与既有 terminal-witness
selector 文档。P1 已有 selection → model/real → latent/native 三断点、误差展开和
stage+reach 命题16；不重复其1056候选选择器，不重复 MAIN 的 O−T 和 secant16 标量分析。
所引用实际源码/理论文档精确 SHA 在新模块 `PINS`，build 前全部核对；不导入其算术。

关键区别：实际 P3 先要求 `I > I_zero && R < R_zero`，再在 **原 family 内**按
`(I,index)` 升序交付；不是按 `I+R`，也不是按 R 排。P1 的 beta1 反例只能说明
sum-selector 可拒绝有益重配置，不能据此解释本次实际 P3 的 immediate false。
实际 immediate false 也不能从“模型里 eligible”推出是 selector bug；真实目标、
时间、动作和 metric 的对接仍需分别检查，不能签“stage 权重是唯一根因”。

在 innerdocs、外层 docs、现有 StageC 协议中检索 `settling/settle/no-op tail/静置/free settle`，
未找到同十支的已实现统一尾段。已有 PushT feasibility 文档反而明确 settle 必须计入 H；
旧 cube gravity settling 是另一任务，不是此处已有对照。不声称穷尽所有历史缓存。

## 2. 最小可执行 slice：同一完整 bank 的零新模型影子检查

四组顺序 `(slot,root,H)=(0,0,1),(1,0,3),(2,1,1),(3,1,3)`。
每组原 R8 + 原 secant4 + **同一个原 paid-zero1**，合计52槽；两 family 分别与同零比较，
零不重复收费，不合并 family 伪称原 union selector。缺失、退化、失败槽均保留；不补方向/半径。
只有两个 known roots，fresh roots=0；H1 与 H3 终点不同，禁止跨 H 排 Q。

`analyze_bank(rows)` 接受精确52行 RAM projection，非“任意字符串自签 authority”：

- identity：`slot/root/horizon/family/index` 必须逐项匹配 `roster()`。
- `status=complete,reason=null` 时 `immediate/reach/terminal` 为源已存非负标量，
  `edge=(5H,2),suffix=(5,2)` 为原 float32 精确值；全 raw 域 [-1,1]，禁 clip/tanh/重算替换。
- 缺项必须 `status=unavailable,reason=原原因`，五个数值/动作字段全 null。
- `source_pointer` 是明确 JSON pointer，仅可定位，**本函数不认证其所指字节**。
  零 action 须正 float32 零位型。完整新 source-bound adapter 尚未实现，不可拿 RAM 输出生产。

MAIN 后续 adapter 所需真实字段已经存在：R report 每组 `control/rows[8]`，secant report
`rows[slot].result.rows[4]` 与 prepare `source_groups[slot].zero`；完整 task/local task、
root_context、原 goal/Q、mean/std、模型/源闭包、整行 array bits 和 full receipts 必须先认证。
`terminal` 映射原 `bounded_suffix_cost`，suffix 映射原 `suffix_actions`，不是凭 J 推出真实收益。
若 ABI 不同必须拒绝/升级新 adapter，不能猜字段或用默认 false。

每个 family 保留四种完整稳定排序：原 eligible→`(I,index)`；固定 beta1 的 `I+R`；
R-only；原有限 suffix 非线性 model terminal T。后三者只是**不可部署影子检查**，不择优选新策略。
shadow support 包括原零，tie 时零排在原候选之后；原资格严格 `>`/`<`，无 epsilon。
Fraction 精确表示输入标量，输出 `stage_plus_R_delta`、`terminal_minus_R` 有理数；
不是新神经重算分数，更不是提高科学门的真值证书。

|固定观察|可检验的最小解释|不能推出|
|---|---|---|
|某候选 T 低于 zero，但 `I<=I0`|原 stage 条件排除了 model-terminal 更优的库内候选|物理上有益、应该改 gate|
|T 更优但 `R>=R0`|reach 条件与有限 suffix model T 有断点|J 错误、全域不可达|
|eligible 但原 I 排序与 T 排序不同|交付排序与 model T 不同|可按实际结果重选|
|原 eligible 空/退化|当前固定支持集未激活原策略|机制被物理证伪或连续空间无解|
|模型 eligible 而原实际 immediate false|预测/时点/metric 对接须分别审读|P1 sum-selector 权重造成|

本 slice 只用继承的完整付费 bank，新增 NN/AD/physics=0。不得读取 evaluator-only
物理 outcome 来排序或改变候选。所有4 logical C/8 comparisons/3 no_witness原封不动；
新影子排序不会给缺失 C 补 witness。P 基线不塞入 H3 bank，不补不存在的 raw15。

## 3. 如需新付费验证：所有52槽相同两布局，禁止 padding/退款

`replay_schedule()` 固定每行 whole 后 packet_segmented；完整原 edge 后接**原已存 suffix5**。
每次从该组同一 actual-cp5 context 独立初始化。raw 长10/20，即 H1/H3 的2/4 packet；
全104验证槽、最大312 predict，AE/predictor/pred_proj各312，AE和predictor context tokens各936，
pred_proj rows936，stage+terminal 各每次一项，总208 metric。J/VJP/solve/SVD/encoder/projector新增0。
unknown FLOPs、scalar_backward 保留 null，不宣称 equalFLOPs；继承 J/VJP 原账另列不能退款。

原 `NativeTree.replay` 支持整 packet 长度，但 `NativeDelivery` 仅5/15、H3 split[5,10]；
**不能把10/20传给后者**。这里新 packet-segmented 是每5步重建，非旧 delivery 布局。
两个布局均须独立保存 native 全 IO/完整移窗 context/raw/token，并核所有边界与原
edge path/suffix_path；不以仅末帧/allclose 或继承另一次调用通过替代。布局不同若位级失配
保留失败，不放宽原 exact 门。新 driver/consumer 和出版体积证明尚缺，不能宣称可直接 run。

`ReplayRAM` 仅证明固定顺序和整单元最大 reservation；失败后该槽 actual=null、不能重试或退款。
它的 reservation 不是实际运行资源；实际 driver 仍必须逐模块 intent/return/failed计量，
包括部分 native 调用。缺失原候选的验证槽应由未来 adapter 明确标不可派发，不造调用凑312。
这些最大整数成本用于共同数据集验证，不是公平算法算力比较。

## 4. 新时间对照：全十支统一原生正零5步

尾段长度固定为**一个原控制 packet=5步**，理由是既有时间分辨率，不据速度曲线选长度。
public control10Hz，因此新增0.5秒，不是原时间预算内免费追加。
十槽顺序为原五 endpoint `(r0C,r0Z,r0P,r1Z,r1P)` 各 U/Z2，全部同一正零 raw5。
原 PushT `relative=True, action_scale=100`：raw零令每次 public step 的 PD 目标为当时 agent位置，
仍有速度阻尼、接触和物体运动。不是冻结环境，也不是只干预“速度分量”的因果实验。
安装原生链的 PD 为 kp100/kv20、dt=.01；`_get_obs` 的原native7末两轴是 **agent.velocity**，
不是 block velocity。`_set_state` 的局部变量虽称 `vel_block`，实际赋给 agent.velocity；
本件不据变量名换轴。所有velocity误差/归因均只指agent速度，不能称object速度改善。
更重要的是原 goal7 的末两维也必须逐位继承和认证，**不可假设目标速度为零**。
MAIN 报告 root0 goal agent velocity 非零（其独立source/snapshot核验仍在进行）；在核验完成前
不据此扩张为两个根的结论。`velocity_error=||state_agent_velocity-goal_agent_velocity||`，
不是 agent 速度幅值；零动作让 agent 更静止也未必使该误差、native7距离或reward改善。
各分支以其原实际完整 goal7 为准，goal与state的速度轴不替换成零、也不改原reward/gate。
所有动作仍按原 native 路线，不用 normalized零冒充 raw零，不 clip/强行终止/重置速度。

必须全十个 fresh env 从原 reset/placement→warm10→原 raw15→tail5 重新执行；
不能 hydrate cp15、拼旧结果或在已结束 env 上偷偷续跑。原 control6/11 guards 保留，
tail control16 前另核原 cp15 **完整已记录 state/snapshot/三PNG/history/goal/actions**，
guard fsync 完成后才能写 step16 intent/dispatch；state-only相等不足。未记录缓存仍 unknown。
`TailRAM` 只测试该控制顺序，`cp15_exact` 布尔必须来自未来真实完整 validator，不能自行自签。
原生 terminated/truncated 即合法停止，所有缺 cp20 保持 null；到20才停止者仍有真实 cp20。

|全十完整时上限|原15步配方|新20步配方|增量（不能只付此增量）|
|---|---:|---:|---:|
|env/constructor/close|10/10/10|10/10/10|0|
|reset / set_state|20 /40|20 /40|0|
|public (warm+plan)|250|300|50|
|physics (reset4/支+每public10)|2540|3040|500|
|off/on/off renders|810|960|150|

goal_render20、goal_bind10、goal_setter30；没有新 NN/encoder。每支 warm0..10 与plan0..20，
共32位置×3=96PNG。不能以 old810 已生成而只付150新帧。
沿原PNG单帧172056上界，96帧16517376 bytes；原38MiB metadata+8MiB reserve 的纯字节和
为64751872<64MiB，**仅是PNG/原quota算术，不是新增事件已证明能放入metadata quota**。
未来实现须重做完整事件/guard/报告预算与临时不可压缩PNG回归；超额先报告不默扩。
建议沿原每支 wall900s、output64MiB/reserve8MiB、diskfloor200MiB作开发约束，
全十总预留640MiB、顺序墙时上限9000s，不把每支900当整批900。须 MAIN 另审批才可执行。

固定原 cp0/5/10/15 全表与新 cp20（absolute30）；步16..20全部返回保留，不取最优时点。
native success、原native7距离/reward、jointXY、objectXY、agent velocity、raw-angle、diagnostic wrapped-angle
各列，不统一成新标量。每个时点五对均以 `error(Z2)-error(U)` 定义方向，正值偏U，tie/null另列；
原 C 对 Z/P 的四逻辑槽/八比较继续保留，三 no_witness 的六比较不可补零。
cp20−cp15 的逐轴变化只能描述“统一控制后的持续性”；不能宣称隔离了速度惩罚的唯一因果来源。
cp15及原即时代价条件即使 false 也保留；cp20绝不替换原 G/gate/同预算结论。
不做新增真实编码；若以后需要，另批10encoder/projector，不偷计入本native诊断。

## 5. 可执行命令、测试与明确未完成项

固定根 `/home/gbk/code/LeWorldModel/le-wm-main`；所有CLI参数必须canonical绝对路径。
本件build只是封存本协议/资源调度/source closure，不生成任何 ready/release authority。
MAIN 可把新输出路径改为独占结果路径；worker仅运行临时盘fixture。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_nongreedy_counterfactual_contract_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_nongreedy_counterfactual_contract_20261029_v1 build --output /tmp/agent_p3_nongreedy_counterfactual_contract_20261029_v1_MAIN_contract.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_nongreedy_counterfactual_contract_20261029_v1 verify --output /tmp/agent_p3_nongreedy_counterfactual_contract_20261029_v1_MAIN_contract.json
```

测试覆盖52槽八panel、严格近tie/codec、P1反例与实际P3排序区别、104验证槽资源整数账、
失败不退款/actual unknown、十尾段全生命周期、guard持久化失败零派发、原生早停null，
真实 subprocess build→排他write/fsync→verify与篡改拒绝。全部为RAM/临时文件，不是科学证据。
尚缺实际52行source-bound adapter、付费验证driver/完整consumer，以及tail driver/consumer；
本件没有生产run命令，未发NN/physics/fullstrict、未读新结果调配置、未新增agent。
模型/solver/DAMPING/CAP/chart/metric保持原值；没有改原科学flags。所有新增准入flags均false。
