# P3 shared-panel：最小结果审读判别表

范围：仅依据冻结设计/源码解释本次实验；未读取运行中 report、日志、partial 或 outcome，未调用生产/full strict，未改 runner、summary、阈值或选择配置。本文写于 MAIN 生产期间，不冒称新增事前预注册。使用 scientific-critical-thinking 技能区分局部可识别命题、合同覆盖与科学推断；不扩展论文或新实验。

## 1. 审读对象与固定比较

两 known roots × 四臂 = 八 episodes。每根顺序：fixedH1_endpoints、fixedH1_midpoints、adaptive_endpoints、adaptive_midpoints。每个到达的 b5/10/15 都支付 H1/H3 两个完整4候选面板，策略仅决定采用哪个面板的赢家首5动作；共享原 prefix5，物理目标20步。adaptive 保留原策略完整字段，但本 driver **只实现 horizon 的应用，不执行原 width/depth/type 全部决策**。

全部比较方向 right−left；两根均保留下面四对，共八对，不根据收益删根或选 bank。

|固定对照（每根）|可识别的有限命题，须先满足身份/执行完整性|不识别的命题|
|---|---|---|
|fixedH1_endpoints → adaptive_endpoints|在原 endpoint bank、同每5步反馈及共享付费流程下，采用原 adaptive horizon/拒绝规则这一**整体策略干预**的实现轨迹与任务轴差异|不是“增加反馈”的效果；不是纯固定H3−H1；不是完整 C3/reconfiguration 或自然成本最优性|
|fixedH1_midpoints → adaptive_midpoints|同上，但限定原 midpoint bank|不能只保留此 bank 若它较好；不能外推另一 bank 或未见任务|
|fixedH1_endpoints → fixedH1_midpoints|固定H1策略下，预先指定候选支持改变的整体轨迹差异|不是纯 horizon/调度收益；bank 改变可同时改变动作、轨迹及后续局部动力学|
|adaptive_endpoints → adaptive_midpoints|同一原 adaptive 规则下，bank 支持改变的整体闭环差异|不是仅“初始动作幅值”效应；后续状态/信号/拒绝也可能随轨迹改变|

“同流程”不表示两臂后续 context/J/候选数值相同：动作分叉后的状态是策略干预的后果。原生确定性与相同初始身份经 MAIN 完整 strict 认证后，可以描述这两个具体任务上的配对程序差异；固定顺序不提供独立随机样本或总体因果平均值。仅看两个不同状态节点的横向赢家，不能隔离该节点的 horizon 因果效应。

## 2. 直接用于审读的最小判别表

各行可并存：代码故障、覆盖缺口、指标不一致不是互斥的全局解释。先身份/执行，再覆盖，再逐轴效果；任何未核验项写“未知”，不得选最顺眼的单因。

|观察类型（条件描述，不是本批结果）|必须核对的固定字段/证据|允许的归类与措辞|禁止的跳跃|
|---|---|---|---|
|源/任务/goal/Q/normalizer/history/raw/PNG/ledger 不闭合，或 full strict 不完整|实际 prepare/report/strict SHA、原生动作与 RGB 接线、attempt/return/终态保存|工程身份/实现完整性未成立；受影响效果结论暂不可解释|把 consumer/接线失败直接算作理论或 adaptive 科学失败|
|adaptive 原始 `program_type` 或 `extension_type` 请求非 progress，例如 reconfiguration；`supported=false`，结果如约 refused|保留 `receipt.policy.decision` 全字段、`supported/refusal`；两面板完整付费；`selected_horizon/commands=null`；拒绝后无该节点 suffix dispatch，其他 episodes 保留|当前 progress-only 合同的**合法覆盖拒绝**；代码可正确实现当前有限合同，而原完整研究要求的程序类型仍未实现/未被本实验检验|称为代码 bug；也不能称“reconfiguration 执行后无益”或“P3 理论已被证伪”|
|`continue_expanding=false` 或其他不支持条件导致拒绝|逐项检查支持谓词：两 type 均 progress、continue 为 true、horizon∈{1,3}；保留原理由|按实际不满足项归类；不是所有拒绝都叫 reconfiguration|仅凭 `unsupported_original_policy_type_or_continuation` 通用字符串猜具体原因|
|应支持的 progress/H1或H3 被拒，或非支持请求仍派发/偷偷回退H1|原 policy 输出与冻结支持谓词、封存 commands、实际 dispatch chronology|有明确谓词/动作矛盾时是当前 driver 实现故障；无矛盾前只列待核|用事后回退“补齐20步”，或把未执行的候选结果当物理结果|
|合法拒绝让某臂更早停止、实际资源更少、cp20 缺失|`contract_refused`、拒绝边界、全八臂/24节点账、event-time terminal 与 cp20 分开|该有限适配器的操作覆盖受限；可报告拒绝发生和完成覆盖，不删槽；尚无同时间 cp20 任务差值|称为省计算优势；把缺失当0收益/失败 success；用停止时状态替代20步|
|progress 在支持域正确执行，完整实际任务轴较差/未改善|同根身份、封存选择、实际动作闭环及可比较检查点；success/XY/angle 原样分栏|若如此，可报告“该固定策略在这些已执行、可比较任务上的收益未显示/出现损害”；这是**有限实证不利证据**|由 known2 宣称所有任务/所有 horizon adaptive 无效，或宣布完整 C3/P3 一般理论失败|
|支持域内 adaptive 始终选H1，动作/轨迹均相同|每个到达节点的原 decision、raw signed bits、观察与终态身份|干预在这些节点未产生 horizon/动作分叉；可报告零实现差异及激活情况|把“未激活差异”当作 H3/reconfiguration 已执行且无益；也不自动证明 adaptive 有益|
|model-Q 改善而 actual-Q 或物理轴不改善|独立 mean192 Q、原 RGB latent、相同时点及动作见证；native 各轴|保存层级间出现不同向证据，待区分模型误差、表示/目标度量、动作路径及渲染等来源|唯一归因 encoder、策略或动力学；用 latent-Q 替代原 native success|

因此必须分开三个判断：①当前 progress-only driver 是否正确；②原始研究所需程序类别是否被覆盖；③已覆盖且真实执行的策略是否有任务收益。合法拒绝可同时满足①并暴露②的缺口；它本身不足以回答未执行类别的③。拒绝行为也可成为该适配器完成能力的有限不利证据，不能因“合法”而从整体分母抹去。

## 3. 统一审读约束

- **指标分栏：** native success 保留原联合 agent+object XY 与 native angle 判据；joint/agent/object XY、原 angle term、wrapped angle、velocity/native7 distance/reward、latent-Q 均分别报告。某轴改善不升级为整体任务成功；wrapped diagnostic 不替换原判据，velocity 不偷从原 native7 metric 删除。
- **时间分栏：** cp0/5/10/15/20 只比较同时间真实存在项；早停成功的 event-time success 与 cp20 success/null 并列。不给缺失造反事实20步标签，不做 complete-case 选根。可比对数量另列，原设计八臂/八对不改。
- **预测见证：** 已执行H1首包且对应终点实际 latent 存在时，才能讨论同动作模型→真实 endpoint 误差；H3在b5预测到20，但中途反馈可改动作，不能冒充全H3同动作验证。b10/b15的H3终点25/30超出本实验真实20步，actual/error 保留 null，不能作为失败预测计数或虚构真实收益。
- **资源分栏：** 完整每节点44 predict/2J/384VJP、每臂三节点132/6J/1152VJP只是完整路径合同参照。核实际完整向量（含 AE/predictor/pred_proj、solve/SVD、replay、native/encoder、失败attempt不退款）；提前终止或拒绝后不补工作凑数。AD-primal独立计量、scalar backward、J元素、engine backward/JVP、未测wall/FLOPs保留unknown，不由 J/predict 推算。共享付费长面板不是自然最便宜 fixed 基线；共同 cap 不等于实际墙时相等，更不等于等总算力/FLOPs。
- **证据层级：** 测试/summary verify/完整 strict 通过证明对应工程与保存层闭包，不自动证明任务收益。两个历史 known roots 不变成新根、八独立根或独立确认；无新置信区间、gate、通过阈值或 P1/P3/C3 签发。若全部工程闭合但覆盖/收益不足，仍应如实报告不足，不以新“宽松通过”替代原合同。

## 4. 冻结依据与交付状态

仅核读源码/设计，未读本批生产结果。锚点路径均相对 `le-wm-main`：

|冻结依据|本次核验SHA256|
|---|---|
|`experiments/spectral_reachability/agent_p3_shared_panel_20261029_v1.py`|`8cc8f796f313d4cc3f209ce231ebcaec4808f096acc02781cd6df734800dc464`|
|`experiments/spectral_reachability/agent_p3_shared_panel_summary_20261029_v1.py`|`69eb17b59cc5430a1644a604fcff7eec79ffcbe4c2488e2eae92a33b198e0e21`|
|`innerdocs/agent_p3_shared_panel_summary_20261029_v1.md`|`0806c5956e4bcf99923e88e9de7bf13955c97470445e65cbaf3b01bcf4d03896`|

纯文档交付，无新增可执行代码、无需新增测试命令，未重跑旧测试。实际结果归类等待 MAIN 完整结果与认证；本表不预填任何本批 outcome。
