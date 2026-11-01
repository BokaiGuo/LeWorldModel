# P3 动作见证与整段 H3：有限解释及固定审读口径

2026-10-29。仅新增本文，使用 scientific-critical-thinking 技能区分观测、模型误差、策略干预及非识别边界。只读 MAIN 已完成的 action-witness 对齐产物、shared-panel/activation 及对应源码；未运行 NN/physics/full strict，未读取任何未来整段 H3 结果，不改旧文件/阈值/选择。本文是已有 shared-panel 结果之后、整段 H3 结果之前的分析约束，**不倒签原实验预注册**，也不替代 Nash/MAIN 最终冻结的执行合同。

## 1. 精确来源

以下路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`，SHA 为本次只读实际核验值，不是缩写或未来值。

|对象|SHA256|
|---|---|
|`experiments/spectral_reachability/main_p3_action_witness_alignment_20261029_v1.py`|`0fbb74386c6c90a920d3e000f5df9a96f29c8ef1ce0f8173bb507cba63a6257d`|
|`results/main_p3_action_witness_alignment_20261029_v1_MAIN01.json`|`7b0031e127848ccacc474699ae8e98d7aaa9b3cb633f00d740b1ee201d4d5911`|
|`results/agent_p3_shared_panel_20261029_v1_MAIN01/report.json`|`668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a`|
|`results/agent_p3_shared_panel_20261029_v1_consume_MAIN01.json`|`d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd`|
|`results/agent_p3_shared_panel_summary_20261029_v1_MAIN01.json`|`4e89812a384a352840f41b84ef9a0ac1ea07622288949c6d5ff6c5795cc28224`|
|`results/agent_p3_horizon_activation_summary_20261029_v1_MAIN01.json`|`c2e933932b2b6a924f565295b72e676849470c7ac21f98dbb21a1086acf9e2ab`|
|`innerdocs/agent_p3_shared_panel_interpretation_20261029_v1.md`|`2dfe23c50a83f41b0ad59e897616a6c113d878d0345cad404a42d401e1a11d17`|

这里只读继承指定 strict 的认证；没有自行重建原 native/encoder/preprocess 或完整 strict。action-witness 工具核原 float32 raw signed bits、winner身份、完整动作长度和实际返回区间，不新增真实轨迹。它不是第二次独立物理重复，也不从原结果重新选赢家。

## 2. 本批究竟执行了什么

固定两个历史 known roots × 四臂八 episodes，24节点、48面板。activation：24 sealed、0 refused、0 not_reached；chosen_H1=dispatched_H1=12，chosen_H3=dispatched_H3=12。四条 fixedH1 分支均 `[1,1,1]`，四条 adaptive 分支均 `[3,3,3]`；每次只释放所选赢家 FIRST5。八对已观察的 raw/state 首分叉均在step6；raw前5步在 activation 的 summary ABI 内未知，不将其缺项自行填成相同。

因此本批 horizon 干预确实激活，不能解释成“始终H1所以没测到H3”。但它也不是完整15动作 H3 的承诺执行：adaptive在b10/b15重观测、重规划，改变此前H3的后续动作。24 progress 节点被支持不证明未出现的 reconfiguration/type/width/depth 已实现，原完整 C3 覆盖缺口仍独立存在。

### 动作见证全48槽

|冻结对齐分类|数目|可解释的误差|不能解释|
|---|---:|---|---|
|H1 same_action_endpoint|12|全部为 fixedH1，b5/10/15分别预测至10/15/20；相同已执行 raw5 的 encoded endpoint/Q 误差|不能外推 adaptive 当前状态下H1反事实精度或H3误差|
|H1 changed_action_trajectory|12|全部为 adaptive 的未采用H1；只能描述不同动作程序预测与实际轨迹之差|不是H1动力学预测失败，也不是与H3的真实同状态动作效果比较|
|H3 changed_action_trajectory|20|存在至少一个已观察raw差异；其中b5的8槽有actual20，后续12槽终点超20且缺actual|有actual20也不能直接叫整段H3同动作预测误差；无actual更不能计预测失败|
|H3 incomplete_action_witness|4|adaptive b15：已观察首5相同，但剩10步未执行，终点30未知|不能把“已比较动作全相同”升级为完整H3见证或拿cp20代替cp30|

重要：`changed_action_trajectory` 与 `future_actions_missing` 可并存。20个changed H3中，8个b5是完整长度但动作不同，12个b10/b15既已不同又未观察完整未来；另4个b15仅缺未来而尚未出现已观察差异。不能把20+4误写为24个可评分终点。

四条 adaptive 的b5 H3均从step11第一次不同于原完整winner raw15；b10 H3均从step16第一次不同且缺最后5步；b15首5匹配但缺后10步。四条 fixedH1 的未选H3从当前下一步即不同。**当前零个完整同动作H3终点**；activation将全部H3保守标为descriptive与新精确对齐不冲突。以后若逐位完整H3动作偶然都相同，也可以形成同动作见证，不能先验规定“有反馈必不可能匹配”。

## 3. 同动作误差能说明什么

令同一实际观察context、同一goal192下，`p`为原模型执行指定动作后的预测latent，`a`为同动作真实RGB编码latent，`g`为原目标；`Q(z)=mean192((z-g)^2)`。原独立summary给出：

`Q(p)-Q(a) = 2 mean((a-g)*(p-a)) + mean((p-a)^2)`。

- 12个合法H1槽可讨论该输入/动作/时间下的 model→observed latent fidelity；Q差的正负表示该槽目标距离的过高/过低预测，不是physical success的正负。
- 小Q差可能来自交叉项与非负平方误差抵消，不能据此说latent预测准确；非零latent误差不必造成Q排序错误。单条动作的误差也不能证明候选之间发生排序反转，需要各自真实同动作终点和margin。
- H1同动作并不唯一归因“动力学模型”：表示、完整context可观测性、隐藏状态、渲染及其与目标的关系仍可能共同作用。位级动作匹配排除的是本槽动作见证不一致，不能排除其他来源。
- 不同动作时，以上代数恒等式仍成立，但`p-a`混入动作干预效应，不能把交叉项/平方项当纯模型误差。这是现有H3描述值的主要识别限制，而非证明模型一定准确或一定错误。
- native success/XY/angle/velocity/native7与latentQ分别报告；不能由latent误差或modelQ改善直接推出物理损害、encoder普遍不对齐，或P1非贪心机制成立/失败。

## 4. 四分支整段 H3：结果之前固定的对照与前置

只读 Nash 开发中的 `agent_p3_h3_witness_physical_20261029_v1.py` 的 `ORDER/select_witnesses/contract` 接缝：四分支为 `(root0,off,endpoints)`、`(root0,off,midpoints)`、`(root1,off,endpoints)`、`(root1,off,midpoints)`；固定取对应原 **adaptive b5 已封存H3 winner**。独立root reset/warm10，原prefix5 + 原winner raw15，control20；不重新选候选、不修改后10动作、不在中间反馈重优化。开发源码不在本文冒称最终冻结authority，执行prepare/report/strict SHA及实际最终源码SHA由 MAIN 后续绑定，缺一不以本文补发认证。

### 身份与同动作前置，先于效果解释

1. 四槽全保留，task/root/off-renderer/native goal/goal192/Q/normalizer、原warmup/prefix动作及完整原b5 request/whole winner均绑定原authority；完整raw20等于prefix5+literal winner raw15，逐位无clamp/补齐/重排。
2. 从root真实重放，在第一条suffix dispatch前验证cp5 state7/snapshot/PNG与原对应adaptive分支精确一致。失败是新driver重放/身份不闭合，不能当H3预测负证据。不能只在执行结束后回顾检查。
3. 新整段H3与原adaptive的step6..10本来就是同一FIRST5，因此cp10原生state/snapshot/PNG应构成同动作控制；不一致先定位初始化、native执行、渲染或认证，不直接解释成策略收益。若原生早停导致未到达，保留missing，不造一致性。
4. 原b5模型输入是当时真实编码的三帧/两个history tokens，不用新预测或只替last latent；cp5物理重放一致是必要控制，不自动证明所有模型输入身份。原request身份和编码链仍需继承/核验。
5. 只有真实完成选定raw15、实际cp20存在，并有同renderer真实RGB→原encoder/projector192的严格产物，才能计算整段H3同动作latent/Q误差。纯物理driver `new_encoder=0`，**只有PNG/state时不得伪造actual latent或报H3 Q误差**；需要后续明确授权的RGB消费，本文不新建runner。

### 固定比较与展示

|比较|固定分母与时间|允许解释|非识别边界|
|---|---|---|---|
|新commit-H3 → 原adaptive MPC（方向统一commit−MPC）|每root×bank各一对，共4对；cp0/5/10/15/20|从同b5状态与同首包出发，承诺旧H3后续动作 vs 后续反馈重规划的整体程序差异|不是纯反馈感知效应；两者都使用H3规划，不是H3−H1因果效应|
|原b5 H3模型预测 → 新commit真实同动作路径|固定4条、原whole三edge分别对应cp10/15/20；无新预测|完整latent/历史绑定且实际编码存在时，逐edge/终态同动作模型误差|未编码不是0误差；不把不同状态下晚节点H3预测混入这4条|
|新commit-H3 → 原fixedH1，同root/同bank|辅助4对，同cp分栏|两个已封存完整程序的有限任务轴差异|与主4对分开；不是候选随机分配、纯horizon效应或新根确认|

主比较一开始即保留两root×两bank，不看效果挑bank/时间；不把辅助对照中较好的一个升格主分析。主方向commit−MPC：Q及距离负为改善、reward正为改善；success用左右bool/null和配对转变，不强行压成连续综合分。不给新最小改善阈值，不做新CI/p值或显著性签发。

逐槽完整呈现 native success、joint/agent/object XY、原native angle term、wrapped angle、velocity error、原native7 distance/reward，以及真实返回步数/terminated/truncated。成功判据沿原native，不用wrapped angle或删velocity重定义；encoded Q仅另列。cp间不选最佳时刻替代cp20。

早停成功/失败均保留四槽：event-time terminal success/step与cp20 success/null并列，不将停止状态复制为后续检查点，不把未完成15步当完整H3预测失败。真实完成长度、可比对数、same-action eligibility分别报告；缺帧/标签/动作认证失败与合法native早停区分。全槽错误、partial、拒绝及资源attempt保留，不退款/补跑筛配置。

新commit物理成本与原MPC计算账单独报告：warmup/control public、physics、render、encoder/projector、全部模型predict/AE/predictor/pred_proj、J/VJP/grad/raw backward、solve/SVD/replay/metric/schedule及wall的scope。原shared-panel每臂132predict是原工程参照，不是新commit必须支付或等FLOPs证据；复用旧计划的生成成本与本次执行增量分列。未知AD-primal、scalar backward、FLOPs不补0，不用无新planner调用宣称自然算法更廉价或公平算力优势。

## 5. 结果审读最小判别表

|未来观测（条件，未读未来结果）|可支持的有限结论|仍不能推出|
|---|---|---|
|cp5/同首包cp10或raw绑定失败|相应重放/接线身份未成立，先定位工程层|H3机制、模型预测或P3理论负结果|
|完整同动作H3 latent误差大，Q偏差或margin翻转|在这些固定动作/任务/上下文上的预测泛化问题；排序翻转须真实两动作成对验证|唯一根因、任意任务失败，或原P1 gate自动可放宽|
|latent预测接近而native轴不利|两层目标未产生相同收益的有限证据，进一步区分目标表示与动作实际效果|单因encoder错误或整体任务成功|
|commit优于MPC|这些固定程序上，后续反馈重规划没有带来同向任务收益/可能损害，按实际各轴限定|一般反馈有害、固定H3普遍最优或full C3通过|
|MPC优于commit|这些固定程序上后续反馈重规划带来有限收益|H3本身预测正确、策略总体最优、独立根确认|
|各轴混合、零success或缺项|如实分轴/分母报告收益边界、早停与工程缺口|挑一个轴宣布P1/P3成功，或以两个known roots证伪一般理论|

整段H3新增的是此前缺失的**同动作物理见证路径**，不是自动的P1非贪心机制、强基线统计或freshness证据。完整C3原type/width/depth/reconfiguration覆盖仍须独立实现验证；本文不签P1/P3/C3，不改研究目标/门槛。

## 6. 交付状态

纯文档，未运行测试（没有新可执行代码）、生产、NN/physics或full strict；只新增本文。未来四分支的最终source/prepare/report/strict/RGB authority和结果尚待 MAIN，本文不猜未来SHA、不提前认定正负结果；原封存文档和所有源码/结果保持不变。
