# P3 shared-panel 独立 saved-only summary v1

## 范围与当前状态

仅本前缀 source/tests/innerdoc；保留全部旧源码与产物。仅标准库+NumPy，不 import producer/producer 算术，不运行训练、神经、物理或 full strict；不使用子智能体。不调 policy/Q/threshold，不依据 outcome 选择根、bank 或指标，不签 P1/P3/C3/freshness。

使用 experiment-code 技能的独立实现、保存记录与回归要求；本任务明确禁止生产和实验，因此不生成训练/运行/图表等技能通用模板。Nash `agent_downstream_resource_contract_20261029_v1` 三件套不改，也不新增 import 耦合；本工具独立展开原27资源轴并保留其未测量项为 unknown。

实际输入 ABI 已直接核读 Bohr 新 `agent_p3_shared_panel_20261029_v1.py` 的 `contract/node_accounting/plan_node/episode/consume`，非旧 effective-schedule 重标。2026-10-29 文档更正时重新核验当前 producer 字节快照 SHA：`8cc8f796f313d4cc3f209ce231ebcaec4808f096acc02781cd6df734800dc464`（与上次快照相同）；当前 consume 各分支均含 `prepared`，资源行亦含 `source_slot/mode`。快照不替未来生产 source 签发身份；实际分析逐个核验 prepare 的完整 sources SHA。会话没有消息工具，未伪称直接发消息给 Bohr。

**未完成项：** MAIN 审读与独立测试；Bohr 终态三件套冻结；实际 prepare/report/完整 strict 路径和 SHA；MAIN 生产后 saved analyze/verify。worker 未运行任何生产消费。没有真实运行数据，fixture 不作为科学证据；未知/不匹配 schema 直接拒绝，不自动回退旧 ABI。

## 已对齐的真实 ABI

- schemas：`agent_p3_shared_panel_20261029_v1.{prepare,report,consume}`。
- prepare：`contract/tasks[2]/sources/expected_model_hash`；model task 含 `goal[192]/projection[192,192]/root_context/action_mean/action_std`。
- report：`prepared/prepare_sha256/rows[8]/nodes[24]/counts/per_episode_neural/journal_sha256/journal_chain_tail`，及实际 model/source/wall/memory 字段。
- 当前 strict 含 `prepared` 字段；它对本 summary 是可兼容的 optional 元数据，本 summary 不读取或要求它。必需项为 `status=saved_shared_panel_verified_NOT_admitted`、`complete=true`、`report_sha256`、`prepare_sha256`、完整 `per_episode_neural`，并核 own schema/分母/不准入标识。report 的 `prepared` 实际绝对路径、`prepare_sha256` 以及 `execution_binding.json` 另行精确绑定。缺实际 SHA 不可用状态词代替。
- node：`request.task`、`receipt.panels[2]`（H1、H3）、`receipt.policy.decision/supported`、`receipt.selected_horizon/commands`；没有杜撰顶层 `receipt.decision`。
- panel：完整4 candidates、`bank.actions/fractions`、whole winner 对象、`selected_path`。重新计算全部候选终态原 Q，验证原 stable `(cost,index)` 赢家及独立 saved replay，不重新生成动作或发布新 selector。
- 当前 `per_episode_neural` 同时含 `episode_id/source_slot/mode`；本 summary 按 `episode_id` 标识并核序，`source_slot/mode` 对本 summary 为可兼容的 optional 元数据，不是缺失字段。实际 report 与 strict 的完整资源行仍须精确相同。
- 原 latent receipt 是普通 float32 `.npy`，含 `path/file_sha256/array_sha256/shape/dtype`；不是其他 RGB companion 的 GZ schema。检查原文件 SHA、raw array SHA、完整 NPY 无尾字节和实际192维。

## 八臂、固定配对与缺项

每根顺序：fixedH1_endpoints、fixedH1_midpoints、adaptive_endpoints、adaptive_midpoints；roots0/1，共8 episodes、24 nodes、48 panels、192 candidates 的固定设计分母。未到达节点保留 null panels/resources，不记完成零候选。每根四项预声明对比、共8对，方向统一 right−left：

|对照|left|right|解释|
|---|---|---|---|
|同bank policy|fixedH1_endpoints|adaptive_endpoints|共享已付面板上的policy差异|
|同bank policy|fixedH1_midpoints|adaptive_midpoints|同上|
|同policy bank|fixedH1_endpoints|fixedH1_midpoints|候选支持差异，不是纯horizon因果|
|同policy bank|adaptive_endpoints|adaptive_midpoints|同上|

每对保留全部0..20步骤，另显式给 cp0/5/10/15/20。native success 用左右独立 bool/null；XY（joint/agent/object）、wrapped angle、原 native angle term、velocity、native7 distance/reward 独立分栏 delta/sign，actual latent Q 另栏。零差为 tie，真实缺项为 missing，不以容差改符号。reward 高为优，其余距离低为优，不合成总分。

提前终止不补帧、不补 latent、不用停止终态替代 cp20。单列真实 event-time terminal step/success 与 `native_success_at20`，例如14步成功时后者仍 null。native warmup 中止而无计划观察时，各控制槽均 null。干净 unsupported-policy refusal 保留已付双面板与全部候选，未来控制槽 null；失败/partial full strict 不获成功 summary。

仅两个历史 known roots，8臂不是8个独立根；不计算独立根 CI/p-value，不签独立确认或 freshness。

## 独立算术与任务绑定

原 identity192 Q：float64 `mean((z−goal)^2)`，分母192，不是 sum。核全部候选原 cost、实际 encoded Q、真实 terminal20 Q。核原全3 observed latent 滚动和两个 incoming action-history token 的原 float32 `(raw−mean)/std`，无 `/100`、clamp 或拟合。

有实际同时间 latent 时独立计算：

`modelQ−realQ = 2 mean((actual−goal)*(pred−actual)) + mean((pred−actual)^2)`。

residual 界仅用于 float64 恒等式舍入：k=16×192+64 的 gamma(k) 乘事前固定绝对值尺度。它不是科学 tolerance，不将小 margin 置零。

**H3不能偷换为同动作见证：** b5 H3预测到20，但中途新反馈可能改变未来动作；只作描述性 forecast-vs-real。b10/b15 H3分别预测到25/30，物理只运行到20，对应 actual/error 项必须 null，明确 `beyond_executed20_lookahead`。只有实际采用的H1首包具备相同动作预测终点语义；仍不自动证明物理收益。

native7 从 strict 认证的原观察 goal/state 独立重算：包含 velocity，保留原 `min(abs(angle),2π−abs(angle))`（可负）及原 success `jointXY<20 && native_angle<π/9`；wrapped angle 单独诊断，绝不替换原成功标准或原 state distance。核实际 reward/terminated、每条 dispatch/return raw signed bits、同根各臂 native goal 一致。

## 完整资源向量与 unknown

独立按 resource/shape/directions 展开原27轴，逐操作重算 attempted/completed、phase、caps 和 inclusive wall，拒绝仅给 predict 的账。分别报告 API/samples/context tokens、AE/predictor/pred_proj、J/VJP/grad/raw callbacks、solve/SVD、compile/replay/metric/schedule；native/visual 从原全事件另计，包括 warmup public steps，不把 control20 当全 public 成本。compiler encoder0与真正 RGB encoder 成本分账，不覆盖实际采集账。

原完整每节点44 predict/2J/384VJP，每臂3节点完整132/6J/1152VJP；这是完整路径预算参照，不替代早停实际。原完整运行的 fixed 未用H3仍付费，是受控 shared-paid pipeline，不是自然最便宜 fixed baseline。

failed attempt 不退款，returned 子操作保留完成。ledger按entry顺序：父failed可在数组中先于已发生的子entry；按嵌套时间区间和最早失败边界验证，不能当return排序误拒。失败后新 dispatch 拒绝，pending native attempt 不伪装完成。

`scalar_backward/AD_primal_packets/jacobian_elements/engine_backward/independent_JVP`、真实视觉 samples/frames、total_FLOPs 均显式 `status=unknown, attempted=null, completed=null`；不凭192VJP或J shape猜测、补0。每episode whole wall 未独立实测，scope=`whole_episode`, status=`unknown`。node `Ledger_lifetime` 的 wall_ns 与原 operator inclusive wall分别报告、不相加；whole-job wall 单列，原公共 wall cap另列，不要求实际墙时逐位相等。不运行 Nash `validate_shared_panel`，因此不凭缺失的六SHA identity发资源匹配证书；不声称 equal_total_compute/equal_FLOPs/equal_wall。

## 认证边界与 CLI

本工具不是 full strict：由 MAIN 给定完整 strict 的外部实际 SHA。核 prepare/report/strict 强SHA、全部 prepare sources（包括实际producer源）、execution binding、原 SHA-bound journal、实际调用的原 NPY；结束 rehash。journal 原生逐event payload/hashchain认证属于指定上游 strict，本工具只检查其完整字节 SHA、已知事件、序号/连接/bookend与独立资源/标量，不冒称重颁发 PNG/preprocess/native/AD closure。

下面在 `/home/gbk/code/LeWorldModel/le-wm-main` 执行。全部路径传 canonical absolute，SHA 使用实际完成值；尚无实际新报告/receipt，不预填未来值。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_shared_panel_summary_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p3_shared_panel_summary_20261029_v1 analyze \
  --prepared /ABS/SHARED_prepare.json --prepare-sha ACTUAL_PREPARE_SHA \
  --report /ABS/SHARED_output/report.json --report-sha ACTUAL_REPORT_SHA \
  --strict /ABS/SHARED_consume.json --strict-sha ACTUAL_STRICT_SHA \
  --output /ABS/agent_p3_shared_panel_summary_20261029_v1_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p3_shared_panel_summary_20261029_v1 verify \
  --summary /ABS/agent_p3_shared_panel_summary_20261029_v1_MAIN01.json \
  --summary-sha ACTUAL_SUMMARY_SHA
```

输出仅新本stem文件，排他创建≤128MiB；输入JSON≤256MiB，原journal≤2GiB，原 NPY≤2MiB；verify 确定性重建整个 summary，不触发 producer.consume 或 NN/physics。

## 实际测试记录与冻结

首轮1 failed/26 passed：fixture 的早停配对断言误用root0 pair索引，修正为root1的第8对；不改数据/科学阈值。随后48 passed /13.58s；加入合法嵌套失败、不退款、unknown与固定对照方向回归后最终 **51 passed /14.55s**。

覆盖全8/24/48布局、早停14、干净拒绝全付44、未到达null、候选/whole winner/独立replay/滚动context/raw/goal/reward/各资源轴/phase/wall篡改；临时真实文件 writer→reader→NPY→analyze/verify链，严格fixture刻意省略 optional `prepared` 字段，测试本 consumer 的最小必需 ABI，不表示当前 producer 缺少该字段。纯合成fixture、未执行原生模型/环境，不作为真实P3效果或 full strict 证据。MAIN 指出的字段描述已仅在本文更正，source/tests 保持原冻结字节；本文更新 SHA 后再次冻结，未重新运行测试或任何生产消费。
