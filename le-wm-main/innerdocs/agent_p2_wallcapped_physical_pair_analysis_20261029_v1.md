# P2 wallcapped 四槽物理逐步配对分析 v1

独占新增 source/tests/innerdoc。仅 saved 数值分析，不做代码审计、不执行 NN/CUDA/物理、不改任何 producer、模型、Q、normalizer 或旧结果。全部四槽、两已知根和两同根配对固定，不按 outcome 过滤或选择。不是 P1/P2/fresh 科学准入。

## 职责与 authority

`analyze(output, report_sha, physical_receipt, receipt_sha)` 必须传实际物理 report 和 strict receipt 的外部 SHA。核自有 physical `.report/.consume` schema、report 绑定、全部 rows/plans/counts/错误、实际 prepare、planner authority 及 receipt 中全部继承 bindings，分析后再重 hash。这里接受 MAIN 已完整签出的 strict receipt，不调用 producer consume 重跑全部上游；明确 `upstream_full_strict_rebuilt_here=False`。外部 SHA 锁定的是实际 authority，不是自签报告。无 receipt 不可消费。

source 只 import 标准库与 NumPy，不 import producer/旧 consumer 的计算函数。独立重算原生指标、rawbits 与 journal 操作计数。PNG 内容/渲染 guard、模型预测/J 与搜索完整性仍由先前 strict 保证；本分析核原 PNG 等文件 bindings 不变，不冒充第二次独立视觉或模型认证。

## 完整表

顺序 `(root0,joint),(root0,composition),(root1,composition),(root1,joint)`，均 off。每行保留 plan steps0..15 全16槽、warmup0..10 全11槽、15动作槽和 cp0/5/10/15；缺项为 null，未 attempt/return 的动作身份显式 false，未观测 actual bits 一律 null。原生早停不 forward-fill、不补足终态。export=None 不执行且保持行/配对分母。

每步重算完整 signed native7 delta、native7 distance/reward、jointXY、agentXY、objectXY、原生 angle term、独立 wrapped angle、velocity error、原生 success。原生合同仍是 jointXY<20 且 min(abs(angle),2π-abs(angle))<π/9；包括原函数可能负的 angle term，不用 wrapped 替换其门槛。object-only success/coverage 保持 null，不将它和 whole-body 原生成功混同。warmup 使用 bind 前旧 goal，plan 使用绑定后的 task goal，不能混入旧 reward。

独立检查原 raw15 float32 lossless witness、planner rawbits、请求 intent→实际 return rawbits、state/snapshot/goal/reward、所有逐步 off RGB 状态与 cp/最终行连接。保留每个动作 planned_raw、attempted/returned、dispatch_bits_match/actual_bits_match。原 model terminal Q 仅作为继承值单列，actual_latent_Q=null，本模块没有编码真实 RGB。

每根固定 joint vs composition：所有16时点和4cp的 composition-minus-joint 误差向量、原成功二元值、计划/实际逐步动作相等性和 state exactness。只有同时间双方均真实观测时才计算；terminal15 在任一缺15时为 null，不拿不同终止步的最后状态混作同长度比较。每行 final_observed.step 是独立 event-time 终态标签；例如14步原生早停只保留其实际14步终态，对臂15步终态另列，同步比较仅0..14，cp15=null。该分母规则不依赖生产结果。same_root_state 显式报告；不从更低误差挑赢家、不报告显著性或新独立样本。

全 journal 严格 spaced JSON 单LF/canonical/hashchain、已知事件枚举、nested intent/return/failure 与原 report 操作 receipts 独立对账；按 branch/resource 分开 attempted/completed，失败不退款。source 没有根据 model/physics 输出调参数的入口。非 matched 费用不推成等 FLOPs/等总计算。

## MAIN 运行入口

worker 仅运行明确的 RAM/临时文件 fixtures。物理生产 report/receipt 尚未提供给本模块，不虚构 SHA。MAIN 实际 strict 完成后使用下列命令（必须新输出）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p2_wallcapped_physical_pair_analysis_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p2_wallcapped_physical_pair_analysis_20261029_v1 \
  --output /ABS/ACTUAL_WALLCAPPED_PHYSICAL_OUTPUT \
  --report-sha ACTUAL_PHYSICAL_REPORT_SHA \
  --physical-receipt /ABS/ACTUAL_WALLCAPPED_PHYSICAL_STRICT_RECEIPT.json \
  --receipt-sha ACTUAL_PHYSICAL_STRICT_RECEIPT_SHA \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_wallcapped_physical_pair_analysis_20261029_v1_MAIN01.json
```

writer 独占新 stem、128MiB metadata、200MiB disk floor，文件内 flush/fsync 后 fsync_dir，不覆盖旧结果。最终测试结果与冻结三 SHA 见交付消息。未运行项：真实生产 analyze/消费及任何神经或物理实验，全部由 MAIN 执行。
