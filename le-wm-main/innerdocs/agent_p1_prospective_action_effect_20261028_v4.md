# P1 prospective action effect v4：只读阶段目标消费

独占新增 source/tests/本文，v2/v3 三文件、producer、report/journal 均不修改。无 prepare/run，无模型加载、神经或物理调用。当前实现只认证完整40行；partial 明确拒绝，不签发部分完成证明。

## 已知失败与限定修复

原始 v2 consume 在 line630 比较 enriched journal terminal triple 与 Store.observations 原始 body 而失败；v3 独立验证并精确投影仅 index/previous_sha256/event_sha256 三项 envelope 后继续，但在 frozen v2 line635 失败。两次失败保留，**不声称原始 v2 或 v3 已通过**。

真实 producer 先 placement/native_goal 下 warmup10，再 bind_goal(final task goal)，再 prefix11..15 与 suffix1..10。原始 line635 对全部25步使用最终 row.goal，故 warmup 全400条比较对象错误。首行 warmup1 保存/placement-goal distance=56.512040741808306，final-task-goal distance=242.84466686189273。这个 consumer 协议问题不是原 P1 teacher15 的12反转解释。

## 精确验证与局部适配

逐40行、全部1000 raw_return：验证完整 signed journal 事件、唯一 scope、递增 index、phase/step 顺序、placement 两次 reset、仅一次 goal_bind、三次 goal_setter 分别严格包含于对应 reset/reset/bind；warmup10 return 之后绑定、goal_attached 之后才有任务 public_step。goal_attached 的 canonical goal=最终 row.goal，RGB hash=原 row.goal_pixel_sha256，绑定 snapshot=warmup10 snapshot；root triplet 在绑定后且保持该 snapshot。任务目标仍由完整 frozen consume 从原始 sealed source 重建，不能由 report 自封。

每步独立从相应 provenance 选目标：warmup 用 placement_recipe.native_goal，其余用已绑定目标。重算 native 原公式（前4维位置<20及原角差表达式<π/9；distance 为 full7 norm，包含速度），精确核 reward=-distance、terminated=native_success、truncated=False、不得 stop 后续填充。无阈值/metric 变化、无浮点容差。NPZ info 的 SHA、完整数组 inventory、dtype/shape 独立验证，再逐步核 goal_state/goal_proprio、pos_agent/vel_agent、block_pose 与 snapshot/state7；info block angle 原生未包裹，state7 使用原式 angle%(2π)。

局部 AST 克隆完整 pinned v2.consume，只允许唯一 `native_labels(row['goal'],e['state'])` 调用变为 `phase_labels(row,e)`。全部 require/原 native helper/其余代码不变；测试逆替换后 AST 与原函数一致。传入完整实际事件 canonical（含 index/job/phase/step/digest），与事前独立验证队列逐次精确一致，并核 row index/goal；结束要求1000次用尽，无多用/跳过。**绝不按 (goal,state) 查表**，重复静止 state 跨阶段不碰撞。返回标签仍由原 v2.native_labels 计算，无 flags 豁免。

同时复用 v3 全未投影事件白名单、像素→原 original224 NumPy preprocessing 精确验证、terminal envelope payload 精确投影。原 journal hashchain、全部 producer artifacts SHA、原资源/模型/闭环/80对分析均继续执行。输出仍 P1_admission=False，不构成新 science gate、selector、部署修复或 fresh-root 认证。

## 验证及 CLI

TDD：先新增实际形状/篡改测试并确认缺实现时报红，再实现。v4 共36 tests；与冻结 v1/v2/v3 联合 **119 passed in 11.34s**。覆盖实际首行、完整1000条、绑定目标/原生目标/行目标/RGB/绑定 snapshot/时序/phase/action dispatch、奖励距离flags、独立 info、逐次调用过量/遗漏/换序及重复 state 跨阶段。

worker 完整只读 saved consume 已 **exit0**（未写 receipt），绑定原 report SHA `2e7794d32602ca2b045202099fedb6b99ac5988d6255ee68870032219530084c`。结果：`strict_phase_saved_adapter_audited_not_P1_admission`；40行/1000 raw（warmup400、prefix200、suffix400）；native_calls=1000、terminal_rows=40、AST_call_substitutions=1；pixels preprocessing checked=88；fixed_pairs=80、available_pairs=80、unknown_pairs=0。此处是新的 actual20→25 固定80对，不应混同旧 teacher15 fixed80 的76/4。最终所有原 artifacts 与 report SHA 再核一致。MAIN 可用下列 CLI 独立复核并签发新的 v4 receipt；原始 v2/v3 失败状态不变。

从 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_prospective_action_effect_20261028_v4.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_prospective_action_effect_20261028_v4 consume --output results/agent_p1_prospective_action_effect_20261028_v2_MAIN01 --report-sha 2e7794d32602ca2b045202099fedb6b99ac5988d6255ee68870032219530084c --receipt results/agent_p1_prospective_action_effect_20261028_v4_MAIN01.json
```

receipt 为独占 v4 新文件、必须在 producer 目录外，使用 xb 禁止覆盖；省略 --receipt 时只读返回结果。receipt 绑定 v2/v3/helper 及本 v4 三文件 SHA。源码/test/doc 冻结 SHA 由交付消息报告，避免文档自哈希。
