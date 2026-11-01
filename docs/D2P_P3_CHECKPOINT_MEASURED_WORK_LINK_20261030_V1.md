# P3 同检查点任务与已计量资源联结

## 现有资产及缺口

修复后的 online binding 已连接四槽 HOST、strict consumer 和完整 20 步的共同 caps/实际数值资源核验；原 RUN_MAIN01 两个已知根也已有真实反馈与保存消费。此前检查点任务指标存在，但完整末端资源匹配不能直接当作第 5/10/15 步资源匹配。已有 commitment-control 的 fixedH1/fixedH3/adaptive 六臂是另一协议，明确实际预算不匹配，不能重标为当前 H1+H3 同资源负控。现有 shared-panel 也研究已付双面板利用，不等同当前在线单面板顺序。

还缺的对照包括：在新独立运行中事前固定的反向 fixed [H3,H1] 与 signal 路由对照；打乱或反转信号路由的可控负对照；fresh roots 下固定分母复核；以及同 wall／总算力的可验证工作合同。前两者要求新 producer、相应源封存、完整物理/资源与 strict saved consumer，不能把既有 RUN_MAIN01 行重标、重复执行凑账或用事后选根代替。现有 runner 的 roster 和 policy 固定，新合同并未授权或实现这些新臂。本次**没有实现这些新在线控制臂**；本次新增的是可实际运行的 saved-only 同检查点结算入口，为判定哪些现有对照符合已测资源条件提供证据。

## 新入口

`experiments/spectral_reachability/p3_feedback_checkpoint_budget_20261030_v1.py` 接受 SHA 固定的 run、原 strict receipt、原 prepare 与预算合同；CPU 独立重跑原 strict consumer 并要求 receipt 全字段相等，再调用修复后的 binding 进行完整向量、共同 cap 和物理 meter 结算，再调用合同任务指标。传入未来 `--plan/--plan-sha/--reference` 时，额外使用修复后 binding 核计划、输出路径和 run coordination；三个参数必须同时给出。它不调用生产 runner，不生成或宣称预注册，plan 引用不自动证明可信时间戳。

数值资源从原 numerical event 的 attempt/return/failure 位置分别累计；不把 failed/未完成当 completed，也不把嵌套指标求和。物理 meter intent 与观察 encoder/projector intent 各自累计，hook `(resource,shape,directions,status)` 多重集分别比较。截点定义：

- cp0：同根 accepted 后、后续数值规划之前；
- cp5/10/15/20：对应实际控制步返回、该边界观察编码完成之后；
- 最终整槽结果另外由修复后 binding 核 ledger caps、actual 及全槽物理 meter。检查点物理累计不含后续 close 等工作，不能混为全槽总账。

每对各时间点同时输出原生任务指标和资源匹配标志。非 strict/partial 保留四槽和 unknown，不补零；错 SHA、伪 receipt、不同计划路径、缺失检查点或违反末端预算时拒绝。两个本次根均为 known，且截点相同控制步不等于相同 wall 或 FLOPs。

## RUN_MAIN01 回顾性结果

对真实保存文件重新运行 strict 消费并结算，结果 `le-wm-main/results/p3_feedback_checkpoint_budget_20261030_v1_RETRO02.json`，SHA `7e9cf07cec428bb37170045970f7bdc34c1f268b5818fff334f7015e11132783`。RETRO01 是添加任务联结前的初稿，保留不覆盖，以下使用 RETRO02。

| 同控制步 | 两根 signal−fixed predict attempted | 数值向量及 hook shape | 物理 meter 与观察编码计数 |
|---:|---:|---|---|
| 0 | 0 | 匹配 | 匹配 |
| 5 | +18 | 不匹配 | 匹配 |
| 10 | -17 | 不匹配 | 匹配 |
| 15 | -17 | 不匹配 | 匹配 |
| 20 | 0 | 匹配 | 匹配 |

这是两个已知根的已记录资源；不证明总计算、时延、FLOPs 或能耗相同。第 5/10/15 步原生任务差异是同控制步描述，**不是同计量数值预算对照**。第 20 步完整已测轴和 hook 形状匹配，但仍不能单凭它证明 signal 的因果优势、完整 P3 机制或泛化。该比较在结果之后生成，不是预注册。

## 使用与测试

新入口的回顾性运行需明确实际 receipt、contract SHA 和独占输出；未来如果与新在线计划结合，还需 `--plan ... --plan-sha ... --reference ...` 并先通过原严格消费。源文件和计划应在任何未来数据产生前由独立权威审读封存；本入口没有改变原科学门。

CPU 测试命令：

```sh
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p3_feedback_checkpoint_budget_20261030_v1.py \
  tests/test_p3_feedback_online_binding_20261030_v1.py \
  tests/test_p3_feedback_budget_contract_20261030_v1.py
```

**33 passed**。新增测试覆盖截点前尝试／截点后完成、数值/shape 分开、missing axis、错边界、跨槽事件、伪 strict receipt、partial unknown 与错误计划路径。合成测试不是在线运行证据；此次没有使用 GPU、模型或环境。
