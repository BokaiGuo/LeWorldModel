# P2 control15 feedback v2：独立 saved scalar summary

冻结范围：新增同 stem 的 source、RAM tests 与本文；不改 producer、选择器或任何旧产物。MAIN 执行实际 analyze/verify。本 worker 未执行实际 analyze、fullstrict、神经、物理或训练。

## 固定分析与证据边界

固定 2 known roots × 2 原 arm × I/F = 8 jobs；每个 base 的 O/I/F 共 12 policies。不是 8 个独立根。逐 job 保留全部 5 candidates（共 40 个 Q）、全部 27 forwards、1 个继承 AD witness；不丢弃 0 direction、0 step、index0 或 tie。O 仅继承原 openloop witness/raw30，`new_model_Q=null`，不为 O 补算/伪造新预测。

独立导入仅标准库和 NumPy，不导入 producer/kernel 的算术或消费者。直接从保存 context 的最后 192 维向量及原 goal 计算：

- `Q_fsum = math.fsum(float(delta_i)*float(delta_i))/192`，delta 为 float64 subtraction。
- `Q_numpy_original = float(np.mean(delta**2))`，逐候选严格等于保存的原 Q，不用容差。
- 原已封存选择必须严格为稳定 `(Q_numpy_original,index)` 最小值；另栏 `descriptive_fsum_selected` 仅描述舍入对排序的影响，绝不替换计划。
- gain 定义为同 job 的 incumbent Q 减原 selected Q；正值表示模型目标下降，负值/零分别保留。分别输出 fsum 与 NumPy gain。不跨 I/F 的不同输入 context 相减来称真实胜负，不推断物理收益。

用原 float64 tanh→float32 chart 逐字节核 parent raw30、prefix15、选定 suffix15、full raw30；精确连接 task/root/arm/policy、parent witness、job input、goal、model hash、receipt plan、gzip plan，独立重建 compact-JSON+newline 的新 witness。I/F 同 base 的原 task/goal/parent 等身份相同，但 context 不强制相同。逐 forward 核 tag/index、incoming/output full context、原 raw packet descriptor 和 request digest，核独立 selected/whole/segment replay 与 AD primal 的 context 连接。保存 gradient 仅用于核固定 proposals 的 direction/slope/alpha 算术；不读取 J 或重新求导。

只读取下列四个显式 SHA authority 和报告所在目录的 `slot0.json.gz`…`slot7.json.gz`，八文件各自 SHA 取自已绑定 report。只接受 canonical absolute paths、完整单成员 bounded gzip、无重复键/NaN 的 JSON。每文件压缩字节及解压结果均限 64 MiB。不会读取 event ledger、J NPY、模型权重或上游 authorities，不调用原 fullstrict；上游/原生真实性沿用 MAIN 已完成且显式 SHA 绑定的 receipt，不冒称本工具重做了全链认证。

资源分开保留 `actual_new`、report 全资源 census、各 job 向量及 inherited resources。新付费原实验为 216 predict / 8 J / 1536 VJP / 46080 J elements / 24 AD primals，三个 module 各 216。FLOPs/engine backward 仍为 null；不把 predict 单轴当等预算，不把继承费用再次算作新费用。本消费者新增模型调用为 0。报告墙时与 after-report-fsync-and-hash completion inclusive 墙时精确连接。全部科学/部署准入 flags 仍 false，不签 P1/P2/C3、fresh 或 equalFLOPs。

## 已提供的实际 authority（仅核 SHA/ABI，未跑实际分析）

共同绝对根：`/home/gbk/code/LeWorldModel/le-wm-main/results/`。

| 输入 | 相对上述根的路径 | SHA256 |
|---|---|---|
| prepare | agent_p2_full30_control15_feedback_20261029_v2_prepare_MAIN01.json | 498b561855d7859681a18ed47e0157908a146a796145b929b5436a1e73b79630 |
| report | agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/report.json | 6e2c82633140e071c880aed92c3cde025635cf7dd92c6687d9e0334bc30db3a8 |
| completion | agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/completion.json | 8622879f9c674d6acffcbc6646fbaccf25023486b0534626c3b63a5bab1d7329 |
| full receipt | agent_p2_full30_control15_feedback_20261029_v2_consume_MAIN01.json | 4e7d5e1f6b4b03b28b68eb8a31016211f7d1e0ff3115671923b6df2e97f3d893 |

CLI 默认 SHA 为上表；也支持显式 `--prepare-sha/--report-sha/--completion-sha/--receipt-sha`，没有未来 SHA 或猜测 flags。输入是完整 receipt 而非 status 字符串：核 schema、flags、authority joins、完整 8/12 roster、plans、费用、source/model-before-after、completion boundary。

## MAIN 命令（全部文件参数为 canonical 绝对路径）

在 `/home/gbk/code/LeWorldModel/le-wm-main` 下运行：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_full30_control15_feedback_summary_20261029_v1.py

feedback_args=(
  --prepare /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_control15_feedback_20261029_v2_prepare_MAIN01.json
  --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/report.json
  --completion /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/completion.json
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_control15_feedback_20261029_v2_consume_MAIN01.json
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_control15_feedback_summary_20261029_v1_MAIN01.json
)
.venv/bin/python -m experiments.spectral_reachability.agent_p2_full30_control15_feedback_summary_20261029_v1 analyze "${feedback_args[@]}"
# 将 analyze 打印的实际 sha256 填入；不预填未来输出 hash。
.venv/bin/python -m experiments.spectral_reachability.agent_p2_full30_control15_feedback_summary_20261029_v1 verify "${feedback_args[@]}" --summary-sha ACTUAL_SUMMARY_SHA
```

analyze 使用 exclusive-create，不覆盖旧输出；verify 重新读四 authority/八槽重建整份 summary，比对完整内容而非几项 checksum。verify 是本独立 summary 的 saved 重算，不是 producer fullstrict。

## RAM 测试记录与未完成项

首轮 34 passed / 1 failed：fixture 内共享 Python context 引用导致单 forward 篡改同步改变对照，未模拟持久化 JSON 的独立对象。修复 fixture 为 JSON 往返后，最终 **35 passed / 5.30s**。覆盖全8/12/40、零方向/原样保留、稳定 tie、fsum/NumPy 舍入分栏、task/raw/prefix/context/witness/forward/AD/资源/科学 flags 篡改、真实临时文件 gzip 写读、CLI analyze/whole verify、错误 SHA/多 gzip 成员/重复键/非有限/非 canonical 路径拒绝。所有均 RAM/临时文件 fixture，不是生产证据。

待 MAIN：完整审读、独立回归、实际 analyze 与 whole verify。无 worker 生成的实际 summary 或科学结论；不支持缺槽/失败的该轮冒充 complete，缺文件、差异或未闭合 receipt 均 fail closed，不补默认值、不容差修复、不换选择。
