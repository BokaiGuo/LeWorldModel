# E1 封存事件逐轴审计

结论：六槽共同 controller 的保存 journal 可按槽、phase 和 hook 重新派生事件集
摘要；它只闭合已保存事件的局部投影，不闭合 proposal、继承、物理或端到端预算。

只读来源为 E1 v2 保存 report SHA
`cf3cb2a506ec1739ca53f58faa00bb07972c6e546cd75e02e0c609dccd7b7578`，
其指定的 gzip journal SHA
`2a3570a54333ff728136b03b4cfebbf203f42a48cc07d3567dbb20441762f5df`，
以及既有保存消费 receipt SHA
`1ca8e949cc4b219b9b03adb3d34ee859e924e0078f9f81c519be2c03ed77be3d`。
工具不跟随其路径引用，不打开 P1 资产、模型、数组文件或其他输出。

核验 gzip 完整性、逐行 canonical JSON、连续 index、前向 SHA 链、最终链摘要和
report 中的事件总数；再按六槽与 `controller/replay` 解析实际 hook attempt/return，
检查 ticket 序号、关联字段、返回配对、samples 和 receipt 各轴计数。
真实 journal 为 9,382 条；12 个槽/阶段组中，controller 每槽四个模块各 144
attempted/returned、predictor samples 144；replay 对应均为 4。模型模块轴嵌套，
不能相加作 FLOPs。

每个派生摘要的输入为按排序键、无空白 JSON 加换行序列化的对象：
`schema=codex.e1.axis_event_set.v1`、原压缩 journal SHA、`slot`、`phase`、
`axis` 及按 journal 顺序排列的 `records`（每项 `index` 和原链记录 `sha256`）。
对该字节做 SHA256。摘要由本次审计**派生**，不是原 producer 预先签发的 event-set
证书。每个 measurement 输出的 `attempted/completed/event_set_sha256` 字段可用于
既有 ledger 的相应轴；`predictor_module_calls` 不在原 ledger 轴中，独立保留。
`predictor_samples` 与 predictor module calls 引用同一组事件、只按 axis 名区分摘要，
**不同摘要不证明事件集不相交**；不得将这两项相加或利用摘要差异绕过去重审计。

示例：槽 0 controller 的 `predict_api_calls` 事件集摘要为
`9004ef70ea41ef0cda8dc0f6c474194d354bd4ee2eca71bd362fb10dd4ff8f73`。
该值可从固定 journal 重现，不是对外部真实性或完整训练/历史成本的证明。

核验继承固定 receipt 的原 fullstrict 权威，不重新验证数组 bytes、payload、全部
模型输入输出语义或运行时真实 hook 的无遗漏性；也不能检测日志之外未计工作。
不核算或分摊 Hi 四槽共享账、不生成 proposal/继承账的事件摘要、不生成排名；
所有科学准入和等预算标志保持 false。未来若要把输出写入 ledger，须另行绑定对应
slot/target 和 receipt、审查各阶段重叠及祖先成本；本工具不会修改旧 ledger。

验证（纯 CPU、只读，测试包含实际封存文件和内存篡改）：

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_e1_sealed_event_axis_audit_20261030_v1.py tests/test_codex_e1_end_to_end_ledger_interface_20261030_v1.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.codex_e1_sealed_event_axis_audit_20261030_v1
```

26 项通过，覆盖摘要复现、原文件 SHA、压缩截断、链篡改、重复尝试、无返回、
receipt 计数/槽不匹配和未识别 dispatch 事件拒绝。
