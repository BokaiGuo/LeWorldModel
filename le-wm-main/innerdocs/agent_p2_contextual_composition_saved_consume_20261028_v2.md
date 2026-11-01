# P2 contextual strict saved consumer v2

v1保持冻结。本v2独立新入口，不修改producer、原prepare/report或任何日志bytes，不重跑CUDA/physics。

## 两个真实ABI接缝

原canonical已经包含LF，Store.event又追加LF：严格要求每条canonical JSON record后恰好一个空LF分隔。
实际177846232 decoded bytes / 19188 records / 19188 empty separators；拒绝任意多空行、缺失空行、非空坏JSON、
重复字段、非canonical字节。完整记录compressed/decoded SHA，不写回原日志。

原run_arm还直接发送无kind的候选终结字典。实际320条joint、160条conditional，共480条，条件臂多suffix_compilation字段。
原check_module_event_order和array_saved筛选直接索引kind会抛KeyError。
v2先将每槽全部120无kind记录与原results两臂80/40 attempts按canonical字节逐项精确比较，保留float符号零、
完整actions/cost/logits/witness/suffix_compilation/耗时；缺失、重复、乱序、篡改、未知记录均拒绝。
只有完成这项完整对账后才向原typed模块顺序helper和array_saved断言提供带标签视图；不使用任意get/skip。
原Gate仍收到每条原native event，原逐入口/返回/AD/成本/数组/tensor/replay及480候选全部验证保持。
RAM AST回归仅允许parser、两处显式typed-view接缝和返回包装变化，不删除其他原断言。

## 实际输入与CLI

report: results/agent_p2_contextual_composition_runtime_20261028_v1_cuda_MAIN01/report.json

report SHA: 498b45ed9f4c580064d8f2a5dfd00ac8542b7ce9d0b86e6587d9ab6a9639d49c

冻结producer source SHA: 32db978cd943c571756b675be95e159020e2c4ffda1f926cdd4e8bd45896d205

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_contextual_composition_saved_consume_20261028_v2.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_saved_consume_20261028_v2 analyze --receipt results/agent_p2_contextual_composition_saved_consume_20261028_v2_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_saved_consume_20261028_v2 verify --receipt results/agent_p2_contextual_composition_saved_consume_20261028_v2_MAIN01.json --receipt-sha ACTUAL_SHA
```

独占writer持久化全部native_strict_receipt八plans；stdout只有摘要/真实receipt SHA，不输出巨大plans。
32MiB cap、200MiB floor、exclusive create及file/parent fsync；verify重建全部消费结果，不仅核hash/status。
未知FLOPs仍null，不把reserve当计算；4slots/2known roots/8plans/480候选/1328predict原分母不变。
P1/P2/physical_release全部false。未终结、失败、pending、封存漂移不生成成功receipt。
物理handoff必须独立绑定MAIN实际receipt SHA；本入口不授权物理、不提供新科学证据。

## 本工作位实际验证

v2与冻结runtime联合76 tests passed（35新consumer+41runtime）。RAM回归使用原候选终结完整字段形态，
复现原check_module_event_order的KeyError，验证typed-view后原helper通过并拒绝模块入口index变异。
实际只读analyze函数已完整返回strict_saved_verified_science_blocked、8plans、19188records；未写生产receipt。
原compressed SHA 0fad105b86c3f4735e0274a54106968fa17f85eda3c0ffb3dcd9198bebf5e16e，
decoded SHA 98e005b651e528cef7b9666d4c0e8269e7dadd05bc621d6f9e0930a20b4f0137。
测试不代表新CUDA/physics证据；MAIN仍需独立审读及运行上述analyze/verify持久化入口。
