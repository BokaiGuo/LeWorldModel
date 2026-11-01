# Actual-cp5 R gate saved decomposition v1

仅解释冻结 actual-cp5 四槽的零对照+八候选：36项/32candidate全部保留。
复用 `main_p3_R_gate_decomposition_20261029_v1.scalar` 的算术；**不是第二份独立数学实现**。
不调用其旧 cohort 的 analyze/explain，不伪造旧 status、schema 或 C3 字段。
不重跑 producer、完整 consume、模型、谱计算、优化或物理；不会生成 physical release。

## 固定 authority

- prepare `results/agent_p3_actual_cp5_r_delivery_20261029_v1_prepare_MAIN01.json` SHA `b4b2f9d3ca1867a86edbb84c753cef3e52d757582299b6f851c3e8c9244a042d`。
- report `results/agent_p3_actual_cp5_r_delivery_20261029_v1_cuda_MAIN01/report.json` SHA `deb69d78579b4cdc0ffb3ae4c671dd71f9ecd9b9e02729484085cb3d0c0dd19d`。
- receipt `results/agent_p3_actual_cp5_r_delivery_20261029_v1_consume_MAIN01.json` SHA `8d55a55e108aca1695a550861d125a0b0baac2b225cba914d0af4b47d30159a5`。
- 真实 own schema `.prepare/.report/.consume`，report status `four_slots_settled_not_admitted`，receipt status `actual_cp5_R_model_deliveries_verified_NOT_physical`。
- 必需 `full_C3_demonstrated=false` 等所有原 false flags；不以不存在的 `C3` 替代。
- 绑定原 actual-cp5 resolver 输出 SHA `ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7` 及原协议 SHA `deed88fd70073c1aaca96d32c81968b7f02a7a3841e3d8a7be00d4e4d573275f`。
- 逐字节读取并 SHA 校验三个产物；检查 prepare 的完整 sources 映射全部当前文件字节（前后各一次），保留整个映射。此操作可能需要磁盘时间，但不是 fullstrict 重算。
- pin 原 producer 三件套及 MAIN 算术源码；将本三件套和依赖 SHA 写入输出。冻结前测试不运行生产 analyze。

## 输出与边界

固定 root0/H1、root0/H3、root1/H1、root1/H3；两真实 task 在各 horizon 双身份复用，四槽不能筛选。
逐槽保留完整 task/provenance、local/original task SHA、真实 absolute5/10/15/control−5/0/5 chronology、四原 branch、raw10/prefix5、资源 receipt/sourceclosure。
每项保存 immediate、nominal_cost、reach、nominal_drift、bounded_linear_gain、linear_terminal_cost、ridge_penalty 和相对 zero 的全部差值。
每项 `report_origin_pointer` 精确指向固定 report 的对应 control 或 candidate JSON 位置。
候选保存 nongreedy/better_reach/eligible、gain_improved/drift_increased；所有拒绝因素独立列出，可能两项同时失败。zero 不参与资格，布尔值为 null（不冒充 rejected candidate）。

exact gate 直接比较冻结 saved 数值 `immediate > zero.immediate && reach < zero.reach`。
scalar 复算容差只验证既有算术；不改变 strict inequality，不作 gate epsilon。
验证恒等式 `Δreach = Δimmediate + Δnominal_drift − Δbounded_linear_gain`。
完整 eligible list 和原 `(immediate,index)` 排序、receipt 状态/delivery/resources 必须一致；no_witness 不发布动作。
这是有限已见根/固定 bank 的拒绝分解，不是新 bank 设计依据、全空间无解或物理收益证明。
未重新验证完整 native journal/NNIO；其 authority 来自精确绑定的 MAIN 已完成 strict receipt，明确 `upstream_fullstrict_rebuilt_here=false`。

## MAIN CLI

在 `/home/gbk/code/LeWorldModel/le-wm-main` 运行：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_actual_cp5_R_gate_decomposition_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_R_gate_decomposition_20261029_v1 analyze --output results/agent_p3_actual_cp5_R_gate_decomposition_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_R_gate_decomposition_20261029_v1 verify --output results/agent_p3_actual_cp5_R_gate_decomposition_20261029_v1_MAIN01.json
```

analyze 仅排他新建，不覆盖；verify 重新从所有固定 saved authority 构建并比较**整个规范 JSON 字节**，不是部分字段或自存 digest。
单个输入/输出上限256MiB；SHA 流式读取完整 source bytes，不加载/反序列化模型。RAM tests 为真实3×192历史、2×10动作、192×10 J、10参数的 JSON roundtrip fixture；cold import/help 禁止 torch/numpy/pickle/producer imports。
worker 不运行生产 analyze/verify/fullstrict。MAIN 审读后实际执行；未闭合项仅为 MAIN 的生产分析及整份 verify，非科学结果。
