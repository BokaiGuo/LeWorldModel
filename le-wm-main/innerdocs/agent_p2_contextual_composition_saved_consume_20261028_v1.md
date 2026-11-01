# Contextual P2 双LF严格 saved consumer

只读封存 runtime 与实际报告；不修改 producer、prepare、report 或 gzip；不重跑 CUDA/物理。
物理 handoff 暂停，直到 MAIN 完整 strict消费验收。源保持原失败证据，不改原consume入口。

已知实际输入：

- producer source SHA `32db978cd943c571756b675be95e159020e2c4ffda1f926cdd4e8bd45896d205`
- report `results/agent_p2_contextual_composition_runtime_20261028_v1_cuda_MAIN01/report.json`
- report SHA `498b45ed9f4c580064d8f2a5dfd00ac8542b7ce9d0b86e6587d9ab6a9639d49c`

## 明确根因与修复范围

冻结 producer 的 `canonical({'x':1})` 实际为 `b'{"x":1}\n'`。
`Store.event` 又追加 LF，因此每条记录是 `canonical(JSON)+LF`，即 JSON record LF + 空行 LF。
原consume第534行把 `raw.splitlines()` 的空分隔当作 JSON，导致 JSONDecodeError。
MAIN实际诊断：decoded177846232 bytes，38376 splitlines，其中19188空分隔，严格交替。

本新parser明确验证**每条非空canonical JSON record恰好配一个空LF分隔**、完整双LF尾部。
不使用 `if line.strip()`、任意filter或异常吞噬；拒绝前导/额外/缺失空行、空record、CRLF、
非canonical空白、重复JSON字段、NaN、非dict及任何非空坏行。保留原gzip与decoded SHA/计数。
不导出改写日志，不声称原consume成功。

`consume_native` 保留原consume全部source/model/prepare/loader/artifact census、四槽480候选、
动作bits/witness/goal/context、逐入口/return/AD/数组/成本、所有模块tensor和双replay验证。
只替换journal parser，末尾把原native receipt与framing证据分栏返回；RAM AST等价测试
验证除此之外所有原语句、断言、helper调用未删改，未patch原module globals。

最终新receipt：`native_strict_receipt` 保存完整原8plan ABI，`framing_evidence`保存准确
原始bytes证据，`bindings_sha256`绑定本新source/tests/doc。P1/P2/physical_release全部false。
失败不写“通过”receipt；未终结报告原有拒绝不变，不补根/候选、不改Q、科学门或模型。

## MAIN命令与小stdout writer

从 le-wm-main：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_contextual_composition_saved_consume_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_saved_consume_20261028_v1 analyze --receipt results/agent_p2_contextual_composition_saved_consume_20261028_v1_MAIN01.json
```

analyze默认使用上述真实report/SHA；也可显式 `--planner-output ... --report-sha ...`。
writer独占创建完整receipt，flush/fsync文件与父目录，32MiB cap/200MiB floor，不覆盖；
stdout只有摘要、完整receipt路径/SHA、framing证据与8/480分母，不输出巨大8plans。
verify使用stdout真实SHA，不预造：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_composition_saved_consume_20261028_v1 verify --receipt results/agent_p2_contextual_composition_saved_consume_20261028_v1_MAIN01.json --receipt-sha ACTUAL_RECEIPT_SHA
```

verify完整独立重算与逐字段比较，不仅验证文件hash或status。生产后续handoff必须绑定
本新receipt SHA及原report SHA，不能再调用已知双LF不兼容的旧consume。

## 边界

RAM测试不代表真实CUDA或科学成功。实际saved strict成功与否需运行完整链并由MAIN验收；
若暴露下一处独立错误，保持blocked并精确回报，不放宽断言。无 fresh/P1/P2 科学放行。
