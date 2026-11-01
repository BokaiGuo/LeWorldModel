# P4 原运行日志的实质证据提取

仅新增独占source/tests/doc，保留旧源码/结果。使用experiment-code的saved输入/验证记录原则；不执行任何日志中的命令，不follow路径，不打开保留根、NPZ/sourcepack或权重；0神经/物理/网络、无子agent、无approval。

## 实际恢复而不是重复阻塞表

coverage `results/agent_freshroots_resume_20261027_coverage.json` SHA `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb` 明确引用两份日志，十个A–H dispatch均在第一份：

- `/home/gbk/.codex/sessions/2026/10/23/rollout-2026-10-23T21-46-56-01a167e2-eca3-7f63-bb9b-93030b5d366c_01a16903-c194-71a3-972b-7a268203b373.jsonl`：绑定508228945bytes、SHA `0a2c097548ef4bd9818affcf82ecb53102ddfa965498b966c13aa7067efdf70c`，真实完整匹配。
- `/home/gbk/.codex/sessions/2026/10/26/rollout-2026-10-26T12-10-35-01a167e2-eca3-7f63-bb9b-93030b5d366c_01a17667-2cec-7780-b530-fb3aa5acdd59.jsonl`：绑定227482902bytes、SHA `cad9cf6253c2b466fcc48bddda725368426e796f8337326733de175db8b5ee22`。当前文件已追加，**仅上述绑定前缀重核成功**；不读取后缀。

新实质证据是B前两次运行direct tool-return中的traceback frame：JSONL行22794对应 `call_a41c0e8d73624b16959e29b9f05fd4ed`，行22817对应 `call_3d96838206a84aba9b97fe0ce82bf8e4`。可定位原执行的producer `preflight_pusht_d2p_p0.py`、LEWM `wm/lewm/lewm.py`（以及第二次的`module.py`）及torch/transformers调用路径和源行号。失败退出本身此前已恢复，本版本不以重复失败状态冒称新增reset证据；新增的是逐frame原运行路径定位。

每项输出精确日志path、prefix SHA/字节数、JSONL行、原record字节区间/SHA、call_id、JSON pointer `/payload/output/N/text`、text行号、Unicode character span和原literal frame。原始frame路径只作为字符串证据，绝不打开当前文件或拿当前SHA冒充原加载SHA。

## 边界

只检索coverage明确十个call_id，dispatch还需匹配原log/行/command；严格join到顶层response_item/custom_tool_call_output，不把别的记录中提到call_id的文字、源码patch、命令、普通路径warning当证据。输出只含traceback frame，不复制root/state/outcome/score数组。必须先完整验证绑定前缀SHA才返回提取结果。前缀追加可接受，前缀变化/截短/超cap拒绝。

traceback只能证明工具当时返回的部分路径信息，不能证明完整loaded bytes、factory/env/wrappers closure或hash；first capture也不是执行时seal，仍需MAIN来源审读。未从这些direct returns恢复真实reset ledger/count，均为null，不能填0或由代码行推reset执行次数。其他8次direct return无匹配traceback不表示没有执行或reset；A/E/H组合调用中的异步session尚未逐子进程归属，不能把整个返回混当单个任务的证明。不解析追加历史或扩大到全部日志，且不宣称exhaustive、certificate、freshness或P1准入。

## MAIN saved-only接口（inner根）

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_original_evidence_extract_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_original_evidence_extract_20261028_v1 extract --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_original_evidence_extract_20261028_v1_report_MAIN01.json
# ACTUAL_SHA取上条真实输出。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_original_evidence_extract_20261028_v1 consume --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_original_evidence_extract_20261028_v1_report_MAIN01.json --sha ACTUAL_SHA
```

consume重读同一绑定前缀独立重建report，正常仍为 `partial_evidence_verified_not_authority`，reset_blockers_cleared=0。当前历史全覆盖所需ledger及原加载byte-provenance仍未知，下一步必须由原来源提供或MAIN进一步沿明确异步关联恢复，不能靠本文件自证。

## 实际saved验证

新19项测试实际通过（4.89s），包含RAM前缀追加/截断/篡改/误join/伪frame拒绝、真实白名单读取、精确origin record SHA/byte span/JSON pointer/character span复取，以及真实extract→临时保存→consume全闭包重读。另一次只读extract smoke得到10dispatch、2次带路径返回、22个原运行frame：B第一次13个、第二次9个。这里22不是22个不同文件/根，也不是reset次数。没有创建MAIN results报告，没有收集新证据或签发任何授权。临时测试与saved结果均非freshness科学通过。
