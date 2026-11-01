# 历史8来源/16 reset阻断：有界来源恢复交付

本交付只做保存 metadata 与源码文本解析，不调用模型、CUDA、模拟或物理，不采新根，不作P1审计或准入。已完整阅读 experiment-code 技能及 references；按用户限定仅实现 metadata pipeline/RAM mutation tests，不运行其中通用训练、优化或绘图流程。MAIN最新 capability probe 的40分支/32矩阵仍只有2 known roots，不由本工具扩大独立样本数。已读MAIN后续固定方向结果文档：40编码/80预测/320模块是MAIN证据，不是本人执行；object-only方向不能替代同时包含agent/object位置的原生任务判据，本工具不使用任何此类结果来排根。

个人完成条件：交付独占 resolver/tests/本文、实际 metadata closure 与复算命令。goal API现有取消的 offline-fidelity职责仍占用 unfinished blocked goal；不覆盖、不伪标完成，更不代表研究完成。没有派生agent。

## 固定输入、顺序和读取界限

入口为原 `root_exclusion_authority_resolution_20261027_v1_MAIN01.json` 的 SHA `032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00`。只投影 reason=`historical_reset_attempt_bound_not_attested` 的8来源/16 inherited refs，逐一核原report SHA、schema、seed/count/oversample。618 inventory不当coverage，不扫描整个results来宣布历史完备。

解析前先核外层 `docs/chat_freshroots_20261027_preregistration.json` SHA `79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e`，及其冻结protocol SHA。固定4 slots：root seeds203027101–104，reset seeds203027201–204，符号+−+−；每slot一proposal/attempt，不替补、不按结果选择。原排除规则和XY1e-9/angle1e-12通过整个prereg SHA绑定，未改科学阈值。

新增证据入口为两份固定保存会话（绝不执行其中的命令）：

- `/home/gbk/.codex/sessions/2026/10/23/rollout-2026-10-23T21-46-56-01a167e2-eca3-7f63-bb9b-93030b5d366c_01a16903-c194-71a3-972b-7a268203b373.jsonl`
- `/home/gbk/.codex/sessions/2026/10/26/rollout-2026-10-26T12-10-35-01a167e2-eca3-7f63-bb9b-93030b5d366c_01a17667-2cec-7780-b530-fb3aa5acdd59.jsonl`

只JSON解码function/custom tool-call的arguments；不解码tool results、messages、native outcomes、latent或trajectories。旧report由3份SHA固定的旧lexer投影白名单protocol字段，outcome字段仅保留**预留seed/count/oversample metadata**，不读成功、成本或物理标签。JSONL按时间/路径/行号固定顺序处理。每log≤650MiB、总≤1100MiB、每行≤8MiB，report/metadata文件≤128MiB，stdout closure≤512KiB。超限/截断/原SHA或协议不符直接错误，无静默部分coverage；CLI不创建目录、不写磁盘。

源码patch仅在7个白名单历史路径重放，绝不import/exec历史源码。Add/Update按精确context重放；同context多命中采用cursor后首个精确匹配，并逐项保存所有匹配位置/选择规则，**不是唯一性或实际apply成功的证明**。无法匹配则清空该路径lineage，不偷拿当前文件补base。shell默认output只可来自已重放旧shell字节，不读当前shell。git读取固定commit `d26e4bff84f31138adf5d1bda809bccafd9be10a` 的4个factory链blob并校SHA；这仅约束代码族，不反向绑定旧运行。

## 本轮已实际解析的新事实

首次metadata运行恢复8/8来源对应命令、共10次dispatch intentions，及全部schema相符的producer候选；21次白名单patch段，0 recovery errors。这里的dispatch仅保存工具调用的意图，**未解码返回值，不冒称实际进程返回或reset数量已验证**。

以下行号均对应第一份固定会话，完整path/SHA/call_id/AST collector/reset表达式保存在新closure：

|ID|dispatch行|历史producer候选SHA|
|---|---|---|
|A|58279|5215f0b49b772d8def14997996774bfd49db614b52b9f615c164271591f7b83b|
|B|22791/22814/22850|22518d203282b7b1a3aed99446f2262c11a9b42be4f6f2eab394e8c20801b9cd / 2830d8ace84341e48423b2fdb635f19ea4449c2b5edfc99806e5d1581a6c372a / 6170c69352462b8755b729b0e8dd32cdb873364944a60be9756f6d8935ad1962|
|C|23599|567bcbe1b4edc9241806f1b60882de5d971cab694620153d992199d984b5f3ac|
|D|23773|24171e996addc6910d30f800f8ad7c03817f9296b95ae4c675cf9dd3168006f3|
|E|58307|80b9674063360a2f51d2e146c830c29a5399bb53b9086b177f3b3f2dc3b0d5e4|
|F|23658|bc3b93478c3a29bf6f3dc0b0f4b478b6be9f963e1fb0384ec5776b9cf5d4b2d6|
|G|23730|06d2c0d2d279f43bdd094006d8d4095a800366cfdc2fef36342b8ca261359908|
|H|58363|6dc263e6f69840479daedd92a064bead928b1b1b80e1628f46e4c44964525806|

实质推进而非重新列unknown：F的v2 producer候选现在能从原Add+Update链恢复，不再用v3替代；B有3个版本/3次调用必须纳入历史尝试覆盖，不可只看最终report。B最终/C/D候选与先前Git字节吻合；A/E/H候选与先前postrun/后续seal保存字节吻合，但“吻合”仍不能把后续seal变成原运行前seal。两个恢复出的G/D v3版本不同，schema并非版本唯一标识。

## 逐项认证缺口与不可认证清单

8份原report/16引用全部核验，但8份都未取得原执行factory/env绑定、全部reset尝试证明和global scope authority：`original_execution_bounds_certified=0`、`coverage_complete=false`。每source输出自己的原report SHA、命令行/调用ID、producer候选SHA与`coverage_request`，明确至少要覆盖的dispatch数（B=3，其余=1），不把最终selected roots当全体尝试。

仍缺的具体材料：

1. 对每个A–H、每次dispatch的原producer/P0依赖、4层factory与当时installed env/wrapper字节的运行绑定。源码候选/patch proposal不能证明工具apply成功、无其它编辑、进程加载的就是该版本；当前安装env不能补历史证明。
2. 各次dispatch所有失败/未选/拒绝reset ledger，或独立可审的穷尽有限包络 authority。固定代码族可推导`base..base+8n-1`的保守for-envelope，另列seed0显式state重建；它是假设该族/分支/默认参数的**conditional**排除区间，不是历史实际覆盖。seed+70000、projection seed不是env reset，不胡乱并入；radial while不能因另分支存在而套入此族，反之缺真实分支绑定也不能擅称从未执行。
3. 原618 cutoff后活动的明确delta及global完整性责任边界。本轮只补两份固定会话，不宣称全机器、全部会话或其它线程已穷尽。
4. 将来固定4slots实际state7/wrapped pose/snapshot/RGB3与历史/批内排重。当前未生成candidate state；新seed不保证新根，goal/velocity/renderer变化不构成新根。

候选清单保留4固定slots，条件seed区间与预留区间跨root/reset namespace检查只可给拒绝信息；无冲突也**不是clear**。所有`certifiable=false`、`dispatch_authorized=false`，0可认证candidate，不可据此启动confirmation或伪签global freshness。knownroot校准和MAIN真实P1状态不由metadata测试改变。

## 命令与验证

从 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_freshroots_resume_20261027_resolver.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_freshroots_resume_20261027_resolver resolve
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_freshroots_resume_20261027_resolver verify --saved results/agent_freshroots_resume_20261027_coverage.json --sha256 SHA_FROM_HANDOFF
```

resolve只输出JSON；本交付实际closure通过apply_patch新增独占文件，不shell覆盖任何旧结果。verify重新读取同一固定输入并完整重算，包括源码/tests/本文SHA、所有log SHA；比较完整语义，不是只看status。旧结果漂移、漏一次失败dispatch、伪complete、版本替换、顺序/seed/阈值变化、截断/超限和重复JSON键均有RAM回归。测试无落盘tmp；测试通过不是物理覆盖或科学准入。

本轮实际RAM测试：42 passed（0.16s，禁pycache/pytest cache，无tmp）。metadata实际执行恢复8来源/10 dispatch/21 patch段/0 recovery errors；最终保存closure记录这三文件与全部输入SHA，供MAIN用上述verify复算。

交付冻结后MAIN可先执行verify，随后针对每source的`coverage_request`收集原运行材料。缺材料不能通过重跑旧generator补成历史事实；不得把新执行产物倒灌原证据链。本工具无prepare/run神经入口，元数据认证未知不由用户选择阈值解决。
