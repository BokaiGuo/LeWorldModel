# P2 组合准入与对比证据审计（2026-10-27，v1）

本模块是独立、只读、fail-closed 审计器，不是实验 runner。仅支持已存在的四个
校准根、composition/flat 与 full20 CEM/random 四臂的历史 schema；不扫描结果目录，
不打开未来 heldout、不选方法、不调整门槛、不运行模型或物理。无 AGENTS.md
存在于本次检查的外层/内层及祖先目录。共享 dirty worktree 不清理、不覆盖。

使用 experiment-code / experiment-design 技能的固定来源、配对基线、显式失败账本
和回归验证要求；用户范围优先，不训练、调参、画效益图或启动 GPU/physics。

## 准入语义

`evidence_integrity_valid` 只表示本版覆盖的保存内容/来源一致性，不是实证真实性认证。
`P2_admission` 永远 false：现有历史 schema 无法提供独立 P1、预封存独立根的组合
增量及严格同预算对照。没有通过补 `passed=true`、sampling 完成或官方成功字段来
打开准入的接口。未来证据必须另加经测试的独立审计适配器，不能沿用本版报告追认。

沿用外层 `D2P_RESEARCH_CLOSURE_AUDIT_20261025.md` 的 P1 依赖关系与严格有利区间
要求，以及 `d2p_p2_comparison_contract.py` 的完整预算、候选槽位和物理观测要求；
本版不创造新的最小根数、CI 阈值或成功定义。报告列出七项必要科学证据及可执行
前置操作，但不会执行它们，也不会把缺少证据称为理论失败。

## 实际核验范围

- 固定 compilation 原 SHA；composition protocol、声明、源码字符串镜像与当前源码
  SHA；full20 prepared/protocol/report 绑定；两类物理协议的全部显式 input bindings。
- full20 sampling、full20 physical、composition physical 的历史 source seals，逐份
  manifest、args、archive/current source 哈希及大小；审计器、测试、本文和指定报告
  的本次 SHA。结束时再次核验全部已登记字节，拒绝运行中变化。
- 按稳定 `(root_sha256, arm)` join，检查四根四臂，不依赖数组顺序；跨报告完整
  root/goal/context 摘要必须一致，拒绝重复、缺失或错接身份。
- 原始完整 predictor context，selected receipt/edge 内容与 SHA、动作域、严格十步
  action witness；composition 两边与非组合单边语义。这里的 compose 回调只取保存
  context，绝非新的神经 replay；原 replay 仅是历史收据声明。
- 每臂 80 planning+4 replay dispatch 与 168 个 predict/action_encoder 实际入口
  元数据逐项对照；full20 全部40候选、time-major `[2,5,2]` 到 `[10,2]` 映射及保存
  incumbent；旧 bank selected path 与 compilation 原 bank 逐值一致。
- 物理封存动作、full20 selection SHA、root/goal RGB、执行 prefix、步数/终态、
  terminal PNG 与可用 journal SHA。旧 guard / unavailable endpoints 全保留，
  不当作固定 H2 成功；官方 success、pose、official distance 不互相替代。

边界：本版不重算神经预测、官方模拟器指标或 FLOPs；不把 hash 自洽当执行真实性
或前瞻封存证明，不重新计算或解释真实 encoder Q，不声称逐候选物理结果已齐。
审计输出的来源清单是 audit-time 快照，不能用它伪造 outcome 打开前的外部封存。
原始细粒度 per-root 文件/逐事件 journal 的语义并非全部独立重建；完整 source seals
只证明保存代码字节一致，不证明实际运行代码必为这些字节。

## 运行与退出

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；output 必须不存在，包括空旧目录也拒绝。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_composition_admission_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_composition_admission_20261027_v1 --output results/p2_composition_admission_20261027_v1_audit01
```

输出仅全新 `report.json`、`report.md`，包含完整哈希清单与前置条件。退出2表示
证据一致但科学准入拒绝；退出1表示缺失/篡改/非法证据，仍保存拒绝报告。不存在
退出0的阶段成功路径；输出复用在任何读取前抛 FileExistsError。测试使用非神经
fixture、保存校准 JSON 和文件哈希，不代表效益验证；测试不会写已有实验目录。

## 当前真实缺口

full20 对照完成不等于 P2 完成。旧 composition/flat 是离散16-command bank，
full20 CEM/random 是连续20维 box；预测样本匹配不等于 proposal 或总预算匹配。
缺强 no-composition SRA、完整上游/编码/FLOPs/共同 wall cap 成本、全部候选的
配对物理结果、独立 P1 与独立根组合机制/真实增量统计。不得据四个已用根事后
挑指标、选方法、删中止根或换成功定义。

本次保存证据复核：旧 composition/flat 为5个完整可用H2分支、仅2个完整成对根；
full20 为8个完整分支，合计13/16，official success 为0/16。报告按根列出与三个
baseline 的12条描述性差值及固定端点标记，所有因果/总预算可比标志保持false。
不把这些差值解释为已验证效应，不计算已用校准根的前瞻显著性。
