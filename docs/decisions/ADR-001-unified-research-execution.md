# ADR-001: 统一科研实验执行契约，保留历史入口

## Status

Accepted

## Date

2026-10-14

## Context

项目同时有 `run_full_spectral_transfer.sh` 和
`run_unified_research_protocol.py` 两个总入口。两者都有效且被历史协议、测试和
结果路径使用，但目前分别实现 stage 编排、resume/skip、日志和产物判断。仅按
文件存在跳过会在数据、checkpoint、代码或 schema 改变后错误复用旧结果。六目标
矩阵还会读取未由本次 unified manifest 声明的历史目录。

## Decision

保留两个入口名和历史行为作为兼容面；新增轻量
`experiments/research_runtime/`，统一提供 input fingerprint、原子 artifact IO、
execution stage state、manifest identity 和输出验证。新 protocol 通过显式
evidence records 构造 claim matrix。历史结果只能作为显式的
`external_prior_evidence` 输入。

迁移按小步进行：先治理和验证，再共享执行器，最后才考虑提取科学 package。不会
一次性移动/重命名上游 LeWM 或删除负结果实验。

## Alternatives considered

### 删除一个总入口

拒绝：会破坏现有复现实验、用户脚本和结果路径，而且不能自动修复历史 evidence
耦合。

### 建立新的超级 runner

拒绝：会复制科学逻辑，进一步制造第三套 stage、缓存和证据语义。

### 一次性重构目录

拒绝：checkpoint/config/import 和历史协议风险高；渐进迁移更容易做固定输入的
行为回归。

## Consequences

正面：输入变化可检测，失败可恢复，dry-run 可审计，execution/evidence 状态分离，
claim matrix 的证据来源明确。

代价：短期会同时维护旧入口和共享 runtime；旧 artifact 需要逐步补充 provenance，
并且已有脚本不能立即获得完整 schema 验证。任何缺少输入锁的历史结果都必须保持
历史/外部状态，不能自动晋升为本次 protocol evidence。
