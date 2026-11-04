# v3 依赖封存边界

Reacher 真实 RGB 原 manifest 未绑定以下实际执行依赖：响应与 CEM
核心、反馈 CEM、预算分配及 Meter。原 manifest 的哈希检查和后续
预测重算不能证明这些文件在原运行时的源码哈希；不补写原 manifest，
也不将现在的文件哈希说成历史证据。

新增 [显式依赖封存工具](../le-wm-main/experiments/spectral_reachability/pc_v3_dependency_seal.py)，
将上述文件、v3 入口、Reacher 适配器和热启动对照纳入最低声明集；
后续新运行需在规划前把 `capture` 结果写入自己的不可覆盖 manifest，
并在执行后调用 `verify`，同时将实际协议、入口、配置、模型与统计
文件通过 `extra_paths` 传入。仅存在工具不等于已有 runner 已接线。

这不是完整依赖闭包：模型实现、数据、动态导入、环境包、GPU/后端、
底层库及系统配置仍须根据新运行单独封存。token 仍不是总成本，
不得因这个清单而声称完整成本或独立复现已认证。
