# spectral8 单臂主机 CUDA 工程运行

2026-10-30 的已知根单臂运行由主机服务
`spectral8-engineering-20261030-main01` 完成，退出码 0。报告
`le-wm-main/results/codex_spectral8_host_cuda_20261030_HOST01/report.json`
SHA-256 为 `410ab431f03d2663d613dbc5f83da1095e94f29e0c56644df8f6a3829f9d1005`。
模型设备为 CUDA RTX 5060 Ti；运行前后模型权重及 buffer 哈希一致。

报告记载所有八个谱候选经过两步 raw 后缀优化及最终 replay，选中索引 0。
账本消费者记录 75 次 action encoder／predict、208 个 VJP、一次 Jacobian
标记、一次 SVD 和八次 replay／selection。神经计算在 CUDA，所用 float64
SVD 为 CPU NumPy。账本状态为 `closed_unverified`，自身明确表明实际调用
覆盖和公平性未认证；上游编码和所有计算成本没有完整计入。

这是探索性的 spectral8 H3 extension 单臂工程验证，不是原始 D2P 的完整
reconfiguration 程序，也没有物理执行、独立根或其他对照。不能据此宣称
P1 通过、真实控制改进或等预算优势。服务总耗时 2 分 55.094 秒，
报告内部 elapsed 约 60.97 秒，两者覆盖范围不同。
