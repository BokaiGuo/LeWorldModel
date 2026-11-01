# E3 cell02 部署预检及 manifest 交接

新工具只观测固定 cell02 claim/output、可用磁盘和 inode、主机
MemAvailable、进程资源限制及 CPU affinity，并从现有 transport 模块
读取合同预算。它不遍历其他 cell、不分配 CUDA、不做模型前向或物理。
这是一份时点观测，不是资源预留、排他证明或运行授权。

预算以实际导入模块为准：输出上限 512 MiB，final reserve 64 MiB，
磁盘剩余地板 200 MiB，创建前至少 712 MiB，transport 墙钟 1800 秒。
分类上限见 JSON 的 `contract.category_caps_bytes`；各分类并非可相加
的独立保证。墙钟从 lifecycle 开始计算，非性能预测；文件系统空间、
inode、宿主机内存及 rlimit 可能变化，MemAvailable 不代表容器/cgroup
可用内存。GPU 显存、CUDA 分配、native renderer、其他入口冲突及
GPU 协调均未检查。原运行时 guard 仍是最终合同，不放宽科学门槛。

预检要求项目根目录为工作目录，清除 PYTHONPATH、PYTHONHOME、
PYTHONSTARTUP 并设置 PYTHONNOUSERSITE=1、PYTHONDONTWRITEBYTECODE=1。
`observe` 只输出 JSON；`generate` 要求 CUDA 未显式禁用、固定 claim
及 output 均不存在（含悬空 symlink）、磁盘地板及 inode 检查通过，
并在 draft 前后重复观测。它用同一解释器和继承的环境，在全新
`-m ... draft` 进程生成候选 launcher 原生 canonical manifest，
独占创建文件并 fsync 文件与目录；输出 manifest 的完整文件字节 SHA。
失败后若写入已开始，文件可能存在或不完整，保留并检查，不覆盖。
单独的 preflight 工具及 transport 文件摘要也写入观测报告；该报告
本身不是签名或 manifest 的一部分，需要独立保全和审阅。

主线程准备时，在最终工作目录、最终解释器和拟运行的完整环境中，
先人工创建项目根目录下专用目录 `deployment_manifests_e3_cell02`，
确认其为真实目录而非 symlink，再执行：

```sh
.venv/bin/python -B -s -m experiments.spectral_reachability.e3_cell02_deployment_preflight_codex_20261030_v1 observe
.venv/bin/python -B -s -m experiments.spectral_reachability.e3_cell02_deployment_preflight_codex_20261030_v1 generate --manifest-out /home/gbk/code/LeWorldModel/le-wm-main/deployment_manifests_e3_cell02/e3_cell02_manifest_REVIEW01.json
```

上例只是生成 manifest，并不启动生产；最终环境须由操作者预先设置。
需要保存完整 JSON 报告、manifest 原始字节和生成 SHA 供独立审阅；
不要将 SHA 重算后直接当成外部批准。候选入口核对完整环境摘要、
解释器、父模块、预检文件及源码，因此运行时环境或文件变化会失败；
使用相同环境、工作目录、解释器与 `-m` 入口，并以独立批准的 SHA
调用原候选入口。preflight 与实际启动有竞争窗口，固定 claim 只能
约束此候选入口；必须再次独立排查其他入口并协调 GPU。失败后保留
claim 和输出，不自行重试或清理。

本工具不宣称 runtime CUDA census、导入同字节证明、完整 saved strict、
生产授权或科学准入。改变候选源码、环境或 authority 都要求重新
生成并审核，不可沿用旧 manifest。CPU 测试中的临时 manifest 不是
最终环境 manifest，也不能作为生产授权。

验证：部署预检、既有 cell02 候选入口及 transport 三组 CPU 回归
40 passed；之后新增的异常 draft 和独占路径测试另行执行，4 passed
（含一个已有测试的扩展）。真实 CPU checkpoint census 为 6127 条，
无 neural forward 或物理调用。一次现场 `observe` 显式禁用 CUDA：
cell02 固定 claim/output 当时不存在，剩余空间 32,316,231,680 字节。
这些时点观测不保证后续启动时仍然成立。
