# cell02 独占 HOST 候选：主线程验收

本件新增独立候选源码和 CPU fixtures；未启动 CUDA、模型或物理，
未创建生产 claim/output，未访问 cell03 claim，也未改原 runner、authority
或科学合同。**这是待审批的生产前置，不是已授权、已证明安全的生产运行。**

## 实现

- `draft` 只读重建 cell02 原 authority/source guard 与 saved setup。
  在新 CPU 子进程执行预检，记录其项目树内当时加载文件的路径和 SHA；
  父进程同样采集加载文件，并检测预检期间的父来源变化。候选 manifest
  另绑定原 authority、固定 claim/output、Python 调用路径及解释器文件、
  全环境摘要，以及预检、runner 子进程、transport audit 三段精确代码
  的 SHA。环境只保存摘要，不保存值。
- `run` 要求外部提供 canonical JSON manifest 及 SHA、确认字符串，
  再重建草案并逐项比对。草案或确认字符串**不代表独立授权**。
  CUDA 显式禁用时拒绝；`PYTHONPATH`、`PYTHONHOME`、`PYTHONSTARTUP`
  必须为空，并关闭 user site 和 bytecode；工作目录固定。
- 已验证后对固定 claim 使用 `O_EXCL|O_NOFOLLOW` 创建、文件与父目录
  fsync，记录 manifest SHA、authority SHA、输出和 PID。已存在输出
  或悬空 symlink 会拒绝。创建后任何错误保留 claim/partial，不重试。
  之后复核父模块、解释器、环境及预检模块路径/哈希；runner 在全新
  Python 子进程执行，继续使用原 authority/source guard 和原 census。
- child 返回 partial 或异常产生非零状态；信号转为非零。子进程零退出
  仍须由另一个隔离子进程使用原 transport consumer 验证 publication、
  cell02 identity、authority 路径及 SHA、唯一 terminal 且无 launcher_failure；
  完整 publication 还须重新验证原 authority 和保存的 before/after
  module census（包括当前来源字节及前后模块一致性）。
  audit 失败同样非零。通过仅表示 transport 和 census 检查通过，**不是 full
  saved strict、任务成功或科学准入**。没有父进程完成 receipt；崩溃
  或被杀时必须从 claim、输出和日志人工判定未知/partial。

claim 仅约束合作本入口的进程，其他入口/输出名可绕开；固定文件名也
不能证明其他 cell02 作业不存在。运行前仍需主线程全范围去重和与 GPU
使用者协调。此候选不会检查/锁住其他人的 GPU 作业。

## 真实边界和待主线程决策

1. 父/预检来源清单是**时点文件哈希与已加载模块快照**，不是已加载
   字节证明：import 后哈希不能排除加载与哈希间的替换、瞬态模块或
   文件恢复；不覆盖解释器外的标准库、外部 native 库及所有环境影响。
   原 runner 的前后 census 也有同类范围限制。要宣称更强的封存，需
   有经审查的可信启动/同字节加载机制；本候选尚未实现它。
2. manifest 必须在最终预期环境中生成、由主线程独立审阅并外部固定
   字节 SHA。当前 `draft` 是 stdout，不自动写 manifest；样本以
   `CUDA_VISIBLE_DEVICES=-1` 验证，不可用作 GPU 启动 manifest。全环境
   摘要严格匹配，任何变量变化都会拒绝；不要绕过比较或重锚未知漂移。
3. 需要单独批准一次 cell02 known-root HOST CUDA/native、协调实际
   GPU、检查其它潜在输出及进程、磁盘/renderer/native 依赖；原 runtime
   资源及 source guards 继续生效。保留已完成的其他15格和所有早停。
4. parent、子进程及 audit 脚本运行时仍可能因 source drift、存储、
   超时、CUDA、native 或原 guard 失败。异常测试不能替代真实加载后
   的 before/after census、native 语义和完整失败路径验证。
5. 旧 frozen authority 并未绑定这个新 launcher、预检或 audit 模块；
   新 manifest 是额外外部封存，不改变原 authority，且需要审阅者
   决定以上弱点是否满足实际运行资格。不得将本文件当成该决定。

## 本轮验证

从 `le-wm-main`：

```sh
env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP CUDA_VISIBLE_DEVICES=-1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -s -m experiments.spectral_reachability.e3_cell02_exclusive_candidate_codex_20261030_v1 draft
env -u PYTHONPATH -u PYTHONHOME -u PYTHONSTARTUP CUDA_VISIBLE_DEVICES=-1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q -p no:cacheprovider tests/test_e3_cell02_exclusive_candidate_codex_20261030_v1.py tests/test_e3_non03_host_readiness_codex_20261030_v1.py tests/test_agent_e3_transport_publication_20261029_v1.py
```

当前 CPU 样本草案含 4 个父进程和 246 个预检子进程项目文件；
此数字与环境及导入有关，不是完整依赖闭包。24 项测试通过，包括
claim 并发、已存在输出、fsync 失败、源码漂移、子进程异常/信号、
真实 transport audit 的 completed/partial/矛盾记录和真实 CPU 子进程
module census。一次在预检期间编辑候选源码的草案运行按预期拒绝
`parent census drift during precheck`，没有 claim。

## CPU 加载后 census 与出版物退出追加验证

新增测试先在隔离进程生成实际 CPU 草案，再在另一个进程以原 cell02
authority 和其 `prepared.bindings_sha256` / `additional_sources` 验证
来源，使用原 loader 在 CPU 读取真实 config 和权重、实例化模型、
校验权重文件字节及模型张量哈希。由于原 `frozen` 强制 CUDA，仅在这个
一次性测试子进程中替换为等价的 CPU eval/no-grad 断言；原源码未改。
没有 neural forward、CUDA 或物理调用。

草案 `precheck_sources` 与 authority 共有路径的哈希一致；加载后原
`module_census.snapshot` 验证了全部已加载项目树模块与 authority 静态
bindings。本次 6127 个模块条目，其中 5784 个不同的已加载文件路径
不在预检子进程快照内，而由原 authority bindings 覆盖。移除任一测试
绑定或替换摘要时原 census 均拒绝。这些计数只是该 CPU 环境的一次
时点快照，不是 CUDA 加载后的闭包或模块瞬态轨迹证明；候选 manifest
的 precheck snapshot **不单独覆盖**运行时依赖，两者通过固定 authority
SHA 关联。

另用原 `_run_lifecycle`、真实 Store 和原 transport consumer 在临时
目录做完整生产流程的 *fixture 接线*：authority 和 episode 在子进程
中替换为测试函数。成功 publication 与 `completed` child exit 0 一致；
fixture 异常出版 partial、child 非零且 audit 非零。独立测试确保缺失
completion、被篡改 journal 和额外 overrun 标记均非零。没有启动真实
episode，publication completed 仍不等于任务成功或科学准入。

更新后完整 CPU 回归 **30 passed**。运行时项目文件、解释器及外部包
可能漂移，必须在批准生产前重跑并重新评估；原 CUDA-only frozen
guard、实际 GPU 模型加载及 physical/native 路径尚未验证。

## 保存 census 的完成判定接线

独立 audit 对完整 publication 重新加载并验证 cell02 authority，合并其
prepared bindings 与 additional sources，调用原 `module_census.saved`。
缺失任一快照、来源绑定不符或来源字节变化均拒绝，即使 transport
本身完整。partial 仍先按 transport 状态拒绝，不被当作完整 census。
这些检查仍是时点文件证据，不证明 import 同字节、瞬态模块或外部 native
依赖，也不构成 full saved strict 或科学准入。audit 代码摘要已改变：
任何旧 draft 不可复用，生产前必须在最终环境重新生成并独立审阅固定 SHA。
上述三组 CPU 回归更新后为 **34 passed in 72.14s**；真实 checkpoint
加载后的 census 为 6127 条，预检快照之外的运行时路径为 5784 个。
