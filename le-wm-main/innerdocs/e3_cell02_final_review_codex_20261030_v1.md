# E3 cell02 最终 manifest / 资源审核交接

只读审核器接受专用目录中经外部固定 SHA 的 cell02 manifest，先读取
并验证 canonical 字节，在本进程环境下观测资源与固定 claim/output，
隔离启动原候选入口的 `draft`，逐字段比较已批准 manifest 与新草案，
然后重新观测资源、确认预算和审核工具来源摘要未变并重读 manifest。
任何本地检查或比较失败返回非零；不创建 claim、输出或 manifest，
不调用候选 `run`。成功仅意味着两次时点检查通过且新草案相同，
不是生产就绪或批准。

工具用原 transport 常量报告输出 cap 512 MiB、final reserve 64 MiB、
剩余地板 200 MiB、创建前最低空间 712 MiB、墙钟 1800 秒以及分类
cap。报告包含磁盘、inode、宿主 MemAvailable、进程资源限制、
CPU affinity、预检与 transport 源码摘要。不得将这些观测当成
容量预留或 cgroup/GPU 可用性。最终审核示例，在 `le-wm-main`、
实际拟用解释器和完整最终环境执行：

```sh
.venv/bin/python -B -s -m experiments.spectral_reachability.e3_cell02_final_review_codex_20261030_v1 --manifest /home/gbk/code/LeWorldModel/le-wm-main/deployment_manifests_e3_cell02/e3_cell02_manifest_REVIEW01.json --manifest-sha256 REVIEWED_EXTERNAL_SHA256
```

占位 SHA 必须由审阅者用已批准的完整 manifest 文件字节 SHA 替换。
manifest 由现有部署预检工具生成；审核 JSON 输出应由主线程独立保全。
审核器和预检工具的来源也需独立审阅及固定，打印摘要本身不是批准。
CPU 测试或 `CUDA_VISIBLE_DEVICES=-1` 环境生成的 draft 不能当作最终
GPU 环境批准；现有入口会将整个环境、解释器、父来源和预检来源
纳入比较，即使 shell 修改环境变量也可能导致拒绝。

## 未消除的阻断

- 主线程尚需单独批准最终 manifest、审核器和 saved 验收检查器，
  明确一次 cell02 运行权限；本工具永远返回 `production_ready=false`。
- 需要与 GPU 所有者协调，确认 HOST CUDA 设备选择、排他和显存；
  本工具不调用 CUDA，也不证明没有竞争 GPU 作业。
- 需要排查其他入口的同 cell02 冲突以及当前生产状态；本入口 claim
  只约束遵守同一 claim 的调用方，不是跨入口锁。不检查 cell03。
- 磁盘、inode、cgroup/宿主内存和竞争负载不是预留；审核到启动有
  TOCTOU 窗口，现有 launcher 并未原子绑定本资源审核。运行时原
  transport 和 CUDA guards 仍生效，不能用审核报告替代。文件系统
  写权限、配额及 fsync 可用性/持久性也未通过写入探测验证。
- 本审核器只枚举 transport 字节和墙钟预算，未独立汇总 physical
  meter 和数值节点各操作次数 cap；实际执行仍受其原封存合同约束，
  主线程须在批准前检查这些预算，不应把 transport cap 当成全资源预算。
- native renderer、CUDA 模型加载、真实物理/反馈未在此验证；项目
  模块 census 只是时点文件证据，不证明 import 同字节、瞬态模块、
  解释器外库或 native 依赖。是否满足启动风险要求须人工决定。
- 完成后须确认进程确已终止、独立固定 publication 的 report 和
  completion SHA，再走 cell02 saved consumer；partial/失败保留 claim
  和输出，不自行清理或重试。保存验收仍非 full saved strict 或
  scientific admission，早停不能补齐。
- 本次只审 cell02，不改变 E3 固定 16 格分母、已封存 authority、
  原 runner 或科学门槛；单格接线通过不能证明全矩阵完成或支持比较结论。

## 测试边界

临时目录 fixture 覆盖 claim/output、磁盘、inode、CUDA 显式禁用、
manifest 哈希/规范格式、草案及前后观测漂移。隔离子进程调用真实
候选 `run`，执行真实 authority CPU 预检、draft、manifest 比较、
独占 claim 和源码复核；在子进程创建之前拦截 CHILD 和 AUDIT，
只验证实际命令参数、顺序、claim 保留和重复拒绝，不执行生产 runner。
另一测试使用真实独立 `draft` 检查 manifest 重建与审核器接线，但
将 claim/output 资源观测改为临时目录；这些测试不能证明生产运行、
GPU 可用性或完整生产端到端成功。

四组相关回归曾通过 **65 项**；补充审核器源码漂移保护和阻断说明后，
新版审核器测试（不重复耗时的 launcher 接线测试）为 **14 passed，
1 deselected**。中途一次编辑审核器触发源码漂移拒绝，未计入通过。
