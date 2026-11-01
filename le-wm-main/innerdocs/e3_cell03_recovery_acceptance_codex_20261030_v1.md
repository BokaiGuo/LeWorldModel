# E3 cell03 恢复验收准备：未授权提案

本文件和伴随脚本仅做只读检查、候选身份设计及 CPU 测试；不是新的
production launcher，不创建 claim，不授权重试，不改变科学门槛。旧
MAIN01 的 claim、output、authority 均保留。旧合同明确要求失败后取得
恢复方向，不能以本提案或 GPU 空闲替代该决定。

## 前次失败的可复核边界

`e3_cell03_recovery_acceptance_codex_20261030_v1` 调用原 transport
consumer 验证 report、completion、journal 哈希与事件链，并额外严格
核对 16 个事件的顺序及 index 15 的确切 census 错误。通用 consumer
的 operation failure 计数不包含 `launcher_failure` record，故必须
单独检查该记录。本次符合已记录的 CUDA loader、binding seal、随后
wrapper 未预封存错误；序列没有环境或物理 episode 事件。此检查不是
full native strict，也不是科学准入。

## 候选 identity 和来源

只读提案名称 `CELL03_RECOVERY01_PROPOSED`，列出独立的候选 claim、
output，绑定旧 claim/report/completion/journal 的 SHA、原 authority SHA，
以及当前旧 wrapper、CPU precheck、原 runner 和 child `-c` 精确 UTF-8
源码摘要。提案 SHA 是规范 JSON 字节的摘要；它是可复核的**草案**，
不是经授权的封存，更不是可执行的新入口。代码不会占用这些路径；
后续必须重新验证它们均不存在，包括悬空链接，也须审视所有其他可能
重复的运行和 job，不可仅靠名称或 claim 判断。

真正实现新的 launcher 前，至少需要另建并经独立审批的不可覆盖
attempt manifest，绑定：

1. 恢复决策及其授权人/时间/范围、旧失败证据、明确的 attempt 身份、
   路径和原 authority；审查同一 cell 是否允许再做物理尝试。
2. 实际新父入口及预检的所有项目内加载模块、子进程启动文本和原
   runner/依赖的明确路径与 SHA，以及 Python 可执行文件、启动参数、
   工作目录与影响导入的环境；不得把当前草案的四个摘要误认为完整闭包。
3. 外部可信预先固定的 manifest SHA。仅在运行时重算哈希并写入同一
   manifest 不能证明事前封存；源码或绑定漂移必须拒绝，不自动重定锚。
4. 父进程在启动前核验自身和项目模块实际加载内容，并记录封存及
   稳定性证据；子进程从新解释器执行**封存的精确启动字节**，继续使用
   原 authority 和原 before/after census。还需处理检查到加载之间的
   文件替换风险；简单 `hash(path)` 后再按路径 import 不是同字节证明。
   加载后再次核验只能检测部分漂移，不证明瞬态模块或外部 native 库。
5. 先完成资源/GPU 人工协调，再对新 attempt 独占创建持久 claim（含
   manifest SHA 与前驱 SHA），同步文件及目录；output 由原 runner
   独占创建。异常、崩溃、信号、publication 失败均保留证据，不自动
   清理或重试。此锁仅约束合作入口，不能替代跨项目 GPU 协调。

## 严格失败状态与验收矩阵

未来实现须分别验证并持久关联：父进程退出码、子进程退出/信号、
claim、原 transport report/completion/journal、overrun 标记，以及
真实 output 是否存在。无 claim、claim 后无 child、child 启动失败、
无 publication、partial、有 pending、failure record、非零退出、
矛盾的完成状态、额外文件及任何哈希/链错误均不得解释为完成；
保留为可诊断的失败或未知。report 和 completion 的原 ABI、独立
consumer 与 census 复核仍适用；有 `publication_overrun.json` 时不可
忽略或抹去（原 consumer 会拒绝额外文件）。仅 child exit 0 不够；
即使 publication 通过，native/feedback/科学验收也仍需原独立审计。

CPU 测试需覆盖并发竞争、precheck/claim/exec 各边界崩溃、SIGKILL、
部分写入与 fsync 失败、矛盾状态、哈希和事件篡改、源码/环境漂移、
子进程未封存模块注入、现有路径及软链接。实际 CUDA 模型加载后、
物理执行前及清理后的 census 仍需另行获得运行授权并按原合同验证，
不能以 CPU fixtures 代替。

## 本次复验

从 `le-wm-main`：

```sh
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.e3_cell03_recovery_acceptance_codex_20261030_v1
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q -p no:cacheprovider tests/test_e3_cell03_recovery_acceptance_codex_20261030_v1.py tests/test_e3_cell03_exclusive_host_codex_20261030_v1.py tests/test_agent_e3_transport_publication_20261029_v1.py
```

当前草案 SHA `8600e37059c42bcecb40b53134fa36dea87f9812316c13387b4935dc7b7e2404`；
这是本次只读计算的快照，任何输入变化会改变它。测试 24 passed：
包含真实旧 partial 审计、真实 authority/CPU census、合成的已重新封存
但语义不匹配事件、各前驱文件篡改、子进程 failure exit。并非生产运行
或完整父子来源封存验收。
