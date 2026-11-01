# E1 learned 新运行预封存合同（待授权，非运行记录）

## 身份与范围

固定来源为 `le-wm-main/results/agent_hilewm_rgb_high_low_20261028_v1_prepare_MAIN01/manifest.json`，
SHA256 `a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9`。
原 runner `verify` 在 CPU 上完整重建 manifest、来源、输入和上游收据；本地已实测通过。
这是已存在的历史 prepare，作为**新运行输入**，不将其变成新运行结果。

固定顺序为 `(root0,off), (root0,on), (root1,off), (root1,on)`，全部四行；
每行执行 RGB 编码、Hi CEM/FIRST，之后还运行 low 固定候选 panel。
两条 off 行与既有 E1 learned 目标的保存来源有关，但新运行不能假定输出逐位一致；
需事后严格检验。on 行和 low panel 不得计作 off FIRST 的专属 proposal 成本。
这是 known roots 的 integration，不是 fresh-root 泛化、物理闭环、官方实现 parity、
公平预算基线、E1 科学准入或三臂排名。

固定运行 ID `codex_e1_learned_durable_EXCLUSIVE_01`，arm `learned`；固定输出和
journal 路径见下面命令。二者必须原本不存在，不能重用、续写、覆盖或补写旧账。
任何失败、部分输出或不完整日志均保留；后续重试需新合同、新身份和新路径。
日志中 attempt、return、failure 与未匹配事件分开解释；尝试不必然表示模型已调度。
调用计数并非 FLOPs、时间、显存、金钱或完整祖先成本；共享费用 unknown / unallocated，
不得按零或事后虚构分摊。运行 journal 不覆盖模型构建、全部祖先和跨阶段去重，
也不提供完整端到端费用收据。

## 独占与授权门槛

本合同只排他创建指定文件和目录，不提供系统级 GPU 锁。未来操作人必须另行获得
MAIN 对这一次 CUDA 执行的明确授权，并通过外部调度器或全体使用者共同遵守的锁
独占指定物理 GPU，确认其他计算进程不存在且运行期间保持独占。仅检查一次空闲、
设置 `CUDA_VISIBLE_DEVICES` 或创建 journal 都不能证明独占。设备身份、驱动/软件版本、
授权、锁、开始结束及失败证据应由操作人独立保全；缺任一条件即不运行。不得据此
声称 GPU 时间或使用量已由本 journal 计量。

原 runner 有输出 192 MiB cap、200 MiB free floor、900 秒 guard；journal 不受该
runner cap 覆盖，需另预留空间。runner 本身的来源校验、输出目录排他创建及
journal 的 O_EXCL 不消除预检和启动间的竞争窗口。预检失败、身份变动、空间不足、
设备独占无法保证、预定路径已存在或外部授权缺失时停止，不修改既有产物。

预封存检查脚本 SHA256：
`092e6571b08d2e39af93f832874c865f7a155dc891368714b46436341fe132c7`。
脚本校验现有 manifest、原 runner 的完整 verify、四行顺序、两个新路径和以下新插桩：

- durable runner SHA256 `f9bb88cb410d73b37da9ee1653efb602d74e1122837dd9be9930c0ce50f38b25`
- journal SHA256 `aab370f280c0a714eb8cfc65711cc4e04cbbc4a98fde55f87181363bd712fdf6`

合同是本地固定方案，不是可信时间戳、独立签名或事前公开注册；需要外部见证时，
须在任何 CUDA 执行前独立保存此合同及其摘要。代码变动需重新制订合同。

## CPU 预检（允许，无 GPU）

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
sha256sum experiments/spectral_reachability/codex_e1_learned_preseal_20261101_exclusive.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_e1_learned_preseal_20261101_exclusive
```

须人工核对脚本摘要与上面一致且预检成功。临近执行时重做；预检不会锁文件或 GPU。

## 后续 CUDA 命令（本次禁止执行）

只有满足上述所有门槛后，操作人选择已独占的物理设备，设置 `CUDA_VISIBLE_DEVICES`
为该设备的设备标识，再从同一目录执行：

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_e1_learned_durable_runner_20261101_exclusive \
  --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_rgb_high_low_20261028_v1_prepare_MAIN01/manifest.json \
  --manifest-sha256 a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_rgb_high_low_20261028_v1_cuda_CODEX_E1_DURABLE_EXCLUSIVE_01 \
  --journal /home/gbk/code/LeWorldModel/le-wm-main/results/codex_e1_learned_durable_EXCLUSIVE_01.events.jsonl \
  --run-id codex_e1_learned_durable_EXCLUSIVE_01 --arm learned --main-authorized
```

`--main-authorized` 是技术标志，不等于实际授权。完成后保留原始 report、所有数组、
完整 journal 和独占证据；先以 `read_events` 严格校验完整链及未匹配尝试，再按原
runner 的 consumer 验证 report。原 consumer 的收据不能弥补新增日志或祖先成本缺口。
只有完成对照分析并事先定义统计和费用口径，才可判断该新运行能支持什么结论。
