# 空间清理核查 — 2026-10-27

## 第三轮：旧 npx 缓存清理

追加删除六处闲置 npx 下载/展开缓存（firecrawl-mcp、两份 pnpm、
DeepSeek dsh、desktop-commander、next-devtools-mcp）；保留正在使用的
filesystem、memory、sequential-thinking 三处 MCP 缓存。

- 删除前逐文件检查 inode、设备、mtime、大小、链接数与路径；文件均超过一天未修改。
- 检查可访问的当前用户进程命令行、映射、打开文件及 cwd；未发现目标引用。
  三个受保护系统服务（systemd、sd-pam、tailscaled）只能检查命令行，
  未能检查映射/打开文件，限制已写入回执；不声称完整覆盖所有进程。
- 实际删除 64,344 项文件/符号链接及 7,383 个空目录；错误 0。
- 单链接文件释放占用块 717,017,088 bytes（683.80 MiB）；
  不含空目录块，也不把全盘并发变化归因于本操作。
- 逐文件清单：`STORAGE_ROUND3_MANIFEST_20261027.json.gz`；
  执行回执：`STORAGE_ROUND3_RECEIPT_20261027.json`。
- 缓存原件未备份；需要时可由 npm 重新下载软件包，不承诺网络上历史版本永久可得。
  未卸载全局软件或改动 MCP 配置，没有删除模型、数据、环境、论文及实验证据。

当前大头仍是 Cube/Reacher 原始 H5、其余数据和约 37 GiB 实验结果；
没有建立可替代性，因此继续保留。临时目录内的版本审计源码和失败下载也保留。

## 本次追加清理（后续核查）

重新核查后，按逐文件清单删除 7,499 个可再生成缓存文件：

- 4,500 个有保留源码对应的 Python 字节码：27,000,374 bytes。
- 2,999 个超过一天未修改、单硬链接的 pip / pnpm / node-gyp
  下载、元数据或构建头文件缓存：125,292,320 bytes。
- 合计逻辑大小 152,292,694 bytes；单链接文件占用块
  170,516,480 bytes（162.62 MiB）。删除前再次校验文件身份、
  路径及源码；变化或缺失候选为 0。未删除任何缓存目录本身。
- 清单：`STORAGE_FOLLOWUP_MANIFEST_20261027.json`；SHA256：
  `3ddbe1d2d4dc94ab3f3a503ead620e268ba2d0f5911251aba825b90abfb69304`。

删除的缓存不保留原文件副本，但字节码可从源码再生成；软件包缓存
可重新下载或构建。没有卸载虚拟环境、CUDA 库或其他软件。

清理后根文件系统仍显示约 60 GiB 可用（94% 使用率）；
可归因释放量以上述文件占用块为准，不以并发变化的全盘差值计算。
项目约 255 GiB 的主要占用包括 Cube 原始 H5（约 95 GiB）、
Reacher 官方 H5（约 92 GiB）、其余数据及约 37 GiB 实验结果。
它们没有被证明冗余，因此全部保留。模型、论文、源码、封存证据、
失败结果及无源码配对的字节码同样保留。

## 本次完成的清理

仅清理 Python/pytest 缓存，不清理模型、数据、源码、实验结果及负结果。

1. 前一轮删除四处缓存，约 28 MiB：
   - `.pytest_cache/`
   - `le-wm-main/.pytest_cache/`
   - `le-wm-main/experiments/spectral_reachability/__pycache__/`
   - `le-wm-main/tests/__pycache__/`
2. 后续采用逐文件清单，仅扫描 `le-wm-main/.venv/` 和
   `le-wm-main/experiments/` 的 `__pycache__/*.pyc`。排除符号链接、
   无对应源码、Git 跟踪文件；删除前再次检查源码存在、路径范围及
   设备号/inode/大小/mtime 未变化。
   - 删除 7,908 个源码配对字节码文件。
   - 文件逻辑大小：154,525,179 bytes（147.37 MiB）。
   - 单链接文件占用块：172,175,360 bytes（164.20 MiB）。
   - 删除时变化或消失的候选：0。
   - 清单见 `STORAGE_CACHE_MANIFEST_20261027.json`，SHA256：
     `394fcef04143b971812496830d36db5aa7e4e459c5d838cb6b4053faa05dbc95`。

第二轮缓存可从保留源码重新生成；没有重装或删除虚拟环境。
并发任务仍可能重新生成缓存，因此占用块不是全盘净释放量的保证。

## 第一轮的孤立字节码例外

第一轮删除测试缓存前没有完成逐文件源码配对，判断不够严谨。
其中有两个对应源码在更早时候已被主动删除的孤立文件：

- `test_pointmaze_finite_intervention_repairability.cpython-312-pytest-9.1.1.pyc`
- `test_audit_endpoint_stability_repairability.cpython-312-pytest-9.1.1.pyc`

已找到完整历史源码文本，但没有找到可验证相同的原始字节码副本，
不能声称原文件已恢复。为避免撤销历史上主动撤下测试的决定，
没有把这些源码重新放回当前测试集。

历史完整源码位置（会话记录，不是当前测试）：

- `/home/gbk/.codex/sessions/2026/10/21/rollout-2026-10-21T02-13-04-01a15a84-55aa-7ac1-9b95-0ebb7076f145.jsonl:155`
  的成功 FileChange 删除记录含第一份源码，7,048 bytes、5 个测试。
- `/home/gbk/.codex/sessions/2026/10/22/rollout-2026-10-22T15-08-17-01a05ac0-4315-7fd1-963b-d838715aa2c4_01a16270-6c4a-7bc2-a3e6-22a45dc21dc1.jsonl:881`
  的成功 FileChange 删除记录含第二份源码，1,798 bytes、1 个测试。

## 明确保留

- Cube / Reacher 官方大型 H5、所有模型权重和环境安装文件。
- 所有已封存输出、真实执行图像、负结果与协议归档。
- `.cache/dinov2` 内的模型权重：目录名带 cache 不代表可直接删除。
- PointMaze RGB float64 数据：未建立其可安全替代性，不删除。
- 无源码配对的剩余字节码：不删除。
- 两处 spectral transfer 的 `latents.npz`：先前已确认同一 inode、
  nlink=2，是硬链接；删除其中一个不释放其数据块，不作为回收目标。

## 空间归因

本次复核开始时 `df -hT` 显示根文件系统可用约 47 GiB。
此前从不足 200 MiB 到数十 GiB 的变化来自本清理之外的并发操作，
不能归功于本次删除。可归因的操作仅为上列约 28 MiB 缓存和
第二轮 164.20 MiB 占用块；清理后实时 `df -hT` 显示可用约
59 GiB（94% 使用率）。47→59 GiB 的增长同样包含并发外部操作，
不全部归因于本清理。
