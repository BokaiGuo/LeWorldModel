# Variable CEM 双根保存记录独立复核

日期：2026-11-01。只读核验既有保存记录；未运行 GPU、模型或物理环境，未修改运行来源。此记录不改变历史封存。

| 项目 | root0 | root1 |
|---|---:|---:|
| report SHA-256 | `ec644cb8961d3e673edb2baa877bbe9faa530921fd9ce4f5889a197214333c68` | `1c919ae38784d03822b948479593700f161d4ae7f71126d9e11daade7a4959d7` |
| 已执行控制步 | 17 | 6 |
| 已封存节点的 predict 账面预留 | 195, 194, 193, 193 | 195, 194 |
| predict 账面预留总数 | 775 | 389 |
| 每包执行步数 | 5, 5, 5, 2 | 5, 1 |
| 未启动边界 | 无 | 10, 15 |
| 最终前四位置坐标联合 L2 | 13.61210916243796 | 16.830586123550493 |
| 最终原生角度项 | 0.17624880204953985 | 0.0001533068597270315 |
| 保存记录重算首次原生成功 | 第 17 步 | 第 6 步 |
| 物理日志 / CEM 日志事件数 | 834 / 4814 | 506 / 2078 |

成功判据为前四位置坐标联合 L2 小于 20，且角度项小于 π/9；不使用速度阈值。终点报告的七维 distance 分别为 497.2838883504739 和 171.50818872432123，不是该成功阈值。

## 核验范围

核对指定 report 文件哈希、prepare 哈希与协议绑定、保存的根/目标/模型身份一致性；检查物理日志完整哈希链及 report 对文件与链尾的绑定；检查节点账本的哈希链、合同、intent/return 顺序和累计预留；重新生成各轮采样、精英更新和胜出候选，核对包封存、逐步动作派发与返回、观测 PNG 文件哈希和逐步成功判据。已封存节点账本均为 `closed_unverified`，`actual_usage_verified=false`、`coverage_verified=false`。

CEM 事件文件检查了内部哈希链及采样、分数、选择与封存包之间的语义关系，但 report 未直接锚定该事件文件哈希；其全部内容不应描述为被 report 外部认证。未重新计算模型输出或物理轨迹，未解码 PNG 像素哈希，也未逐一重新认证 prepare 的全部来源。保存日志的自洽性不是独立物理复现实验或独立 GPU 使用真实性证明。

两根均为 `recorded_known_root_not_admitted`，是已知根探索性结果，不是预先确认的独立泛化证据。节点数字是账面预留上界，不是已验证实际使用量；早停使两根累计账面开销不同，不主张等成本或将结果后选择的配额视作预先确认。

## 复现

在 `le-wm-main` 目录，仅 CPU 执行：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_p1_variable_cem_root0_audit_20261101_exclusive
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p1_variable_cem_root0_audit_20261101_exclusive.py
```

测试结果：5 passed；包括两根保存记录及对胜出者、精英、缺失分数、包内容和成功公式的负向检查。
