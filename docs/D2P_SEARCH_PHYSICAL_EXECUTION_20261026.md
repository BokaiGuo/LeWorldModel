# 封存搜索计划的真实执行

主线程在 2026-10-26 完成主机 CUDA + PushT 真实执行接入。不是成功率或 P1/P2 通过证据。

输入为 `pusht_program_search4_v2_20261026_cuda/report.json` 已封存的最佳动作见证；执行前核验根、目标、源码和输入绑定，逐条验证动作等于保存 primitive 索引拼接，不改选择、不补零、不因 outcome 换候选。

runner：`le-wm-main/experiments/spectral_reachability/run_pusht_search_physical_cuda.py`。

产物：`le-wm-main/results/pusht_search_physical4_v1_20261026_cuda/report.json`，`integration_completed`，进程退出 0，7 项完整性门全 true。全部四分支保留；终态编码使用冻结官方模型主机 CUDA，无 CPU 神经回退。终态向量、完整实际命令、RGB 哈希、状态与两种评价标签均保存。

| 根 | 计划长度 | 实际执行 | 状态 | official 成功 |
|---|---:|---:|---|---|
| 0 | 10 | 10 | completed | false |
| 1 | 10 | 10 | completed | false |
| 2 | 5 | 5 | completed | false |
| 3 | 10 | 5 | observation_unavailable | false |

根 3 的首 primitive 仍把推杆带出画面，实时 guard 阻止后续 5 条盲命令。所有候选均使用同一规范 RGB presence guard；真实物理状态只记录作评价，不进入模型搜索或 guard。四根 official/object-only 成功均为 0，但这些已知校准根和很短的不等长计划不是正式闭环 benchmark，不能从中推出理论不成立。

这一步新增的是“模型搜索→带动作见证→真实执行→保留失败”接线证据；没有解决出画面预防、恢复控制、物理任务与 latent 目标不一致，也没有独立新根与等预算强基线优势。先前新 runner 的阶段名与评分接线错误已单独修复归档；不能把这次通过完整性检查后的控制失败说成已被那些错误解释。
