# PointMaze C3 V26：高运动配对簇校准的冻结复验

V26 是 V25 的新鲜根 successor。它不是对 V25 已打开 outcome 的阈值重调，也不复用 V25 的训练、校准或 outcome 根。

## 修复的协议错误

V25 的高运动条件由同源的 `+/-` 速度对构造，并将完整对保留在同一个 split；但当时仍将 128 条根作为 128 个独立 conformal/Bootstrap 单位。独立审计证明，这只有 64 个独立源对，因此 V25 的高运动 0.8828125 根级覆盖不能支持其原先的 90% 有限样本解释。

V26 保留相同可见历史机制和固定 candidate bank，但预先声明：

* 高运动校准单位：每个源对两根最大 contrast error；
* 高运动效用 bootstrap 单位：每个源对两根平均 paired regret gain；
* 直接零运动：每个唯一来源根仍为一个独立单位。

## 新鲜运行与结果

CUDA 主机运行使用新根：高运动 35080--35527、直接零运动 35528--35911。静态 preflight 通过 candidate 支持、可见历史、位置代理拒绝、动作对反转、历史根隔离以及直接根唯一性检查。

| Stratum | Independent calibration units | Outcome coverage | Release rate | Paired regret gain |
| --- | ---: | ---: | ---: | ---: |
| high motion | 64 source pairs | 0.953125 (61/64) | 0.9296875 | +0.163212; 95% cluster bootstrap CI [0.146493, 0.177091] |
| direct zero motion | 128 roots | 0.921875 (118/128) | 0 | 0 |

高运动的 split-conformal rank 为 59/64，对应有限样本边际覆盖下界为 59/65 = 0.907692。在本协议范围内，这支持新鲜 fixed-bank 高运动 C3 转移结果；它不支持 conditional coverage、permit-conditional safety、adaptive/CEM candidate 或在线 MPC 的结论。

## 证据

* 运行报告：`le-wm-main/results/pointmaze_c3_clustered_high_v26_20261023_cuda/v24_c3_report.json`
* 独立审计：`le-wm-main/results/pointmaze_c3_clustered_high_v26_20261023_cuda/v26_clustered_high_audit.json`
* 静态 preflight：`le-wm-main/results/pointmaze_c3_clustered_high_v26_static_preflight_20261023.json`

结论边界：V25 的高运动覆盖短缺是配对簇伪重复污染的负结果，不能作为反对理论机制的证据。V26 提供修复该协议后、在新鲜离线固定银行上的正向 C3 证据；C1/C2 的未见连续动作与 RGB-only MPC 仍须作为独立的新鲜研究阶段验证。
