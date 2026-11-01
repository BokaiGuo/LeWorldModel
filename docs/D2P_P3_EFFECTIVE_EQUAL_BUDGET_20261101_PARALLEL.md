# P3/C3 有效等轴调度对照（2026-11-01）

## 结论

本轮解决了一个明确的工程缺口：在固定预算点内，`signal` 与
`reverse` 不再因配额取整而退化为相同策略。两臂使用相同候选总量、
相同已记录资源调用向量和相同反馈机会，但把候选预算按相反方向分配给
两个根：

| 预算点 | signal | reverse |
| ---: | ---: | ---: |
| 20 | `[15, 5]` | `[5, 15]` |
| 30 | `[22, 8]` | `[8, 22]` |

该合同在结果出现前固定为同时比较 20 和 30 两点，不选择有利预算点。
实际在线权重为 `[162.32405169996156, 156.2867146609139]`，不是相等权重
或取整相等情形。任何配额、权重、调用向量、winner、来源 SHA 或 journal
链变化都会 fail closed。

这仍不是 P3/C3 科学通过：两根均为已知根，只评价冻结模型代价，没有
真实物理 outcome、独立确认或跨根统计；FLOPs 与能耗仍未知。

## 为什么不再使用 K1/K17 作为等预算结论

历史 K1 六槽在资源预检处停止，没有完整计划；K17 六槽完成了不同大小
的搜索树。K17 中 fixedH1、staticH3、adaptive 的实际预测调用和已分配
请求数不同。K1/K17 使用的总 cap 也不同。因此它们可证明调度路径不同，
但不能证明同一实际预算下的收益。

本轮改用同一预算点内的 `signal/reverse` 配对。它只比较配额方向，不把
K1 的预检拒绝与 K17 的完整搜索混为性能曲线。

## 实现与验证

新增：

- `le-wm-main/experiments/spectral_reachability/p3_effective_equal_budget_20261101_parallel.py`
- `le-wm-main/tests/test_p3_effective_equal_budget_20261101_parallel.py`

入口提供两种模式：

1. `verify`：只读取保存产物，重建 18,898 条 journal hash chain、任务与
   候选银行哈希、每臂配额、逐轴调用、winner 与模型代价。
2. `run`：只允许主机 CUDA，复用冻结模型执行器，写入独占的新结果目录；
   运行结束后必须再次通过同一个 saved-only 重建。

旧 consumer 内部已经验证每个 resource attempt 都有匹配 return，但其
保存摘要省略了 `returned` 字段。新入口只在完整 journal 重建成功后补充
`returned == attempted` 认证；不会从不完整摘要猜测或补零。

历史 host prepare 的来源闭包曾依赖当前 Python 进程的已导入模块集合，
从新 verifier 调用会产生无关的假漂移。新入口改为验证 prepare 保存的
固定来源清单，并独立重建 predecessor、任务哈希、候选银行哈希、模型
路径和模型哈希。

测试：

```text
11 passed in 16.48s
```

覆盖固定反向配额、相同候选总量、相同 attempted/returned 向量，以及对
allocation、weights、attempted、returned、winner、journal chain、来源
SHA、wall 合同和缺失臂的篡改拒绝。科学 gate 在所有路径保持 false。

## 新主机 CUDA 封存

运行目录：

`le-wm-main/results/p3_effective_equal_budget_host_run_20261101_parallel/`

结果：

`le-wm-main/results/p3_effective_equal_budget_host_result_20261101_parallel.json`

关键 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `report.json` | `bc2f01289535f5fe01c1d032349b941354e7b3747a222e6fb871f9a530587232` |
| `prepare.json` | `eaebcd04943442e403749c00daed0395da18183c6ff3e9a77cfbe2e5e00a06d5` |
| `events.jsonl` | `124157cad08b3217e81ef13d2a4ccafb49dbb32bc54a34f7421ae9fabf7862db` |
| 新结果摘要 | `f95c07943dad455c67424d4006dff14538733fa455e513a1b39be0d97b0d68b0` |
| 历史 saved-only 重建摘要 | `de279df2bb5696b48f316b0ba0745f64101319ec9e2af9d523746a4b92cc02a7` |

设备绑定为
`GPU-4e8973ff-0278-53f3-2bae-2615d461e441`，运行前可用显存
15,763 MiB。峰值 allocated/reserved 分别为 141,664,768 /
144,703,488 bytes。模型前后哈希均为
`4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`，
确认未训练、未更新冻结模型。完整 host 报告时长为 26.1651 秒。

## 实际调用与预算曲线

每个表格单元均为该 arm 的实际 attempted 和 returned 数量；两者相等。

| 预算点 | action encoder / predict / predictor / pred proj | Jacobian | VJP / grad / raw backward | SVD | metric |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 各 48 | 2 | 各 384 | 2 | 20 |
| 30 | 各 68 | 2 | 各 384 | 2 | 30 |

`signal` 与 `reverse` 在同一预算点拥有完全相同的上述向量。不同资源轴
不能直接相加解释为 FLOPs。

模型代价越低越好：

| 预算点 | signal 两根 | reverse 两根 | signal - reverse 均值 |
| ---: | --- | --- | ---: |
| 20 | `[0.97235115, 0.22972827]` | `[1.00846794, 0.22972827]` | `-0.01805839` |
| 30 | `[0.87730485, 0.22972827]` | `[1.00846794, 0.22972827]` | `-0.06558155` |

改善只来自根 0；根 1 在两个预算点均无差异。该结果说明信号方向配额可
在已知根上改变搜索结果，不能据此声称跨根稳定收益。

主运行的 instrumented resource span：

| 预算点 | signal span (s) | reverse span (s) | 每臂 resource calls |
| ---: | ---: | ---: | ---: |
| 20 | 3.3759 | 3.6309 | 1,368 |
| 30 | 4.5650 | 4.5106 | 1,458 |

span 只覆盖已计量资源调用，不包含未计量 host 工作，不能替代完整 wall。

另一个 resolved 执行合同保存了每臂约 23--24 秒的进程 wall-time，但其
配额为 `[11,9]/[9,11]` 与 `[16,14]/[14,16]`，不同于主运行。新结果把
它放在 `auxiliary_resolved_wall_contract` 中单独报告，不与主在线 CUDA
代价曲线拼接。

## 科学边界

- `P3_admission=false`
- `C3_admission=false`
- `physical_outcome=false`
- `independent_confirmation=false`
- `known_roots=true`
- `model_cost_only=true`
- `full_total_compute_equal=false`
- `FLOPs=unknown`
- `energy=unknown`

相同记录调用轴不等于相同总计算。当前也没有证明 signal 相对公平
uniform、固定 horizon、CEM/GRASP 或真实闭环策略的优势。

## 下一前置

下一项需要预先封存一个真实闭环 wall-budget 合同：在未参与开发的新根
上，让动态信号调度、反向控制和固定/均匀调度获得相同 wall 上限、反馈
频率、动作时域与物理执行机会；完整保存模型调用、调度耗时、编码、真实
动作、早停与任务 outcome。必须报告所有固定预算点和失败分母，并在结果
前冻结根清单、统计判据与缺失处理。

只有动态调度在该独立物理闭环中取得任务收益，且总资源差异被预算曲线
透明覆盖后，才具备打开 P3/C3 科学 gate 的前置证据。
