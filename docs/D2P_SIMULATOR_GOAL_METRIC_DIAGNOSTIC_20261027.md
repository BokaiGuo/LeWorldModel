# P1任务度量：从编辑RGB到真实模拟器目标的核验

## 当前结论

真实renderer目标替代干净sprite编辑，仍未修复所测目标距离的物体任务排序。
这缩小了代码/协议归因范围，不证明D2P或非线性任务度量不可能；P1仍未通过。
本实验没有选择新动作、改变threshold、训练模型或开放新outcome根。

## 已执行证据

主机运行产物：
`le-wm-main/results/pusht_simulator_goal_catalog_20261027_cuda_v1/pusht_simulator_goal_catalog_20261027_report.json`。
状态 `diagnostic_completed`，7完整性门全true；4个已知校准根×256个位置，
每根239位置合格、17位置因几何穿透无效。全部1024位置/图像保留；有效956图
在主机CUDA冻结encoder中编码，60次batch调用。参数、buffers、eval、输入与源码
字节均核验；没有predict、反向或训练。

不是RGB拼图：调用官方reset/render，并在生成goal的render当刻捕获实际七维状态、
碰撞穿透与可见几何。`_set_state`会走一步物理，故资格同时检查请求/实际object
xy与模2π角度一致、agent位置/速度一致、无穿透和画布内，容差1e-9用于数值一致性，
不是任务成功阈值。不得用reset返回的已恢复state冒充生成goal时state。

此生成器读取task specification与模拟器几何，仅用于特权诊断。其提供的是额外
目标RGB目录，不是single-RGB部署；object-only等价目录也不等于官方含agent位置
的success合同。全部资格先于读取保存probe/control outcomes确定，不按收益删根/格点。

## 共同ID对照：控制目录覆盖而非直接比较不同最小值

主线程另实现独立离线比较：
`le-wm-main/results/pusht_simulator_vs_edited_goal_common_ids4_20261027/report.json`。
按稳定root与grid ID连接，每根共同229位置，排除旧clean目录额外5个原始目标；
不用数组位置、不按探针成本选子集。四根同ID embedding平均MSE约0.000149–0.000293，
说明渲染路径有数值差异，但不能仅由这个差异推出任务改善。

对物体位姿严格不变的真实5探针，goal-set minMSE最大减最小为：

| 已知校准根 | 干净编辑共同229目标 | 模拟器共同229目标 |
| --- | ---: | ---: |
| 0 | 0.01552259 | 0.01407307 |
| 1 | 0.09976188 | 0.09884879 |
| 3 | 0.01276568 | 0.01232002 |

变化略减，但物体静止时仍有明显agent相关响应；根2的selected探针物体不静止，
不放入该表。完整239目标和共同229目标结果分别保存，不能混用为同一覆盖范围。

每根保留10个原控制分支（含两条不同根的中止），共同ID目标距离与物体pose cost
的描述性Spearman：

| 根 | 编辑共同目录 | 模拟器共同目录 |
| --- | ---: | ---: |
| 0 | −0.71910 | −0.73161 |
| 1 | −0.66908 | −0.66908 |
| 2 | −0.00614 | −0.00614 |
| 3 | −0.35642 | −0.40020 |

这是保存终态的诊断，包含中止、重复pose与不同完成长度，**不是固定时域效能或
独立泛化统计**。未恢复正向任务排序，不能宣称修复成功；也不能将其归因为
尚未验证的理论失败。RGB编辑不是本批现象唯一充分解释。

## 与其他已排除解释的关系

原始20个目标变体也已由独立render hook审计：所有图像hash与历史封存一致，
实际object geometry漂移0/20。因此不能把旧有限Q失败解释为已发现的reset碰歪物体。
另一方面，当前Q线性代数和goal0检查一致，不保证全局actor invariance；把全部
干净目标差分放入线性nuisance span会rank192，原compiler正确拒绝零度量。
以上都是有限样本/接口证据，不能相互替代为普适不可能性定理。

## 复现与验证

运行脚本与完整主机命令见 `docs/PUSHT_SIMULATOR_GOAL_CATALOG_20261027.md`。
所有已有输出独占不可覆盖；神经重跑只准主机CUDA。主线程独立比较可执行：

```bash
.venv/bin/python -m experiments.spectral_reachability.audit_pusht_simulator_vs_edited_goal_catalog_20261027 \
  --edited-report results/pusht_clean_goal_catalog4_v1_20261027_cuda/report.json \
  --simulator-report results/pusht_simulator_goal_catalog_20261027_cuda_v1/pusht_simulator_goal_catalog_20261027_report.json \
  --probes results/pusht_spectral_response_physical4_v1_20261026_cuda/report.json \
  --control results/pusht_task_spectral_bank4_v1_20261026_cuda/report.json \
  --output results/pusht_simulator_vs_edited_goal_common_ids4_20261027
```

cwd为 `le-wm-main`。上面保存目录已存在，复跑须另新目录；该比较只读保存embedding，
没有CPU神经实验。主线程实际运行共同ID比较7项非神经回归通过，官方render目录
10项测试通过（含真实模拟器而无神经），原goal几何审计6项通过。
