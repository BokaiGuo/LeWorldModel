# P0预测样本匹配：已执行的校准核验，不是效能准入

默认P0真实调用审计发现每根D2P/GN预测样本536、CEM256，旧receipt公式混用
样本与调用单位。新计数器已接入；本轮只根据计算账决定CEM每轮67样本：
67×4轮×2预测步=536。D2P提案数显式固定128，其余默认设置、目标、seed
20261040/41保持不变。参数由计算量决定，不由物理收益选择。

## 主机CUDA结果

两根仅校准作业正常退出，7项原完整性门全部通过。每根D2P与GN动作仍完全
相同；冻结参数hash不变，物理执行RGB根身份和合法动作盒通过。

| 根seed | D2P predict调用/样本 | CEM调用/样本 | D2P物理关节终距 | CEM终距 |
| --- | --- | --- | ---: | ---: |
| 20261040 | 122/536 | 8/536 | 0.03695256 | 0.03860314 |
| 20261041 | 122/536 | 8/536 | 0.04466771 | 0.03382845 |

这只是已知两根的描述性诊断，不是独立outcome。两根各有胜负，不继续依照
这些物理值调样本、阻尼、seed或目标。动作AD/反向、编码与运行时间不匹配，
禁止称为相同总算力；预测样本是一个明确但不充分的资源坐标。
随机仍只是一个随机程序、restart-GD并非完整GRASP；强基线缺口不由这次
CEM预算修正自动补齐。原P1与依赖阶段未打开。

## 可复现命令

在`le-wm-main`主机CUDA执行：

```bash
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/lewm-matplotlib .venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --roots 2 --calibration-roots 2 --calibration-only --seed 20261040 \
  --d2p-proposal-samples 128 --cem-samples 67 \
  --output results/reacher_d2p_p0_predict_sample_matched_calibration2_v1_20261026_cuda
```

报告：`le-wm-main/results/reacher_d2p_p0_predict_sample_matched_calibration2_v1_20261026_cuda/report.json`。
计数receipt已核对completed=true、无failed调用、两根均为上述数值。
此旧runner尚未保存完整预执行源码快照，不能把本轮冒充具有全新封存链的
正式效能实验。下一次正式新根比较必须先补足源码/协议绑定与全资源计数。

## 新发现的准入实现缺口

当前`d2p_stage_admission.p1_admission`虽然要求相对“matched CEM”的有利
区间，却仅验证原六项实现门，未独立核对实际资源收据。由此，预算不同但
偶然区间有利的未来报告可能错误开放P1。这是阶段协议代码缺口，不是理论
负结果。本轮已交由独立worker实现严格实际预测样本核验与回归；在完成集成
及验证前不使用这一旧入口开放任何新阶段。

这里样本匹配仍只是必要而不充分条件：即使536=536，也不证明编码、反向、
墙钟、反馈访问和基线忠实性均匹配。不能把补上一个gate宣布为原强基线要求
全部完成，不能追认历史报告。下一轮正式准入必须逐项保留这些限制。
