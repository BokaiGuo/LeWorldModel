# 独立对话：强基线对比

推进 GRASP、Hi-LeWM、LeWM、PlanWAM 对比。先核查已有产物避免重复；作者报告值与本地实验分栏，协议不同不得直接比较。

主线程负责 P1 和代码审计，不重复其当前任务。不使用子智能体；仅写独占新文件，保留已有改动和封存结果。神经实验只用主机 CUDA，不降级 CPU；不训练冻结模型、不放宽科学门槛。

工作目录 `/home/gbk/code/LeWorldModel`；先读本目录 README.md。用个人有界goal推进“基线证据表与可执行补实验交付”，不是宣称整套研究goal完成。独占新文件前缀 `chat_baselines_20261027_`，可放在 docs/、le-wm-main/experiments/spectral_reachability/、le-wm-main/tests/、le-wm-main/innerdocs/、le-wm-main/results/；不修改现有共享文件或sealed产物。编辑用apply_patch。

先查现有文档及其明确引用的真实报告：

- `/home/gbk/code/LeWorldModel/docs/D2P_REPORTED_BASELINES_GRASP_HILEWM_20261026.md`
- `/home/gbk/code/LeWorldModel/docs/D2P_REPORTED_BASELINES_LEWM_PLANWAM_20261026.md`
- `/home/gbk/code/LeWorldModel/docs/D2P_REPORTED_BASELINE_NUMBERS_PRIMARY_LOCATORS_20261027.md`
- `/home/gbk/code/LeWorldModel/docs/d2p_baseline_reported_values_20261027_v1.md`
- `/home/gbk/code/LeWorldModel/docs/D2P_BASELINE_FAITHFULNESS_AUDIT_20261026.md`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_moving_baseline_comparison_20261027_v3.md`

交付：四方法原文来源、页/表/版本定位、环境/数据/checkpoint/动作历史/horizon/预算/指标/样本数可比性矩阵；作者值与本地值明确分栏，不在不匹配协议上算提升百分比。已有本地实现要标official/reimplementation/local variant，缺实现写缺失，不造数值。针对可补齐项实现独占adapter/协议/测试并执行安全的独立验证；神经运行先核主机GPU资源，与主线程当前P1产物完全隔离，不重跑既有实验。最终给文件、SHA、实际验证、缺失项和可消费接口，不仅给计划。不得自动提交/推送或删除文件。
