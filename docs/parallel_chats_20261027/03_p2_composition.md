# 独立对话：P2 程序组合

推进多程序组合实现与验证准备；P1 前置未通过时不得宣称组合机制成立。

主线程负责 P1 和代码审计，不重复其当前任务。不使用子智能体；仅写独占新文件，保留已有改动和封存结果。神经实验只用主机 CUDA，不降级 CPU；不训练冻结模型、不放宽科学门槛。

工作目录 `/home/gbk/code/LeWorldModel`；先读本目录 README.md。个人goal为“可执行组合及有效对照准备”，不是科学机制通过。独占新前缀 `chat_p2_20261027_`，仅创建 docs/ 和 le-wm-main/{experiments/spectral_reachability,tests,innerdocs,results}/ 下该前缀文件；apply_patch编辑，不改主线程及worker现有producer。

先查既有代码/协议，避免另造平行框架：

- `/home/gbk/code/LeWorldModel/docs/D2P_P2_MATCHED_COMPOSITION_20261027.md`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/moving_p2_composition_prospective_queue_20261027_v1.md`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p2_composition_paired_20261027_v2.md`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p2_composition_predict_frontier_cuda_20261027_v2.md`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p2_no_composition_sra_20261027_v2.md`

补齐程序动作见证、边界history/状态/目标一致性、组合搜索与无组合SRA/强采样对照、原始world-model API与batch samples分别计量、失败不退款和完整候选分母。真实实现应有prepare/verify/run/consume及封存、预算、错误保留；不只写研究规划。可执行RAM契约测试和纯saved验证；科学P1/freshness缺失时确认性队列fail closed，可保存工程准备但不得称P2机制成立。与Nash/Bohr当前moving_overlay_capability_probe_*范围不重叠，不接管P1八计划。交付源码/测试/说明/实际验证/SHA与精确待放行条件，不修改阈值、不用outcome选组合规则，不自动删除/推送。
