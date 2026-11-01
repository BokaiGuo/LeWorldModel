# P1 回溯动作的 CUDA 重放结果

准备清单 `le-wm-main/results/p1_backtrack_cuda_replay_20261030_prepare_MAIN01.json` SHA256 `f245d30ae0b28591a9c7308817951b2a38f7c89b81f24d6f6262434f6044310e`；主机 CUDA 报告 `le-wm-main/results/p1_backtrack_cuda_replay_20261030_MAIN01.json` SHA256 `83ea957ecb38025e6b5489a0bf843b112906b0954b8b6c62238c10461b31cfa4`。16/16 行均报告两路径逐位一致、第一包和历史与原候选匹配、终点与保存结果匹配；实际 96 predictor / 96 action encoder，模型和所列源码前后哈希未变。

独立保存消费器重建来源并核 16 行身份、分段状态／动作历史和与原中间及终点的比较，结果 `all_exact=true`，未重算模型。**连续路径的完整数组没有保存，仅保存了运行时相等布尔值**，因此消费器不能从保存数组独立重算 whole/segmented 相等；所记录 token 本身也未独立重做 GPU normalization。源码清单覆盖列出的项目文件并继承原绑定，不是整个进程和所有底层库的闭包证明。完整终点证据、GPU token 和环境部署尺度仍需后续物理接线审计；本次不改变 P1 或物理 release。
