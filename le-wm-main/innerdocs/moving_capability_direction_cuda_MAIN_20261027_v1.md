# 固定能力方向预测：MAIN独占实现

预声明见外层docs/D2P_FIXED_CAPABILITY_DIRECTION_PROTOCOL_MAIN_20261027.md。
固定2 known roots、8锚点、40全部探针。新40真实endpoint编码及80 teacher/open H1预测，实际模块hooks分别限40/40/80/80/80，不训练、不做AD或物理，无CPU神经fallback。历史28/16视觉编码成本另列，不能称零成本。

prepare先严格消费原RGB与编译器结果，核真实40物理分支与journal、全部PNG与来源；teacher三帧为warm5/warm10/真实prefix5，open末帧来自对应arm保存的prefix预测，between-state history严格相同。输出PNG/pixels/preprocess/257hidden/CLS/projector/full预测以及完整module输入输出。全部预测先封存；本runner不执行任何评分或新动作选择。

verify为外部prepare SHA＋其全部输入文件hash，不调用神经/物理。run仅MAIN显式授权，900s含验证/加载/写入，64MiB输出/4MiB最终保留/200MiB磁盘floor。新路径独占，不重试覆盖；失败仍保留固定40行与partial报告。错误不能记为理论负结果。

CLI（填实际prepare SHA，不预填未来hash）：

```
.venv/bin/python -B -m experiments.spectral_reachability.moving_capability_direction_cuda_MAIN_20261027_v1 prepare --output results/moving_capability_direction_cuda_MAIN_20261027_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.moving_capability_direction_cuda_MAIN_20261027_v1 verify --prepared PREPARE --prepare-sha SHA
.venv/bin/python -B -m experiments.spectral_reachability.moving_capability_direction_cuda_MAIN_20261027_v1 run --prepared PREPARE --prepare-sha SHA --output NEW_OUTPUT --authorization MAIN_FIXED40_CAPABILITY_DIRECTION_CUDA_ONLY
```

RAM测试只验证接口，不证明CUDA已运行。最终报告及模块账需另由MAIN saved consumer检查全部桥接后才评分；没有consumer验证前不得称方向预测可靠或P1成立。已知根不是fresh确认，原整体goal未完成。
