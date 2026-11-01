# P1/P2 阻尼解 SVD 符号回归检查

针对混合初始化器 RUN01 的单侧 SVD 变号错误，定向检查三种现有实现：

- `pusht_task_spectral_bank.synthesize` 同时翻转左右向量。
- `p1_two_stage_witness.compile_response` 只规范化右向量，但用该右基
  重新计算正规方程右端，不依赖未经翻转的左向量。
- `agent_p2_wallcapped_cuda.ridge_from_svd` 使用另一套完整、未经
  单侧变号的 SVD；对称谱候选使用的右向量来自单独的 SVD。

新增 `tests/test_p1_p2_svd_sign_regression_20261031.py`，用非对角
Jacobian、满秩／秩亏矩阵及有效／无效范数约束，将这些路径的阻尼
解与独立正规方程比较。P1 的部署动作有明确的 float32 舍入及边界
向内投影，比较按该实现精度设置容差；首次用 float64 级容差的检查
因这一已确认的投影行为失败，并非新的 SVD 错误。

本次新增测试、混合候选测试和两个已有相关测试共 26 项通过。
这是纯数组数值验证，不包含神经 CPU 推理，也没有启动新物理实验。
结果仅排除这些函数在所测输入上存在同类符号错误，不证明全部
调用链、保存结果、P1 机制、P2 优势或独立泛化成立。
