# P1 cp5 预处理主机 CUDA 对照结果

使用固定的 24 条已知根 cp5 真实 RGB，按
`D2P_P1_CP5_EVAL_TRANSFORM_PROTOCOL_20261031_MAIN.md` 比较 P1
图像路径与当前仓库 `eval.py` 的 `img_transform`。实际设备
RTX 5060 Ti，`cuda:0`；冻结模型前后哈希一致，执行 48 次编码。
24/24 输入 float32 张量逐字节相同，24/24 输出编码逐字节相同，
两项最大绝对差均为 0。输入和编码逐行哈希保存在
`le-wm-main/results/p1_cp5_eval_transform_cuda_20261031_RUN01.json`，
SHA-256 `4552ec81d25af9a20e256e9cdcbb128e2113fd85ddad4bd58a5f34905f9dfc6e`。
比较器的形状、dtype、非有限值及位模式差异测试 2 项通过；
测试只涉及数组，不含 CPU 神经推理。

结论仅是这些 cp5 图像上，当前两条评估预处理及其编码无差异。
两条路径共享图像和框架依赖；本实验没有比较训练时实际源码、
训练数据及其 normalizer，也没有重建目标图像编码或时序历史。
不能据此认定所有代码正确、归因模型幻觉、证明独立泛化或放行 P1。
