# 精修链的预测、真实末帧编码与物理结果

按 `PC_PREFIX_NOVEL_ACTUAL_ENCODING_DEV_PROTOCOL_20261105_CODEX.md`，
对已完成同机会精修的 47 对链重新执行完整 25 步，记录真实
末帧 RGB，并用数据集同轨迹目标图、冻结 LeWM encoder/projector
及原生代价的平方误差和计算目标代价。真实图像和物理成功只用于
事后分析。即使提前成功也执行到第 25 步，另单独记录首次成功；
首次成功时间与之前停止于成功的物理重放逐条一致。

47 对中，模型预测代价与真实末帧编码代价的前缀配对排序：
**29 对一致、18 对相反**。仅 6 对恰好一条链在 25 步内曾
物理成功；按模型预测代价选择成功链 5/6，按真实末帧编码
代价也为 5/6。样本小且已有开发选择，不能据此声称整体
评分可靠；预测与编码排序分歧也不能单独定位为动力学错误，
因为仍需核验预测末态与实测末态的 embedding 及目标输入一致性。

实现审计发现最初脚本直接将环境渲染输入张量缩放，而正式 World
先通过 `AddPixelsWrapper` 的 PIL 双线性缩放；新版本已按
原生管线修正，并通过随机非方形图逐像素对照测试。本批环境
输出的 94 张实际末帧上，修正前后编码代价逐项相同；早期输出
仍保留，不再用未经认证的路径作为结果来源。编码模型状态
SHA-256 为 `ce70853a275c1c957a021c65186abcdc690bf9dfa237186da5eecfbccf0a585f`，
前后未变化，但该哈希本身不完成 checkpoint 来源认证。

有效输出：
`le-wm-main/results/pc_prefix_novel_actual_encoding_native_pixels_v2_seed42_20261105_codex.json`；
可用 `python -m experiments.pc_prefix_novel_actual_encoding_analysis_20261105_codex`
复算。三项实验的四个针对性测试通过；不改变 PC gate。
