# P1 共同 Adam 后缀规划：CUDA 运行时调用核对

对既有 24 条固定 cp5 输入，在主机 RTX 5060 Ti 上重新执行四起点、
每起点 20 步的冻结模型 Adam 规划。新观测器分别拦截模型 `encode`、
`predict`、action encoder 前向入口及张量 `backward`；结果与原先按
代码路径推算的数量一致：

| 计数类别 | 24 条重放观测总数 | 固定代码路径每条 |
| --- | ---: | ---: |
| encode | 24 | 1 |
| predict | 4032 | 168 |
| action encoder | 4032 | 168 |
| backward | 1920 | 80 |

观测器记录的是全程总数，没有单独记录每一行的运行时计数；
每条列是固定代码路径推算，不是逐行 hook 测量。

逐行对照原封存规划结果：24/24 身份、四个候选分数、赢家索引及
赢家动作 SHA 相同；模型参数与 buffer 前后哈希相同。对比使用
原结果 `p1_uphill_multistart_adam_20261030_main_RUN01.json`，其
SHA-256 为 `018dffcb53e5e4dd40e6323fc3665ca53321cb027023c0cc4c1edb5daa06229f`。
观测器的纯替身行为测试 1 项通过，不是神经实验。

- 重放规划文件：`le-wm-main/results/p1_adam_runtime_counts_20261031_plan_RUN01.json`；
  SHA-256 `8c0a2a72422708490378df18edcef333c292b0fbba7fcb86d726e38665ede391`。
- 运行时计数：`le-wm-main/results/p1_adam_runtime_counts_20261031_receipt_RUN01.json`；
  SHA-256 `57119cf99183130e730ac636247acd7f0c7bd3cb217bd8cccabb24b22440c92c`。

这是**另一次重放的入口调用计数**，不是原实验当时的独立 hook
记录，也不是 FLOPs 或端到端资源认证；不含前缀生成、动力学
结构计算、物理交互及额外反馈规划。计数类别不能直接相加，
也不能因相同调用次数推断等耗时或等预算。重放未使用物理结果
选择动作，未改变模型或历史产物；已知根和已有物理结果的证据
边界不变，不打开 P1。
