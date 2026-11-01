# P3 rank-reserve：预声明工程对照

2026-10-27；在新候选评分前固定。不是旧比例调度的追认修复，不是独立科学复验。
只使用冻结模型信号 w=(1-chi)E；不读取物理 outcome 或候选成本决定资源。

固定 U,D_rank,S_rank,N 顺序；20 blocks，每 block 3 个完整 H2 候选。
U/N 为10/10且入口不接收信号。D 每根先保底5，余10给严格较大 w 的根；
S 反向。精确相等（包括双零）均分；不加温度、epsilon、gain或随机seed。
rank-only 丢失幅度信息，会放大小权重差/噪声，这是明确的方法限制。
本幅度来自既有5..15可行域的两端，不根据候选/真实收益选取。

所有臂使用同root、goal、Q、bank，按各自bank索引前缀消费；不删除zero/GN/D2P锚点。
先交替保底5 blocks，再按slot 0,1轮询至耗尽。稳定(cost,index)选优。
同候选也独立付费；不因结果相同退款。碰撞、失败、同winner均保留，不换根、不重试。
固定主要比较D-U；次要D-S、D-N。模型Q不是物理收益。

完整执行预计每臂120 search predict+8独立replay；共享8 predict、2J、384 VJP、2SVD。
predict API、predictor samples、encoder、AD、SVD、wall分别计，不宣称等FLOPs。
历史编码8/8、anchors288 predict及历史AD另列；缓存重放不冒充新CUDA结果。
新的可执行队列接口只消费已绑定候选，未知score保持missing，不从旧winner补齐。
新神经执行必须主机CUDA、冻结模型、完整计量与双replay；本交付不自动启动神经/物理。

当前已知两根只作工程队列验证，不具有freshness。P1、freshness、P3、compute/wall优势全关闭。
科学阶段缺：认证fresh root/goal有序清单和完整input SHA；通过P1的冻结动力学；
新schema主机CUDA执行器与durable全模块ledger/双replay；独立physical/RGB消费者及预算。
不得将本队列消费者的缓存评分比较解释为这些条件已经满足。
