# E1 同目标信息干预：接线验收

主线程审读双臂 backend 接线、来源计量与说明，联合回归 16 项通过。实际调用来源 join 对两条既有根重读 task、编译产物和既有完整 receipt：根 0 的 target SHA256 为 `e0b9a70dff3b37349cbe49f479424ddab17b055db4604897e82e018441801109`，根 1 为 `373c830c2d72df5472e9bbdf4cc516298e695aa278263bf0efe16fbd2923621c`。来源认证继承旧 receipt，没有在此重新执行旧 full strict，也不能说明见证已在物理中可达。

新双臂设计在同一目标和原 72 候选下，对照额外零动作候选，干预额外动作见证前缀候选；每槽计 144 原候选、2 额外 packet、4 replay，完整四槽各模块轴 600。代码中 target/terminal 评分与原控制器相同；非神经 fixture 验证成对 bank、失败计费及 saved 重建。这是候选信息干预，不是原三方法对照、等 FLOPs 或等信息比较。

当前尚无完整四槽 production CLI、全局封存 barrier、实际 CUDA IO 和完整 saved consumer，也没有新物理或 RGB。双臂工具返回的科学准入为 false；真实根身份、模型参数和完整来源需要生产入口重新绑定。不得把已有 fixture 或已见根提升为前瞻新根证据。
