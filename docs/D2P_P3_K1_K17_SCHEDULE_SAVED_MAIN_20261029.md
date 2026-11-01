# P3 K1/K17 调度与预算保存重建

对已有主机 CUDA 的固定 12 槽保存产物进行调度和 27 轴资源汇总，继承原完整消费 receipt；汇总自身再次重建验证通过，SHA-256 `93a3b3ea7e1e2d0c8db8206ad42ae82fffc910400fce62a743b1735a56036a12`。文件为 `le-wm-main/results/agent_p3_c3_budget_schedule_saved_20261029_v1_MAIN01.json`。关联汇总和预算推导各 8 项独立测试通过；不重复加载模型或物理执行。

K1 六槽均因队列限额无完整计划。K17 六槽均有完整模型计划：两根的 fixedH1 每槽 36 个已分配请求、114 次预测，staticH3 每槽 36 个、198 次；adaptive 两根分别 68/52 个、446/278 次预测。第一根 adaptive 有四次 H3 reconfiguration composition 队列，第二根没有；固定策略没有这一分支。由此能确认该批内调度路径确有变化，但策略实际计算量不同，不能因共享 cap 就声称等实际预算或调度收益。

事后探索的资源上界推导表明，原 K17 向量可覆盖原有最大 17 父节点树及其验证预留的资源轴；这是有条件的算术上界，不保证真实硬件、存储、有效 witness 或最终物理成功。重复同一已知根的 K17 不是独立实验；实际物理 root/goal/history 与动作映射仍需认证。P3/C3 科学门不变。

主线程复核 K17 saved physical handoff：12 项测试通过，实际六槽分析及重新构建 verify 通过，结果 `le-wm-main/results/agent_p3_c3_k17_physical_handoff_20261029_v1_MAIN01.json` SHA `3b2caf555f390426234b34a96d8bd84670933da5364eba4a1f22df9a424a6df9`。它将六个模型计划与旧物理 root/goal 记录连接，且原 raw10 一次 float32 归一化与两段保存历史 token 逐位相同。这只是保存来源匹配，不等于新环境的 live reset/history/RGB 守卫或实际动作映射通过；提交时域、资源和早停仍需事先固定，物理未授权。
# 追加：K17 paid replay 与 live guard 的来源桥接

主线程复跑 bridge、live guard 和 handoff 联合回归 52 项通过；对实际六槽执行保存分析和重建验证，分析 SHA `91831cd0392f575fddb99e5a73e3a13da2ed44e89fa43d2c2c483f4fc52884e9`。该分析从 whole/segmented paid replay 的原 native IO 核对 step0 token 和实际 action-encoder 输入，保持六槽身份；继承上游已封存 strict receipt，并非本次重新运行神经模型或物理。

这只闭合已保存 IO 到未来 guard 的接缝。实际 live capture 的来源与计量、六槽物理预算以及持久 writer/consumer 仍缺失；分析明确 `execution_authorized=false`，不能据此宣称 P3 科学通过、等实际预算或真实控制收益。

后续新增独立 128 MiB 有界存储与 native capture 接缝，主线程复跑三组相关测试共 14 项通过。固定类别配额总和约 109.38 MB，低于 128 MiB；但不保证任意真实 native 快照可装入单类配额，也不保证 I/O 故障时失败记录一定落盘。当前存储检查只核 journal 和 PNG 哈希，不是六槽来源、动作、早停或真实物理 consumer；禁止将它视为生产派发授权。

六槽调度核心、meter、episode、store 和来源 guard 已完成一轮集成 fixture；主线程独立复跑 59 项通过。工程失败后未执行槽仍保持 `not_started`，不会补齐为物理结果。目前尚无真实来源封存的 prepare、完整终态发布及独立 saved consumer，也未运行物理；该核心不得用于生产派发。

后续代码审计发现 `material()` 对六槽分别调用来源重建，而每次重建均包含全部六槽。现改为每次材料准备重建一次新鲜快照，再对六槽逐一执行身份与来源核验；公开的单槽 authority 接口仍自行重建，不接收调用方提供的快照。独立回归 19 项通过，包含一次构建计数、根身份不匹配及缺失或重复槽拒绝。尚未完成真实来源全量耗时测量或生产消费，此项是工程效率修复，不是 P3 科学结果。

新的六槽持久 lifecycle 与 saved consumer 可在 fixture 中保留启动失败、槽内失败和未开始槽；主线程与 native outer 联合回归 10 项通过。该组件尚未认证真实回调来源、动作、PNG、预算或物理早停，发生存储失败亦不能保证失败日志落盘；不能用于宣称真实物理或科学通过。实际来源和 native adapter 集成仍在推进。

