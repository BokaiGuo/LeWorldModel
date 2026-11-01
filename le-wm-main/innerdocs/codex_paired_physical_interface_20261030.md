# 同协议物理接口：非派发原型

独占 `codex_paired_physical_interface_20261030.py` 是标准库 RAM 状态机，不 import
旧 runtime、模型或环境，不提供 GPU/物理生产入口；也不声称它已连接 Full D2P 或 SRA。
输入为两臂共同 identity（root/task/goal/history/model/normalizer/reset/controller/
renderer/guard 的 SHA）、相同三帧及两段 raw 历史、相同 horizon 和 packet 长度、
两臂各自的逐轴整数 cap、外部持久化回调。identity 只是调用者提供的摘要：本代码
不解析来源、不证明摘要与源材料对应，也不验证 task 真实合法。

每臂按 `observe → select → begin_step → finish_step` 推进。规划请求只含本臂
RGB 哈希、执行动作历史和共同身份，不含物理 state/reward；选择及每个动作 intent
须先由回调确认内容摘要。一次成功 packet 才更新本臂边界历史。早停、异常、未返回
动作与未到达时间保留在固定两臂×全部时间 roster，缺失 checkpoint 为 null。
state7/reward 只在实际返回时由外部提交，root 的物理 label 不伪造。
模型/API/AD/编码/物理/setup/IO/墙钟按轴保留 attempted 与 completed 下界和 pending；
轴 cap 不等于 FLOPs 或总成本等价。不存在公平、科学或 dispatch 自动放行。

测试持久化回调是 RAM fixture；其摘要确认不证明 fsync、事件闭包或真实性。当前
`finish_step`、resource 事件及停止事件没有实现 durable 写入；没有进程重启恢复、
完整来源封存核验、native guard、真实 encoder、planner 预算 hook 或事件消费器。
step 抛错后保留 pending，但内部工作量未知。无物理状态成功判定，无动作择优或 planner
正确性证明。此原型不能直接接入真实执行，生产前须新版本实现可恢复 durable
journal/写前资源预留、完整输入身份闭包、严格 native 适配、失败时不可重复 dispatch、
完整 saved consumer 与独立审查/授权。Full D2P 的科学前置仍须单独满足。

在 `le-wm-main` 可运行非神经回归：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_paired_physical_interface_20261030.py tests/test_codex_full_d2p_physical_preflight_20261030.py
```

fixtures 仅验证接口和错误路径，不是新神经/物理记录，也不改旧封存。
