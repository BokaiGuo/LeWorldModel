# D2P 在线真实反馈协调层

`experiments/spectral_reachability/d2p_feedback_controller.py` 是面向后续
通用 compiler 的 CPU/Python 协调接口。它不导入模拟器、模型或 CUDA，也不是
固定 root runner，因而可以由不同 compiler 提供 `plan` 回调。

## 接口

```python
result = run_feedback(
    initial_context,
    plan=plan,
    execute=execute,
    observe=observe,
    packet_size=10,
    max_replans=3,
    event_sink=journal.append,
)
```

`plan(context) -> Plan(actions, prediction)` 必须返回至少一个 packet 长度的
完整二维有限数值动作数组。`prediction` 只进入 journal，永远不进入下一轮
context。`plan` 收到的是 RGB/历史等 caller 提供的 context；协调器拒绝明显的
`reward`、`state`、`*_state` 和 `*_reward` 字段，但不能检查闭包、编码后的数组
或 callback 内部变量。因此 callback 的数据来源隔离仍是调用方的信任边界。

`execute(packet) -> Execution(actual_actions, rgb, terminated, truncated, error)`
接收一个 packet，返回实际执行的动作和 `uint8[H,W,3]` RGB。实际动作可以不同于
计划动作；它是唯一传给 `observe` 的动作。完整 packet 是实际动作数恰好等于
`packet_size` 且返回元数据有效、没有 `error`。只有完整 packet 才调用
`observe(old_context, actual_actions, rgb)`，其返回值才替换 context。预测终态、
计划动作和 partial/error 回执均不能替换真实反馈。

## 停止、失败与计数

初始 planning 后，每个完整 packet 最多触发一次 replan；`max_replans=N` 表示
初始计划加最多 N 次反馈重编译。native `terminated`/`truncated` 会停止后续
planning。partial packet 必须带停止标志；动作已报告的数量仍保留。callback
抛异常、格式错误或 execute 未返回可验证动作时，结果为 `failed`，且
`executed_action_count_fully_known` 为 `false`。不重试、不补值。

结果包含完整 `events` journal：每次 `plan_return` 保存完整计划与 prediction，
并记录 packet、实际动作、RGB、observe 前后 context 及错误。`event_sink` 收到
事件副本；sink 失败会 fail-closed。`counts` 只表示协调器可观察的 callback
attempt/return、完整 packet 和已报告动作数，不是完整模型/物理预算认证。
结果中的 `P1_admission` 固定为 `false`。

当前交付只包含 CPU 假后端测试；未运行物理、模拟器或 GPU/CUDA。主线程接入时，
应由 compiler 将真实 RGB/历史构造成 context，将真实执行回执映射成 `Execution`，
并把 `events` 持久化；不得用预测终态或整体 reward 代替真实反馈。
