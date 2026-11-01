# Observed-source native terminal angle risk — pre-execution addendum

本说明新增而不修改冻结 producer、PushT env、已封存 source 或实际证据。
此时没有 observed-source actual 结果，不宣称已经发生误拒绝。

## 可复现判据错误

本地封存 PushT `eval_state` 使用：

```python
pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
delta = abs(goal_state[4] - cur_state[4])
angle_diff = min(delta, 2*np.pi - delta)
terminal = pos_diff < 20 and angle_diff < np.pi/9
```

没有先取 modulo。比如 position distance=0、goal angle=-2、current angle=5：
abs delta=7，native angle_diff≈-0.716814693，native terminal=True；正确 wrapped
角差≈0.716814693 > π/9，故 wrapped terminal=False。这是纯数组可复现错误，非实际
本批误拒绝的证据。`_get_obs` 将 current angle 限制到 [0,2π)，但保存的 goal angle
可为负，因此仅约束 current angle 不能排除此风险。

## 必须保持的安全协议

producer 仍以 native terminal/step terminated 为硬停止；不得改 env、偷偷 wrap goal、
吞掉 terminal、继续 warmup 或补换根。即使可确认 native_false_positive，原固定协议
下该根仍为失败，不能升级成功/准入。truncated 也必须停，不能归为 task success。

现有冻结 producer 报错文本 `native terminal observed source` 或
`public step terminal/truncated; no continuation or replacement` **只说明停止原因，
并不自动证明真正 task terminal**。它没有结构化输出下列 discrepancy 分类；需 MAIN
或后续独立只读审计从原 journal state/goal/flags 计算并单独报告。不要声称现有
producer 已自动区分 wrongterminal。不得为了添加分类修改此次已冻结 source closure。

## 实际拒绝的独立只读分类

对同一次拒绝使用同一 observed state7 和原 goal7。step 路径取
`operation_completed(resource=public_step).receipt.state/goal/terminated/truncated`；
at0 路径取 placement reset receipt、checkpoint state、`placement_request.goal`。
不要把另一时刻的 state 与该 goal 混配，也不要从 rewards/outcomes 选择来源。

```python
position = float(np.linalg.norm(np.asarray(goal)[:4] - np.asarray(state)[:4]))
delta = abs(float(goal[4]) - float(state[4]))
native_angle = min(delta, 2*np.pi-delta)
wrapped_angle = abs(float((float(goal[4])-float(state[4])+np.pi) % (2*np.pi)-np.pi))
native_recomputed = position < 20 and native_angle < np.pi/9
wrapped_terminal = position < 20 and wrapped_angle < np.pi/9
```

必须报告原 terminal flag（at0 是 eval_state 触发停止）、native_recomputed、position、
delta、native_angle、wrapped_angle、wrapped_terminal、truncated，不只一个 label。

| 条件 | 分类及允许结论 |
|---|---|
| native=True, wrapped=False | native_false_positive / wrongterminal；仍按 native-stop 失败，不放行 |
| native=True, wrapped=True | wrapped_criterion_task_terminal；仍只是源构建时终止，不是下游任务收益 |
| native=False, wrapped=True | native_false_negative / discrepancy；单独报告，不称普通 nonterminal |
| native=False, wrapped=False | 两判据均非终止；若 truncated/其他错误，另列具体原因 |
| saved terminated 与 native_recomputed 不同 | flag/state/goal chronology mismatch 或其他 wrapper 原因，unknown，不强行归类 |

特别注意：这里 wrapped task terminal 的 position 距离必须保留 native 的 **前4维**，
阈值也保留20和π/9，只替换角差定义，才能隔离 angle bug。
`source.initial_domain` 的 nonterminal 使用 **block position 两维** 和 wrapped angle，
是另一项独立 source-design 检查，不等同上述 wrapped task-terminal 判据，不能直接
拿其 accepted/nonterminal 反证 native task terminal。

若后续打算改变生成协议，必须另开版本、预先冻结新规则并重新生成，不能追随实际拒绝
结果修改本批协议。此风险与 global freshness unknown/calibration-only、缺少 plateau/
two-stage witness/confirmation 等原准入限制相互独立。
