# P2 已保存失败槽的后验域外删失与同时间配对

本次合同固定在查看原失败之后、生成新分类和完整重读之前；不是原运行的预注册。原报告仍是 `engineering_failed`，原 strict 仍拒绝，新的 `domain_censored` 只是有明确条件的后验探索视图。

| base | 返回步 | agent x | 返回动作 float32 little-endian hex | 原 off 帧 | agent 颜色像素 |
|---|---:|---:|---|---|---:|
| 0 | 29 | 512.069093363224 | `eeef173eb61e4c3e` | `b1_plan_29_0.png` | 2 |
| 1 | 21 | 512.9542836307869 | `a0f3cc3edf490e3d` | `b1_plan_21_0.png` | 9 |

对应原 guard 的拒绝条件为颜色像素为零或 native state 前四维之一不在闭区间 [0,512]；两帧均有非零颜色像素，且返回和帧的 state/snapshot 相同，因此已保存证据足以确定越界条件触发。两步 requested、actual、计划动作位串一致，原始 PNG 字节和像素哈希一致；返回的 terminated/truncated 为 false，之后无公共 step。没有补步，失败帧不被当作已验收 checkpoint 或可用反馈。

新独立入口在隔离的函数命名空间重用原保存 consumer：仅在已证实的失败帧允许该帧存在于 journal 而尚未进入聚合 trajectory，并仅对该帧保留原 guard 拒绝而继续核验 PNG；其余完整 journal、步骤、标签、图片、来源、资源、cp15 守卫等检查不放宽。它不是原 strict 的通过，也不是重新执行物理或神经模型。四目录均完成此适配保存复核；两个完整目录仍运行原 consumer。

同时间物块指标差定义为固定候选 1 减 selected；缺失为 null：

| base | cp20 位置 / 角度 rad | cp25 位置 / 角度 | cp30 位置 / 角度 |
|---|---|---|---|
| 0 | 0 / 0 | 0 / 0 | null |
| 1 | +22.869357 / +0.165693 | null | null |
| 2 | +25.808427 / +0.245472 | +71.294906 / +0.227288 | +93.170671 / -0.099181 |
| 3 | +25.808427 / +0.245472 | +71.294906 / +0.227288 | +93.170671 / -0.099181 |

base2/3 是同一个已知根和相同反馈上下文，重复值不是独立证据。各已完成的六个槽终点原生成功均为 false；两删失槽终点成功为 null，不把上一个成功标志或返回时状态冒充 cp30。总实际公共 step 310/310，physics step 3132/3132；神经和物理新增调用均为零。模型候选的共享生成费用不是独立等总预算，两已知根、后验合同及非随机删失均阻止机制或 P2 通过结论。

保存复核 `results/codex_p2_candidate_domain_censor_20261030_v1_ANALYSIS01.json` SHA256 `99e6da71ab93adeb8604f6b5cfdcce8b78bfabdccb70ba8ed57b50b57ec1f605`；同时间结果 `results/codex_p2_candidate_censored_pairs_20261030_v1_ANALYSIS01.json` SHA256 `cf81a9244f98d59aabe459bc41913f625a213a3d5dd1dd7da39650a508fd9b90`。两组测试分别 7 passed、2 passed，前者包含原错、动作、状态、终止、图像和后续步篡改拒绝。完整复核文件可用 `--verify-sha` 原样重新构建。
