# TwoRoom：独立 RGB 规划进程烟雾结果

## 实现

新增 [spawn 规划进程](/home/gbk/code/LeWorldModel/le-wm-main/experiments/pc_rgb_spawn_planner_20261107_codex.py)。评估环境、物理状态与成功判定留在父进程；规划进程接收白名单形式的当前 RGB、目标 RGB、合法未执行旧尾、剩余时域、随机种子和策略。动作 normalizer 的均值、尺度作为初始化参数；规划进程独立加载冻结 TwoRoom LeWM。未知字段以及 `state`、`proprio`、`contact` 和未来成功字段会被拒绝。

此实现隔离了本实验的对象和消息数据流，**不是操作系统安全沙箱**：规划进程仍具有当前用户文件系统权限，恶意或被修改的可信代码不受阻止。RGB 字节本身的来源语义也不能仅靠形状验证。不得将此结果描述为对所有历史运行器的权限认证。

## GPU 与真实环境检查

按[协议](/home/gbk/code/LeWorldModel/docs/PC_TWOROOM_RGB_SPAWN_BOUNDARY_PROTOCOL_20261107_CODEX.md)，使用相同的已曝光 TwoRoom 任务、`lookahead` 搜索、候选种子和 5 步反馈，对照旧的同进程闭环记录：

| 根 | 执行步数 | 重规划次数 | 查询次数 | 真实成功 | 对照 |
| --- | ---: | ---: | ---: | --- | --- |
| 10 | 15 | 3 | 48 | 是 | 所有规划动作逐值一致，物理轨迹和停止状态匹配 |
| 11 | 18 | 4 | 64 | 是 | 所有规划动作逐值一致，物理轨迹和停止状态匹配 |

21 次模型评分调用均保留输入收据；工作进程关闭时模型指纹与加载时一致。模型评分按预定 0.002 绝对容差核对，逐步距离按 1e-4 容差核对。进程启动约 5.20 秒；两任务分别约 0.289、0.118 秒请求往返时间之和，包括本次规划和 IPC，但不构成重复测量或系统成本优势。

边界单测 **2 passed**，检查额外特权字段、图像格式、时钟和非法旧尾拒绝。实际 GPU 烟雾使用相同环境和既有轨迹，不是独立物理重现；它没有比较新方法和强基线，也不能证明 PC4 真实闭环收益或更改任何 gate。

[运行器](/home/gbk/code/LeWorldModel/le-wm-main/experiments/pc_tworoom_rgb_spawn_feedback_20261107_codex.py) · [逐步记录](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_rgb_spawn_feedback_20261107_codex.json)。
