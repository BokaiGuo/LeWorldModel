# TwoRoom 原生缓存接口：实际反馈回归结果

按照[事前协议](PC_TWOROOM_NATIVE_CACHE_FEEDBACK_REPLAY_PROTOCOL_20261107_CODEX.md)，
在此前事后选择的已曝光 roots 38、39 上，使用 GPU 和 TwoRoom 环境重跑真实
RGB 与模型想象两臂。搜索、随机种子、环境、前缀和旧尾规则与历史诊断一致；
替换的是 latent rollout 路径。新版本调用 LeWM 原生缓存接口，不修改模型
的 `encode` 方法。历史运行器、历史结果均未覆盖。

| root | 真实 RGB | 想象 | 每臂环境步数 | 每臂重规划次数 |
| --- | --- | --- | ---: | ---: |
| 38 | 失败 | 失败 | 25 | 4 |
| 39 | 成功 | 成功 | 22 | 4 |

四条轨迹相对历史记录的派发动作、动作哈希、选择、查询数、模型动作包
步数、终止和截断均一致；所有记录的候选评分和逐步物理距离最大差均为
**0**。评估时数据状态仅用于复位和真实评价，规划评分使用 RGB 编码、
动作与目标编码；规划器和模拟器仍在同一进程，此结果不构成进程级隔离证明。

新增接口及回归比较共 **9 项测试通过**，包含评分与动作篡改、重复 root、
跨轨及非法输入检查。运行依然发出 Gym reset 类型与动作转换警告。程序对
新包装、旧运行器、缓存接口、协议和历史对照分别记录哈希；历史模型及
数据文件指纹也由原运行器保存。`experiment-code` 用于固定输入、复用
原运行器并留存逐步结果，没有将技能模板的“必须正结果”等要求作为研究标准。

这是两个已曝光失败案例的同环境实现回归，不是独立复现、完整成本比较、
行动后果描述增量或新的 RGB-only 成功率证据；PC0–PC5 不改判。

[运行器](../le-wm-main/experiments/pc_tworoom_native_cache_feedback_replay_20261107_codex.py) ·
[原始轨迹和评分](../le-wm-main/results/pc_tworoom_native_cache_feedback_replay_20261107_codex.json) ·
[逐项比较](../le-wm-main/results/pc_tworoom_native_cache_feedback_replay_20261107_codex_audit.json) ·
[测试](../le-wm-main/tests/test_pc_tworoom_native_cache_feedback_replay_20261107_codex.py)。
