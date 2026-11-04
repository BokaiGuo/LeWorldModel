# 统一入口精确动力学链与既有机制诊断的绑定

通过[绑定程序](../le-wm-main/experiments/pc_unified_exact_existing_mechanism_binding_20261104_codex.py)
核对来源文件哈希、任务、实际完整选中链及各对照前缀：统一入口精确
Wall 根 0 的首轮链，与既有多方向精确开发计划及其事后机制诊断中的
选中链完全相同。该绑定没有重新运行规划或干预，不改变历史停止规则。

这一链的五步前缀恰好等于同库 greedy：起始距离 1.00499，前缀后
距离 0.71882，未满足原非即时条件；选中与 greedy 的后继响应差为 0。
选中与 zero 的后继响应差为 1.11553，但不能据此声称相对 greedy 的
重配置机制。选中尾部作为共同后缀时，两者终距均为 0.50672、相对各自
零后缀的变化均为 -0.21210；两者的共同优化器重规划终距也相同。
各控制另有独立等反馈重规划记录；所有差异均见绑定产物。

该结果说明当前统一入口的这个精确开发实例没有提供非即时机制见证；
不代表所有根或其他方法均失败。原机制协议在首轮规划之后制定、
任务已暴露，所选共同尾部源于原规划，历史随机桥接和 lookahead
也不是事前独立对照；所以不能提升为 PC2 确认。该绑定也不将精确
状态任务与真实 RGB 轨迹拼成同一条物理链。

[绑定数据](../le-wm-main/results/pc_unified_exact_existing_mechanism_binding_20261104_codex.json)
SHA-256 `d81882b37eb653a0ddc6db107577eff5865c40e8e4b1a6421ff71a5f707ee013`。
既有机制诊断覆盖的 24 个开发根和其限制见
[原诊断](PC_MULTIDIRECTION_SUCCESSOR_SAME_CHAIN_DIAG_RESULT_20261103_CODEX.md)
及[等反馈诊断](PC_MULTIDIRECTION_EQUAL_FEEDBACK_MECHANISM_DIAG_RESULT_20261103_CODEX.md)。
