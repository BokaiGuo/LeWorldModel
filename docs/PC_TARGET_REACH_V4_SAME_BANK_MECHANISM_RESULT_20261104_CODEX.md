# v4 自身选中链：同库与局部响应事后诊断

按[诊断协议](PC_TARGET_REACH_V4_SAME_BANK_MECHANISM_PROTOCOL_20261104_CODEX.md)，
对 24 个已暴露根使用 **v4 自身最终选中链**，对九个原候选和
selected 前缀分别做同预算后缀优化及共同尾部评价；全部
[29,040 条完整链](../le-wm-main/results/pc_target_reach_v4_same_bank_mechanism_20261104_codex_audit.json)
已用同一精确动力学重算。另对 selected、zero、greedy 和同库
lookahead 做两尺度局部响应及阻挡序列，独立输出并以同代码重建
[4,032 条扰动链](../le-wm-main/results/pc_target_reach_v4_same_bank_mechanism_20261104_codex_local_response_v2_audit.json)。

- 24 根中，21 根 selected 前缀非零；其中 3 根同时满足原距离非改善
  定义（根 4、7、16）。selected 尾部在 24 根均非零。
- selected 前缀在原候选库中完全匹配的有 21 根；按各前缀相同
  120 次评价的有限 CEM，selected 对同库 lookahead 的自行规划
  终距严格更好为 **0/24**，lookahead 严格更好为 21/24。
- 对非即时、共同尾部相对零尾收益、自行规划收益、双尺度响应差
  与阻挡分支稳定性的交集，selected 对 zero 和 greedy 为 **0/24**；
  再加入 lookahead 也为 **0/24**。这是对已暴露面板的事后指标，
  不替换历史事前合同，也不自动解释理论。

共同尾部只覆盖零尾和 selected 尾部，不是完整尾部空间；每个
前缀只有一套有限 CEM 搜索，而 lookahead 从九项搜索中择优，
有选择偏差。响应计算对首五步名义尾部做安全裁剪，最大裁剪幅度
`0.1`；局部响应不能无条件外推到原尾部。部分 selected 前缀
本身属于候选库，也使 lookahead 与 selected 的对比不是独立对手。
同代码重建不等于独立物理或模型复现。

第一次响应诊断在写入时因 NumPy 布尔类型无法 JSON 序列化而
失败，已保留原 manifest 与部分文件；修复为原生 `bool` 后使用
新 v2 路径重跑，相关测试共 5 项通过。该错误属于诊断输出层，
没有证据表明它影响了已封存 v4 规划。当前结果不能宣称 PC2
通过，也不能将 v3 的 1/24 见证嫁接给 v4。
