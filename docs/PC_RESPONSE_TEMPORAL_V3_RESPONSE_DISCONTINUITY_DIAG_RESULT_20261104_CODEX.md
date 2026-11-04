# 事后响应尺度与阻挡分支诊断

按[诊断协议](PC_RESPONSE_TEMPORAL_V3_RESPONSE_DISCONTINUITY_DIAG_PROTOCOL_20261104_CODEX.md)，
重新执行六个非即时根的中心及正负扰动，保存逐步阻挡序列，并核对
所有差分矩阵与此前归因产物一致。

根 0 的 selected 前缀、选中尾部附近：尺度 0.01 时 10 对正负扰动
中无阻挡序列分支差异；尺度 0.05 和 0.1 时各有 2 对差异。
对应 zero 前缀三个尺度均无此类分支差异。因此，根 0 在较大尺度
观察到的响应差异确实伴随 selected 侧的离散阻挡分支变化；不能将
该大尺度中心差分无条件解释为单一光滑局部导数。

根 6 的 selected 和 zero 在三个尺度上均无正负扰动阻挡序列差异，
其选中尾部名义下的响应差异则在 0.01–0.1 范围内相近。这是对
事前零名义失败的归因线索，不能追认机制 gate 通过；不论结果如何，
原事前交集仍为 0/24。更不能从精确 Wall 后端推断 RGB 物理机制。

[完整逐步阻挡轨迹](../le-wm-main/results/pc_response_temporal_v3_prospective_mechanism_20261104_codex_response_discontinuity.json)
SHA-256 `a07bf6b20af9d5aa2ed04b4114f7bc556b12e11687e9de4fec3f9d12c847c1a6`。
