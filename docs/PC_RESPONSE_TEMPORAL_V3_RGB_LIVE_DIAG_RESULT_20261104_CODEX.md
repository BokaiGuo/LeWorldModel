# v3 固定终点真实 RGB 开发诊断结果

按[事前协议](PC_RESPONSE_TEMPORAL_V3_RGB_LIVE_DIAG_PROTOCOL_20261104_CODEX.md)，
在已暴露 PointMaze 根 0 上，v3 response 和 smooth CEM 两臂各自在
真实 RGB 反馈下执行三次五步。账本审计显示六次均选到其记录中
完整时域评分最低的链；额外审计根据实际 CEM 批次定位边界，三次
response 精修起点都是此前记录的完整链最优值。

response 三轮 token 为 1920、1919、1920，代理分数分别为
0.00241138、0.00223538、0.000260981，未弃权；smooth CEM 三轮
均为 1920，代理分数为 0.00629652、0.00218478、0.00174566。
原生后端重放热身后第 0、5、10、15 步的距离：

- response：1.74559、1.22135、0.54135、0.31790；
- smooth CEM：1.74559、1.59344、1.15181、0.09485。

这些结果在数值上与该根先前 v1 诊断相同，不是独立重复或新根
性能证据。审计没有独立重算模型预测；原生动作重放未逐状态匹配
原 RGB 执行。代理分数不等于物理终距，单根不能证实规划优势、
PC4 或跨模型泛化；训练及 normalizer 来源独立性仍 unknown。
本次真实 RGB 轨迹亦没有同链机制干预。

- [运行入口](../le-wm-main/experiments/pc_response_temporal_v3_rgb_live_20261104_codex.py)
- [账本审计](../le-wm-main/results/pc_response_temporal_v3_rgb_live_diag_20261104_codex_audit.json)，SHA-256 `41eeb22e91aa2b2f4790fa1299dfa9b58aa8f8ac3fc387b2f3dd05171f9122c9`
- [原生动作重放](../le-wm-main/results/pc_response_temporal_v3_rgb_live_diag_20261104_codex_physical.json)，SHA-256 `385359c5eb67ef6154b94b5754a99234083cb4bdc7aef0264e3a3c55003c7a29`
- [精修边界审计](../le-wm-main/results/pc_response_temporal_v3_rgb_live_diag_20261104_codex_refinement_audit.json)，SHA-256 `60db600e917a9548493ad9022ba4eaa8266a02fcb87aa4b570704ef19d604c8c`
