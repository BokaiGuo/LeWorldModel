# 时间相关响应反馈：真实 RGB 开发诊断结果

按预先协议，在已暴露 PointMaze 根 0 上，v1 response 与 smooth CEM
分别使用自身真实 RGB 反馈完成三次固定终点决策。每次实际执行五步，
总计 15 步；这是单根接线诊断，不是独立确认或充分调优的基线比较。

| 臂 | 三次预测 token | 三次选中模型代理分数 |
| --- | --- | --- |
| response | 1920、1919、1920 | 0.00241138、0.00223538、0.000260981 |
| smooth CEM | 1920、1920、1920 | 0.00629652、0.00218478、0.00174566 |

账本审计记录六次选中链均为各自记录的完整时域已评分候选的最低分；
response 均未弃权。审计核对记录，不独立重新计算模型预测，也只绑定
列出的依赖。原生 MuJoCo 2.1.2.14 同场景动作重放，热身后第
0、5、10、15 步的目标距离分别为：

- response：1.74559、1.22135、0.54135、0.31790；
- smooth CEM：1.74559、1.59344、1.15181、0.09485。

重放尚未与原 RGB 执行逐物理状态核对；模型代理分数不等同物理距离。
两臂只有一个暴露根，不能推断优势或通过 PC3、PC4。该轨迹与 E0
机制见证不是同一链，不能拼接；模型训练和 normalizer 来源独立性
仍为 unknown。

## 后续发现的 v1 缺陷

在 10/5 步响应生成中，部分有限差分探针本身是已评分完整链，但 v1
未确保将其纳入最终 incumbent。受控反例中探针分数为 -1 而 v1 返回
0；因此 v1 的一般性全局最优保留声明不成立。上述六次 RGB 决策及
既有 E0 的 288 次决策账本审计均未发现该缺陷在所记录决策中实例化；
不修改既有结果，也不能由此排除其他输入上的缺陷。

独立 v2 版本在最终选择时跟踪已评分完整链，并预留真实计费的预算
重验零前缀弃权。其联合精修仍由 v1 内部 incumbent 初始化，不保证
围绕最佳探针精修；v2 不能直接继承 v1 的性能或机制结论。

证据：

- [事前协议](PC_RESPONSE_TEMPORAL_RGB_LIVE_DIAG_PROTOCOL_20261104_CODEX.md)
- [账本审计](../le-wm-main/results/pc_response_temporal_rgb_live_diag_20261104_codex_audit.json)，SHA-256 `06eb4a7d7845a3f36d4b2e3905d6d82d2b6e2909f531d80e3cf2f68454a4c134`
- [原生动作重放](../le-wm-main/results/pc_response_temporal_rgb_live_diag_20261104_codex_physical.json)，SHA-256 `b466a6bf4b5f62843e44eddcb9231768537a9fbcac9c05ee99c1c8752e568185`
- [v2 实现](../le-wm-main/experiments/spectral_reachability/pc_response_temporal_feedback_v2.py)，SHA-256 `8dc31afac9d936b8b223131726db38e75381a90379df983587beac0269b491bb`
- [反例与修复测试](../le-wm-main/tests/test_pc_response_temporal_feedback_v2.py)
