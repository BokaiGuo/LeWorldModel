# TwoRoom：带历史程序组合的梯度和后端检查

## 实现边界

上一轮的单 latent 缓存接口有意将初态表示 detach，且只提供一帧；
它适用于此前单帧重规划诊断，**不应直接用于跨前缀反传的联合精修**。
新增[带历史接口](../le-wm-main/experiments/pc_latent_history_rollout_20261107_codex.py)：
同时接收前缀预测表示、产生这些表示的动作、后续动作并保持计算图；
调用原生 LeWM rollout，不改模型权重和方法。调用者仍须保证它们来自
同一 root、按相同动作时钟对齐；接口无法仅凭形状证明这些语义。

## GPU 结果

按[固定协议](PC_TWOROOM_LATENT_HISTORY_GRADIENT_PROTOCOL_20261107_CODEX.md)，
两个已曝光根各 8 条合法原始动作链，经过动作 normalizer，分别测试
前缀 1、2、4 个包及历史窗口 1、3，共 12 个条件：

| 后端 | 满足预定数值容差 | 轨迹最大差 | 目标代价最大差 | 动作梯度最大差 |
| --- | ---: | ---: | ---: | ---: |
| 默认 GPU 后端 | 4/12 | 1.55e-6 | 6.11e-5 | 0.04525 |
| math SDPA、禁用 TF32 | 12/12 | 1.72e-6 | 6.11e-5 | 4.96e-5 |

前者默认三帧窗口 6/6 未达梯度容差；后者使用相同输入和原容差，是
观察失败后的[后端复核](PC_TWOROOM_LATENT_HISTORY_PRECISE_BACKEND_PROTOCOL_20261107_CODEX.md)，
不是独立确认。后端复核同时改变注意力实现与 TF32，不可归因于其中
某一项；现有导数后端作用于进程全局，不能与不同后端设置并发混用。
接口自身没有强制启用该后端，正式用于动作梯度前必须明确管理此约束。

负对照中，默认三帧窗口仅保留最后 latent 会使末端表示最大差达
**0.1551**、动作梯度最大差达 **5.617**；中间表示 detach 会丢失
跨前缀的梯度路径。单帧历史窗口并不等价于默认三帧协议，不能因为
单帧测试通过就将它当作相同的多步程序。

相关接口、单帧缓存及后端恢复共 **16 项测试通过**；模型参数指纹
运行前后相同。原始结果分别为[默认后端](../le-wm-main/results/pc_tworoom_latent_history_gradient_20261107_codex.json)、
[精确后端](../le-wm-main/results/pc_tworoom_latent_history_precise_20261107_codex.json)
及[后端来源记录](../le-wm-main/results/pc_tworoom_latent_history_precise_20261107_codex_manifest.json)。

这解决的是当前 checkpoint 和所测链的程序拼接数值问题，不证明真实
动力学、候选保留、实际链机制或控制收益；也未审计所有历史搜索器是否
误用单帧接口。`code-debugging` 用于定位历史和计算图边界并保留失败结果。
PC0–PC5 不改判。
