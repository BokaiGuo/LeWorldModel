# JEPA 原生 PushT 静态因子诊断：渲染校验停止

本次是对已暴露单根、两臂实际轨迹的后验诊断准备，不是新规划实验。[初始协议](PC_JEPA_NATIVE_STATIC_FACTOR_PROTOCOL_20261104_CODEX.md)使用保存的 float32 目标状态直接重建时，目标 RGB 有 242 / 150528 个通道不一致，[记录](../le-wm-main/results/pc_jepa_native_static_factor_20261104_codex/render.json)显示在模型评分前停止。

为区分目标状态精度与其他因素，[修订协议](PC_JEPA_NATIVE_STATIC_FACTOR_V2_PROTOCOL_20261104_CODEX.md)用原 seed 和固定目标输入重放原生目标 reset。目标 RGB 和评价状态均逐元素匹配原始记录；但将该环境的双精度 agent 和物体位置、角度赋给另一个同配置环境，再经原生渲染，仍有 242 个通道不一致。[第二次失败记录](../le-wm-main/results/pc_jepa_native_static_factor_v2_20261104_codex/render.json)保留了异常，未进行任何 JEPA 编码或因子评分。

后续只读检查发现：目标环境重复渲染与原图一致；两个环境公开的 agent、物体坐标及角度和物体世界顶点一致，公开 shape 类型和颜色顺序也一致。这尚不能定位像素不一致的原因；不能断言它由 float32、绘制顺序或特定 bug 造成。上述临时只读检查不是独立的完整渲染审计。

因此目前没有可解释的 agent／物体因子结果，也没有新的 PC gate 证据。下一步需定位渲染状态或画法差异，形成新的事前诊断协议并重新逐像素验证；不得绕过不一致阈值或从 32 张图中选择可重现的子集。两次失败均未重新规划、改变原轨迹或执行 GPU 模型。
