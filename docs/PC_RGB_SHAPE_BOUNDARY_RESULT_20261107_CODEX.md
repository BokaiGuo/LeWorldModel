# RGB-only 图像形状边界：工程与 GPU 核验

旧 `RGBLeWM` 检查输入字段名和张量有限性，但没有在调用模型前要求 `pixels` 和 `goal` 具有 RGB 图像形状。新增 [RGBImageLeWM](/home/gbk/code/LeWorldModel/le-wm-main/experiments/pc_rgb_image_boundary_20261107_codex.py)，在旧边界上增加图像形状、浮点类型、动作包宽度、batch 和历史时钟检查。它允许显式空动作历史；不接收额外状态字段。

- 定向及既有权限测试 **40 项通过**，包括状态向量冒充图像、单通道、额外状态字段、动作包宽度错误、非有限值和 batch 错误的反例；验证拒绝发生在模型调用之前。
- 在本机 GPU、同一冻结 PushT LeWM 上，对一个已曝光图像和两条已保存动作链分别经旧、新边界评分；两个代价为 **2.221642、1.975874**，新旧的评分、预测表示和输入收据逐位一致。模型前后指纹一致。记录见 [GPU 结果](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_rgb_image_boundary_gpu_parity_20261107_codex.json)。

遵照[协议](PC_RGB_SHAPE_BOUNDARY_PROTOCOL_20261107_CODEX.md)，这是版本化接口核验，保留旧实现以维护历史资产。新边界尚未被所有规划器强制采用，单个上下文的 GPU 一致性也不能认证所有形状或整个闭环。图像形状约束不能证明图像来源合法、内容没有编码特权状态，也不是恶意调用者的安全沙箱；RGB-only 主张仍需要输入数据流、动作历史和真实闭环权限审计。

该修正没有显示新算法收益，不能解释此前实验的成功率，也不改变 PC0–PC5 状态。
