# LeWM RGB checkpoint 绑定：真实模型工程核验

对现有 TwoRoom 冻结 checkpoint，加载原兼容适配器和动作 normalizer；以已曝光开发输入第一个根、目标图以及 25 步合法零动作，在 GPU 上经旧 RGB 边界和新增 checkpoint-bound 边界各评分一次。两臂严格使用相同张量，要求代价及预测 latent 一致（绝对容差 `1e-5`）；检查原 state fingerprint 与既有开发结果绑定。严格边界在评分前后将完整运行中模型 state 与磁盘 checkpoint 经相同 legacy key 映射后逐张量精确比较，并校验依赖哈希和冻结状态。记录两次边界调用及指纹。

这是加载和执行接口 smoke test，不是任务成功实验。原 checkpoint 从官方来源下载是用户说明，未在本检查独立认证；该检查不认证构造器、预处理、兼容映射行为与官方实现等价，也不覆盖任何其他 checkpoint 或历史运行。两次评分均额外执行并计数。
