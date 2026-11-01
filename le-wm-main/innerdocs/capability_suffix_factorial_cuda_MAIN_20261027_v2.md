# 固定24后缀编译 v2：实际 CUDA raw 动作唯一权威

v1 完成312次CUDA调用，但严格消费者发现235次token重建不等，原因是CUDA tanh与NumPy tanh的ULP差异。v1保留、不得放行物理；这是接口代码缺陷，不是科学负结果。

v2不改变24case、上下文、谱库、目标、预算、候选顺序或门槛。每一次forward只计算一次CUDA raw，先保存原float32 tensor及收据，再将同一tensor传入transition。candidate、winner和独立重放均引用该实际buffer。保存312个raw与936模块事件，含AD调用。末端重放必须同时满足prediction与raw逐位相等。

saved-only消费者从实际raw收据重建normalization，必须与真实action_encoder输入逐位相等。NumPy tanh只作4*float32 epsilon的函数实现诊断，绝非执行权威；真实模型raw→token→待执行raw全链不容差。所有24分支仍仅为两个known calibration roots，不能开放P1或宣称独立泛化。
