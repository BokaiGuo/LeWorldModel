# 六槽独立计量接缝（部分实现）

本文件实现两个独立计量 phase：controller 保留旧每模块 144 上限；replay 使用新 meter 每模块4上限并复用原真实 native hook 入口，预先发票、失败不退款。replay 使用独立三条 dispatch（whole H2，segment 1+1）以及独立 saved cursor；saved verifier 逐条验证原 module input/output、嵌套、packet、全部16 module 返回和三条 edge，无原地改144阈值或 monkeypatch。

此文件并不是完整六槽 CUDA runner：尚无模型加载、六 target 持久 barrier、动态源闭包、完整六槽生产 writer/consumer、actual raw 文件重载、分数后置 barrier、磁盘/显存/wall 证明和 completion 绑定。不得在生产使用本局部核心或把 fixture callbacks 称实际神经执行。P1/scientific gate 不变。
