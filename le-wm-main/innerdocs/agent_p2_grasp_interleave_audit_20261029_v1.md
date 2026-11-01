# P2 GRASP source-bound interleave 对照设计

本件只检查四份实际源码 SHA 和两个 `soft` 的 AST 时序，不导入 torch，不执行模型；它不是 CUDA parity runner 或生产认证。先固定原 known root0、相同 task/context/goal、slot 与原初始 `x=0`、线性插值 `v`，使用各自原代码路径，不读取结果决定输入。先封存输入/normalizer/model 与四源码哈希，实际权威必须另行完整绑定，不能把本 source 检查当输入认证。

原 Native.soft 每 packet `step` 后立即计算 dyn/dense；HOST GraspBackend.soft 三 packet 后在 score 账本回调中按相同顺序计算。原 Native.step 和 chart 继承复用；HOST 使用局部 ModelFacade 和额外 capture/同步。数学形式相同不保证 CUDA 逐位相同，也未证实错误；固定对照必须分别调用原 Native.soft 和 HOST GraspBackend.soft，不能把同一实现调用两次当独立路径。先核 model eval、同权重 before/after SHA、torch/CUDA、输入 dtype/shape/stride/device，固定执行顺序原→HOST，必要时记录 allocator/同步及实际调用账。每条实际重做三次 predict、AE/predictor/proj，且各一次 backward；不要把它们混入既有搜索预算或称等 FLOPs。

预固定逐位门：分别保存三 packet 的模块输入输出、原始和标准化动作、context/预测、dyn/dense/total、x/v VJP、图输入梯度及其 dtype/shape/stride；对 f32/f64 原始字节精确比较，含 signed zero；不使用容差。保存逐项差异与首个差异，不因不同而丢弃完整结果或自动修补。逐项捕获本身可能改变交错，因此还需保留不增加捕获同步的原路径和 HOST 路径原生日志对照；有/无捕获是不同条件，各自计费，不能仅凭某一条件无差异证明整个优化等值。两条路径各自从相同独立叶张量启动，清空梯度、无训练和随机扰动；若还对 Adam 做验证，必须固定初始 state 并绑定已封存的 lerp/负零算术，单独计账。

后续实现须同时具备 MAIN 授权、HOST CUDA only、真实 loader/输入 authority、排他写入及 partial failclosed、模型前后冻结和独立 saved consumer；目前这些对照执行接口尚未实现，故不能给实际生产命令。特别是原 hook 和 HOST module capture 的事件/同步不同，需设计不 patch globals 的 capture：不能仅复制公式声称调用实际方法。数值差异先定位首个实际输入/布局/操作，不自动归因为交错；不改科学 gate，不据此宣称等 FLOPs 或公平。

只读检查与 RAM 回归（项目根）：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_audit_20261029_v1
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_grasp_interleave_audit_20261029_v1.py
```
