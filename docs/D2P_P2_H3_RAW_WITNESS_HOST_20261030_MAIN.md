# P2 H3 原始动作见证：新主机 CUDA 运行

针对旧 H3 planner 只保存 logits 和归一化上下文、未保存实际
transition 输入原始动作的问题，新增独立入口
`le-wm-main/experiments/spectral_reachability/codex_p2_h3_raw_witness_20261030_main.py`。
它保留旧实验源码和结果，在每次候选 rollout 的每个 transition
前记录候选序号、包序号、实际 CUDA float32 动作字节及哈希。
新合同将该入口及原始动作记录模块纳入源码绑定。消费者先执行
原 journal 验收，再检查每个候选的三包、哈希、动作域和记录紧邻
packet dispatch；winner 通过完整候选见证关联。Jacobian 探针未
被误当成候选动作；保存见证不代表物理执行已完成。

五项测试通过，覆盖成功配对、哈希、候选序号、dispatch 次序和
缺包拒绝。它们是模拟事件测试，不是模型或物理实验。

新固定已知根运行于主机 CUDA 完成 4 根 × 2 臂，报告状态
`exploratory_complete`，模型参数和 buffer 前后哈希一致；按报告
哈希重读 journal 验收 8/8 个候选 winner。准备文件 SHA-256
`15120a07716763f18deb2c8caa2b58d4fe088096ead39211c7488ff9af1c0638`，
报告 SHA-256
`05271deca38756fef3af141e9ecc3e1f3dc4d665dd56b44090ee3235e4ece581`。
输出目录
`le-wm-main/results/codex_p2_h3_raw_witness_20261030_main_RUN01`。

与先前 H3 规划报告逐臂比较，8/8 获胜 logits 和模型 cost 数值
完全相同；本次记录的 8/8 winner 原始动作字节与此前用于物理执行的
CUDA rechart 字节完全一致。这是**新运行与此前重构动作**的同位
证据：它支持该次重构与当前实际 planner 输入一致，但不证明旧
运行中未保存的原始位，也不替代本次新规划到物理的正式绑定或
独立物理重放。不能据此消除所有模型、动作、运行环境来源问题。

新增记录和同步的开销纳入该新运行的实际墙钟，但该已知根探索
仍未认证全流程等资源、独立泛化、P2 科学机制或强基线优势。
P1/P2 admission 均保持 false。

从 `le-wm-main` 验收：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.codex_p2_h3_raw_witness_20261030_main audit \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p2_h3_raw_witness_20261030_main_RUN01 \
  --sha 05271deca38756fef3af141e9ecc3e1f3dc4d665dd56b44090ee3235e4ece581
```
