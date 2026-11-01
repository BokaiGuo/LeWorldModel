# 已保存模型/动作 ABI 与 paired journal 的诊断接线

本独占模块使用 Python 标准库，读取固定 SHA 的 SRA H2 planning report 与 P4 多轮
runtime journal，归一化为 root、原始模型 context 摘要、时间、候选 ID、五步 raw
命令和来源语义；不 import 旧 planner，不执行模型、GPU 或物理。

* SRA 校验 40 个已完成候选、稳定 argmin、一条 H2 witness、输入 context 和完整
  10 步动作一致；输出第一 packet，另保留第二 packet。它是**已保存开环 winner**，
  不表示在线 SRA 已按观察重编译。
* P4 校验实际反馈 selection、10 候选、旧摘要格式、稳定 argmin、原 context/Q 绑定、
  原 prefix 和所选 suffix；还从原 boundary 记录提取实际三帧 hash 和两段 raw
  history，要求与 journal 请求一致。它是**单 packet 后缀校准**，不是 Full D2P。

`bind_saved` 要求 root、step、输入 context、动作长度和指定 slot 一致，先独占写
fsync source-bound receipt，再调用旧 journal 的 `select`；若后者失败，receipt
可能孤立保留，不可当成功选择。`consume_bound` 先运行旧 saved consumer，再要求
每个 select 恰有按顺序提供的一份 receipt，逐一验证摘要、原固定来源、字段及
journal 动作/请求连接。任何没有 receipt 的选择会被拒绝。对 P4 测试中的人工
prefix，基础 journal 可以重建，但整个 bridge consumer 按设计拒绝缺失的 prefix
receipt；这不是一个已完成的真实 P4/Full D2P 执行链。

真实保存数据检查示例：已知 root `23b8de...6050` 的 SRA winner 为
`joint_h2_no_composition_sra_38`，P4 actual step5 为 `suffix_6`；两者**不在同一
决策时刻或同一完整协议**，不能直接比较效果或当成 Full D2P 对照。SRA 的输入
context 已核对，但其 paired journal 初始化帧 hashes 与原 root RGB 的逐帧关系仍须
真实来源桥认证；P4 仅核对现存 journal 中的值。

两个模型结果均为旧记录；独立 consumer 不重做完整原 producer strict、SVD、神经
前向或物理，也不证明 caller 真实执行。上下文的 SHA 和动作绑定是结构检查而非
真实实时编码/模型 attestation。journal 的 `full_d2p` 只是历史通用接口的占位槽，
本桥明确不授予 Full D2P 标签或物理 dispatch 权限。要成为生产接口还需：具体可审
Full D2P 算法、在线 SRA/H2 适配、每臂真实观测/编码与动作归一化、原生 guard、
共同 source/reset 和时钟闭包、真实资源 hook、全选择可消费 receipts、实验授权。
这些缺口未被本模块消除。两方法资源不等，总预算与 FLOPs 未知。

回归命令：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_codex_saved_model_action_bridge_20261030.py \
  tests/test_codex_paired_durable_journal_20261030.py \
  tests/test_codex_paired_physical_interface_20261030.py \
  tests/test_codex_full_d2p_physical_preflight_20261030.py
```

新测试将真实保存模型/动作 ABI 与临时 synthetic journal 混合作为接口夹具；
假的物理返回不是物理证据，也不代表这两种旧轨迹已被同协议配对。
