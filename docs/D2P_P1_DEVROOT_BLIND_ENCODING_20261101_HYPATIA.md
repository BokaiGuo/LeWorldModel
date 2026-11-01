# 已用开发根 blind payload 与 CUDA 编码入口

两个固定 slot 的实际 replay0 RGB 0/5/10、10 个 raw float32 warmup 动作及
construction 第20步真实目标 RGB 已分别写入仅含三个白名单数组的 NPZ。
目标不是 reset goal；NPZ 不含 state、seed、witness、未来轨迹或来源路径。
原 moving encoder 的 `decode_blind` 和 `load_blind` 已对两份真实文件通过；
这证明格式及输入契合，不是正式 source approval。

Evaluator 侧来源与 SHA 位于
`le-wm-main/results/p1_devroot_blind_payload_20261101_hypatia_LOCAL01/receipt.json`，
其 SHA256 为 `ae62273327df4b8190672f0c5f53f42e1d80a2ae1bf7b32bdc239cf210df5586`。
其中绑定 construction/replay 图像、原 source 审计和 builder；producer 在每次
public step 后核对 `latest_action` 与 dispatch 的 float32 bits。NPZ 本身不携带
该 evaluator 侧来源。source audit 验证了两个已用开发 slot 的物理重放和实际帧；
未导出的 solver cache 不在证明范围。

CPU 测试：在 `le-wm-main` 下运行

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q -p no:cacheprovider \
  tests/test_p1_devroot_blind_payload_20261101_hypatia.py \
  tests/test_p1_devroot_blind_encode_cuda_20261101_hypatia.py
```

本次结果 5 passed；测试文件 SHA256 分别为
`c735dcd44d9ac6a775c551d4d28e62cad10a48d9ba586df9ffa17b31748db723` 和
`c3e44c59ea979b8906d461bc3884a8912c7202ae06ade8871244e8c16a407f97`。
测试不执行神经模型或证明 CUDA 运行成功。

独占开发 encoder 使用原 checkpoint、原图像变换、frozen eval、float32 native
projected CLS192、MATH SDP、禁 TF32；按两个固定 slot 的 0/5/10/goal 顺序
最多各调用 encoder/projector 8 次，禁止预测器、动作编码、训练、自动重试、
planner 和 task export。raw 动作在 packet 中保持原始值；后续若使用动作，
只使用原 mean/std 归一化一次，不重复除 action scale。
prepare 中嵌入的原 `moving.contract()` 的 `source_admission` 字段描述原 encoder
的 MAIN manifest 条件，**不是**对此开发输入具有 MAIN 批准的声明；本入口既不
创建也不使用该 manifest。实际准入状态以顶层 `purpose`、`planner_authorized`
和本文的开发用途限制为准。

CPU prepare SHA256：
`7b29da516ea2cbae3f788c80bbf028232fd2cb74c2a6d03ebdcf2bbd86964b94`。
已完成 CPU `verify`，checkpoint 文件和 normalizer 与既有合同的固定 SHA 一致，
未加载模型。以下命令须在拥有可用 CUDA 的主机、对应工作目录执行；它只产生
dev latent 和报告，不生成 MAIN manifest 或 planner task：

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B -m \
  experiments.spectral_reachability.p1_devroot_blind_encode_cuda_20261101_hypatia run \
  --prepare-sha256 7b29da516ea2cbae3f788c80bbf028232fd2cb74c2a6d03ebdcf2bbd86964b94
```

输出目录固定且独占；失败后保留 partial，不自动重试、覆盖或补槽。实际 CUDA
调用、模型 tensor hash、latents 与报告以主机运行收据为准，见下方运行记录。
MAIN source/encoder/planner 准入没有被改写或替代。

## 独立确认缺口

现有 evidence 能追溯探索 seed/request、source state、target-off 重放、终态
witness 和开发 payload；它同时证明这些是已使用、已观察的开发根。历史碰撞
检查只覆盖可见记录；A-H 实际调用、失败重启及其他缺失来源的完整排除链尚未
证明。checkpoint/normalizer 的训练数据来源及排除范围也未证明。
参见 `D2P_P1_SOURCE_CANDIDATE_DISCOVERY_20261101_HYPATIA.md` 和
`D2P_P1_EXPLICIT_COLLISION_CHECK_20261101_HYPATIA.md`；这是有范围的既有
搜索结论，不是对所有可能资料不存在的证明。

后续独立确认需要另有来源授权、固定分母和方案的候选，完整可审计的开发历史
排除及训练来源证据，满足原有预算、基线、统计和 MAIN 准入，再实际执行。
在此之前 freshness、训练排除和 P1 均不能认证；禁止将这两个 dev slot 换目标后
重标为新根。

## 主机 CUDA 实际运行

上述固定入口已在项目 GPU 协作锁下执行，报告状态为
`dev_encoding_complete`：两个 slot 各四张图像均完成，encoder 和 projector
各 8 次尝试及返回，模型 tensor hash 前后一致，设备为 RTX 5060 Ti。
实际报告位于
`le-wm-main/results/p1_devroot_blind_encode_cuda_20261101_hypatia_RUN01/report.json`。
这只是开发输入编码；`planner_authorized=false`、`task_export=false`、
freshness/训练排除 unknown 和 P1 不准入均保持不变。
