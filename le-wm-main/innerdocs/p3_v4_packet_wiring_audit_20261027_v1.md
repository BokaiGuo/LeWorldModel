# P3 v4 packet/context/goal wiring audit

独立 offline code/identity audit，仅新增同名 module、tests、doc、results。
旧 planner、physical、seals、main01 原始产物只读；不加载模型、不创建环境、不重新 encode/predict/step。
与 Meitner 分工：不复算 Q 成本、pose 差值、SVD、residual、allocation、argmin 或计算预算。

## 固定范围及证据强度

覆盖四个既有 root、uniform/dynamic 全八臂和所有保存 candidate context。
检查完整 time-major XY raw packet、float32 PUBLIC_ACTION 标准化后末两包历史、H2 保留的最老 latent、
同 horizon 零动作 nominal 和候选终点 exact identity、whole/segmented/winner context identity。
从原 metric preflight 的 cached inputs 追踪 drop-first 历史、goal_vectors[0] 和原 Q，
绑定 intervention variant0 的 goal vector/RGB receipt、原 RGB references 以及 paired physical reset identity。
不调用原 exporter（避免它重复 SVD/成本/argmin）；只核对保存的 winner 确实属于完整候选集合及 export 转录。

物理侧核对完整八臂顺序、每条 attempt/step_returned/observation、raw command 原序、guard root/观察时点、
初始状态、末次 observed state、末次 PNG、encoding receipt 的 endpoint/goal/Q/selection 绑定。
所有 PNG 文件 bytes SHA 在 prepare 清点并在 audit 前后重验；不将文件 SHA 说成重新做了 RGB stack 或 latent 编码。
sealed source 当前 bytes 和 archive/manifest 一并核验；这不证明历史运行确实执行了这些 bytes。
模型未改变只采信已有 runtime receipt，不重新遍历 checkpoint/tensors。
AST/source-fragment witnesses 是可定位源码证据，不是语义形式化证明。

## 时间及坐标合同

三包零 warmup 各执行五次 primitive env.step。causal frames 是 F0/F5/F10，
root-anchored shift 后为 F5/F10/F15。删除最早 outgoing token 后，两包 between history 分别对应 6–10、11–15 步。
H2 新动作相对 decision 为 1–10 步，相对 warmup 起点为 16–25 步；不是再插入 bootstrap 或按最大搜索 H3 执行。
main01 全八臂赢家实际 H2。reset 内部 integrator 推进不计入此 primitive env.step 时间轴；root fingerprint 命名的是
原始 root/goal，而非 warmup 后 decision state，二者不可直接等同。

live env 是相对 PD target：每次 primitive step 先取 current_agent + 100*u 为固定目标，随后 PD 子步推进。
不是纯 velocity，也没有证据可将 PUBLIC_ACTION 当前自洽性当作 checkpoint 训练 normalizer 合同。
模型 float32 与 physical float64 command 的量化差逐臂记录；不从其非零推断物理损害原因。

## 明确保留的边界

warmup 的 15 次 primitive step 不在 selected-path journal 内；只能以原 whole-stack RGB hash、
raw history、normalized history、末端 state receipts 和封存源码确认边界合同，不能独立恢复完整中间轨迹。
本 audit 不重新恢复 goal render geometry/camera，不声称排除所有 camera/训练语义差异。
Q 原本去除 goal-agent nuisance，不等于官方 success 或 object-pose 代价；该已知 objective 范围不是本次新发现的 causal bug。
Q/pose 分歧本身不证明接线错误；若所有 identity check 通过，报告应为“未建立可复现因果代码错误”，
不可写成代码全排除、理论 negative、物理收益或 P1/P3 admission。两者保持 false。

## 执行（只读原产物）

在 le-wm-main 根目录：

```bash
.venv/bin/python -m pytest -q tests/test_p3_v4_packet_wiring_audit_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p3_v4_packet_wiring_audit_20261027_v1 --prepare --output results/p3_v4_packet_wiring_audit_20261027_v1_readonly01
sha256sum results/p3_v4_packet_wiring_audit_20261027_v1_readonly01/prepare.json
.venv/bin/python -m experiments.spectral_reachability.p3_v4_packet_wiring_audit_20261027_v1 --audit --output results/p3_v4_packet_wiring_audit_20261027_v1_readonly01 --prepare-sha256 REPLACE_WITH_RECEIPT_SHA
```

运行时使用完整 receipt SHA。
prepare exclusive 新目录，提前固定 input/source/test/doc hashes；audit 完成后再验全部输入并独占写报告。
非神经 fixtures 只测试合同与篡改拒绝，不是 efficacy 或 simulator 复现。
