# E1 六槽共同控制：saved-only prepare

新独占 source/tests/本文；只准备，不提供 CUDA/物理 run。所有六槽 root0/1 × arbitrary、已有 learned-FIRST、compiled 均须在任何新模型加载/dispatch 前封存 target；共同采用原 P2 工程 mean/std 和原 raw→token 一次合同，**不是历史 Hi scaler、官方 Hi 复现或科学门通过**。已存 FIRST 是继承输入，不伪称新生成或免费；proposal 算法与费用不同，不称等 FLOPs。

## 实际来源校验与限制

准备读取并逐文件 SHA pin 旧 controller prepare/report/完整持久 receipt、Hi prepare/report/完整持久 receipt、fixed known2 parity/pairing receipt、原 controller/normalizer source，以及两个原 task gzip、blind NPZ。每个 Hi/off 的实际 FIRST 和 low z_subgoal NPY 逐文件 SHA/长度、NPY header、`<f4 [1,192]`、无尾字节、有限值、原 tensor 数据 SHA、**真实字节相等**与 receipt 的 target/low SHA 均校验。两 root 的 blind/task ID 和对应正常持久旧 arbitrary/compiled target/见证严格连接。metadata receipt 按已保存字节和指定字段检查；本模块不重做上游 fullstrict，原加载字节及原 Hi scaler 不由本模块认证。先完整验证后将六 target 原 `<f4` 字节存入准备文件并合并算 seal SHA；生产 runtime 尚未实现或授权。

task 的实际 mean/std 和 raw10 在每根分别核验，沿用 P2 原 controller 的 3×24 H2/固定每根种子、原 Q、同一个 controller/上下文及无 witness 特供 warmstart。FIRST 不赋物理时间标签；compiled witness 是模型动作见证而非原生到达。历史 parity 只涵盖已实测 known2 输入，不能推成所有模型输入全域等价。未来相同模型构造、input/task/normalizer 实际加载字节和目标交付还需 MAIN 新 runtime 核验。

## 新的事前资源范围与未完成证明

* 新六槽 controller：每槽 72 H2 candidates，144 predict API/samples、144 action encoder、144 pred_proj；总各 864。controller 配方不将 old winner cost 冒充额外 replay。
* 另外要求每槽先将 literal raw10 持久重载、一次 normalize 后，从同完整原 history 付费执行 whole H2（2 packets）和 sequential 1+1（2 packets）：每槽 4、六槽各 24 predict API/samples、AE、pred_proj；controller+replay 各轴上限 888。全部 replay 输出封存后才做 replay score。whole/seg 保存 bitwise、maxabs、ULP；不以非 bitexact 宣称 parity，不能容差升级。这里是拟实现的具体接口及预算，不是既有 job 支出或已实现 kernel。
* 若以后按旧物理 adapter 结构完整六分支、无早停/额外操作，逻辑上限为 warm10+control10 每槽，共 public 120、physics `6*4+10*120=1224`、explicit PNG `6*22=132`，constructor/close 6、reset 12、set_state 24、goal_render 12、goal_bind 6。要实际使用必须对新源码/原生 hooks、所有早停及异常做新 census；不得将此估算作为新 runner 已认证额度。
* 新 NN encoder/AD 在上述 controller/replay 计划中为 0；所选 FIRST、arbitrary、compiled、原 root encoding 已付账分别继承，Hi 四槽 receipt 的整体开销不任意除二。proposal、资源轴/FLOPs 不按总算力相等报告。
* **存储输出 bytes、磁盘 floor、显存、wall、以及实际 native meter 的新上限未证明，明确为 null；不是零或沿用旧四槽 64MiB 的证明。** 生产 source 尚缺六槽 array/journal 原精度复制数与完整 PNG 最坏界、实际 runtime 与失败 durable intent/return。未来执行须在每次动作/模型调用前收费、失败不退款、无重试/删槽/补齐，并先获得 MAIN 对完整 census 和 caps 的批准。不得用本 prepare 开生产。

## MAIN 命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_six_slot_common_prepare_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_six_slot_common_prepare_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e1_six_slot_common_prepare_20261029_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_e1_six_slot_common_prepare_20261029_v1 verify --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e1_six_slot_common_prepare_20261029_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_SHA256
```

prepare 排他写入，verify 重建完整封存对象/全部固定输入；临时盘测试不是实际生产或新研究结果。P1/scientific admission 恒 false。MAIN 需另审新 runner、物理和完整资源证明，再分别授权执行。
