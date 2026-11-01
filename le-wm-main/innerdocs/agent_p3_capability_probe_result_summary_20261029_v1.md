# Stage C 固定十分支 saved 数值分析接线

本模块只新增 saved 元数据消费者。冻结的 result_protocol 是纯算术核；本模块显式导入该核，
不导入 Pauli 物理 producer、不调用 consume/verify/native/模型、不读取 PNG 或事件 journal。
原 fullstrict 的事件/像素/因果 guard 证据严格继承 MAIN 的完整 receipt，而非再次独立认证物理。
这里只新做 receipt 与完整原始 row、report/completion/binding、原始 U 和历史 witness 的接线及数学复算。
冻结协议内 arithmetic_only 的 authentication_performed=false 保持原样，不能把它改签成物理认证。

## 固定 authority 与 ABI

MAIN 已提供（worker 不运行这些真实文件的分析）：

|输入|SHA256|
|---|---|
|Stage C prepare|8fdecdc9fe5fcb1f00b03c30e9971566da03c2a523ccc652d86065c9b442e95f|
|Stage C batch|b70022a70a32c0c5f743684276e42f9040e228ae8296bb6daeae5fb70ce28c1b|
|Stage C full receipt|4e7d35ec52dba109c9e2c181ad46d754f44bc0c0ef41a362c30a68776c66f3b1|
|Stage C source|f070b6afe4fe0ebba66e05d366f929467602b98805980f86f27e2344445a8344|
|Stage B signal summary|88587a2d9f610df775c321eda6497bf3602c6ccd993b8779d7f36c6ad4744a4d|

Pauli 当前 `.consume` 有 prepared/prepare_sha256、jobs[10]、source_authorities、完整 inherited_stageB_resource_receipt，
没有 batch_sha256。因此 CLI 强制单独提交 batch 文件及实际 SHA。batch 含每 job report/completion SHA；
逐 job metadata 的完整 row 必须与 receipt.row 相等。source SHA 必须同时存在于 prepare.bindings。
当前工具无直接 agent 消息接口；本 ABI 通过 Pauli 已落盘 v2 source/innerdoc 对齐，并向 MAIN 明示。
不猜测 future 字段或虚构收到直接消息。

原 Stage B 四项文件、五份实际导出 U、完整 actual-cp10 witness 均按 prepare.authority 的路径与全文件 SHA 读取或散列。
stageB 源/权重/encoder/PNG 不递归加载；不重复其 fullstrict。原 signal summary 与物理 receipt 的资源向量完整相等，
不只比较 predict，也不把 unknown FLOPs 置零。物理全轴 attempted/completed 从已认证 row.operations 描述性重计。
transport_failed 分支 counts/terminal/cp15 保持 null，即使 partial declarations 提供观测计数也不补零。

## 固定数学与分母

固定 root0 C/Z/P、root1 Z/P，逐 endpoint U/Z2 顺序十分支；五个 G、八个 C-control、四个 logical C。
三个原 C no_witness 全留。已知 immediate-cost dual false 不被新的 delayed outcome 覆盖。
G = 原 reward 距离 D15(Z2) − D15(U)，正值偏 U；各固定轴分开同向相减，不作跨单位总分。
独立 stdlib math.fsum 的 native7/XY/objectXY/velocity/raw angle、historical native-angle bug 与 wrapped diagnostic
来自事前冻结算术核。原 native reward/label 与 fsum discrepancy 并列，不以复算公式覆盖 native_success。
native_success 不等于小 native7 距离/大 reward：速度、角度及位置保持分轴。
合法 early stop 的真实 terminal 另列；无 cp15 不外推、不补15、不替代同时间结果。
工程失败可保留已返回 cp15，但不纳入 eligible G；transport publication 失败明确非完整认证。
不作因果中介、独立根、P1/P3/C3、释放准入或 equalFLOPs 结论，所有科学 flags false。

raw15 = 原 causal raw10 + 实际 Uraw5 或正 float32 零 Z2。warm10、完整 source task、goal、endpoint_identity、
prefix cp0/5/10 state/full snapshot/PNG file/pixel SHA、原 public returned actionbits 与所有 response label/reward 均接线。
不 clip、再归一化或重选。physical selection_digest 使用 default-spaces JSON；Stage B witness digest 使用 compact JSON；
own 输出 compact JSON+newline。普通对象的两 codec 不同及 −0/+0 位差有回归。

## 读写和验证界限

每个 JSON/输出 256MiB 上限；所有 CLI 路径必须 canonical absolute、无 symlink。
完整 input SHA 和 source/tests/doc/math-kernel 三件套在 build 前后复核。
exclusive-create、fsync 文件与目录，不覆盖既有产物；失败保留已写文件、不冒签成功。
verify 对整个 summary 重新 build 并逐字段比较；这是本分析的 saved rebuild，非上游 fullstrict。
本模块无生产 run 入口、无神经 CPU/GPU、无 physics；不承诺统一 CPU wall/disk floor，输出失败只属分析工程失败。
临时全十槽 fixture 有明确 fixture_only，CLI 不接受 fixture 绕过真实 pins。

## MAIN 命令

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行，以下只由 MAIN 实际运行：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_capability_probe_result_summary_20261029_v1 build \
 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_prepare_MAIN01.json \
 --prepared-sha 8fdecdc9fe5fcb1f00b03c30e9971566da03c2a523ccc652d86065c9b442e95f \
 --batch /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_MAIN01/batch.json \
 --batch-sha b70022a70a32c0c5f743684276e42f9040e228ae8296bb6daeae5fb70ce28c1b \
 --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_physical_20261029_v2_consume_MAIN01.json \
 --receipt-sha 4e7d35ec52dba109c9e2c181ad46d754f44bc0c0ef41a362c30a68776c66f3b1 \
 --capability /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_signal_summary_20261029_v1_MAIN01.json \
 --capability-sha 88587a2d9f610df775c321eda6497bf3602c6ccd993b8779d7f36c6ad4744a4d \
 --physical-source /home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_p3_same_time_capability_probe_physical_20261029_v2.py \
 --physical-source-sha f070b6afe4fe0ebba66e05d366f929467602b98805980f86f27e2344445a8344 \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_probe_result_summary_20261029_v1_MAIN01.json
```

build 输出自身实际 SHA，随后：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_capability_probe_result_summary_20261029_v1 verify \
 --summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_probe_result_summary_20261029_v1_MAIN01.json \
 --summary-sha <MAIN_build_实际SHA>
.venv/bin/python -m pytest -q tests/test_agent_p3_capability_probe_result_summary_20261029_v1.py tests/test_agent_p3_capability_probe_result_protocol_20261029_v1.py
```

## RAM 回归记录与剩余项

本轮联合回归 71 passed / 1.86s（新 summary 35、原冻结 math protocol 36）。包含临时盘完整十槽
build/write/verify、原整 row 与 source 变更、重封外层 SHA 后的 counts/guard/action/snapshot/goal/response 拒绝、
工程失败/transport unknown、合法提前停止、warmup 尚未附原 goal 时使用实际 terminal goal，以及成功标签与速度距离分离。
干净子进程确认本模块不导入 torch/numpy/PIL/stable_worldmodel。

中间一次 fixture 调整产生 1 failed / 70 passed：把成功样例位置置零后，距离恰为256，但测试仍断言 >256。
已将合成成功位置固定为1（仍满足原成功几何条件），速度256，恢复 >256；未改源数学、阈值或真实结果。
此错误只是 RAM 测例断言，不是任何实际 native 运行失败。

实际 Stage C 文件只检查了字段 ABI 与大小，未运行实际 build 或上游 fullstrict。
剩余工作由 MAIN 完整审读、独立回归并实际 build/verify；本轮没有科学分析结果可报告。
合成 fixture 绝非 Stage C 科学证据。三件套交付冻结后不再改写，避免 sourceclosure 漂移。
