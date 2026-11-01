# P3 physical commit 已有 authority 接线 v1

仅新增 source/tests/本文。saved-only bridge；不导入 physical/encoder/cp5 producer，不调用 fullstrict，不启动 NN/physics。MAIN 运行生成新输出。只读现有文件的 ABI，RAM fixture 不是实际实验证据。

## 实际接通的字段

|真实来源|source-bound 投影 / 明确派生|
|---|---|
|H3 physical report `/rows/0`、`/rows/2`|固定 endpoints 两根，原 root/task/off/goal/goal_pose；不按收益选根，不冒称已经执行 fixed/adaptive|
|`/rows/{i}/prefix51/frames`、`checkpoints` 中唯一 step5|51 帧顺序、off/on/off、pre/post 同状态、cp5 state/snapshot/PNG/pixel 精确相等；只重新解码实际历史所需三个 PNG，其余完整覆盖继承显式 MAIN fullstrict|
|`/actual_operation_receipts/{j}/raw_bits`|逐 public step 小端 float32 解码 warm10 和 prefix5；prefix 与原 sealed commands bit 相等；不误把 incoming raw10 当后续 full10|
|warm5/warm10/plan5 的 off receipt|绝对时间 5/10/15、control -5/0/5；路径为 report.parent + 单一合法相对 PNG 文件名，时间和路径属派生，不能冒充原 JSON 字段|
|H3 encoding report rows0/8 + consume context chains0/2|实际 cp5 原 gzip NPY float32[192]、array/file SHA、strict actual192、重编码 anchor 相等；原 raw10=warm 尾5+prefix5，按原 mean/std float32 一次归一化生成两 token，模型/task 精确连接|
|Nash actual-cp5 `.tasks` 实际输出|两 task/provenance，原 task/prefix/raw/context/goal/Q/normalizer/root 连接；原 shared-panel prepare/report/receipt 外部 SHA、源码 SHA、逐八个 b5 request digest/observation/task binding/audit 再连接|

physical 原完整 strict 的 rows/counts、外部 report SHA、source bindings 必须一致；另独立核对所有已完成 operation receipt 的连续 index 和 resource census。`saved_PNG_verified` 必须为整数计数，不把 bool true 当完整认证。history 原 prepare 必须绑定同一 physical report/strict 和冻结 encoder 源。

所有指针保留真实 schema。pure projection 与 derived time/path/raw bytes 分栏；不创建假旧 report/receipt，不 monkeypatch。physical/helper/encoder/cp5 源均有固定 SHA，cp5 还要求 MAIN 显式传入与冻结 SHA 一致的源码描述符，不接受任意字符串声明为 authority。

## 必须仍然 BLOCKED

`typed_contract_partial.authorities.planner=null`，不复用 H3 report 当 R planner。四槽（两 known 根×未来 fixed/adaptive）全保留；`ready/physical_release/execution_authorized/P1/P3/C3/freshness=false`，不签 strict receipt。

缺项逐项输出：实际 R report+fullstrict、release/model IO/resource ledger；新 env 的 causal cp5 pre-step6 guard；typed 源投影尚需显式支持派生 time/path/warmraw 及 rebased task identity。最后一项不是数据损坏：原 physical task_id 是 base task，Nash actual-cp5 task_id 是合法重绑定，不能伪装相等。当前桥只交付可核验 source-bound 部分，并保留 frozen typed manifest BLOCKED。

未给 history 时明确缺 encoder report/fullstrict；未给 Nash actual 输出时第一二帧的 encoder 来源仍 pending。全部给齐时 `all_three_frames_actual_verified=true` 仅表示**继承原完整 strict/原 task export 的 saved 实际历史并与 Nash 输出及 cp5 独立 NPY 接合**，不是本桥新编码三帧，也不是新 renderer 因果或新 restore 证据。原 task exports 的细粒度检验继承冻结 Nash builder，不在此重跑 producer/fullstrict。

这里的 actual-cp5 task 只接 physical restore 历史。**不得偷换 R group 预注册 root context**；两 known 根不是 fresh/独立根。输出不授予执行权限或科学准入。

## MAIN CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；所有输入参数显式文件+SHA。以下完整接入现有 authority，输出新文件且禁止覆盖：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_physical_commit_authority_bridge_20261029_v1 \
  --physical-report results/agent_p3_h3_witness_physical_20261029_v1_MAIN01/report.json \
  --physical-report-sha c8fa06be8eaaf6f7515e6e9550c900c71e04f2141d745699b33d1585eb21a2ce \
  --physical-receipt results/agent_p3_h3_witness_physical_20261029_v1_consume_MAIN01.json \
  --physical-receipt-sha 48a1a5ecadf7043e3162ca104d8f43b0d59baf88655a076f675c9f96d697b9f7 \
  --history-report results/agent_p3_h3_witness_encoding_20261029_v1_cuda_MAIN01/report.json \
  --history-report-sha c3195ea6ba35898bfb0eaef8f9dd69e7e2f4e01834c7a2a43c6c73302352c539 \
  --history-receipt results/agent_p3_h3_witness_encoding_20261029_v1_consume_MAIN01.json \
  --history-receipt-sha 671e48b0e953e6adeb8e07381ecaa965572afa8801d29d961c344eb82a3e505e \
  --cp5-tasks results/agent_p3_actual_cp5_task_builder_20261029_v1_prepare_MAIN01.json \
  --cp5-tasks-sha ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7 \
  --cp5-source experiments/spectral_reachability/agent_p3_actual_cp5_task_builder_20261029_v1.py \
  --cp5-source-sha 703f5f552c1911159b2144f72711c542ca48c408ed4e306b1fc17b019df889f1 \
  --output results/agent_p3_physical_commit_authority_bridge_20261029_v1_MAIN01.json
```

不提供后四对 optional history/cp5 参数也可只接 physical，缺项明确列出；每组必须成对。输出上限16MiB、exclusive write/fsync，不修改输入。没有 run/神经/物理入口。R 尚无 actual authority，不猜未来路径/SHA。

## 验证与冻结

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_physical_commit_authority_bridge_20261029_v1.py tests/test_agent_p3_physical_commit_preconditions_20261029_v1.py
```

自身 RAM 测试第一轮30 passed/2.45s；补 actual-cp5 全 join 与十类 mutation 后41 passed/4.79s。最终与冻结 preconditions 联合82 passed/7.18s。覆盖 schema/status/hash/源绑定、bool/int PNG census、prefix chronology/renderer/cp5 signed-zero/raw/box、history NPY/tokens/model/context、Nash 上游 witness 连接、CLI new-only、无 torch/pymunk/producer import。未执行真实 build/生产/fullstrict；MAIN 待实际生成，BLOCKED 为预期状态，不是模型负结果。

源码快照（本文不自含自身 hash）：source `fd136726fc67eac13e91b3b21514eb8a2049d398b2e86fee0dadfdca588b6647`；tests `d444145176ec0987131df8778cdec3be10700577d40d54b246faf2acc30a64d6`。旧 preconditions 三件套不改。
