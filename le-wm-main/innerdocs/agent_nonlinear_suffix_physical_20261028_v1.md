# Nonlinear suffix：四槽独立 native raw10 adapter

## 范围与真实 ABI

本实现仅负责 model-only 优化后的执行对照，不预言 P1 通过。固定 known root0/off、root0/on、root1/off、root1/on；不读物理 outcome 选计划、根或库。每槽全部新环境执行原 literal float32 raw10，无 whole-row 继承或 prefix 拼接。native 成功/截断即停止，不补步、不换动作；失败仍保留四分母，缺失写 null/显式 missing。

上游是 `agent_p1_nonlinear_suffix_library_20261028_v2` 的真实 prepare/report/consume，不是 terminal selector。MAIN 通过 CLI 给真实 producer source、prepare、report、receipt SHA；不绑定尚不存在的未来 SHA。prepare/verify/consume 均实际调用 `suffix.consume(output, report_sha)`，其 verify→build→原 allmode source.consume/verify 重建完整原来源，再逐值比较 MAIN 持久 receipt。complete/status/all168 sealed/order/four finals/3376 predict+1344 scalar reverse 任一不全均拒绝；不能将 partial 或 physical_release=false 重贴为已有物理证书。上游 physical_release=false 表示未做物理；本 adapter 的 MAIN 原生执行权限独立，不生成科学准入。旧 suffix v1 在 candidate0 前因8个内置 module 源码绑定缺失而 failclosed，是工程失败，不是模型负结果；本版仅导入真实 v2，拒绝 v1 prepare/report/consume schema，不跳过完整 build。

四个新 execution_plan 与真实 report.finals/consume.plans 精确对应。核对 stable terminalQ 胜者、iterate、完整 commands/float32 bit SHA、预测 context、原 prefix5；从封存原 allmode prepare 重算 task_from，对齐原 task_id/task SHA/blind SHA、root context、goal192、identity Q、原 normalizer/model。native goal7/root/goal_pose/条件 goalRGB 由原 source_material 的完整来源验证绑定，绝不将 latent goal 当 native goal。report/receipt/artifacts/prepare/模型与 source/tests/doc 全部进入哈希闭包，并于执行后复核。

## 低层、实际计量与边界

复用冻结 terminal adapter 的 Meter 和 mixed.episode/native_hooks/原 recipe placement、warmup、goal attach；本版本有独立 prepare/report/consume/authority/executor/strict validator，不调用 terminal.execute/consume，不伪造其 schema 或 receipt，不修改导入模块 globals。constructor 返回后才安装 native hooks；通过原 constructor_contract 绑定依赖源码与 inert-constructor 检查，不声称 constructor 前已装 hooks。constructor 失败保留 unknown_constructor_work。

每臂两 reset、warmup10、一次真实 native7 goal bind，执行 sealed raw10。物理控制固定 with_target=False；所有实际 warmup0..10 和 plan0..10 各采 off/on/off triplet，condition 决定本臂观察历史和 goal 图。goal_pose 不改。完整无停步四槽预算：constructor/close 各4、reset8、set_state16、goal_setter12、goal_render8、goal_bind4、public_step80（warm40+plan40）、native physics_step816、explicit_render264/PNG264。attempt/return/failure 和嵌套真实调用均记账；native stop 的实际成本较低，绝不伪造等量计算。上游原 allmode 计算与新增 3376/1344 成本分栏保存，不称 equalFLOPs。

独占 output 上限32MiB（末尾预留4MiB）、wall900s、磁盘 floor200MiB；prepare/receipt 最大64MiB，producer report 最大其封存 final_reserve64MiB。计量遵循冻结低层 scope，完整物理范围仅在 inert constructor 和实际返回证据满足时声明。真实 native success 使用 joint agent+object XY 与 native angle；native7 reward 包含速度，wrapped angle 仅诊断。

strict consume 重建 producer/source、检查完整四分母及动作 dispatch、native7/root/goal/renderer、state/snapshot/RGB 轨迹和 step0/5/10、所有 PNG 文件和像素 SHA、off/on/off 同物理状态、guard、原 warmup 证据、native stop、不明 event 拒绝、journalchain/嵌套 attempt-return、全资源 census、文件 census、执行 binding、前后 source seal。原 source 没有的中间 on warmup 证据保留 unknown，不伪造配对。消费自身新增 physics/neural=0，不表示已有执行从未花费物理资源。

## MAIN 精确命令（仓库 inner 根目录）

以下变量必须由 MAIN 用已冻结且完整的真实产物赋值，不能拿示例值或未来 SHA 替代：SUFFIX_OUTPUT、SUFFIX_REPORT_SHA、SUFFIX_PREPARED、SUFFIX_PREPARE_SHA、SUFFIX_SOURCE_SHA、SUFFIX_RECEIPT、SUFFIX_RECEIPT_SHA。路径均须绝对路径；本版本不提供默认 future artifact。

```bash
M=experiments.spectral_reachability.agent_nonlinear_suffix_physical_20261028_v1
PREPARED="$PWD/results/agent_nonlinear_suffix_physical_20261028_v1_prepare_MAIN01.json"
PHYSICAL="$PWD/results/agent_nonlinear_suffix_physical_20261028_v1_MAIN01"
.venv/bin/python -B -m "$M" prepare --output "$PREPARED" \
  --producer-output "$SUFFIX_OUTPUT" --producer-report-sha "$SUFFIX_REPORT_SHA" \
  --producer-prepared "$SUFFIX_PREPARED" --producer-prepare-sha "$SUFFIX_PREPARE_SHA" \
  --producer-source-sha "$SUFFIX_SOURCE_SHA" \
  --producer-receipt "$SUFFIX_RECEIPT" --producer-receipt-sha "$SUFFIX_RECEIPT_SHA"
# MAIN 从实际 prepare stdout 取 PREPARE_SHA（不预填）
.venv/bin/python -B -m "$M" verify --prepared "$PREPARED" --prepare-sha "$PREPARE_SHA"
.venv/bin/python -B -m "$M" run --prepared "$PREPARED" --prepare-sha "$PREPARE_SHA" \
  --output "$PHYSICAL" --authorization MAIN_NONLINEAR_SUFFIX4_RAW10_PHYSICS_ONLY
# MAIN 对实际 report.json 取 PHYSICAL_REPORT_SHA，再消费，写新的排他 receipt
.venv/bin/python -B -m "$M" consume --output "$PHYSICAL" --report-sha "$PHYSICAL_REPORT_SHA" \
  --receipt "$PWD/results/agent_nonlinear_suffix_physical_20261028_v1_consume_MAIN01.json"
```

`run` 唯一在显式 MAIN authorization 后导入 gym/native；自身无 CUDA/模型调用。上游 saved consume 可能耗时，因为必须重新验证完整168及原 source，不可用未验证 receipt 跳过。

## RAM 测试与未决

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_nonlinear_suffix_physical_20261028_v1.py
```

最终实际 v2 import 联合回归115 passed in41.71s（本版67项 + 冻结 terminal adapter48项），命令为上述 pytest 后追加 `tests/test_agent_terminal_witness_physical_20261028_v1.py`。仅 RAM/metadata（包含假 env 真实低层调用及 PNG 字节），不是科学/真实物理证据。覆盖 partial/遗漏 candidate/final、错误 raw/context/task/Q/成本/模型/source SHA、完整四槽新执行、native/工程停止、重封 journal 变异、PNG/额外文件拒绝、cold import 禁 torch/gym/pymunk/foreign runtime、真实 material/verify 入口和持久 receipt 精确比较，以及 prepare/report/consume 三类旧 v1 schema 拒绝。切换上游 v2 前单版64 passed in24.49s及含旧 suffix RAM 的152 passed in48.55s仅作为工程过程记录，最终适配不接受旧 v1。

未运行 production prepare、producer CUDA、physics 或真实 saved consume。关键前置：Nash 最终冻结 source/tests/doc 及与其精确匹配的成功 all168 prepare/report/真实 strict receipt；实际 SHA 均由 MAIN 提供。若其最终 ABI 改变，本版拒绝，不默默适配/降低校验。MAIN 审读、原 frozen native 依赖、可用磁盘和完整真实资产满足后才执行。P1、freshness、独立确认和科学 admission 保持 false；四 reserved fresh roots 不涉及。
