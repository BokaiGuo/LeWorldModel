# Matched-domain action-witness beam producer v1

本独占交付只实现新 producer、RAM tests 与本文。无真实模型/物理运行；MAIN审读并执行。已有个人 goal API 被旧 offline-fidelity blocked goal 占用，不覆盖或虚标其完成；本次交付条件为三文件、RAM验证、源码闭包与可执行接口。

## 事前固定数学合同

原两个 moving off task，不选根、不读 native outcomes 或 witness。原历史确实非零：前5 raw=[.1,0]、后5=[0,.1]，保持真实三个192维状态、两个 normalized incoming token 和原 goal。上下文为 states_with_between_state_actions_v1；逐 edge 保留完整三状态/两动作历史，而非 latent 最近邻捷径。

每 slot PCG64 seed 分别20281028、20281129。一次抽取 normal(19,10)，再 uniform(19)，方向单位化、半径 cap*U^(1/10)，cap=.75*sqrt(5)。追加 index0 全零，得到20个10维 packet logits；每个经 float32 tanh→5×2 raw，各时刻可不同。相同 bank 用于两层；没有 constant-packet 限制，没有旧 rawbank 成绩复用。bank、seed、draw order、原始 logits/raw SHA 在任何评价前固定并写 durable journal。训练 normalizer 沿原 task；只归一化一次，不额外 /100，不重新拟合。球约束认证基于源 logits，不由反解 tanh 重新定义。

先20条 H1 edge，按 (mean192Q,path) 稳定排序保留6；依次各扩展20条，得到120条 H2 terminal。只在全部120个完整H2中选 terminal Q 最小者，稳定 path tie-break；不能发布H1赢家。每 node 具 parent/path、edge logits/raw、累计可执行动作、完整 successor context、cost。选中路径先 seal，再从原 root 独立 whole2 和独立 segmented1+1；两者中间和终态 context 必须与树逐字相等。

|阶段|每根实际predict API/样本|高层dispatch|
|---|---:|---:|
|首层|20|20|
|第二层|120|120|
|whole2重放|2|1|
|segmented1+1重放|2|2|
|合计|144|143|

两根固定科学分母始终2，实际总288 API/288 predictor samples/288 action_encoder/288 pred_proj。batch1、context3；不把 API 与内部 predictor 相加。每原生 packet 4资源，各自 predispatch cap144，真实 hook charged attempts/returns/samples；失败不退款、不重试、不 padding。原 source encoding8encoder/8projector 继承成本单列，不新增 encoder/image projector、AD/J/VJP/SVD/训练/物理。

## 实现与 authority

自己的有限 beam loop 和新 schema/arm，不 monkeypatch globals，不冒用旧执行入口。只复用 sealed moving_prediction_matched_sampling_cuda_20261027_v1 的加载闭包、Store、Meter、CUDA guard 和逐 packet native _step；固定该源码 SHA72195df2a7f1d10948c2bf5c494dd26e66400eaa2e269737332cbd876e599d40。d2p_kinodynamic_program_search 仅用于完整 context/scalar 验证，源码 SHA c92a945ca884d503b918c7f7357be7ee55fa36377d9de60e2dddf68b42208b7d。

prepare 重建并绑定本三文件、模型/config/官方API/原task/source及所有上游闭包 SHA；不得传不同任务。verify 严格重建 equality。prepare ≤256KiB，父目录须存在，新文件独占。run只 HOST CUDA0、MATH、TF32off、autocastoff、frozen model，无 CPU fallback。output NEW，64MiB总cap（512KiB final reserve）+200MiB free floor；900s为 Store 创建起 dispatch/write 检查，不含更早 build/verify，也不能抢占已阻塞 native call。

每 edge 先 durable attempt/full inputs，再真实 module ticket inputs/outputs、successor、return；bank/node/frontier/selection/replay均保存。第一次失败 latch 禁止继续，两个 row 事前预置，未启动/partial不删。finally分别尝试模型、源闭包、journal finalize、各artifact hash与report；不能保证磁盘故障后report一定可写。成功输出严格7文件：binding.json、events.jsonl.gz、两根result.json.gz、两根plan.json、report.json。consume外部SHA验证后，独立以已保存真实资源输入输出重建全部树、排序、winner和两种 replay；不调用模型。此处 saved replay 仍共享本 producer 搜索实现，不是外部科学独立证明。

## CLI（MAIN执行）

在 /home/gbk/code/LeWorldModel/le-wm-main 下：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_kinodynamic_matched_domain_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_domain_20261028_v1 prepare --output results/agent_kinodynamic_matched_domain_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_domain_20261028_v1 verify --prepared results/agent_kinodynamic_matched_domain_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_domain_20261028_v1 run --prepared results/agent_kinodynamic_matched_domain_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01
.venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_domain_20261028_v1 consume --output results/agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA
```

consume返回两条model_plan，含source_slot/arm/task_export/task_sha256/blind_sha256/raw10 commands/prediction/model_terminal_cost/selected_path/selection_sha256/whole_run_report_sha256。后续物理consumer需要MAIN独立接入新producer authority/arm/完整manifest；旧physical whitelist不能假冒为其他算法。本次不写任何物理consumer。

## 验证与边界

RAM测试包含两个完整140node搜索、144实际fixture资源调用计数、143dispatch、完整action witness、history-sensitive dynamics、全部 saved journal重建、seed/球内/逐时刻动作、stable ties、14类以上损坏/非法输入、失败计账无重试、固定两根partial分母。fixture不是神经或实际效果证据。真实 saved build只哈希读取（若执行则交接单列）；不调用prepare/run。

实际验证：上述专属RAM测试 **25 passed in 3.61s**；只读真实 saved `build()` 成功：2slots、132bindings、canonical prepare 43494bytes（小于256KiB）。该 smoke 没有创建prepare/output，没有加载模型。两bank semantic SHA分别 `9dfa7d1f9b8bb975704c08e42d04e2324b07085aa719707188363d792074717d` 和 `01552a0bc69ac761b8869928d0009d406516af84795d093af8ce7d443e8fedc7`。最终doc字节变化会使MAIN实际prepare字节数微变；MAIN重新build绑定最终三SHA。

这是有限动作空间 kinodynamic beam，不是 RRT、不是官方论文忠实复现，不代表所有 kinodynamic search。仅与 matched CEM/nominal-MPPI 在predict API/样本这一轴对齐；不宣称与AD方法或其他方法等FLOPs、等总成本、等搜索覆盖。有限bank合法但不穷尽球。两known root不是独立样本；零/负结果保留，P1未准入。缺少真实CUDA成功report与后续physical/RGB证据时绝不宣称方法成功。
