# Terminal-witness mixed222 native physical4（MAIN-only）

最小架构：只绑定Pasteur的`agent_p1_terminal_witness_selector_20261028_v1.py`真实`analyze/verify`接口。没有selector prepare/consume；不伪造这两种receipt/schema。physical自己提供prepare/verify/run/consume。analysis是model-only替代选择法，改变的是“stage+own ridge reach”的替代目标→“原封存非线性terminal Q”；并非重新优化suffix、改Q或科学门。MAIN已给mixed222新索引[51,47,61,69]，旧[51,51,23,23]；索引在metadata回归中核对，但生产策略仍由完整222稳定argmin确定，不将这些索引硬编码成选择算法。不用物理outcome解释/选择算法或库。

## 精确authority与复用

固定analysis路径`results/agent_p1_terminal_witness_selector_20261028_v1_MAIN01.json`及SHA `a3a317b557da8bb6eb7a38d582f7df4d230b1b849cced00eca1add85403e63d9`。准备前校验analysis的source/tests/doc等source_sha256，再调用真实`selector.build()`完整重构逐值比较整份analysis。build包括原allmode42和mixed222两份model证据的saved数值验证，这是原analysis闭包成本；physical决策**无条件只取family=mixedmode/index0..3**，不跨库择优。selector不提供full上游consume，不能伪称已调用；本适配器准确继承其原每行execution validator/全数值证据的authority范围。

固定原known2×off/on全四槽。再绑定原mixed prepare SHA `8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b`、原model report SHA `d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2`、原source report SHA `85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`，逐槽绑定task/blind/context/goal/Q及raw_float32_sha256。新的`execution_plan`/selection_digest独立封存analysis SHA、terminal分数与完整raw10；不是把旧surrogate winner重命名。原模型/normalizer/Q/动作不改；不重新tanh/clip/scale。

直接调用冻结`agent_mixedmode_physical_20261028_v1.episode`，复用其env/native_hooks/Meter/Store低层，不调用旧execute/prepare/consume、不改其globals，不转换旧schema/receipt。Store只替换新排他输出路径检查；Meter原caps保持一致。native hooks是在factory返回后安装，**不声称constructor前拦截**；原constructor源码hash/AST检查无reset/step/render/setup，若constructor失败内部工作仍unknown。

## 原生执行及严格消费

默认四臂全新env，不实现whole-row继承（可选且非必要，避免引入旧物理outcome依赖）。原两reset＋warmup10，off物理执行；真实off/on/off triplet用于所有history/root/goal/eval，selected condition用于trajectory/checkpoints。一次原goal attach，goal_pose不改。逐一执行已封存全部raw10，原生终止/截断保留、不补步、工程失败不换动作/不换根。264 RGB triplets成员PNG上限（4×22×3），原source只有17帧warmup证据：中间on缺历史参照保留unknown，不虚构配对。

完整无提前停止新运行的实际资源上限：constructor4/close4/reset8/public_step80（warmup40+task40）/physics_step816/set_state16/goal_setter12/goal_render8/goal_bind4/explicit_render264。attempt→callback→return/failure均原生durable计量；嵌套资源分轴，不能相加成FLOPs。0encoder/predict/AD/GPU。32MiB输出、4MiB末尾reserve、900s wall、200MiB磁盘floor。失败仍四分母，缺失null/明确missing而非0。native success保留原joint agent+object XY/角度门；native7 reward包括速度，不以object-only或wrapped诊断替代。

consumer重新verify自身prepare（含全selector.build），检查新authority、计划digest、原source、真实操作LIFO intent/return/failure、未知event拒绝、完整资源分母与原始float32 dispatch bits；逐一读原PNG检查文件SHA/像素SHA/224RGB/agent guard、无未知文件、condition-specific完整triplets、state7/snapshot/returned transition/goal/原生reward与success对应。完全新schema消费，不将旧报告投影为新schema。native-stop真实不足10保留；不得称全10已返回。原production artifacts保持只读，只可写独占新receipt。

## CLI（inner根目录；仅MAIN实际prepare/physics）

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_terminal_witness_physical_20261028_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_physical_20261028_v1 prepare --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_terminal_witness_selector_20261028_v1_MAIN01.json --analysis-sha a3a317b557da8bb6eb7a38d582f7df4d230b1b849cced00eca1add85403e63d9 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_terminal_witness_physical_20261028_v1_prepare_MAIN01.json
# PREPARE_SHA取上条真实输出，不编造未来SHA
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_physical_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_terminal_witness_physical_20261028_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_physical_20261028_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_terminal_witness_physical_20261028_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_terminal_witness_physical_20261028_v1_MAIN01 --authorization MAIN_TERMINAL_WITNESS4_RAW10_PHYSICS_ONLY
# REPORT_SHA取实际新report；消费不运行物理
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_physical_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_terminal_witness_physical_20261028_v1_MAIN01 --report-sha "$REPORT_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_terminal_witness_physical_20261028_v1_consume_MAIN01.json
```

每次prepare/verify/run前verify/consume都会完整saved重构selector，耗时不是物理/神经成本，不能谎称0计算；实际耗时尚未worker测量。no-cache复算用于拒绝漂移，不靠未经验证的自签receipt。没有额外新增approval/cert门；MAIN显式authorization只用于真实physics入口。真实结果未知；P1/P3/freshness/scientific_admission/independent_confirmation均false，known2不是四fresh根，不打开reserved4。worker仅实现和RAM/saved metadata测试，未生产prepare/消费新结果、未跑GPU/physics。

最终实际回归48 passed in17.26s；包括真实封存analysis＋原mixed prepare的metadata/newplan接线、RAM full222稳定terminal选择、material/verify入口闭包、四新env完整raw10/264PNG/816physics计数、原生stop/工程失败、重链变异拒绝、PNG/未知文件、冷import隔离。首轮45 passed in17.17s，随后只补3项回归和真实metadata接线，无科学阈值改动。source539行/tests260行/doc35行三件套现在冻结，不再原地写入；MAIN据此prepare/run，后续变更必须新版本。RAM成功不代表物理或P1成功。
