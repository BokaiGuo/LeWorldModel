# Common-resource 六计划 × cp5/10/15 RGB companion

独占新 source/tests/本文；冻结 physical/runtime/旧RGB源码及所有结果不变。worker仅RAM/fixture回归，不运行生产prepare、consume、神经或物理，不读取MAIN新物理结果。MAIN负责审读、实际HOST CUDA及生产strict。两个known/off根不是fresh，不放行P1/P2机制或科学结论。

## 权威、槽与缺失

唯一物理上游为 `agent_p2_common_resource_physical_20261029_v1` 自有schema，不能传旧wall4、standalone GRASP或旧RGB收据。CLI提供实际physical_output/report SHA、完整physical receipt及SHA、physical source SHA；先核字节seal，再完整调用其consume，与持久receipt canonical全内容相等；再verify物理prepare，继承其完整common-resource runtime consume/verify、source/task/model/raw/账本绑定。原runtime report/prepare/receipt SHA与各轴resources、accounts、timings均显式保留，绝不填未来receipt。

固定producer六计划顺序：root0 joint/composition/grasp，root1 grasp/composition/joint；每计划依次cp5/10/15，总18槽。模型对照为该计划原 `contexts[0/1/2]`，不是旧终态或另一臂；与真实原PNG同step、native7 state/snapshot/response、task/goal/root/raw15及因果raw10认证。原goal192和identity192 Q不变、不重编码goal、不变normalizer。

null runtime export保留该计划3槽，model_Q/predicted/动作/real_Q均null；native提前停止保留尚未达到的checkpoint，model预测仍保留而actual来源/real_Q/causal_raw10为null，已封存计划的planned_raw10与实际历史明确分开；即便warmup提前停、没有物理root/cp5也不补图。不按success筛选、不用最后一张图代替cp15、不从旧latent补新终态。正常native提前停并非工程错误；实际缺失只能由完整physical strict决定，合同不硬编码某个槽必缺。

## 原生编码与strict

仅可用真实checkpoint各一次batch1 encode/projector：上限18encoder+18projector、0predict/AE/predictor/pred_proj/AD/新physics/优化/训练。复用冻结contextual RGB的Trace、PNG加载、标准预处理和native guard；自身build/verify/run/consume，不冒用旧schema、不改module globals。旧native guard按4图块计数，新局部子类收紧为精确实际图数0..18，不能偷跑第19图；Trace及roster另独立限制同一预算。

主机CUDA入口显式授权，无CPU fallback；原模型参数冻结，前后tensor hash与source/weights closure验证，inference_mode、autocast关闭、matmul/cudnn TF32关闭、SDPA MATH。保存每张原pixels、原预处理、完整encoder输入/hidden257/CLS192/projector输入输出/最终192、所有module attempt/return和禁止predict/AD的guard events。配置从原224/patch14推导257而非197。strict逐值独立复算PNG pixels/preprocessing、hidden→CLS→192接线；不以容差代替任何native数组身份。全event顺序/数量、artifact census/数组SHA/原图SHA/原source前后seal均核；部分失败保留但完整consume不认证失败prefix。

输出6个root×checkpoint三臂组，及18个固定两两差值（composition−joint、grasp−joint、grasp−composition）。组内要求同task SHA/ID、condition、goal/Q、模型和完整原root context；相同的是**计划相对步数**，不是同步wallclock或相同物理状态。model_Q、actual编码Q与native指标分列，任一端缺失的差值null。全组/全槽保留，不能据此直接声称物理收益或equal FLOPs。

输出64MiB、final reserve8MiB、floor200MiB、wall600s、metadata128MiB；排他新路径，不重试、不覆盖。全18图和17图都可运行；实际调用数只由已认证可用PNG导出。consume本身0NN/0physics，继承物理实际费用不抹成0。

## MAIN命令（inner根，变量必须为实际值）

```sh
.venv/bin/python -B -m pytest -q tests/test_agent_p2_common_resource_rgb_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_rgb_20261029_v1 prepare --physical-output "$P2_COMMON_PHYSICAL_OUTPUT" --physical-report-sha "$P2_COMMON_PHYSICAL_REPORT_SHA" --physical-receipt "$P2_COMMON_PHYSICAL_RECEIPT" --physical-receipt-sha "$P2_COMMON_PHYSICAL_RECEIPT_SHA" --physical-source-sha "$P2_COMMON_PHYSICAL_SOURCE_SHA" --output "$P2_COMMON_RGB_PREPARE"
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_rgb_20261029_v1 verify --prepared "$P2_COMMON_RGB_PREPARE" --prepare-sha "$P2_COMMON_RGB_PREPARE_SHA"
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_rgb_20261029_v1 run --prepared "$P2_COMMON_RGB_PREPARE" --prepare-sha "$P2_COMMON_RGB_PREPARE_SHA" --output "$P2_COMMON_RGB_OUTPUT" --authorize-host-cuda
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_rgb_20261029_v1 consume --output "$P2_COMMON_RGB_OUTPUT" --report-sha "$P2_COMMON_RGB_REPORT_SHA" --receipt "$P2_COMMON_RGB_RECEIPT"
```

metadata/output basename须以本stem加下划线开头，输入路径绝对。未运行任何生产操作；最终physical完整strict receipt实际SHA是下一前置，由MAIN提供，不能用尚未终态收据。初轮RAM41pass/1fail为测试文本转换误将128MiB写成188MiB，已定向修正，未改变production上限或门槛；保留失败记录，不当作科学负结果。

新增build接缝测试首轮55pass/1fail：fixture将receipt和模拟consume返回共享同一字典，mutation同时改两侧，拒绝发生在后续schema检查而非预期的全收据不等检查；已改为独立deepcopy，保留原拒绝断言，未动production校验。

实际回归：RGB与冻结physical测试联合113passed/72.10s；最后补充planned-vs-actual历史区分后，最终RGB源码独立56passed/5.58s。包含全18图、17图、null export、全缺失、三臂同时间join、197token拒绝、完整arrays/events篡改、预测/AD拒绝、RAM落盘consumer及新schema全build接缝。MAIN已另行提供真实完整physical receipt SHA；仅CLI输入，不在合同内按结果硬编码缺失槽，worker未读取该生产report/receipt或调用生产prepare/consume。三件套交付后冻结，后续修改必须新版本。
