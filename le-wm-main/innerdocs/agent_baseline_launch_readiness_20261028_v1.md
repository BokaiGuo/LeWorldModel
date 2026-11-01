# 固定四基线launch readiness（仅离线前置，不自动launch）

独占本stem source/tests/本文。消费Jason执行矩阵SHA `d4e0a2a49dc911d37b534f86451435d8c7d261154bce50ccbf24dc0e1ff1cf3f`；不改矩阵/旧源码/结果，不递归扫描、不下载/解pickle/加载模型，不执行生成的任何命令。标准库hash/JSON检查不是新科学证据。

## 可直接执行

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_baseline_launch_readiness_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_launch_readiness_20261028_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_launch_readiness_20261028_v1 --method GRASP
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_launch_readiness_20261028_v1 --method LeWM
```

默认stdout输出机器可读报告，不创建新results文件；exit0=所选方法本地工作流可进入MAIN dry preflight，exit1=存在blocked，exit2=工具/参数错误。默认包含Hi-LeWM/PlanWAM，因此预期exit1，不表示GRASP/LeWM也不可消费。作者表读取失败只影响独立evidence栏，不会让模型ready升降；矩阵/实际工作流依赖漂移则failclosed。

每个ready工作流生成带实际cwd/env/argv/shell的`main_dry_preflight`两条命令：复用原runner `verify --prepared <实际文件> --prepare-sha <实际SHA>`，及原`consume --output <已有真实目录> --report-sha <实际SHA>`。**本模块不执行它们，不重写原consumer算法或把hash检查称strict consume通过。** MAIN自行审读执行，消费已保存结果优先于重跑。

|方法|实际本地入口|硬检测/缺项|官方结果身份|
|---|---|---|---|
|GRASP|`p1_moving_grasp_baseline_cuda_20261027_v1`|固定prepare/report/source、139项closure、6项artifact各自SHA，原2root task SHA、P1/training/with_target冻结|bounded H3 adaptation，非官方DINO-WM完整复现；官方checkpoint/config/原表协议绑定unknown|
|LeWM|`moving_prediction_matched_sampling_cuda_20261027_v1`和`p1_moving_gradient_baseline_cuda_20261027_v1`|各129/121项closure、10/6项artifact及prepare/report/source，原任务与冻结scope；gradient必须feedback_MPC=false|同predict CEM/nominal-MPPI仅两者144 API/samples匹配；gradient开环，不是三solver Table10官方复现|
|Hi-LeWM|复用`agent_hilewm_asset_verify_20261028_v1 receipt`诊断MAIN终态失败|完整main object122424517B/CRC d59f865b缺；官方成员SHA未知。MAIN整包HTTP200/curl18早EOF；已有config不等于对象/normalizer/-C bank/依赖认证|官方推理命令null；诊断预期exit1，不自动网络、续传或解包|
|PlanWAM|未认证本地官方入口|实际code revision/checkpoint/config路径与digest、许可、4相机/ego数据、NAVSIM-v2/HUGSIM split/evaluator均unknown|不造`planwam.run`，不把PushT或同名旧PointMaze脚本冒充官方驾驶方法|

来源枚举只沿三份实际prepare的`bindings_sha256`及三份report的`artifact_sha256`读取，hash跨工作流缓存一次，**不解析候选/outcome挑选根**。共用本地flat权重字节可hash但不反序列化；保留`.venv/bin/python`路径，不resolve symlink成系统Python。精确失败路径/expected/actual SHA或协议字段在errors内，缺失不是0。

作者表、local comparison、budget matrix分别在`evidence_columns`绑定原SHA，不拷贝作者数值给本地，不按结果提高readiness。local readiness仅叫`ready_for_MAIN_dry_preflight`；`strict_verifier_executed=false`、`cuda_launch_authorized=false`、官方`baseline_runnable=null`、P1/freshness false不变。closure文件哈希不认证CUDA硬件/版本/可用显存，也不替代原verify/consume的TOCTOU与完整语义检查。

## 如MAIN确有新实验目的：只生成真实NEW run命令

不是默认动作，不重跑已完成基线。MAIN提供实际NEW绝对目录与新目的：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_launch_readiness_20261028_v1 \
 --method LeWM --new-run-workflow matched_sampling \
 --new-run-output "$MAIN_NEW_ABSOLUTE_RESULTS_DIR" --main-purpose "$MAIN_PREDECLARED_NEW_PURPOSE"
```

只有ready且NEW目录不存在、parent存在、位于project/results、剩余>=64MiB+200MiB才生成`actual_run_command`；不创建目录/prepare、不执行run、不授权CUDA。三个workflow选项为moving_grasp/matched_sampling/moving_gradient；既有预算不改，墙钟600/900/600s。MAIN必须再完成原strict verify/consume、code review及HOST CUDA独占/版本/内存准入；命令与模型规划都不自动dispatch物理/RGB。API/samples/AD/encoder/physics/walltime不相加，不称equal FLOPs；原known2不是fresh确认。

## 验证/限制

实际RAM测试**13 passed（0.16s）**，用固定recipe的小型元数据注入，不CPU神经、不运行原consumer。真实只读检测已实际执行：191个唯一hash读取，134,008,791 bytes；三个工作流分别139/129/121项closure全匹配、各生成两条MAIN dry preflight命令；GRASP/LeWM为ready_for_MAIN_dry_preflight，Hi-LeWM/PlanWAM blocked。未执行原verify/consume、模型或官方eval，默认全方法exit1是预期缺资产状态。三文件冻结SHA见交付。官方表级复现协议、Hi-LeWM/PlanWAM资产仍缺；本工具不会补造未来hash、训练normalizer、改变阈值或P1。不能因作者原表数字存在而宣布本地可运行。
