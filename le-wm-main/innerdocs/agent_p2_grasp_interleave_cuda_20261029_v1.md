# GRASP 固定输入原路径与 HOST 对照（审计候选，未生产）

固定 root0/off，同一原 task、Q、normalizer、模型和固定初始化；原 `Native.soft` 先执行，随后独立叶张量运行 HOST `GraspBackend.soft`。双方各真实三 packet、一次 backward；不是训练，不是优化结果或等 FLOPs。模型经已封存 loader 原权威与 425 tensor 守卫，不改旧函数或全局。运行必须 MAIN 授权且 CUDA only。

保存原生事件中每 packet 的输入、完整三个模块输入输出、预测及 context，loss/dyn/dense 和 x/v 梯度保留原始 bytes、dtype、shape、stride。局部子类还记录 packet 前后 state/history/action 的 shape、stride、dtype 和 device；模块内部 IO stride 未保存，不能声称模块布局等值。saved consumer 对事件、值和已记录布局逐位比较，含 signed zero，无数值容差；不重跑神经或梯度。上下游来源通过原 common build 绑定。评估执行交错时，捕获与同步也可能影响执行；当前只覆盖原路径自身 hook 和 HOST 自身 facade 捕获条件，不能声称无捕获条件下等价。

固定预测6、backward2；事件逐条 durable，发生失败保留 report/已封存 side，未完成或资源异常拒绝完整 consume，不修补/重试。HOST saved consumer 以原 SavedGrasp 与 SavedTape 重走三 packet、score、backward 账本及全部事件，梯度仍只验证记录而非重算。原 HOST Tape joint unit 不调用 optimizer/commit，单独记录资源而非冒称完整搜索。64MiB 输出上限、200MiB floor、900s；任何缺失或失败不能声称 production-ready。

RAM/tempdisk 覆盖 prepare→verify→两侧 fixture→consume、外层重哈希后的账本篡改、失败 report 留存，以及真实 runner 控制流的 fake-CUDA 失败 finalizer。另用非神经 tensor 分别执行原 Native 的三次 step/hook 和 HOST 的三次 step/Tape，并重建 HOST 完整账本。整链 fixture 的原侧由事件构造，**不是**双真实模型执行；测试不证明真实 CUDA 数值、原模型加载或内核一致性。仅交 MAIN 审读候选，实际 host 行为须 MAIN 授权执行并完整 consume 后才能报告。

## MAIN 命令（需先自行审读并接受限制）

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_grasp_interleave_cuda_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_cuda_20261029_v1 prepare --protocol results/agent_p2_compute_matched_protocol_20261029_v2_prepare_MAIN01.json --protocol-sha fbc151b975409eff95bffa084707531f20d1659cbcf08ea23e5da1b490db4bd2 --output results/agent_p2_grasp_interleave_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_cuda_20261029_v1 verify --prepared results/agent_p2_grasp_interleave_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_cuda_20261029_v1 run --prepared results/agent_p2_grasp_interleave_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p2_grasp_interleave_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_P2_GRASP_INTERLEAVE_CUDA_ONLY
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_grasp_interleave_cuda_20261029_v1 consume --output results/agent_p2_grasp_interleave_cuda_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p2_grasp_interleave_cuda_20261029_v1_consume_MAIN01.json
```
