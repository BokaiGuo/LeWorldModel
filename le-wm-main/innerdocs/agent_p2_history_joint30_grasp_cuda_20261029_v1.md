# H3 / joint30 bounded raw15 GRASP：明确算法适配，非官方复现

只新增本stem源码、测试、本文。MAIN审读/运行；worker不生产prepare、不加载模型、不执行NN/AD/physics、不下载、不训练、不修改封存源/结果。使用experiment-code先固定模型输入、算法和资源，再RAM回归；不采用其通用训练/正结果模板。

## 事前算法与输入

选择已由MAIN明确：history-aware joint30-bounded GRASP变体，非原官方last-state solver，非论文表级复现。复用官方LeWM适配94be3c42aa2f0867c6cb1e38bda498b293b6921e的750 joint、lr_s=.1/lr_a=.005、interval50/sync10/lr=.05、固定虚拟状态Gaussian sigma=.005、dense goal weight1、无schedule_decay。保存全部轨迹，无best-iterate/outcome选择、重启或早退。

来源为旧prepare `agent_p2_three_program_20261028_v1_prepare_MAIN01.json` SHA86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461，two known roots/off，全2分母。原task_00/task_02 gzip、goal192/identityQ、root10真实latent0/5/10、warmup非零raw10、mean/std、checkpoint/transform身份不变。原export的horizon2不改，本runner自有future3/raw15合同。prepare重建原P2输入/编码authority，不读取planner或physical outcomes选参数；不打开四保留fresh根。

两份virtual192按官方linear interpolation thirds初始化。lifted边0输入原H3；边1输入[original5,original10,virtual1]，边2输入[original10,virtual1,virtual2]，所有latent输入detach，target virtual仍live；between-state action history始终含原真实历史及本次live前动作。loss严格为sum3 MSE(pred,nextvirtual或goal)+sum3 MSE(pred,goal)。sync从原H3 serial3保持完整graph，terminal MSE，fresh action Adam10；保留virtual值但重建joint Adam，不刷新virtual。噪声仅joint Adam后加入virtual，使用官方randn_like，事前global seed20261029+101*slot；保存实际noise，不以统计分布测试声称随机流重算证书。

**适配差异明确：** 优化30维float64 logit而非官方无界normalized action；每次Adam后投影ONE `.75sqrt15` 球，非逐包球。forward严格NumPy tanh(float64).astype(float32)，原normalization一次；一阶surrogate raw梯度×(1-tanh64²)一次，不声称对量化map数学导数。原real model/virtual为float32，logits/Adam action moments为float64；这是事前固定的坐标适配，不声称官方浮点等价。初始化rawzero不等于官方normalizedzero。原normalizer provenance未知不补fit；未来physical release仍false。

## 原生证据与独立saved消费

每次forward保存实际full latent3/action2/raw5；hooks保存AE、predictor、pred_proj所有原精度输入/输出及attempt/return，predict API另计。直接调用原model.predict，不替换其方法，不monkeypatch任何globals；禁止新encoder/projector。precision原MATH、两TF32关闭、autocast关闭；before/after实测守卫。使用既有reviewed immutable-buffer loader，保留其自己真实loader receipt schema，不伪造新receipt；config/weights bytes SHA、model state/all buffers前后hash、项目与installed class/optimizer源码前后绑定。

每joint/sync保存loss各分量、actual raw backward、gx/gv、installed Adam step/m/q前后、未投影proposal、投影后x/virtual、noise；sync/reset事件完整。消费不导入torch：逐边重建完整native输入/rolling与actual model outputs；独立从这些outputs重算loss，不只信任producer score；验证raw→logit梯度chain、stopgrad目标侧virtual梯度、Adam moments/proposal及全局球/噪声加法、重置时序。**不重算神经梯度或模型输出。** NumPy/PyTorch reduction/Adam只允许保守浮点运算计数误差（f32/f64各自eps），从不对native数组引入model-parity tolerance；逐元素native连线、最终whole/segment边界均exact。噪声本身保存而不伪称std重算通过。

固定最后actions后fresh whole15+三段重放，共6predict，所有3边latent/action历史逐位相等。source最终核、全量hashchain/未知event拒绝、exact file census、全2任务分母；任何partial不生成成功consumer/plan，不补零、不refund，原attempt证据与失败输出保留。计划消费仍physical_release=false，需MAIN另审完整raw15物理接缝。

## 固定前瞻预算（未运行）

每根joint2250+sync450+final6=**2706predict**，AE/predictor/pred_proj各2706（嵌套，不相加当API）；8118predict context tokens与8118pred_proj rows。**900 scalar backward**=750+150，不冒称900×192方向；Adam900、virtual noise750、joint optimizer reset15。two roots总5412predict/1800backward，0新encoder/projector/J/SVD/physics；原paid编码、source verification/loader开销另列，不伪称免费总pipeline。actual wall/FLOPs尚unknown。

明确不与旧common-cap1902/wall450称equal预算。这里wall上限3600秒（verify到最后source检查，边界watchdog非抢占），CUDA4GiB allocated/6GiB reserved；NEW输出512MiB含8MiB final reserve、200MiB disk floor。journal gzip增量写+fsync，最多2GiB解压trace/1MiB每event，严格单流EOF/无trailing；不是解压到磁盘或模型storage导出。错误保留partial且不重试；最后报告真实attempt/return event census，未出现的partial返回不伪造为已完成。

## MAIN CLI（worker不执行）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；所有SHA必须取实际输出，不能预填未来report。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_history_joint30_grasp_cuda_20261029_v1.py tests/test_agent_p2_grasp_raw15_prepare_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_history_joint30_grasp_cuda_20261029_v1 prepare --output results/agent_p2_history_joint30_grasp_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_history_joint30_grasp_cuda_20261029_v1 verify --prepared results/agent_p2_history_joint30_grasp_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_history_joint30_grasp_cuda_20261029_v1 run_cuda --prepared results/agent_p2_history_joint30_grasp_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p2_history_joint30_grasp_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_HISTORY_JOINT30_GRASP_HOST_CUDA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_history_joint30_grasp_cuda_20261029_v1 consume --output results/agent_p2_history_joint30_grasp_cuda_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt-output results/agent_p2_history_joint30_grasp_cuda_20261029_v1_consume_MAIN01.json
```

实现完成不等于科学通过。RAM fixtures仅接线回归，不是优化或NN实验；真实loader/installed Adam/actual gradient/CUDA内核及完整5412调用需MAIN审后主机运行。P1/P2/freshness保持未准入；物理接入、官方process/论文checkpoint配对及新的等资源对照均未完成。

## 实际回归 / 冻结

上述联合命令实际 **86 passed in 1.84s**（本stem40项、已冻结raw15prepare46项）。覆盖原真实task gzip只读、H3全部三边输入、whole/segment请求、完整750+150 schedule/全部资源轴、全30球与float64chart、原生数组与重新hash后的错误连线、coherent错误loss、Adam moment/proposal突变、未知event/跨根/额外事件、single gzip EOF、partial attempt不伪造return、真实installed源码闭包，以及独立冷进程拒torch/foreign/network。没有worker生产prepare、完整优化、NN/AD、GPU或物理运行；没有以测试代替科学证据。三SHA见交付消息；冻结后任何变动NEW版本。
