# P2 full15 全42候选模型机制统计（saved-only）

独占本stem source/tests/本文。原三程序planner、MAIN报告/prepare、物理adapter及所有封存证据不改。只处理模型侧，不加载torch/权重、不运行NN/physics/network，无新选择或调参。experiment-code技能仅用于固定分母、严格输入闭包与轻量回归；不作额外图表或按结果改算法。

## 实际authority与运行门

绑定原planner三件套SHA，实际prepare `86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461`，实际CUDA report `56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef`。MAIN strict receipt路径/SHA必须使用真实后续产物，不能预填未来SHA。运行先要求report完整终态，再实际调用原`planner.consume`重建全部arrays/J/资源/事件时序/全42评分/replay，随后与MAIN持久receipt全对象比较；不是status-only或仅核日志hash。报告缺失、partial或receipt未齐不分析。绑定上游全部artifacts、原prepare及其完整source/model/blind/tasks闭包和本三件套；verify重跑完整saved闭包及分析，不执行模型。

## 全分母表，不再选择winner

固定2known roots×off×42seed=84 paired rows（168臂候选），不把84行当84独立根。每行保留两臂literal full15 raw、原seed、cp5/10/15模型Q、原root10 Q、各完整context/hash及两次局部compiler的全部5候选成本、固定alpha/保旧身份。原prefix raw5必须逐位相同，cp5完整context相同；H2 incumbent必须是原joint剩余路径，H1 incumbent必须是选中H2最后一路。保留所有平局、不按效果/root/word/成功过滤。

每行精确分解原joint terminal Q → cp5重编H2选中terminal Q → cp10重编H1选中terminal Q；用已存float64成本的Fraction差分认证telescoping，不设置epsilon或拟合新Q。全root内42choose2=861，合计1722条配对排序关系（同时记录两臂方向，相当于3444个单臂排序对）；strict reversal、双方tie、单侧tie、同向逐项分栏。原planner四winner仅附带原selected index/selection digest/commands SHA，不输出新动作选择。

`strict_model_prefix_retreat`仅指imagined cp5相对原root10的模型Q变差；不指真实物理暂退、可控集合扩张或非贪心能力。`proposal_logit_gradient_norm`是原continuous tanh链后的proposal梯度，不冒称raw梯度。cp5/cp10增量是先后依赖的模型侧分解，不是两因素独立随机因果效应，不把guard单调性当新机制证据。native success、各物理指标、actual latent Q与物理非贪心效应全部null。本分析不读取物理标签；后续物理join须独立实现并有精确动作/起点authority。

## MAIN命令：strict实际完成后

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_p2_three_program_model_mechanism_20261029_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_p2_three_program_model_mechanism_20261029_v1 analyze --planner-output results/agent_p2_three_program_20261028_v1_cuda_MAIN01 --strict-receipt ACTUAL_MAIN_STRICT_RECEIPT --strict-sha ACTUAL_MAIN_STRICT_SHA --output results/agent_p2_three_program_model_mechanism_20261029_v1_MAIN01.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_p2_three_program_model_mechanism_20261029_v1 verify --report results/agent_p2_three_program_model_mechanism_20261029_v1_MAIN01.json --report-sha ACTUAL_ANALYSIS_SHA
```

新报告16MiB cap、200MiB floor，exclusive新stem results文件、文件open内flush/fsync后fsync目录；不调用旧全parent目录累积cap writer、不覆写任何失败报告。RAM测试仅无模型加法响应、完整168/84/1722表及临时authority/报告篡改拒绝；不是实际CUDA或科学通过。真实分析待MAIN strict完成和执行授权，P1/P2_mechanism/fresh/physical_release始终false。
