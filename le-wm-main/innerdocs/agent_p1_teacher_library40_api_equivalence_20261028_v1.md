# P1 fixed40：相同literal输入的public rollout / cached suffix API

这是完整可执行的HOST-CUDA adapter、prepare/verify、严格saved consumer，不是只有预注册。仅MAIN审读后调用`run-cuda`；本工作位不运行神经/物理。不改冻结模型、Q/goal/normalizer、阈值或根，不碰reserved4fresh。不修改任何旧源码/封存/共享文档。父goal不覆盖，本有界交付与科学准入分开。

## 复用边界与固定实验

已只读确认P2 stageB compare receipt有4roots/8arms/16routes，实际rollout8、AE/predict/predictor/pred_proj各16；复用其已验证的public ABI、Ledger原dispatch、Hooks、saved数组/精度工具，**不调用P2 collect/prepare/run，不使用P2 root或latent/goal/action**。新Ledger只配置自己的40×2顺序，不patch旧globals/kernel。consumer独立验证P1 40槽/80路线/360原生entry，不把P2分母换标签当结果。

输入绑定P1 history-localization NASH01 SHA `7d171fea5ad553541dfd6393e4bf2530992ef916c92fcc39295590de7337d3a2`，重新完整saved重建（含rank-error `e23e60588fa9766b690e605022e089656c4c80ad185853773e23a942b5e0228f`、RGB39/teacher/physical的既有strict闭包）。固定4task `(root0/off,root0/on,root1/off,root1/on)` ×Z/S×`zero,uZ,-uZ,uS,-uS`，每槽public先、cached后，不筛、不重排、不去重、不重试。**index12的预测输入存在，仍执行两路；缺的只是actual terminal-on RGB/latent/Q，所有相关actual指标及完整五成员rank继续null。** 不补图、不拿off冒on。

两路使用相同原保存float32 `z5/z10/z15`、warm6–10/prefix11–15/suffix16–20经原mean/std得到的三个outgoing token，原goal192不变。public路线：

```python
model.rollout({'pixels': pixels[:,None], 'emb': lat[:,None]}, tokens[:,None], history_size=3)
```

cached路线：`model.predict(lat,model.action_encoder(tokens))`。public显式提供相同已保存emb，因此**这是public缓存入口等价实验，不是重新RGB编码或end-to-end视觉rollout**；公共API H=T=3，n_steps=0但其final循环仍真实predict一次，输出`[1,1,4,192]`，最后一个为绝对20/未来5步端点。禁止encode/encoder/projector/AD，若API意外走这些路径立即拒绝。

两路均提供并持久化真正同condition已保存PNG5/10/15、原始RGB及原p0 transform真实CUDA结果，精确匹配saved NumPy预处理。public像素ABI `[1,1,3,3,224,224]`、emb `[1,1,3,192]`、actions `[1,1,3,10]`；AE `[1,3,10]`→`[1,3,192]`；predict/predictor input/output `[1,3,192]`；pred_proj `[3,192]`。checkpoint image224/patch14仍257token，但本轮encode预算0。模型hash使用原P1 `teacher.enc.tensor_hash`，不能拿P2 helper另一种序列化hash算法冒充原checkpoint tensor SHA。

## 真实调用、失败与saved闭包

预定80个语义预测；planned rollout40，AE/predict/predictor/pred_proj各80，共360个entry，encoder/projector/encode0，physics/AD0。实际attempt/entered/returned/saved、输入axis0、temporal rows、inclusive wall分别由真实回调记录，不把planned或RAM fixture当实付。预测API/AE/predictor各80batch/240temporal；pred_proj240flat rows；rollout40batch/120history rows。API/predictor/projection时间嵌套，不能相加假算FLOPs，equal_FLOPs/equal_total_compute仍null。

每次dispatch先durable attempt，再保存真实CUDA输入，再durable enter才调原API；return/error立即存真实输出/错误与journal，禁止后续补根/重试。单次失败保留已用entry及全部80行，后续blocked_not_attempted；setup失败也补齐80个pending分母。strict consumer只接受完整completed，不将中途文件当完整证据。native异常后已耗资源不归零。

闭包包括固定输入/report/prepare/原数组SHA、模型config/weights与实现源码、adapter/tests/doc、stageB Ledger/helper依赖。运行前后完整verify、冻结模型hash/precision一致；MATH-only、TF32/autocast关闭、cuda:0，无CPU神经回退。saved consumer验证整个artifact census、durable hashchain及**精确嵌套attempt/enter/return拓扑**，逐一重算真实source→AE→predictor→pred_proj→public/cached输出数组、ABI和分轴预算，不仅验证日志hash。

数值差异作为结果保留，不用“必须相等”筛掉槽：报告每个模块buffer精确相等性、maxabs/MSE、首个不同buffer，两个路线各自原mean Q与39actual误差分解、固定80组内/20ZS匹配，缺12与对应完整rank为unknown。另列与旧teacher输出差异。不调用public criterion（其sum与原mean Q合同不同），不调Q、不拟合。known roots=2，不是40独立根；P1/scientific/full40 release均false，latent Q不代表物理收益。

预算：output上限512MiB（两路完整RGB/预处理与public输入/输出像素不可拿128MiB旧八臂cap硬套），final预留8MiB，disk floor200MiB，wall900s。边界检查不是抢占式watchdog；MAIN可加外部timeout但不得把中断报告说成完整。journal上限16MiB。

## MAIN命令

在`/home/gbk/code/LeWorldModel/le-wm-main`，下列新路径不能已存在。prepare/verify/consume均无神经或物理；仅run-cuda需要MAIN批准。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_teacher_library40_api_equivalence_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_api_equivalence_20261028_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_api_equivalence_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_api_equivalence_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_api_equivalence_20261028_v1_prepare_MAIN01.json --prepare-sha <prepare实际SHA>
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_api_equivalence_20261028_v1 run-cuda --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_api_equivalence_20261028_v1_prepare_MAIN01.json --prepare-sha <prepare实际SHA> --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_api_equivalence_20261028_v1_cuda_MAIN01 --authority MAIN_APPROVED_P1_FIXED40_LITERAL_API_HOST_CUDA_ONLY
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_api_equivalence_20261028_v1 consume --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_api_equivalence_20261028_v1_cuda_MAIN01/report.json --report-sha <report实际SHA> --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_api_equivalence_20261028_v1_consume_MAIN01.json
```

测试仅RAM NumPy/假tensor回调及saved metadata，不加载torch/模型、不声称实际CUDA执行。源/测试/doc最终冻结后方可MAIN prepare；后续修复必须新版本。未决依赖是MAIN真实CUDA执行与审读，不是伪造外部批准/新metadata证书。
