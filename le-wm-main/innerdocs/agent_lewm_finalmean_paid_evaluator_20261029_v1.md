# finalmean 后付费 evaluator：实现与MAIN执行入口

固定协议先于实现落盘：`innerdocs/agent_lewm_finalmean_paid_evaluator_20261029_v1_protocol.md`。
新stem `agent_lewm_finalmean_paid_evaluator_20261029_v1`，只补实际交付均值的两布局评估。
原slice及其`final_mean_evaluated=false`永久不动；不把elite cost冒充mean cost。

## 实际authority与验证边界

- 原CEM report `results/agent_lewm_native_cem_slice_20261029_v1_cuda_MAIN01/report.json`，
  SHA `ab3313d24b6efb3539bb1b1a70417f7aa41df1602157465d1da5cd1d26e8b965`。
- 原prepare `results/agent_lewm_native_cem_slice_20261029_v1_prepare_MAIN01.json`，
  SHA `f8f773572326ddd52c407d14b850584228f29c5c3a14d0f44262129a676d276c`。
- MAIN完整receipt `results/agent_lewm_native_cem_slice_20261029_v1_consume_MAIN01.json`，
  SHA `85af90a95df1481f1bb2491f1db0adfd307be50d0d7a1f636d3e4b0a712e5ada`。

prepare检验上述外部SHA/schema/status/全两root/资源及未评均值标记；
复用旧`verify`进行saved来源闭包，不调用旧`consume`或嵌套fullstrict。
完整旧候选银行的真实性以该持久完整receipt为authority，不声称本模块重做其全部验证。
新job需要的全部原数组（包括实际returned actions、mean、last callback mean）另逐文件SHA验证，
三份均值全元素bitexact；context/warm/goal与原task逐位一致。新prepare再次绑定新source/tests/两doc。
有材料不匹配即拒绝，不更新旧pin以绕过。

继承原CEM成本全轴单列，继承8encoder/8projector来源单列，不回填equal144。
模型来源使用原loader/config/weights闭包，加载后/运行后tensor hash核对，
不是同buffer loaded-bytes证明。加载CPU权重字节不等于CPU神经推理；本代码没有CPU NN fallback。

## 固定运行与账本

固定known/off roots0、1，每root whole然后segmented，无重试/重选。
whole直接调用官方cached `rollout`四token，真实RGB3仍提供；
segmented为独立AE3→predict3→AE3→predict3，第二段只用自己的pred15，绝不伪造RGB。
每次AE/predictor/pred_proj完整输入输出都用原float32 NPY保存；
每root固定11个module dispatch，输入/output shape、CUDA dtype与数值守卫、durable intent/return全核。
encoder/projector hooks拒绝新encode。whole保存五帧predicted_emb与action_prefix；
segmented保存五帧predicted_emb和两次完整full3，consumer逐元素校验全部因果接线。

全部四预测输出已durable封存才发四次原criterion（不重跑预测、不传goal进predictor）。
全job新增8predict/8proj/6AE/20AE tokens，predictor样本8/context tokens24，proj rows24；
rollout wrapper2/segmented predict wrapper4/criterion4分别记，不与内部模块相加。
encoder/projector/AD/physics/training0。最多约16MiB实际fullprecision IO与metadata，
事前仍固定512MiB output含64MiB final reserve、200MiB floor、900s cooperative wall、GPU3/4GiB。
失败partial保两root×两layout及未完成scores；不能消费成成功receipt。

consumer重算两个layout各float64 sum192/mean192，并检查实际原float32 criterion。
float32 reduction对float64只继承旧预声明算术误差界；**不是模型parity容差**。
帧15、帧20和两帧合计全部元素均报告bitwise、maxabs、max_ULP、不同位元素数、signed-zero差异。
ULP用有序float32位映射，±0同ULP但bitwise差别单列；非有限值拒绝。
非bitexact仍可成为完整工程receipt，但原样报告`bitwise=false`，`parity_upgraded=false`，
不通过阈值/allclose或调整布局伪称parity。
requested MATH与观测TF32/autocast/inference分开，不冒充实际kernel证书。

## MAIN CLI（从inner根执行）

```bash
.venv/bin/python -m pytest -q tests/test_agent_lewm_finalmean_paid_evaluator_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_finalmean_paid_evaluator_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_finalmean_paid_evaluator_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_finalmean_paid_evaluator_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_finalmean_paid_evaluator_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA"
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_finalmean_paid_evaluator_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_finalmean_paid_evaluator_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_finalmean_paid_evaluator_20261029_v1_cuda_MAIN01 --authorization MAIN_FINALMEAN_PAID_HOSTCUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_lewm_finalmean_paid_evaluator_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_finalmean_paid_evaluator_20261029_v1_cuda_MAIN01 --report-sha "$REPORT_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_finalmean_paid_evaluator_20261029_v1_consume_MAIN01.json
```

两个变量均取各步真实输出，不预填未来值。所有输出需是独占新stem的results直接child，不覆盖/拒绝symlink。
report保存`rows[].inputs/modules/layouts/scores`、artifact SHA/bytes、journal_tail、上游三authority、
继承成本、模型前后hash、numerical guards及memory peaks/error/time。
consume持久新receipt，重建所有新journal/census/NPY/输入身份与接线；不重旧fullstrict。

## 测试与未完成

实际RAM/临时文件回归：新32项通过；与冻结旧CEM联合61项通过（4.00s）。
仅RAM/临时文件fixtures，含真实Store→完整saved reader、67事件hashchain与seal-before-score，
并故意构造两layout非bitexact确认如实报告；覆盖输入/动作/因果上下文/full3/投影/
cost/计数/文件/partial/source/model/预算破坏，CLI冷启动不导入torch。
另有纯fixture hooks测试：完整11模块shape/次序、禁止encoder、异常退出移除hooks，
以及criterion阶段禁止所有神经模块（仅原criterion tensor algebra可运行）。
worker未执行生产prepare/verify/run/consume、NN/physics/fullstrict或网络。
MAIN仍需审读、实际prepare/verify与HOSTCUDA/consume，结果未知，不能称科学通过或任务成功。
与旧bounded CEM/MPPI/beam/GRASP动作域、资源或时距不一致的部分仍须分栏，不能据本切片直接排名。
