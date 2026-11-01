# GRASP 六槽独立 saved scalar 核验

只新增本 stem source/tests/说明。不改冻结 RGB/physical/planner/results，不执行生产分析、原 strict 消费、NN、torch、CUDA、physics 或训练。使用 experiment-code 技能组织受限实现与测试；不生成实验图、选择候选或根据本批结果调参数。输出是**封存动作后的独立数值复算**，不是新 selector/gate/科学准入、官方 GRASP 复现、等预算或 fresh 确认。

## 输入和固定布局

六个固定位置：root0/off cp5/10/15，root1/off cp5/10/15。`analyze` 必须显式传 RGB prepare/report/strict 三个规范绝对路径及各自实际 SHA。MAIN 已报告真实6encoder/6projector完成及完整 strict 通过：report SHA `34c252e32d92472589dfa224dd6b7d958a9dd059599a80a9bb8daec4d08e55f3`、strict SHA `f93b9082e3b313eaefc4fe56ed0f0390c6e211a44514768dbf56e13d9f3ab22c`；下方填此实际值，prepare SHA 仍由 MAIN 提供，不猜。独立脚本**完全不导入 producer math 或任何 experiments 模块**，只用标准库和 NumPy。

绑定三份完整 JSON、report 的 prepared+SHA、strict 的 report SHA，所有 prepare.bindings、全部 RGB artifact SHA/census/native_seal、原 PNG SHA。沿封存指针读取自己的 physical_v2 prepare/report/strict，再读取原 planner prepare/report/root0/1.json 的 whole；这些文件全部要求实际 SHA，末尾复验闭包未变。原严格 receipt 作为 MAIN 已认证且外部 SHA 固定的输入，**本工具不重新签发或替代其完整 native-hook/preprocessing 严格认证**，也不动态调用原 consumer。这避免把“重用同一数学函数”当成独立复算。

每槽核原 task/model/identity192 Q、literal raw15/history10 bits、原 goal192、whole 对应时刻 predicted192，以及同时间真实 PNG/state/snapshot/response。real192 从 report 指向的原 `.npy.gz` 独立解析，校验压缩文件 SHA、单 gzip、展开上限2MiB、无尾随 NPY 数据、allow_pickle=False、dtype=float32、shape、原数组 bytes SHA；同时要求 real192 NPY 与保存的完整 projector output `[1,192]` 位级相等。goal/pred 原资产是 JSON 中完整精确 float32 值，不伪造不存在的 goal/pred NPY。

无 cp 时保留对应 slot、原 model_Q 和 missing=true，actual 距离/分解/native 轴均 null。不能以其他时刻、terminal 或零向量替代。核对六槽 strict missing incidence、完整 encoder/projector 实际计数和其余零调用；不按 success/time 删除或平均缩小分母。

## float64 独立距离与误差恒等式

令 p=原预测、a=实际 RGB 编码、g=原目标，均在还原原数据后转 float64，维度严格192。逐条重算：

`model_Q=mean((p-g)^2)`；`real_Q=mean((a-g)^2)`；`prediction_error_MSE=mean((p-a)^2)`。

三者和 gap 逐项精确核对原 strict 的 model_Q / real_Q / prediction_vs_encoded_MSE / model_minus_real_Q，**不使用其标量当作计算输入**。MSE 是 mean，sum=192×mean，无新归一化、Q 拟合或 latent/物理单位混加。

令 e=p-a、d=a-g，输出有符号分解：

`model_Q-real_Q = 2 mean(d*e) + mean(e^2)`。

交叉项可以抵消二次项，不据其符号宣称唯一根因或可部署修复。float64 计算的恒等式 residual 和保守舍入 bound 单列：eps=float64 eps、k=16×192+64、gamma=k eps/(1-k eps)，scale 为三项绝对输入平方和的均值。bound=gamma×scale，仅检查这条数学恒等式的算术舍入，**不放宽任何原 scalar equality / native label / gate**。非零相同向量的 residual 可为0而保守 bound 非0。

## 原生七维：各轴分开，不替换指标

delta=state-goal，按原顺序给出7个坐标差与平方贡献；agentXY、objectXY、jointXY、raw angle、velocity 各自分栏，另列 pose5、完整 native7 距离及 reward。完整7维包含两个 velocity 分量，不能拿 pose/XY 替换。

原 native angle 逐字公式为 `min(abs(delta[4]),2*pi-abs(delta[4]))`，没有自动 modulo，可能为负。原 success 仍是 jointXY<20 且 native_angle<pi/9；reward 仍为负的完整7维 Euclidean distance。另列 `abs(atan2(sin(delta[4]),cos(delta[4])))` wrapped-angle 与 wrapped-success **诊断**，绝不覆盖原指标/termination。object-only success 未获定义，明确 null。原 inner native fields、wrapper scalar、reward、native_success/terminated、state/goal 全部独立核验；truncated 原样检查布尔类型。

## MAIN CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。实际 RGB prepare/report/strict SHA 由 MAIN 最终产物提供，不能用 RAM 数据代替：

```bash
MOD=experiments.spectral_reachability.agent_p2_grasp_rgb_scalar_summary_20261029_v1
.venv/bin/python -m "$MOD" analyze \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_rgb_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "${GRASP_RGB_PREPARE_SHA:?actual prepare SHA}" \
  --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_rgb_20261029_v1_cuda_MAIN01/report.json \
  --report-sha 34c252e32d92472589dfa224dd6b7d958a9dd059599a80a9bb8daec4d08e55f3 \
  --strict /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_history_grasp_rgb_20261029_v1_consume_MAIN01.json \
  --strict-sha f93b9082e3b313eaefc4fe56ed0f0390c6e211a44514768dbf56e13d9f3ab22c \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_grasp_rgb_scalar_summary_20261029_v1_MAIN01.json

.venv/bin/python -m "$MOD" verify \
  --summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_grasp_rgb_scalar_summary_20261029_v1_MAIN01.json \
  --summary-sha "${GRASP_RGB_SCALAR_SUMMARY_SHA:?actual summary SHA}"
```

只写新 own stem summary，不覆盖。JSON 读上限128MiB，RGB artifact 总量64MiB、单数组2MiB、summary16MiB；拒 duplicate JSON key、NaN、symlink、路径逃逸、尾随 NPY、额外 artifacts。summary 输出六个独立可读行及完整来源 SHA map，不复制整份 report 或重复192维字符串。verify 从全部显式输入重建 summary，比对 canonical JSON 全文。没有神经/物理执行入口。

## 开发验证记录（不是科学证据）

`.venv/bin/python -m pytest -q tests/test_agent_p2_grasp_rgb_scalar_summary_20261029_v1.py`

首轮1failed/43passed（保守舍入 bound=0 的错误测试期望，改为 residual=0）；新增完整临时磁盘 fixture 时9failed/48passed，其中3项因本说明尚未创建导致自己的 source closure 正确拒绝，6项因测试 NaN 检查误落在其他函数作用域。已创建说明并移回对应测试，没有放宽数据约束。

测试覆盖可手算常数/单坐标 MSE、192 mean/sum、消去、velocity 与 native/wrapped 差异、六槽及全 missing、复制一致但数学错误的 scalar、双 NPY 一致篡改、独立 gzip/NPY/metadata/hash/shape/尾随/压缩 bomb 拒绝、禁止 producer imports，以及真实临时文件的 full binding→NPY→analyze→重建。临时 fixture 的 PNG 是明确 RAM placeholder，不作为真实 renderer/预处理证据；worker 未消费真实生产产物。最终 **57 passed /11.02s**，模块 `--help` 正常；三件套冻结交 MAIN 运行。
