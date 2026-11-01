# Common-prefix native192 fidelity CUDA v1

本新stem只新增源码/tests/doc，不改任何封存producer。依据主线程授权固定协议，使用experiment-code的先声明协议、实际数据接线、逐调用日志与mock故障测试；不做训练/调参/画图，不保证正结果。MAIN独占HOST CUDA；实现者仅RAM/mock/saved验证，不运行神经或物理。

## 固定任务和接线

显式pin实际physical report SHA `450b182e4ed4b02959f74593099cf85c728619bfc5ddc3678ea1940c846e1210`，physical prepare SHA `70616f76f08f560301cd0b5445cad8fcb41a9e53415bcac7a9d0efbf34cf560e`，physical源码SHA `d48047562096a52a077b00e9ab16e109b81974495ce023bd1e1d6d5bef1564dd`。4唯一positions、6角色别名（G=D）、2knownroots、每position全部21probe，84行不可筛选。原zero/坐标0..9先+后−、logitρ=.25、NumPy float32 tanh、time-major5×2完全沿用封存bank。无阈值、无优化/选优、无新的物理执行。

先流式核完整physical journal hashchain、操作attempt/completion与report对齐、每行branch plan/命令/closure/step5signature。PNG必须有manifest/file/pixel SHA，fresh_observation、guard accepted、state/observation/snapshot provenance，prefix签名须与同position zero probe完全相同。仅原批准blindtask读取为模型输入；不解析sourcepack/certificate/witness，也不读原planner候选库。上游封存文件只hash，不重新seal。

固定92图片槽顺序：root0 warm5/warm10、root1 warm5/warm10（取该root第一个position的zero分支，不择优替换），position0..3各zero分支prefix5，然后branch0..83完整guarded terminal10。每图片一次encoder和projector，无内容去重。warm5/10编码必须与原blindtask latent_history[1:3] float32逐值exact；不符在预测前拒绝，不用容差掩盖。原transform/原模型native projected CLS192，identityQ/原批准task goal保持。MATH SDP、TF32off、batch1、float32、eval且requires_grad false，模型tensor hash前后封存，无CPU neural fallback/下载。

各position H1输入严格为真实编码`[warm5,warm10,prefix5]`三state；两action来自原raw `[zero5,prefix5]`，用同原float32 mean/std归一化，历史不后缀重中心化，不重复除100。每个原probe单独完整H1，保留完整successor context；terminal10编码只作label，永不作为输入、方向、取样或选优依据。预测callback只收到context和固定raw probe，不收到terminal/state/goal。终态label不替代step5输入。

## 预算/故障/partial

完整输入固定92encoder+92projector、84predictor+84actionencoder实际module entry；新AD/J/VJP/SVD/solve/physics/训练/选择均0。不将原source/planner/physical成本归零：report独立继承。所有module调用predispatch限额、固定每槽顺序；attempt先durable记录，返回且CUDA synchronize成功才记completion，异常不退款不重试。encoder/projector return失败与sync失败均保留实际进入模块账，不声称任意外部硬件故障都能保证finalreport落盘。

整个output cap8MiB、final reserve256KiB、floor200MiB，启动至少208MiB；lossless gzip日志逐事件Z_SYNC_FLUSH+fsync，包含每张native192、每H1输入/命令/完整输出。模型load、输入漂移、model finalization、journal finalize、artifact hash均捕获错误；first errors不因后续错误丢弃。完全磁盘I/O失败仅承诺已有字节，不保证终结report可写。prepare≤2MiB，exclusive新路径。没有padding调用补预算。

支持partial语义：92图片槽及84行分母不变；无guarded完整terminal10则missing/null，不拿早停帧或最后可见帧补。缺固定warm/prefix则该position21预测缺context，不换其他probe的prefix；仍报告实际预算而非声称92/84完成。当前实际input被显式pin为84/84完整run；partial路径通过mock验证，不修改旧report伪造actual partial。新神经故障整体停止，未开始行保留。真正数值0保持0；缺测和共同方向零范数使用null，绝不默认0。

## 固定算术

每probe预测/实际native192 MSE与原identityQ(goal)分别报告。每position各自相对于本position zero endpoint的预测Δz/实际Δz及误差，禁止跨position共享zero终态。共同latent方向固定为同root批准blindtask的`d=goal-original_root`，同时报告`d·Δz`和单位方向投影；零范数未定义。不是ownprefix残差，也不是physical u（像素/angle），不混量纲。

每position输出预测/实际192×10中心secant `B_j=(z(+ρe_j)-z(-ρe_j))/(2ρ)`、signed投影、`E=z+ + z− −2z0`与secant误差。它是固定有限logit干预响应，不是AD Jacobian/原raw坐标导数。±任一缺则B null；zero缺则Δ/E null，不删除列或归一化缺失列。仅两knownroot有限干预的fidelity描述，不能从误差/反转唯一归因理论或代码bug，G=D不能证明非贪心；P1/P2=false。

## MAIN命令（实现者不执行CUDA）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_common_prefix_probe_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_probe_cuda_20261027_v1 prepare --physical-report results/p1_constructive_common_prefix_probe_physical_20261027_v1_main01/report.json --physical-sha 450b182e4ed4b02959f74593099cf85c728619bfc5ddc3678ea1940c846e1210 --output results/p1_constructive_common_prefix_probe_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_probe_cuda_20261027_v1 verify --prepared results/p1_constructive_common_prefix_probe_cuda_20261027_v1_prepare01.json --prepare-sha MAIN_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_probe_cuda_20261027_v1 run --prepared results/p1_constructive_common_prefix_probe_cuda_20261027_v1_prepare01.json --prepare-sha MAIN_PREPARE_SHA --output results/p1_constructive_common_prefix_probe_cuda_20261027_v1_cuda_main01
```

prepare/verify为纯saved读取与hash。测试无torch/native env实例、无磁盘tmp；actual RAM material验证仅已批准输入，不执行神经。
