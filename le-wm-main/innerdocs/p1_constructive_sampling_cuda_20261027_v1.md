# Constructive blind same-two-task sampling baselines v1

本实现仅新增本 stem 源码/测试/说明；MAIN 独占 prepare、CUDA、consume 和后续物理。未读取 privileged sourcepack、witness 或 physical 结果；无网络、无训练、无 CPU 神经运行。Hi-LeWM 下载仍 blocked，和本任务独立。

## 预声明算法与预算

固定 source_slot0/1、每根 random_shooting 然后 CEM。只消费已有 blind encoder 的 task_00/01.json.gz，经既有 load_export 强校验外部 SHA、无target合同、原始 frozen normalizer/Q/context/checkpoint。原任务为192维 projected CLS、3状态/2动作历史、H2、每packet5个2维raw command，identity192 Q。不加载旧 planner 选择结果，不使用真实效果参数。

seed=20261027+slot*101+arm_index，random=0/CEM=1；NumPy default_rng/PCG64。random生成64个标准高斯 `[2,10]` logits。CEM初始mean0/std1，4轮×16；每轮所有16个fullH2 cost完成后稳定取top4，按已投影logits计算mean、population std（ddof0），std截断[.05,1.0]，不平滑、不注入incumbent/zero。第4次更新同样留存，但不再生成额外候选。

两臂均使用 sealed screen.projected_logits 对每个packet作L2球投影，cap=.75sqrt5，含其既有float32 inward-rounding。rawcommands=NumPy float32 tanh(logits).reshape(10,2)，不另clip/除100。不是均匀raw动作分布，不宣称覆盖完整[-1,1]^20；与原GN同一有界logit域。CEM并不因名字保证真实物理强于GN，须等待实际比较。

最终在全部64槽按(cost,index)全局稳定argmin（不是只选CEM末轮），先journal封存winner，再调用 gn.verified 独立 nominal/whole/segmented 三次H2：候选128predict+赢家6predict=**134 predict与134 action_encoder/arm/root**。四分支总536各；零AD/VJP/Jacobian/SVD/solve，新encoder/projector/physics为0。历史blind编码单独计账，不是零全流程资源；不宣称与D2P等总算力、独立确认或P1通过。

NativeCuda 按实际模块hook pre/return计账和CUDA synchronize，predispatch cap134。所有模型冻结，MATH SDP，TF32关闭；搜索与三重replay的commands/terminal/cost精确相等是准入gate。inference_mode下无求导。全部64个logits/rawcommands/cost/terminal与CEM更新保存；native_packet逐步保存完整rolling context。为8MiB约束，evaluation_complete不重复整段history，但每次native_packet保留完整值。候选bank每轮生成后、评估前fsync封存；CEM仅依据过去模型cost更新。

## 产物/失败合同

复用blind planner独占Store：总8MiB含partial，200MiB floor，额外256KiB终态预留；每event gzip Z_SYNC_FLUSH/fsync，最终完整footer才可消费。四个result.json.gz、四个plan.json、binding、journal、report。rows直接包含source_slot/arm/blind_sha256/commands。没有命令dispatch或物理入口。

失败保留已完成candidate与原始模块attempt；未开始分支不替补、不自动重试。run首错与final-model/source/journal/hash错误分别保存；所有分支、完整资源、frozen model前后hash与源闭包通过后才commands_released。consume要求外部actual report SHA，重建全部64槽随机流/CEM更新、own-Q cost/稳定选择/动作映射、replay合同、产物哈希、完整journal attempt/return及根身份。后续物理必须另行绑定本次完整实际report，不可借用旧endpoint。

空间估计：真实输出硬上限8MiB，启动需208MiB可用；prepare小于256KiB，不复制大报告/模型。mock体积不保证真实压缩率；真实超限失败保留，不能放宽cap续跑。

## MAIN命令（agent不执行）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；唯一新路径均须不存在。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_sampling_cuda_20261027_v1 prepare \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_sampling_cuda_20261027_v1_prepare_main01.json
```

MAIN独立核prepare SHA后填入 `$MAIN_PREPARE_SHA256`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_sampling_cuda_20261027_v1 verify \
 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_sampling_cuda_20261027_v1_prepare_main01.json --prepare-sha "$MAIN_PREPARE_SHA256"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_sampling_cuda_20261027_v1 run \
 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_sampling_cuda_20261027_v1_prepare_main01.json --prepare-sha "$MAIN_PREPARE_SHA256" \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_sampling_cuda_20261027_v1_cuda_main01
```

取得真实report后另给外部SHA：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_sampling_cuda_20261027_v1 consume \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_sampling_cuda_20261027_v1_cuda_main01 --report-sha "$MAIN_REPORT_SHA256"
```

## 非神经验证

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_sampling_cuda_20261027_v1.py
```

测试使用192维NumPy解析数组替身、RAM压缩产物和已有blind export纯校验，无torch模型、simulator、tmp大文件或网络。验证四分支预算、完整候选、种子/投影、CEM因果更新、原始动作/ownQ重建、篡改拒绝、失败attempt留存和replay错配拒绝。mock是工程证据，非真实模型/物理效果。

实际离线测试40项通过；包括完整四分支saved consume的内存端到端校验。全套RAM journal/results/plans/report为2,470,690 bytes（约2.36MiB），不是实模压缩率承诺。只读build对真实两份已批准export通过，110个bindings、prepare预计21,512 bytes；未写prepare。两份输入外部SHA固定为：

- task_00：`714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a`
- task_01：`70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2`

consume同时复算journal每次实际H2的两段context/raw action history/terminal/own-Q cost，核bank在对应评估前封存、winner在全部64cost之后且三重replay之前封存；不是重复receipt相等替代实际数组算术。

experiment-code技能用于预声明run顺序、日志和失败处理；本任务不采纳其要求图表/通用目录/非零准确率的模板，因为用户明确只新增独占stem且禁止本轮实验，不能按真实负结果改算法。交付冻结后任何源码改动必须新版本；本轮不创建prepare或实际结果。
