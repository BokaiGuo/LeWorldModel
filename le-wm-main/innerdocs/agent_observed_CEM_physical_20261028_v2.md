# Observed CEM physical v2：仅 prepare 尺寸工程修复

v1 保留不改。MAIN 已通过其74项RAM测试，但实际 material canonical bytes=540309，
超过旧 PREP_CAP=524288：bindings413851 / prefixes111060 / plans12167 /
arguments1140 / contract1796（这些是 section 尺寸，不是完整对象字节和）。
v1 实际尚未写出 prepare，未启动该物理运行；这是工程容量拒绝，不是科学负结果。

v2源码仅两处变化：NAME 的版本与 PREP_CAP=1048576。独立模块拥有全部函数 globals，
无 import-v1 后 monkeypatch；prepare/report/consume schema 使用自己的 NAME，
material 通过 __file__/NAME 自动绑定新 source/test/innerdoc，同时保留所有旧来源 closure。
run 输出 CAP=8MiB / FINAL=512KiB / disk floor=200MiB / wall≤600s 全部不变。
不增加候选、预算、动作、Q或阈值，不删减closure来挤尺寸。

其余协议完整沿用 v1：固定两根×两种原prefix=4分母；原seed+placement recipe重建，
warmup10和原prefix5逐帧exact后，执行已封存CEM suffix5；无重选、raw变换或旧endpoint回填。
完整运行8reset/80public_step/88explicit_render，实际physics816单列。
native7 reward/success及wrapped诊断不改；success含agent+object位置和原native角度，
无速度成功条件，不偷换object-only指标。全部失败、earlystop、未执行项保留。
P1/P4/scientific_admission/independent_confirmation=false，不称等总算力/FLOPs。

实现使用 experiment-code 技能；按用户限定只交三文件与 RAM/saved-only 验证，
不训练、不新绘图、不执行 prepare/物理/模型/下载，不派生agent。

## 验证

原74项RAM测试逐项复制到新独占测试文件，仅切换被测模块到v2；
新增最小差异、cap/闭包与真实saved material尺寸回归。
真实回归使用下列已封存authority调用 material，不调用 prepare/run；
断言旧cap < canonical bytes < 1MiB、完整4计划/4prefix、v2三文件都在closure。
同时验证反向规范化v2的NAME/cap后与v1源文本相同，避免借版本更新改行为。

测试命令（cwd `/home/gbk/code/LeWorldModel/le-wm-main`）：

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_observed_CEM_physical_20261028_v2.py
```

测试没有实际模拟或神经调用。fake state machine不是native物理证据。
新prepare/物理report SHA尚不存在，只能由MAIN执行生成，不得用v1或上游SHA代替。

## MAIN CLI

以下命令仅由 MAIN 执行，输出路径须不存在；worker 未执行。
参数与v1 innerdoc完全相同，仍是旧四observed-prefix cohort，不接新teacher8。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
MOD=experiments.spectral_reachability.agent_observed_CEM_physical_20261028_v2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m "$MOD" prepare \
 --compiler-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_next_cuda_MAIN01 \
 --compiler-report-sha256 83f9b6fa5ad991022af6bc38063c0fa6c72fca350578692b3c30f4a7e4854a6b \
 --producer-source-sha256 bc8e588bc11a4ab50572ace8a99ef397f05f0ee8b6edfab5ae5a42f022528e10 \
 --consumer-source-sha256 bc8e588bc11a4ab50572ace8a99ef397f05f0ee8b6edfab5ae5a42f022528e10 \
 --consumer-wrapper-source /home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/consume_observed_CEM_MAIN_20261028_v1.py \
 --consumer-wrapper-source-sha256 442a153ca01c525d1315146d4ffeac2debfce7432486c5adb7fb1db171e8c9da \
 --consumer-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baselines_resume_20261027_next_consume_MAIN01.json \
 --consumer-report-sha256 b81abd4c2f6600bae69513228a9b6896e2a67f09c3e0ab46fc6901a4034f47d3 \
 --original-physical-report /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json \
 --original-physical-report-sha256 a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf \
 --wall-seconds 600 \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_CEM_physical_20261028_v2_prepare_MAIN01.json

.venv/bin/python -B -m "$MOD" verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_CEM_physical_20261028_v2_prepare_MAIN01.json --prepare-sha ACTUAL_V2_PREPARE_SHA
.venv/bin/python -B -m "$MOD" run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_CEM_physical_20261028_v2_prepare_MAIN01.json --prepare-sha ACTUAL_V2_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_CEM_physical_20261028_v2_MAIN01 --authorize-physics
.venv/bin/python -B -m "$MOD" consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_CEM_physical_20261028_v2_MAIN01 --report-sha ACTUAL_V2_PHYSICAL_REPORT_SHA
```

wrapper仍为原已封存consume_observed_CEM_MAIN_20261028_v1.py，不能把它重命名成v2。
v2只hash该wrapper，不执行其代码；调用的saved consume仍是冻结CEM producer自身接口。
