# Offline fidelity CUDA v3：仅修复首次磁盘telemetry采样

2026-10-27。独占NEW v3源码/tests/本文；不改v2三文件、旧prepare或实际结果，不新增独立auditor。
使用experiment-code最小错误修复与失败保留原则；本次仅RAM接口/算术tests和只读验证，不运行神经/物理、
不prepare真实输出。MAIN决定是否需要另行实际CUDA重跑。

## 明确的v2问题及历史解释

MAIN报告v2实际状态`stage_A_fidelity_saved_not_admitted`，20 encoder/projector、24 predict/action_encoder/
predictor/pred_proj均完成，模型未变、errors为空。该实际执行结论来自MAIN，本次不重新做结果审计。

v2的run先构造`Limits()`，verify后首次check使用其默认mock探针`lambda:FLOOR+CAP`；稍后才替换为
实际输出目录的disk_usage。因此`minimum_observed_free_bytes=243269632`（232MiB）可能被fixture常数占据，
不能称为实际主机minimum free。MAIN观测当时实际约61GiB不意味着能追溯恢复所有采样的真实minimum。
保留v2 report原字节，不回填或追认其磁盘telemetry。该问题不改变已保存预测、科学输入或MSE算式；
不能以此宣称v2科学结果无效，也不能声称v3已实际复现其预测。

## 唯一运行逻辑差异

v3复制冻结v2实现并只改变NAME/schema/output身份，以及run第一行：

```python
limits=Limits(free=lambda:shutil.disk_usage(Path(output).absolute().parent).free)
p,payload=verify(prepared,expected); limits.check(); store=Store(output)
```

从构造时就注入真实output.parent探针，首次check在Store创建前读取实际文件系统；非200MiB以上即拒绝。
输出目录建立后原有逻辑仍切换到真实store.output探针。Limits的默认模拟探针保留供RAM单元测试，
生产run绝不使用它。metric含义是**离散检查点的最小观测值**，不冒称连续实时最低值。
缺失parent、disk_usage失败、reserve不足都fail-closed，不添加fallback值。

其余与v2完全相同：固定分散[0,6228,12456,18684]、RGB0/5/10/15/20、原action[0:20)不clip/tanh，
W10/W15 teacher-forced F/Z/M，M只mask最后embedding，20 encoder/projector、24 predict及各内部入口、
各72 token positions。原checkpoint/transform、public normalizer trainingunknown、标签隔离、错误分母、
durable journal、评分、600s/3GiB allocated/4GiB reserved、32MiB output/1MiB final reserve/200MiB floor均不变。
v3需NEW prepare/SHA与NEW输出目录；严禁给v3套用旧v2 prepare SHA。
只有版本来源绑定将随v3三文件改变，科学数组/算法不因本修复改变。

四episode ordinal仍不等于独立示范；训练成员关系/collector lineage未知；Z/M不是物理反事实标签；
无paired render B，无训练、CPU神经或P1通过声明。原P1严格要求不变。

## Tests和MAIN命令

tests继承v2全部51个RAM测试，并增加真实run入口的output.parent只读探针测试、首检低空间拒绝、
高空间不混入fixture常数、与冻结v2仅指定行不同的逐字节比较及v2三文件SHA保护。
真实disk探针测试在Store前主动停止，无输出/模型加载；其他是RAM mocks，无tmp fixture/cache/pyc。

在`/home/gbk/code/LeWorldModel/le-wm-main`执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_offline_fidelity_cuda_20261027_v3.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_offline_fidelity_cuda_20261027_v2.py tests/test_p1_constructive_offline_fidelity_cuda_20261027_v3.py
```

只有MAIN决定需要实际重跑后，先prepare并记录打印的实际64位SHA，再以该SHA替换下方`PREPARE_SHA`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_offline_fidelity_cuda_20261027_v3 prepare --output results/p1_constructive_offline_fidelity_cuda_20261027_v3_prepare01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_offline_fidelity_cuda_20261027_v3 verify --prepared results/p1_constructive_offline_fidelity_cuda_20261027_v3_prepare01.json --prepare-sha PREPARE_SHA
timeout 650s env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_offline_fidelity_cuda_20261027_v3 run --prepared results/p1_constructive_offline_fidelity_cuda_20261027_v3_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_offline_fidelity_cuda_20261027_v3_cuda_main01
```

输出必须全新、run启动free≥232MiB，之后每写保持200MiB reserve；本交付不生成prepare或实际结果。

本次验证：v3共55 tests通过；与冻结v2联合106 tests通过（纯RAM和真实磁盘只读探针，未进入模型）。
逐字节差异测试确认源码除NAME与上述初始化两行外，与冻结v2完全一致；v2三文件SHA保持原值。
