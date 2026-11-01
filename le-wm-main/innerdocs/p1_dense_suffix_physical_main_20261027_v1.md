# 固定持续后缀干预：测量支持是否过窄

研究计划：原246分支只有单个时间/坐标的.25/.5 logit扰动；负响应不能外推到完整
后缀动作球。本NEW协议保留同2校准根、同zero/greedy/two_stage角色、同原生任务，
固定8罗盘方向×logit L2半径(.5,1.5)+zero，共17suffix/role、102branch。
五个时间步持续同方向，float32向内舍入；不拟合、选择或部署任何物理赢家。
.5与旧单轴较可比较；1.5同时扩大幅度与时间支持，不称单因素消融。

独立审读发现的失败路径已修复：启动时预建102行，journal/close中断也保留
not_started分母；终态journal失败后仍尽力写report，存储失败不保证落盘；
partial进程退出码2。新增run级factory/branch-close journal/final-journal故障注入。

每分支按原source recipe重建，不使用saved poststate reset。全部10命令实际执行，
honor native terminated/truncated/fresh guard；partial全分母保留，不补跑。
保留0/5/10 RGB和两body动力学，same-prefix逐字段检查；不保证solver cache完整身份。
最多204reset/1020step/1122render/306PNG，64MiB输出、200MiB余量，不运行神经模型。
新增源码/测试/本文封存后才能prepare与主机physics；没有CPU neural回退。

本协议仅排查探针支持/接触能力的测量问题，不证明P1成功、独立根泛化、全局可控性，
也不修改原模型Q或success。后续若构造新控制器，应另行封存、独立根验证并公平计账。

MAIN命令（inner repo）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_dense_suffix_physical_main_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_dense_suffix_physical_main_20261027_v1 prepare --output results/p1_dense_suffix_physical_main_20261027_v1_prepare01.json
# 使用上条输出SHA，不覆盖现存产物；run需要主机physics授权。
```
