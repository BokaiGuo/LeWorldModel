# GN固定缩步：独立saved审计

仅读取MAIN已完成的`p1_gn_step_diagnostic_cuda_20261027_v1_cuda_main01`，不调用producer的
material/extract/evaluate/run、任何神经模型或物理环境。复用纯数据task/context验证器，其余插值、
float32 tanh、normalizer、原Q、事件状态机和选优独立复算。源producer、prepare及所有旧产物不改。

固定actual report SHA为2636b3a89cbe96b97d64dce853f506a837e4191e05b90dd5b754146fde658848；
固定prepare SHA为76a4248cd61d1d6e2a0056e6eb6cb9c41ef1b2b7fd5bd6ba07770a9ecbfa01ec。
输入、source/model/normalizer、B8 task/arm/journal、producer/tests/doc及所有新execution artifacts均hash核对，
审计末重新hash输入，防止读取期间漂移。实际producer目录必须是完整exclusive artifact集合。

逐根核原B8 GN run[7] incumbent、run[8] rejected proposal、same task_id、Q/目标/原packet球与完整H2。
不新增/调整尺度：固定`.5/.25/.125/.0625`，按float64凸插值后一次float32取整生成logits，再float32 tanh。
每根重新执行的incumbent与原保存contexts/cost exact相等；全部四候选成本从保存192维终态经原Q独立重算。
按包含incumbent的五条新成本稳定最小选择（exact tie保留首项），三个winner H2 replay完整contexts/commands/cost exact一致。
全部两packet retained latent和归一化incoming action history逐值核对，不只验证shape。

独立严格消费192条原事件：16个evaluation，每个request/start、两个packet各action_encoder/predict attempt与return、
两份native context，共64个真实module entry，每resource32；顺序、index、root、label、mode、packet及采样数均exact。
每根2 incumbent+8候选+6 winner=16predict/action_encoder，两根32；没有每候选三replay或省略失败试探。
所有eval/selection/result文件、顶层rows/module_entries/resources相互核对；0新AD/image/physics、无新Q或科学passing阈值。

测试读取实际小产物并在内存中篡改副本，不创建tmp，不执行模型。新source/tests/doc/report合计限制100KiB，
report只写新路径、每次保留200MiB，不复制arrays/闭包；本audit消耗纯CPU算术/文件hash，不将其伪装为零开销。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_gn_step_diagnostic_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_gn_step_diagnostic_saved_audit_20261027_v1 --output results/p1_gn_step_diagnostic_saved_audit_20261027_v1_main01.json
# MAIN复验须改用全新output名称，不能覆盖上述报告。
```

实际四点都必须报告：.5在两根均变差；.25/.125/.0625均降低Q，.25是此预声明四点中的赢家。
这证实原GN固定大步被拒绝时，该原方向的有限较小可行步仍可改善模型Q；不能把B8拒绝当方向无改善或stationarity证明。
它不是新根、独立confirmation、物理task gain或P1通过，不证明任意连续尺度/更广domain的性质，也不复验AD/FD正确性。
