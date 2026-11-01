# 独立进程并行验证入口

MAIN新增，仅运行固定九组mock/saved-metadata测试，不执行CUDA实验、训练或物理。
每套测试使用独立Python进程，进程间不共享sys.modules；最多同时四个进程，
各进程BLAS/OMP线程为1，禁止pytest cache和Python bytecode写入。
保留全部测试stdout/stderr的SHA与末8192字节、退出码、耗时和测试源码SHA；
180秒超时按失败记录，不重跑、不忽略失败、不修改封存测试。

实际发现：旧比较器测试断言整个sys.modules无torch等native模块，
而teacher测试收集会导入torch。因此把五套测试放入同一pytest进程会出现
192pass/1fail；该原始联合命令仍有隔离缺陷，不声称已修复原断言。
本入口用独立进程验证每套原测试，避免其他套件的已导入模块污染断言。
它是验证执行方式修复，不是模型/物理结果修复。

在le-wm-main运行：

```bash
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_metadata_parallel_validation_20261027_v1.py
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_metadata_parallel_validation_20261027_v1 --workers 4 --output results/moving_metadata_parallel_validation_20261027_v1_MAIN01.json
```

第二条已实际完成，不可覆盖重跑同名输出；后续须另用新路径。
入口自身6项mock测试通过，九组测试分别42/30/32/38/51/22/29/81/123，
合计448项全部通过，墙钟16.06057秒。
报告SHA `04b0b98dac2f502cbbb3714e5449fd2e436ad847d40ca93b5762a8e8d8e42e90`。
这不代表新增P1/P2/P3科学证据，也不代表所有新模块已获实际实验准入。
输出排他创建、flush/fsync；不删除partial或旧输出，不更改原封存资产。
