# Main：P3 新四臂物理结果独立保存数组复核

仅标准库、NumPy及PNG读取；不导入神经/环境/compiler/runtime，不执行模型或物理。
固定实际report SHA `6c3659e199a8151fee2fa2c020a0cfe63db00b691b947cdbeb762c3c18d6004a`
及consumer prepare SHA `1cfc5eb2ddcfc563176a3cce51a0ef90b88e92d298b3cf35d415c2e420317daf`。
复核16分支、165selected/240warmup/48reset的原始public调用journal、完整动作/状态/
PNG关联与可见性；从七维状态独立重算pose、官方指标与每步reward，逐Q重算同一终点
native192与预测terminal代价；重建16次encoder/projector的attempt/return，不混入新预测。
同根reset完整一致；mixed-H不产生same-H pose差，不跨Q相减。

最终实际小报告 `results/p3_joint_physical_arrays_main_20261027_v1_main03.json`
绑定本审计源码SHA及所有读入文件SHA，结束重验文件未漂移。8项纯算术/接口测试通过。
早期main01/02是较窄审计的保留记录，不覆盖、不冒称与最终源码相同。
此审计不重证模型tensor hash、checkpoint训练normalizer或导数真值，也不重新生成
模拟器/encoder输出；因此它证明保存数据的绑定和算术，不是独立效能复现。

运行（从le-wm-main；新独占输出路径）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_joint_physical_arrays_main_20261027_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p3_joint_physical_arrays_main_20261027_v1 --output results/p3_joint_physical_arrays_main_20261027_v1_main04.json
```

当前0/16官方及object-only成功。旧Q动态调度第2根出现预测/真实Q排序反转；第3根
预测/真实Q均下降但pose变差，提示另有任务代理失配。两类不能混成唯一原因或理论反例。
仍是四个已用校准根；P1/P3准入均false，未建立动态调度的独立真实收益。
