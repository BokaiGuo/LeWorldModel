# MAIN：等上限 journal 读取 ABI 修复

实际HOST CUDA report SHA `8c65d3c5c54ba447b279fd0ad938edfab6eb42f7e1b3dbb4a1624987153cc58d` 无运行错误，原consume在读取journal时拒绝，尚未取得strict证书。219720记录/524758150总bytes，最大记录79913556bytes为完整arm_return；超过继承reader的64MiB单行限制。producer的原总cap1GiB、final reserve256MiB均未超。该失败是新增timing使arm_return变大后的producer/reader ABI不匹配，不是模型或科学负结果。

不改封存source/prepare/results，不重跑CUDA。不monkeypatch任何函数。本新consumer显式独立读取，单行上限使用原producer已经固定的256MiB reserve，总journal仍限原1GiB；readline先限长，完整末尾换行、重复字段拒绝、有限JSON、索引、前序SHA、每记录canonical SHA与最终tail全部验证。没有跳行、filter或截断。

报告/prepare/来源重建、所有artifact字节、all-and-only文件、CUDA preflight、模型/installed classes/precision前后、整job计时/显存/输出上限、全576分母均沿原条件。原producer.validate_saved完整验证全部native数组/原bounded ridge/资源账/原队列重放/计时guard placement及plan release；没有修改原数学阈值或放过partial账。新receipt显式披露兼容层及其source/tests/doc当前SHA，不冒称它们存在于原生产前manifest。

新strict成功前不称完整验收；fixtures仅验证读取ABI和拒绝篡改，包括真实超过64MiB单行。新增神经/物理调用0，P1/P3/C3全部关闭。上限相同不等实际资源/FLOPs/耗时匹配。MAIN命令：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q tests/test_main_p3_equal_cap_saved_consumer_20261029_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.main_p3_equal_cap_saved_consumer_20261029_v1 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_equal_cap_parent_20261029_v1_prepare_MAIN01.json --prepare-sha f12e900b243d0ae775cccd37b06a2228576ef1b22003aebf3cf001f755004ca9 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_equal_cap_parent_20261029_v1_cuda_MAIN01 --report-sha 8c65d3c5c54ba447b279fd0ad938edfab6eb42f7e1b3dbb4a1624987153cc58d --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/main_p3_equal_cap_saved_consumer_20261029_v1_MAIN01.json
```
