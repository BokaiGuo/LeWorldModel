# Fixed5 physical v2：只读严格saved consumer

本stem仅消费已存在的v2报告，不提供prepare/run、不创建环境/加载模型、不重跑物理、不写MAIN产物。旧v1/v2源码、测试、文档和原报告均冻结不改。新源码独立复制v2.validate_saved，唯一验证语义修复是warmup0 placement request到native observation的角度边界；没有monkeypatch旧globals或schema投影。

## 权威和最小修复

冻结v2三SHA（在委托真实verify/material前后检查；消费闭包也保存）：

- source `5b9532d57564472abf07f305ab62ffeef98ba046b9ac7104dac478c42d33dc92`
- tests `207f610b30c1581898dc7eb483f324b8be65d573a05c66be1071603f9e746a8c`
- doc `3e390d1c32ccdaa4c693a1a8ae15225e6d846044d865ffa455f70302449aeff2`

完整consume仍调用冻结v2.verify；该函数重建v2.material并完整调用真实CUDA producer.consume，不以信任report字段替代。另提供只读material委托函数（同样前后pin），无prepare输出入口。原schema仍为`agent_p3_fixed_execution_physical_20261028_v2.report`，原contract和全部source/task/raw5/goal/renderer/PNG/journal/resource/metric/earlystop检查保留。新的receipt schema仅为自身`.consume`，同时记录原physical_report_schema。新增自身三文件哈希入receipt闭包，旧prepare无需包含未来新consumer，不修改旧prepare。

原生`/home/gbk/code/LeWorldModel/le-wm-main/.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py`的_get_obs把block.angle按`% (2*np.pi)`投影到观测，snapshot却记录原unwrapped angle。因此warmup0应重算原生观测，而非直接拿recipe角度与观测角度比较。slot0实际request angle=-5.587802913760431、obs=0.6953823934191554，差恰2pi。原v2 strict consumer此处拒绝是接线bug，不是物理失败或理论负结果。

修复仅限该边界：对原requested native7复制后只令第4索引角执行原生mod2pi，与实际f.state逐位比较（其他六维仍exact，无容差）；同时要求snapshot.bodies.block.unwrapped_angle **以及**该body.float64_bits中角度位置均逐位等于原requested角。不是把物理身份按2pi等价归并。requested/snapshot加2pi仍拒绝，obs加减2pi仍拒绝，其他坐标或角度一个ULP变化仍拒绝。原source snapshot/warmtrace证明继续检查，不能篡改原body身份。

固定四分母(slot0 fixed/adaptive、slot1 fixed/adaptive)、已知2根、原生早停不padding、nested resource不相加、native jointXY+angle成功不同于native7 reward等合同完全继承。P1/P2/P3/scientific/freshness/physical_release等门仍False，完整raw5/测试/consumer成功不代表科学通过。

## 真实saved回归与限制

真实report：`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_MAIN01/report.json`，SHA `5fdfc8cfea7f6b4fe78d3e19b59fe946aad73a537850c52a86f2d395055b5e45`。

真实prepare：`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_prepare_MAIN01.json`，SHA `51837e4f8a97d6bed08031ceea70b549ea1bf893351c77e6061b698dc5763e85`。

测试只读校验这些实际SHA及journal/execution_binding，用冻结v2.validate_saved复现原错误、新独立validate_saved通过原样数据；不改原报告或重跑。RAM边界用例逐项注入2pi/ULP/六坐标/原有科学门和选择/raw/resource等篡改。实际saved bytes/PNG消费回归将verify边界替换为已SHA认证prepare，仅在RAM返回receipt；真实CLI必须调用完整冻结verify（包括producer.consume），由MAIN验收运行。测试不等同于已完成MAIN全闭包验收。

实际本地回归 **43 passed in 4.22s**。其中包含冻结三SHA前置拒绝、真实入口委托及逐字比对原validator仅有warmup0边界变化；未运行实际CLI consume或写出独立receipt。全部测试是saved/RAM工程证据，不是新的物理或科学结果。

## MAIN绝对路径命令（只读，不落结果）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m pytest -q -p no:cacheprovider /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_fixed_execution_saved_20261028_v1_test.py

/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_saved_20261028_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_MAIN01 \
  --report-sha 5fdfc8cfea7f6b4fe78d3e19b59fe946aad73a537850c52a86f2d395055b5e45
```

CLI只输出receipt至stdout，无文件写入入口，不创建任何MAIN结果。若MAIN自行持久化，请使用自己的排他写封装且不得覆盖已有结果。saved消费无新neural/physics，返回new_neural_calls/new_physics_calls均0；原运行资源仍完整保存，不能把0新调用误读为原物理资源0。
