# P1 checkpoint 动作单位与 camera 来源：离线继续审计

独占本模块、同名tests/本文档和新结果目录，不修改此前sealed记录。
使用experiment-code技能；用户限制优先，不训练、不跑GPU/model/physics，不重复identity/Q。
只执行CPU图像预处理/JPEG，读取固定真实RGB及已保存物理receipt；不挑根/轨迹或调阈值。

## 有界检查计划

1. 检查本地HF cache中的README/config/weights对象digest和共同revision；这只是
   本地cache声明，不伪称在线验证或证明训练run。权重只做字节hash，绝不torch.load。
2. 固定读取本地HDF5 episode0的pixels行0/5/10和actions行0:5，加原root1的三帧
   RGB归档，共六张图；大HDF5仅绑定所读hyperslabs及文件stat，不称全档hash。
3. 提取当前 `utils.get_img_preprocessor` 和运行期 `image_transform` 原函数，
   执行相同CHW uint8 tensor的CPU预处理，精确比较（没有任何神经模型）。
4. 独立测试安装版ToImage的NumPy分支是否交换x/y；检查HDF5/Lance reader实际
   返回tensor，防止把未执行分支问题当成P1根因。
5. 用安装版Lance `_encode_frame` 的默认JPEG95，展示同六张真实图的有损差异。
   这是codec路径见证，不能当作checkpoint实际训练codec，也不建议事后加JPEG修复。
6. 读取既有16episode replay的first-transition和80个boundary RGB指标。只汇总
   保存数值，不重跑物理；非零RGB误差不自动归因于camera几何、codec或reset。

## 待证根因与排除范围

installed env执行每primitive command固定 `target=p_start+100*u`，再10次
PD子步（kp100/kv20/dt.01）。不是纯速度控制，不能用velocity解释stats。
原生JEPA `rel_actions.pth/100` 在像素相对位移上做单位转换；LeWM当前live packet
已是u单位。两者经相同mean/std后的代数可相同，但不能证明数据生成/训练分布相同。

当前train源码从dataset实际计算mean/std；模型config缺少fitted scaler。当前data配置
为Lance，安装writer/reader有JPEG路径；本地仅有HDF5，缺少训练Lance及转换版本记录。
现有dataset统计差异不足以推断官方权重训练normalizer错误。当前源码也不能直接冒充
该官方checkpoint的实际训练源码。需要发布者training-run配置、数据修订、conversion
recipe、renderer/codec版本和fitted normalizer绑定到weights digest，不能自造认证。

## 执行

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_checkpoint_action_camera_provenance_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p1_checkpoint_action_camera_provenance_20261027_v1 --output results/p1_checkpoint_action_camera_provenance_20261027_v1_offline_v1
```

独占NEW目录；protocol与source seal在六张CPU图像对照前保存，全部输入结束复验。
报告如实区分confirmed source/branch difference、precise exclusion、unresolved
checkpoint linkage。P1_admission=false，efficacy_verified=false，不给理论negative或代码全排除结论。

## 验证记录

2026-10-27：上述非神经tests实际执行，16 passed（5.44s）。checkpoint三对象
本地cache digest独立检查通过，revision一致；未从网络确认训练来源。

非神经tests与新report是执行记录；本文档在报告前冻结，不回填封存输入。
图像fixture通过不是物理或checkpoint训练证据。没有改变部署预处理或normalizer。
