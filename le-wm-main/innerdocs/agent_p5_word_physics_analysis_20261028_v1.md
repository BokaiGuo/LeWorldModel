# 全12 words 的 saved 逐帧分解（非任务成功）

只新增本 stem source/tests/doc。旧物理 adapter 三文件及所有 MAIN 结果保持冻结。
本模块只读保存 JSON/非pickle NPY，复用已有严格 saved consumers；无神经、physics、网络、
训练、teacher/adapter/readout/repair，不拟合 latent→物理，也不按差异或接触挑 word。
worker仅RAM单测；真实分析输出由MAIN审读后执行，不占MAIN其他文件。

## 实际 authority

固定物理 report `results/agent_p5_action_words_physical_20261028_v1_physics_MAIN01/report.json`，
SHA `796d757c6ffcf623f4e307f094dba9e39eda2a4810a140e5ab3cf0897b3c73a5`。
固定 MAIN strict `results/consume_p5_physical_MAIN_20261028_v1_MAIN01.json`，
SHA `2248899fb03439bc4c52ee741dc0e2b497e15ce692d1d4d306f32fee7d476883`，
要求 all12_saved_native_replays_verified。实际物理 prepare FILE SHA
`d33b539c77b35b4256a5e7099e39cdd1da619d8e8c3466d6f753ca3fded351d9`。
本程序先核上述实际文件、原consumer/source SHA，再调用冻结 physical.consume(saved-only)，
其返回必须精确等于MAIN持久回执；缺失/修改/partial 均fail closed。
复用原consumer核整个source closure、prefix、raw commands、native7/RGB、公用state、journal、
实际attempt/return计量。不重复造环境、重跑fidelity或physics。

模型并列固定实际 v2 report `cdbccbdf894d8d6e7c078e02f7078bd8cd26e802c9a00d59421ad0e4b0883bbd`，
原strict receipt `5237000615227a5cb6741e19c3feb4414c710077983399c46804b96c223b01db`。
同样重算exact producer.consume，不调用模型。按完整 seed/root_sha256 绑定三根，
逐12分支再绑定case/word；producer的端点patch MSE仅独立并栏，不铺成伪逐帧预测、不回归物理单位。

## 固定输出

根2026102702/03/04，各xy/yx/x_minus_x/zero_zero，完整12分母。

- 12条baseline：原prefix结束frame10，读取同一真实branch，不拿旧endpoint回填。
- 120条suffix long rows：每branch的frame11..20，全部十个实际控制帧。
- 60条pair rows：每根XY−YX、cancel−zero，各十个同帧差异；不只比较终点。
- 6条描述性pair summaries：逐轴max、是否有精确native block pose差异、是否有RGB差异。
- 3条model_only_parallel：真实producer端点feature contrasts及原实际meter，各资源轴原样保留。
- 原物理12个call-entry meter、durable intent/return meter原样并列，不合成总成本/FLOPs。

每帧从原double-precision full_public_state读agent位置/速度、block位置/未wrap角度/速度/角速度，
并记录native coverage、历史max coverage、callback points、原返回n_contacts、reward、done、goal。
native7是float32 `[agent_x,agent_y,block_x,block_y,block_angle_mod2pi,agent_vx,agent_vy]`；
本分析不拿末两维当block速度，也不将rounded state7替代body原精度。

每branch另给相对自身frame10的逐轴差值及raw RGB差值。
配对差值方向始终left−right；位置长度、速度、角度、角速度、coverage、contact各自独立。
角度保留native未wrap差值及principal atan2(sinΔ,cosΔ)弧度差，不把弧度和像素相加。
精确native表示差异没有容差，也不是自动科学显著性或物理机制认证。

RGB直接读224×224×3 uint8保存像素，先int16再相减，报告changed pixels、changed channels、
max abs intensity、MAE、MSE及逐channel MAE/MSE。MSE单位为intensity²，与feature MSE不同。
这是完整图像，含agent/物体/native goal；**object attribution=unknown**，无分割、mask或区域归因。
不能将agent轨迹/速度残留或其像素差冒充物体非交换能力；两个word终点相同也不能掩盖中途差异。

contact明确区分：public n_contact_points 是这一个控制步十个physics substeps内累计callback points，
不是唯一物体接触数；原 info.n_contacts=ceil(points/10)，不是相同计量单位。
程序核这个源公式，但不从callback对猜接触对象/方向或推导因果机制。
coverage为原native area fraction；done按绑定源恒False检查，绝不按coverage阈值定义done。
即使coverage达到1，task_success仍null。未认证task-goal/接触前提仍阻塞科学admission。

不宣称控制优势/任务成功/object noncommutativity、独立12根推断或全局等算力。
三根为已知校准根，不fresh；四word同根配对，不是四独立样本。无p值、阈值搜索或模型选择。

## MAIN 执行

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；仅保存分析，不需要物理overlay/CUDA。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p5_word_physics_analysis_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_word_physics_analysis_20261028_v1.py analyze --output results/agent_p5_word_physics_analysis_20261028_v1_MAIN01.json
sha256sum results/agent_p5_word_physics_analysis_20261028_v1_MAIN01.json
```

独立复算已保存分析（MAIN替换实际新分析SHA，不造未来hash）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_word_physics_analysis_20261028_v1.py verify --output results/agent_p5_word_physics_analysis_20261028_v1_MAIN01.json --output-sha256 <ACTUAL_ANALYSIS_SHA>
```

输出exclusive create+fsync、8MiB cap，拒绝symlink/重复键/非有限JSON，绝不覆盖旧结果。
verify纯saved重算完整确定性对象（无时间戳），连来源及120/60逐帧内容一起比较。
RAM测试使用明确标注的数组fixture：验证中途block动而终点同、agent/RGB不同而block不动、
uint8溢出防护、角度单位、contact公式、完整分母/来源mutation等；这些不是实际物理科学证据。
worker未执行实际analyze/verify，尚无本模块科学结果；MAIN真实运行后以其输出为准。
