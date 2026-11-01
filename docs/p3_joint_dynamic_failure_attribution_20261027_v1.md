# P3 old-Q dynamic：第2/第3根真实 pose 退化的 saved-evidence 归因

## 结论先行：两个不同的失配，不是一个已确认根因

本注只读已封存的真实 CUDA 与 physical 保存证据，不重新运行模型、encoder、物理或优化，不改 scheduler/consumer/参数，不新增 physical auditor。主线程的接线与独立审计是前置；这里定位结果含义，不重复宣告一次执行认证。

- **第2根：预测到真实 old-Q 的排序反转，伴随物体被带离目标。** dynamic 预测 old-Q 比 uniform 低 `0.017206227753`，但同一 Q 对真实端点 latent 的评分反而高 `0.209746814516`。dynamic 物体从 x=257.182 移到167.838，目标x=256，末端 XY error 88.172（uniform为15.648）。不能解释成整条动作“没碰到物体、物体没动”。可以确认 model→actual ranking mismatch，不能从保存值唯一归因于某个接触模型或特征。
- **第3根：真实 old-Q 排序保持，但与 object-pose 排序不一致。** dynamic 的预测 old-Q、真实 old-Q 都低于 uniform；物体 XY error 也更小，却转到更差的朝向，角度项增加16.381794492，超过XY项改善4.746622248，pose净变差11.635172244。这对真实端点提供明确的 proxy–pose 排序不一致证据，不能简单归为“模型把两条动作的真实 Q 排反了”。

两根均为 same-Q、same-root、相同初始 context、完整H2/10步；不是混H/partial长度效应。只有四个已暴露 calibration roots，全部16臂 official task success=false；不据此断言唯一根因、理论失效或可推广规律，不据此调参。

## 1. 根身份与固定证据

用户所指 root2/3 在这里按**第2/第3根（一基序号）**理解。物理报告零基 `root_index=1/2` 才是两条退化；零基 `root_index=3` 的old-Q uniform/dynamic动作及终点为tie，不得错配。

| 本注序号 | root_index | root SHA256 |
|---|---:|---|
| 第2根 | 1 | `23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050` |
| 第3根 | 2 | `4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca` |

相对路径以 `le-wm-main/` 为根：

- [真实物理报告](../le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/report.json)，SHA256 `6c3659e199a8151fee2fa2c020a0cfe63db00b691b947cdbeb762c3c18d6004a`。
- [真实规划报告](../le-wm-main/results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_cuda_main01/report.json)，SHA256 `23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950`。
- [规划协议，包含原root/goal及真实两Q](../le-wm-main/results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_prepared01/protocol.json)，SHA256 `dcad803f75c1d0824358960995064793330ea769e3352b7b5656e1c9cfa9479e`。
- physical prepare SHA256 `1cfc5eb2ddcfc563176a3cce51a0ef90b88e92d298b3cf35d415c2e420317daf`；physical-before-encoding报告 SHA256 `206aafcce4d13beee552aa84b4af76e72b5b12a6f47960e4df42baea333b6a2e`。

此次只读定位重新从原192维 latent、goal和old-Q计算 `mean((Q@(z-goal))**2)`；从七维 raw state 计算 `object_XY_error + 20*abs(atan2(sin(theta-theta_goal),cos(theta-theta_goal)))`。检查目标四分支journal及其中44张PNG的文件SHA，人工查看四分支 initial/step5/step10 共12张保存帧。未生成或复制图片。

预算只作归因边界：本轮165 selected +240 warmup =405次公开env.step返回，16 encoder +16 projector，failures=[]；目标四分支各10/10完整返回。既有planner为1066 predict/1066 action-encoder（含shared40、search960、verification66）。不再计一次，不把它叫等总FLOPs；接触子步、历史成本及backward engine API不是这些计数的同义词。

## 2. 同Q预测/真实排序与候选来源

这里所有Q cost只用各根自己的 **old-Q**；不与goal-patch-Q相减。实际Q来自物理报告中 `encodings[].terminal_vector`，不是从raw pose反推，也不是用预测terminal冒充真实endpoint。

| 根 | schedule | winner (H,index) | 预测old-Q | 真实编码old-Q | 实际pose |
|---|---|---|---:|---:|---:|
| 第2根 | uniform | (2,10) | 1.106308585666 | 1.163649220550 | 64.824136750010 |
| 第2根 | dynamic | (2,15) | 1.089102357913 | 1.373396035065 | 136.681238464768 |
| 第3根 | uniform | (2,12) | 1.259073905567 | 1.307942348611 | 61.846075992833 |
| 第3根 | dynamic | (2,16) | 1.149465572163 | 1.284264845744 | 73.481248236739 |

第2根 dynamic−uniform：预测Q `−0.017206227753`，实际Q `+0.209746814516`，pose `+71.857101714757`。第3根分别为 `−0.109608333404`、`−0.023677502867`、`+11.635172243906`。两根现象不同，不把第二种误称为Q预测排序反转。

两根 uniform 均 blocks H2/H3=5/5，候选15/10；dynamic均6/4，候选18/8，search各60 predict。其冻结chi gain分别为0.209450919711、0.202456941804。uniform H2 index0..14仍完整包含于dynamic；dynamic新增15..17，而H3少了8/9。**最终被选中的index15/16都是新增的H2随机尾候选，不是index1 own-Q ridge proposal。** 因而直接的选择路径是“预算分配改变候选覆盖 → own-Q argmin改选新增候选”，不能把两条退化特指为ridge solver或conditional suffix失败；此scheduler也不是conditional compiler。

第2根 predicted→actual Q偏差：uniform +0.057340634884，dynamic +0.284293677152；第3根分别+0.048868443045、+0.134799273581。这些标量偏差不等于latent prediction MSE，也不能单独确定误差发生在哪一步。附加的末端 `mean((Q@(z_actual-z_predicted))**2)` 为第2根uniform/dynamic 0.067337276224 /1.256750066978，第3根0.139891633290 /0.092418107842：第3根dynamic的该误差甚至更小，不能拿“模型更差”统包两根的pose退化。

作为未退化参照，原第1/第4根的old-Q两schedule分别选择同一(2,4)/(2,8)，真实轨迹和终点评分tie。这仍是原四根，不是额外独立样本。

## 3. 第2根：主要是物体平移远离目标，suffix放大

目标物体 `(256,256,theta=0.785398185)`；目标agent `(170.648071,252.877075)`。下表状态坐标为模拟器原单位，不是224像素PNG坐标。theta是保存的原角，误差另按wrapped角计算。

| 分支/边界 | agent XY | object XY | object theta | XY error | angle error(rad) | pose |
|---|---|---|---:|---:|---:|---:|
| 共同initial | (391.986328,249.312778) | (257.181824,260.315643) | 4.569799487 | 4.474537 | 2.498784 | 54.450217 |
| uniform step5 | (324.053300,202.462019) | (243.170300,264.734967) | 4.538569801 | 15.520981 | 2.530014 | 66.121255 |
| dynamic step5 | (368.508378,274.462274) | (230.283529,258.029868) | 4.676308019 | 25.796458 | 2.392275 | 73.641968 |
| uniform step10 | (348.379230,225.505578) | (243.266621,265.095891) | 4.609799452 | 15.648456 | 2.458784 | 64.824137 |
| dynamic step10 | (293.352490,200.002835) | (167.838347,257.359442) | 4.643128227 | 88.172133 | 2.425455 | 136.681238 |

dynamic在step2就出现物体明显左移（x257.181824→244.540362），step5到230.283529；suffix step6..9继续到212.590490、194.990961、174.017061、167.838347。uniform的物体主要在step3左移到243.175458，后续只小幅变化。dynamic终态相对uniform的XY项恶化72.523677218，角度项反而改善0.666575503，净pose恶化71.857101715。**此根不是朝向项主导，而是平移离目标主导。** 两分支相对initial也都没改善pose，dynamic更严重。

dynamic物体从initial到terminal总平移约 `(−89.343476,−2.956201)`，uniform约 `(−13.915202,+4.780248)`。逐步物体pose存在变化的step（精确非零，不设后验阈值）分别为dynamic 2,3,4,6,7,8,9和uniform 2,3,4,7,10。小变化或不变本身不是接触计数；大位移至少排除了整段没有作用于物体的运动这一描述。

同一目标下的预测prefix/terminal old-Q：uniform step5=0.947424402389、step10=1.106308585666；dynamic step5=1.236097630889、step10=1.089102357913。选择依据只有terminal，因而预测中较差的dynamic prefix仍可因预测suffix终值较低而获选。**没有真实prefix encoder latent**，不能给prefix的真实Q排名或声称model误差已被定位在suffix；真实state仅显示物体离目标的恶化在suffix显著放大。

图片证据：[uniform step5](../le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/root_1_old_Q__uniform_rgb_5.png)、[dynamic step5](../le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/root_1_old_Q__dynamic_rgb_5.png)、[uniform terminal](../le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/root_1_old_Q__uniform_rgb_10.png)、[dynamic terminal](../le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/root_1_old_Q__dynamic_rgb_10.png)。灰色T在dynamic末端明显偏到绿色目标左侧；图像与rawstate一致，但不是接触冲量证据。

## 4. 第3根：XY有收益，朝向损失更大；实际proxy也偏好它

目标物体仍为 `(256,256,theta=0.785398185)`，目标agent `(294.070984,444.941315)`。

| 分支/边界 | agent XY | object XY | object theta | XY error | angle error(rad) | pose |
|---|---|---|---:|---:|---:|---:|
| 共同initial | (179.714608,178.608368) | (240.660782,246.718674) | 2.746086359 | 17.928598 | 1.960688 | 57.142362 |
| uniform step5 | (166.551262,167.527682) | (240.569811,246.661328) | 2.917237315 | 18.036117 | 2.131839 | 60.672899 |
| dynamic step5 | (259.151291,165.393652) | (252.605066,247.507050) | 3.587197145 | 9.146354 | 2.801799 | 65.182333 |
| uniform step10 | (201.294015,159.008892) | (241.789527,246.380856) | 3.019702172 | 17.159996 | 2.234304 | 61.846076 |
| dynamic step10 | (281.835224,192.924681) | (268.205486,258.262291) | 3.838791897 | 12.413374 | 3.053394 | 73.481248 |

dynamic的角度误差在step1..5由1.960688增到2.801799，step10到3.053394；物体总角变化+1.092705538 rad（uniform +0.273615813）。dynamic在全部10个观测step都有物体pose变化；uniform主要为1,2,3,10，step4..9的保存物体pose完全不变。故dynamic不是物体没动，而是持续改变物体的平移与朝向，朝向朝更坏的一侧转。

terminal dynamic−uniform：XY error `−4.746622248089`，angle error `+0.819089724600`，乘20后角度项 `+16.381794492`，净pose `+11.635172243906`。prefix step5已经呈现同方向冲突：dynamic XY较好、angle较差、pose仍较差。不能说dynamic“完全无物体进展”，也不能只拿XY进展称成功。

预测prefix/terminal old-Q：uniform 1.380465380730 /1.259073905567；dynamic 1.201464719097 /1.149465572163。实际terminal old-Q也是dynamic更低（表2）。这支持**该root这对实际图像上，现有old-Q目标距离未忠实排序物体pose代价**；不需要诉诸尚未验证的错误接线，也不能声称Q矩阵运行时变了。

图片证据：[uniform step5](../le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/root_2_old_Q__uniform_rgb_5.png)、[dynamic step5](../le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/root_2_old_Q__dynamic_rgb_5.png)、[uniform terminal](../le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/root_2_old_Q__uniform_rgb_10.png)、[dynamic terminal](../le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/root_2_old_Q__dynamic_rgb_10.png)。dynamic明显改变T的朝向；不能由224像素单帧估计精确contact或substep轨迹。

## 5. agent / official 指标不是object pose的替身

七维rawstate为 `[agent_x,agent_y,object_x,object_y,object_theta,agent_vx,agent_vy]`；最后两维是**agent速度，不是物体速度**。当前安装evaluator `_get_obs` / `eval_state` 与此一致：official distance是七维目标差的Euclidean norm，包含agent位置和速度；official success另用前四维位置及角阈值。不能把其变化自动归因于物体改善。

| 根/分支 | terminal agent→goal距离 | agent速度norm | official state distance | object pose |
|---|---:|---:|---:|---:|
| 第2根 uniform | 179.826483 | 262.809032 | 318.850515 | 64.824137 |
| 第2根 dynamic | 133.611600 | 238.787223 | 287.507573 | 136.681238 |
| 第3根 uniform | 300.607579 | 326.288301 | 443.991474 | 61.846076 |
| 第3根 dynamic | 252.313491 | 83.036951 | 265.933541 | 73.481248 |

两根dynamic的agent更接近其goal且terminal速度更小，official state distance因此可以与object pose给出相反排序。第3根真实old-Q偏好dynamic与这些agent/XY收益方向相符，但**没有可辨识的feature归因或干预**，不能断言old-Q具体由agent特征主导，更不能把Q数值拆成agent/rotation权重。两根terminal角误差均远超成功阈值；task0与上述连续量冲突并不矛盾。

“metric proxy漂移”若仅指**proxy与物体pose排序分离**，第3根可直接证实；若指Q参数漂移、encoder权重变化、特定特征语义漂移或训练分布变化，则本记录不提供该因果证据。Q与goal来自固定source，不能把不满意的真实结果追认成运行时Q被替换。

## 6. 接触问题能/不能定位到哪里

- 原始raw动作已按选中H2路径完成，无partial、guard abort或长度差；这里不把安全guard当contact detector。guard认证可见agent，不认证有效推动方向。
- 四分支的initial保存帧显示agent在T附近，之后rawstate与PNG均有物体位移/转动。第2根dynamic的大幅左移、第3根dynamic每步的旋转/平移，不支持“全程动作没有接触物体所以失败”这一笼统叙事。
- 反过来，物体某step不动不能直接证明该step无接触；物体动也不能给出具体接触点、法向、摩擦、冲量或驱动力。原env info可产生 `n_contacts`，但当前这些saved branch trace/journal没有保存它，也没有物体速度或contact/substep日志。不能凭离散state或PNG宣称每一步都接触、接触恰好在某个子步发生，或唯一是接触动力学误差。
- 第2根可将直接失配定位到**新增H2候选的预测terminal排名与实际Q排名不一致**，真实运动错误主要是object translation，suffix扩大；第3根可定位到**实际Q偏好与pose目标偏好分离、角度损失主导**。采样覆盖效应、模型误差、评价proxy语义与接触动力学可同时存在，现有保存证据不能唯一分摊其贡献。

## 7. 可追溯文件与尚缺证据

四条原始journal位于physical输出目录：

| 文件 | SHA256 |
|---|---|
| `root_1_old_Q__uniform.events.jsonl` | `7bc399427f022b830a5b4771008cfd865ff463bd9ba80c2328b54d6f8fae79bc` |
| `root_1_old_Q__dynamic.events.jsonl` | `06190a630b4f53ce70a0ccdfda6aca661c743822d38c6c57ce474541d85369bf` |
| `root_2_old_Q__uniform.events.jsonl` | `353e5e6b2006954f132888d4705b9d3d0caf6eb0250fb3633740869cfd40c2de` |
| `root_2_old_Q__dynamic.events.jsonl` | `b18314a3e49f4acfe4a70fb068b96d08e52347a6c407b043240118586781753c` |

evaluator仅只读源码，未import或构造环境：`le-wm-main/.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py`，SHA256 `c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`，`step/eval_state/_get_obs/_get_info`分别在312/347/360/380附近。

还缺真实prefix encoding（无法区分各段latent误差何时开始）、接触/物体速度/substep诊断（无法精确归因接触模式）、特征干预或有识别力的独立动作对照（无法判定agent/角度特征各自作用）。非选中候选也没有完整physical反事实结果，不能用它们的预测值替代真实结果来推测如果另选会怎样。这里只列这些辨识缺口，**未发起补跑、改目标/阈值/预算、重排候选或重新训练**。

最终结论限于这两条同H2校准配对的具体失败位置：第2根model-to-actual ranking mismatch与错误平移，第3根实际metric-to-pose mismatch与错误转动；不是唯一根因证明，也不是理论命题的反例认证。
