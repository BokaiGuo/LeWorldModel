# 六槽 saved 配对误差分解：固定协议，等待实际 RGB authority

仅新增本文；冻结 RGB/roster/controller/paid/native 源码、tests、docs、结果一律不改。本轮不实现或执行分解器，不重encode、不预测、不physics、不调用任何生产fullstrict，不启动子代。experiment-design技能仅用于固定问题、分母和解释边界，不采用训练、调参或效果择优建议。

## 1. 与已有实现的差别

已封 `agent_p2_six_target_rgb_20261029_v1.panel/consume` 已报告24位置、24pair、双goal MSE、预测—真实编码MSE和native指标；不能把重新打印它们当新实验。本项真正新增的是**保存向量的固定代数分解及同时间pair差值分解**，检查哪些差值来自预测残差/方向交叉项、各槽target到原goal的几何分离。它不识别target质量的因果效应，不拟合latent→物理，不输出winner。

固定源：

- RGB source SHA `5a3ab9162f527ba6e51595b53cff095e095fbbcb596e8567eb29eac88bc2168e`；roster source SHA `eef264e7856c92ca2920aa60039abfdfe7972780f6c64658679cef59e49be444`。
- actual-token paid report SHA `c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd`、full receipt SHA `6e5fa7c57721cd4765430c7379d501d19978bdb44eebe96dce94f7bc68d9a3db`。
- actual native report SHA `ddbb339f87e0f49b1d99d32ae4f41340d72e6c19f8199fe69e6d0261f78bca2b`、full receipt SHA `7579693032046b2427a7d5b492a90131029fe87d39147d06fe49123a3cfead13`。
- 原normalized six consume SHA `592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103`仅作祖先/成本来源；不能拿它的模型分数代actual-token预测。

以上文件位于 `/home/gbk/code/LeWorldModel/le-wm-main/` 下已有各stem results目录，完整路径沿已封RGB协议。**实际RGB prepare/report/completion/full receipt路径及SHA尚待MAIN提供**。源代码已交付/测试不等于RGB成功；若MAIN发现bug并以新版本运行，本协议必须新附录显式锁定其真实source/schema后再实施，不默默接受任意版本。

## 2. 权威边界：只用已消费保存资产，不再fullstrict

未来CLI显式给RGB prepare/report/completion/完整持久receipt的真实路径和SHA，固定已授权MAIN实际版本。该分解器不调用RGB.consume/verify/build或其上游native/paid认证器；不得因“saved-only”就隐藏一次fullstrict。它是**以MAIN已完成严格消费为条件的独立小数组算术复算**，不是第二套物理/模型真实性证明。

局部检查必须包括：

1. 精确文件SHA、真实schema和24行/24pair；receipt.report_sha256、prepare_sha256、completion_sha256与对应文件一致；report→prepare路径/SHA、completion→report SHA/inclusive边界、report.native_seal_sha256→seal及report.artifact_sha256中所用数组全部连接。完成门仍原RGB合同，不自签MAIN审计证书。
2. prepare.imported_paid_receipt和receipt.inherited_paid_receipt整份强等；原controller receipt/authority以及native report/receipt的实际引用固定。存在envelope则依据实际schema取payload，不把envelope当原consume返回值。原fullreceipt不因本算术工具存在而被重写。
3. `index=4*slot+cp_index`，slot0..5按root0/1各 prior_FIRST/program_first_postcondition/arbitrary，cp固定0/5/10/15；task/goal/fullhistory/raw15及动作前缀SHA、native实际step、PNG文件/pixelSHA、snapshot/state/response、target_origin/source_time严格join。每对仅左右同root/同cp；“同动作同时间”指**每槽模型/编码/物理三层对应它自己的同一有序动作和时刻**，并非声称不同策略raw动作彼此相同。
4. 只读取已明确引用的RGB prepare行小向量、report.rows[i].latent等NPY及对应native state7；array reader核file/dataSHA、dtype/shape/EOF，不加载checkpoint/weights/pickle。bind闭包原样保留为上游认证证据，不重新扫描整个bindings或重读全部权重/PNG以冒全链再认证。若局部引用不在封存inventory/原bindings里即拒绝。
5. 从原向量独立复算原RGB各scalar，按原f64归约精确核producer字段；不同归约次序不能伪称bitexact。对引用/schema/字节不匹配应失败，不把篡改解释成正常missing。明确输出`upstream_authentication_imported=true`、`independent_fullstrict=false`、`independent_gradient=false`。

不为未来接口虚造可执行命令；actual RGB authority到位后，才另新增own stem source/tests/doc及build/verify CLI。其verify只重做本文的局部join/算术，不偷偷回调上游fullstrict。

## 3. 固定位置分解：两个独立目标，不混单位

每位置使用原f32向量：`p=spec.predicted192`（actual-token模型预测，cp0是observed history末帧）、`z=actual RGB latent`、`t=spec.target192`、`g=spec.goal192`。原f32字节保持，算术先转f64，不重新normalize、拟合Q/mapper、做whitening或修改评分。定义 `D(a,b)=math.fsum((float64(a)-float64(b))**2)/192`，向量乘积同样f64、fsum/192。

### 3.1 预测残差与原goal误差

令 `e=p−z`，对每个 `r∈{g,t}` 固定输出：

`D(p,r) = D(z,r) + mean(e²) + 2 mean(e*(z−r))`（实数代数恒等式）。

字段：model_goal_MSE、actual_goal_MSE、prediction_residual_energy、signed_residual_cross_term、model_minus_actual_goal_MSE、identity_roundoff_residual。两种r严格分栏，不能用对t的预测偏差解释对g的任务表现。交叉项可正可负，不能将非负残差能量单独当“模型多估/低估多少”，也不将负交叉项称有益控制机制。

### 3.2 tracking与目标几何分离

令 `v=t−g`，分别对 `x=p` 和 `x=z` 输出：

`D(x,g) = D(x,t) + D(t,g) + 2 mean((x−t)*v)`。

字段：own_target_tracking_MSE、target_goal_geometric_separation、signed_tracking_goal_cross_term、original_task_goal_MSE、identity_roundoff_residual。target_goal_geometric_separation不命名为“target quality”；source time不齐和feature/native metric错配都可能影响解释。不同t的tracking MSE不能直接作跨槽物理排名。

所有新增代数项均保留符号，不clamp、改阈值或删除小量。浮点表达式可能不严格满足实数恒等式：原MSE和每项采用明确固定f64运算，额外保存重构和直接值之差。**不以新容差把residual归零或宣布科学门通过**；该残差是报告量，不取代原producer字段精确复核。可在RAM用整二次幂数值验证精确实例，用一般f32验证不隐藏舍入残差；不从生产结果选择算法或阈值。

### 3.3 原生误差的独立平方分栏

原state顺序固定 `[agent_x,agent_y,block_x,block_y,block_angle_mod2pi,agent_vx,agent_vy]`，goal始终原native task goal。逐位置独立f64重算原native7 distance/reward/success及agent/block/jointXY、agent velocity、raw/native/wrapped angle；不复制producer native.summary当独立算术。

以d=state−goal：native7 squared的四组是agent XY squared、block XY squared、raw angle squared、agent velocity squared。保留各项、原native7 squared、组和及roundoff residual。原混单位范数只作原指标；不能把wrapped angle squared替换raw angle项来“改善”原误差，不重定义原native min-angle/success，不把agent velocity当block velocity。不构造latent-Q与native量的统一总分或拟合回归。

## 4. 固定24pair与时间差分

每root固定pairs(0,1),(0,2),(1,2)的root内索引，全部四cp，共24pair，方向总是right−left。只对共同原g分解：

`Δ model_task_MSE = Δ actual_task_MSE + Δ residual_energy + Δ signed_goal_cross_term`。

四项与直接/重构差值都报告。各槽对本槽target的分解仍逐行保留，但不同t的cross_target_tracking_loss_difference固定null，不为凑分解把不同目标伪成共同目标。native每一分量的right−left同时并列，不能解释成由latent分量唯一导致。

每槽相对自己cp0和固定相邻区间0→5、5→10、10→15也固定输出同一组分解项之差及原native gains，正负方向逐字段标注。时间差分从原24行派生，不创造额外独立根/样本，不用“每槽最后有效帧”替代cp15。已封RGB.panel已有cp0 gains：本工具只独立复算并扩展对应误差项，注明不是新增物理收益。

## 5. null/partial规则与报告预算

固定24位置和24pair永不压缩。z缺失则所有依赖z的残差/actual tracking/actual pair分解为null，并列missing reason；p/t/g本来已封存的纯预测/目标几何量可保留但明确无实际观测支持。native frame缺失则所有native量null，不用模型补；native存在而RGB编码缺失仍允许native分栏，但不出跨层归因数。任一pair需要的侧/时间缺失则该项null，保留各轴有效计数，不能一项缺失令所有量变0。

实际RGB report若partial/failed，不能以complete协议出完整分解；先保留其错误并报告输入门不满足。若未来另授权partial诊断，须独立协议，不在此自动降级。源/身份/hash错是invalid，不是正常missing。

设计限额：新neural/encode/predict/AD/physics调用全0；不读权重、不生成新图像。预计24latent仅24×192×4字节，另24位置原目标/预测小向量；完整原报告metadata可按原128MiB单文件上限读取，不扩权重读口。新输出JSON上限16MiB、磁盘floor200MiB、排他写入/fsync；候选实现whole wall120秒（局部文件核验、算术、落盘全含），失败不自动重试/丢行。上游RGB24 encoder/projector、controller72/324、paid36/36、native150public/1524physics等仅按真实receipt导入且分阶段计，不宣称新增或零CPU/IO。FLOPs=null，不称equal-FLOPs/公平generation。

## 6. 待MAIN authority后的最小实施与本轮状态

最小实现顺序：固定真实RGB版本和四文件SHA→纯fixture的局部authority/24位置join→向量代数与24pair/时间差分→真实saved build/verify由MAIN执行。RAM需覆盖signed zero、null/缺侧/错cp/错target/旧normalized预测替换/数组与receipt篡改、正负交叉项、非零舍入残差，以及分解器无torch/fullstrict导入或调用。不得以假fixture标记HOST ready。

本轮仅定向只读已封RGB.panel/consume和native scalars源码，未读RGB新结果、未跑测试/生产/NN/physics/fullstrict；未实现分解器或假造运行/未来SHA。等待MAIN实际RGB权威，不因GPU空闲启动任务。known2非fresh、prior时距unknown、generation预算不公平；P1前置未过、P2机制/scientific_admission保持false。目标质量与controller误差只能给可检验的断点，不能由这组保存代数唯一辨识。
