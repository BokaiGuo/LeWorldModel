# P2六槽局部干预：先固定对照，不凭分解项认定机制或bug

本轮仅新增本文，不读新decomposition输出、不实现或运行实验。沿已封六槽/known2、cp0/5/10/15及24pairs，不删槽、挑动作或改目标/评分/科学门。experiment-design技能仅用于预先固定最小对照及证伪条件，不采用调参、训练建议。以下真实动作实验须MAIN另批准、封存新实现后执行。

## 1. 问题与不变项

每槽自己的同动作同时间三层记为p=actual-deployment-token预测、z=真实RGB编码、native=原生状态，t=本槽封存target，g=原task goal。D为原identity192 f64 MSE；e=p−z，E=mean(e²)，C_r=2mean(e(z−r))，G=D(t,g)。原等式为`D(p,r)−D(z,r)=E+C_r`，r分别取g、t。不得把不同t的tracking损失直接排名，也不得把native7的agent velocity影响计入模型误差。

权威沿冻结RGB四文件：prepare `14aace0fad5c941fce475a41d45bb1dfa1bad9388cba9f767bdd4eb42aa4f393`、report `26e00d955ee2c106e4452b97ddca26a7731d721f7e00605f5750e3a38981460f`、completion `f54b1ba7fc7657db1b738f1fe8be5ca280d790a14d5fae0a1a5326d0ac5bcb23`、receipt `c16d99ba82c8d3adeb68515e2035cc191bd060ab6c273e07ff21cef9a23d870b`；路径见冻结decomposition文档。新decomposition尚由MAIN验收，不编其未来SHA，也不按其结果决定下列配置。

## 2. 先做两项纯代数负对照：不是新预测或物理实验

固定全24位置；缺值逐项null，不缩分母。原p/z/t/g及原分数永不覆盖。诊断派生量仅在独立字段计算，禁止送controller/planner或称新评分规则。

**目标几何重参数化对照**：仅令诊断辅助目标`t_s=t+s·2^-8·u0`，s固定−1、0、+1，u0为192维第0坐标单位向量，不依据e、goal方向或物理结果选轴。p/z/g全固定；用f64派生t_s，不伪称真实f32交付target。逐x=p,z重算`D(x,t_s)+D(t_s,g)+2mean((x−t_s)(t_s−g))`，原`D(x,g)`必须不受该辅助目标影响；E和C_g也不变。G可以变化，但tracking与cross会补偿。这可反驳“几何项单项变化必然等于任务收益”的算术解释，不能证明更换controller target会无效。

**残差幅度/方向对照**：z/t/g全固定，构造诊断向量`p_a=z+a·e`，a固定−1/2、0、+1/2、1（1为原p）。分别r=g、t报告`D(p_a,r)−D(z,r)`及其两项；实数期望为`a²E+aC_r`。±1/2的能量相同、cross相反；a=0两项皆0。向量均用f64算术，不再转f32/encode，也不是纠正后预测、oracle可部署控制器或新模型结果。保留各直接值与重构值的舍入残差，不新增容差或把残差抹零。该控制只检验分解解释，不能独立验证模型/encoder实现正确。

## 3. 最小真实动作干预（待另批准）

沿六槽原完整raw15及原root/task/goal/history/t，固定只扰动**第6公开动作的X坐标**：raw float32逐步加±2^-8，其余29坐标逐字节原样。两符号对每槽都执行，不因G/E/C大小或已见pair反转选槽/轴/步/符号。封存实际float32差值、ULP和完整raw字节，不假定浮点扰动严格对称。任一符号非有限/超原native[-1,1]，该槽两符号均blocked保留分母，不clip/tanh、缩幅或换坐标；此前原baseline不重绑。

全部12分支输入与source closure先封存，实际raw按原P2 normalize_live_packet的同mean/std及分步sub/div付费获得tokens，不借用或认证Hi scaler。每分支固定whole H3一次与segmented[1,2]一次、n=1，无CEM/重新选目标/训练。必须保存模块IO/actual tokens/fullH和接缝，全部预测seal后才消费真实观测及指标。全量原生reset/warm/prefix合同原样复现并保留cache，不用静态hydrate充动力学重启；原三层baseline只作为已认证保存对照，不声称新重跑。

预算上限按12分支：新执行15步共180 suffix public actions（warm/prefix另列，不藏入180）；每分支whole+segmented累计6 predict/6 AE/6 lowproj，合72/72/72；cp0/5/10/15共48 RGB位置各encode/project一次，不因重复前缀免费去重。无AD/搜索；原native各public控制的实际Space.step/render/reset/callback及失败attempt独立计账，不预编造其总数。资源/墙钟上限需MAIN在执行实现前另封存，未授权前不称HOST ready。任何blocked/earlystop/partial逐槽逐cp保留null，不退款、不填充、不自动重试。

固定报告12分支全部四cp：原D(p,g)/D(z,g)、D(p,t)/D(z,t)、G/E/C_g/C_t及带符号差；与本槽原baseline比较，及同槽正负分支对比，不把跨槽不同raw叫“同动作”。native原7维指标、joint/agent/block XY、raw/wrapped angle、agent velocity分栏。24原pairs不删不重选；新分支对比为附加重复测量，不增加独立root数。

## 4. 可证伪检查与修复依据

- 前缀负对照：第6步之前raw、actual tokens及causal history必须逐字节一致。确定性原生重放的cp0/cp5 state/RGB应与本槽baseline一致；确定性同shape模型的对应前缀输入/输出亦应一致。不一致先定位reset/seed/cache、action grouping、归一化精度或未来动作泄漏，不直接归咎cross term。
- 局部动作后若E变小但任务Q偏差未变小，只可检验C是否数值补偿；若native变化不能由feature目标排序解释，保留B/C断点，不能称代码bug。若±干预并未产生可辨认的对象变化，也如实报告，不加大扰动/挑更易接触动作。真实动作同时改变p、z及C，不能据此唯一识别E或C的独立因果贡献；G因t/g固定始终不变，不能声称本试验证实目标生成质量。
- **可支持修复的证据**必须落到已承诺合同的具体违例：错root/cp/动作前缀或目标/原goal SHA；旧normalized预测冒充actual-token预测；raw→token算术/顺序与已封原公式不符；whole/segmented原接缝不一致；模块API/layout或causal masking违反已核源码；相同primitive/归约次序的独立算术与保存值不一致。先RAM复现最小违例及定位字段/源码行，再NEW版本修复；旧源与失败结果保留。
- G、E或C较大、负cross term、pair反转、native7与XY不一致、一般f64恒等式舍入残差，**本身均不是修复依据**。不同归约次序、随机性或kernel路径差异先核协议，不事后改容差。符号仍用比较而非乘积，各断点tie/null各自计数。

结论边界不变：known2非fresh、prior物理时距unknown、生成费用不齐；不报告latent物理winner、不训练、不修评分或gate，不宣称P1/P2科学机制。下一可执行步骤仅MAIN审阅本文，决定是否先实现第2节RAM/保存算术对照；第3节另行授权，不随文档交付启动。
