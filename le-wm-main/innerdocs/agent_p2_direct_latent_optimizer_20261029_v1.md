# P2/E1直接latent优化：最小数值核心与统一tracking接口

仅新增本stem source/tests/doc。没有模型、rollout、物理、动态source加载、CPU神经、训练、
生产prepare/fullstrict或新根。旧文件/科学gate均不改。

## 原协议缺口与本次明确选择

已全文读取Pauli冻结`agent_p2_subgoal_gap_protocol_20261029_v1`源码/doc。
源码SHA `274cbd6f3a8539c866fd62b3ab4d0fff44e37915407a8d596adc8c8ba722a6cd`。
该协议要求直接优化[3,192] latent，**尚未指定原任意latent目标函数、初始化、优化步数**。
因此本组件不是“恢复了原方法确切目标”，更不是已经完成科学合法配对的arbitrary baseline。

本次固定一个可审查的、模型无关的探索性几何surrogate。它保留原goal与identity192
terminal MSE，但新增事前固定的相邻latent平滑项，不声称该平滑项是原提案唯一要求：

令`a`为当前真实context的最后latent，`g`为原任务goal，未来参数`Z1,Z2,Z3∈R^192`，`Z0=a`。

```text
J(Z) = mean((Z3-g)^2) + Σ(k=1..3) mean((Zk-Z[k-1])^2)
Z(0) = [a,a,a]
for t = 0..63:
    Z(t+1) = Z(t) - 16 * analytic_gradient(J,Z(t))
return exactly Z(64), with all iterates0..64 retained
```

三个输出对应cp5/10/15；没有action变量/decoder/endpoint输入，没有拿final goal或随机向量
直接充当生成目标。历史三帧不是未来三帧初始化；只重复当前实际末帧。
smoothness系数1、64步、eta16均在任何新实验前固定，不调beta、不看物理outcome/赢家择参。
全部算术Python binary64，无隐式float32 cast/normalizer/tanh/projection/epsilon修补。
数值非有限或下降检查失败直接拒绝，不改步长重试或选择别的iterate。
若a恰等于g则全程不动仍保留64步；这是退化输入，不伪装成goal替代算法。

## 解析梯度与事前步长依据

定义`Dk=Zk-Z[k-1]`：
`G1=2(D1-D2)/192`，`G2=2(D2-D3)/192`，`G3=2(D3+Z3-g)/192`。
每坐标Hessian为`(2/192)*tridiag(-1,2,-1)`，最大特征值小于`8/192=1/24`；
eta16小于保守的`1/L=24`，无需观察任何结果来定步长。
该凸surrogate的解析极小值为`Zk=a+k(g-a)/4`，仅用于RAM测试核验，**代码不直接返回此闭式值**。
该极小值反而清楚展示：几何平滑并不表达动力学/可达集合，不能拿最优化成功当实现能力。

实际代价：65次标量loss、64次解析梯度、64次参数更新；O(64×3×192)算术，
保留全部历史O(65×3×192)。0神经forward、0神经AD、0模型训练/physics；
解析loss梯度不是实际模型VJP，不能给生产预算退款；controller未来成本须另计。
不同算子FLOPs未知，不声称本几何优化与compiled/prior生成等算力。

algorithm-design技能仅用于公式、循环和实现对应，不扩为论文绘图或科学认证。

## 可直接审读的API

`optimize(task)`：真正执行上述64步，不接受steps/eta/outcome/action-endpoint回调参数。
task必须恰含：

```text
schema = agent_p2_direct_latent_optimizer_20261029_v1.task
root_slot = 0 or 1; condition = off
context = {layout, context_length:3, latent_dim:192, action_dim:10,
           latent_history:3×192, action_history:2×10}
goal = 原192 goal
coordinate_refs = {model,encoder,preprocess,normalizer,context,q,raw_domain,task_goal}
每ref = {path:实际metadata/source绝对路径, sha256:实际SHA}
```

raw动作、privileged state、旧compiled targets、native outcome等多余字段拒绝。
`optimize`只验证输入ABI并算数，不读取上述refs，也不认证真实encoder坐标；
返回完整history、delivered targets与task/context/goal/target摘要、`production_ready=false`。
`optimizer_asset(model_hash,bindings)`返回Pauli的`arbitrary_optimizer`兼容descriptor，
额外绑定本实现source实际SHA与冻结协议source；这只是ABI，不签配对/科学资格。

`sealed_numerical(task_path,task_sha)`可供MAIN运行：只打开≤16MiB的JSON/PY/MD/TXT/H
metadata/source，拒绝weights/NPY/archive；前后hash全部输入/coordinate refs、协议与本三件套。
仍只执行数值优化，**不校验ref内容足以证明latent坐标配对**，也没有三臂真实dispatch入口。
CLI只stdout，不写任何results/共享文件。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p2_direct_latent_optimizer_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_direct_latent_optimizer_20261029_v1 contract
# MAIN构造真实task/coordinate refs后，可执行纯数值入口，非CUDA/controller运行：
.venv/bin/python -m experiments.spectral_reachability.agent_p2_direct_latent_optimizer_20261029_v1 numerical \
  --task ACTUAL_FIXED_TASK.json --task-sha ACTUAL_TASK_SHA
```

worker只用synthetic task测试，没有打开原两根/真实task执行numerical。

## 统一target-tracking seam

```python
targets = core.optimize(task)["delivered_targets"]
request = core.tracking_request(task, "arbitrary_latent_waypoint", targets, controller)
# 其它两臂用同一个函数，targets必须来自各自真实生成器，不能拿上述targets替代它们。
```

controller恰含：`source, predictor_source, model_hash, coordinate_refs, budget,
same_callable_all_arms=true, privileged_compiled_warmstart=false`。
source和predictor_source均为path/SHA；前者是低层tracker，后者才是匹配predict调用边界，
不得将两者混称。coordinate refs和task完全相同；budget使用Pauli原scope/模式：
API或forward samples匹配、外部正cap、batch_limit、model/callable SHA、失败不退款，**没有默认cap**。

请求只交相同public full3 context/两between-action tokens、goal、三delivered targets及来源/预算。
没有compiled raw witness warmstart，也不允许从action endpoint生成arbitrary targets。
缺controller返回`blocked_missing_common_controller`；learned/任意臂缺真正targets返回
`blocked_missing_delivered_targets`，不random/goal/旧动作fallback。
metadata声称同源不等于真实controller通过MAIN审查，因此所有request的dispatch_allowed=false。

`tracking_metrics(request,predicted,observed)`每cp都对**本臂实际delivered target**计算L2/MSE，
不是对task goal或另一个target；未知encoded/predicted用null，完整3槽不padding。
无事前criterion则realizability_ratio=null；native_success恒null，另由原native evaluator评估。

`fixture_track(request,callback)`只接受`kind=RAM_FIXTURE`且source SHA匹配的注入stub，
不动态加载source、不接受HOST_CUDA/CPU神经callback。
fixture返回predicted/observed及API/sample attempted/completed，分别验证cap/batch/不退款；
不能据此认证真实predispatch计量，`real_meter_authenticated=false`，其它资源unknown。
MAIN未来真实controller必须接原真实meter/预算before-dispatch，不能拿该stub计数当权威。

## 已测与精确未完成项

28项RAM测试：固定64步/65迭代、最优解近似与单调下降、全部576坐标的中心差分梯度、
事前谱界、zero方向、action history不驱动arbitrary参数、bool/NaN/shape/outcome/privilege拒绝、
Pauli asset ABI、三臂同请求模板、missing prior/controller、source/coordinate/predictor/cap拒绝、
delivered-target metrics与null、synthetic saved task完整数值入口/绑定漂移、
saved metadata篡改/NPY拒绝、冷import无torch/NumPy/physics。
三臂fixture使用相同synthetic targets只验证ABI对称，绝不是伪造真实learned/compiled输出。

尚未完成且不能由本核心补造：

1. 原科学方案确切arbitrary loss仍未定义；本平滑surrogate是明确的新探索性算法选择，
   是否回答预期科学比较需MAIN事前固定，不宣称它复现了缺失的原目标。
2. 当前P2原encoder/preprocess/normalizer/fullhistory坐标的真实learned-prior配对、raw inverse仍缺；
   不能拿Hi的另一latent坐标强cast，也不训练mapper。
3. 三臂真实同source低层target tracker、配对delivery、实际model/meter/native recipe runtime接线；
   本模块只提供同一请求ABI和RAM数值核心，不假称可立即执行真实三臂。
4. 外部budget cap、真正生成费用与tracker费用、原criterion、真实RGB/物理偏差与native success。
   不造阈值、不放P1/E1/P2、不把known2变fresh或独立样本。

MAIN管理后续生产审查和主机CUDA；本worker不启动任何生产计算。
