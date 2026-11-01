# History3 normalized target-tracker core v1

仅新增本stem source/tests/本文。沿experiment-code技能做source-first日志/回归；用户限定RAM callbacks，本轮不训练/调参/加载模型，不运行NN/physics/fullstrict或网络，不创建代理。原parity完整consume由MAIN报告已成功，SHA `3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480`；本core不修改该链，也不重复生产消费或将其当scaler认证。

## 精确范围与接口

实现一个统一低层tracker，而非三proposal生产runner。B=1，每臂同入口、同spec、同初始化；仅输入真实原normalized `z:[1,3,192]`、between动作 `a:[1,2,10]`、delivered target `[1,192]`。输入均原float32；不造zero history、不重normalize、不借用scaler、不做raw inverse。没有arm分支、witness或外部initial mean参数；初始化只接受显式 `full_horizon_zero_no_witness`，避免compiled特供warmstart。target的producer/lineage由MAIN未来外层验证，本core不签外部authority。

源码 `experiments/spectral_reachability/agent_hilewm_history3_target_tracker_20261029_v1.py`：

```python
spec = spec_from_dict(MAIN_RESULT_BLIND_SPEC)  # 全14字段显式；无生产默认配置
report, arrays = track_fixture(z_numpy, a_numpy, target_numpy,
                              spec=spec, kernels=RAM_CALLBACK_KERNELS)
local_check = validate_saved(report, arrays)  # 只有完整trace可通过

# 仅MAIN受审HOST CUDA集成；此函数无模型loader/数据发现/自动fallback
report, arrays = track_cuda(authenticated_original_graph, z_cuda, a_cuda, target_cuda,
                           spec=spec, expected_state=MAIN_BOUND_STATE_HASHES,
                           main_authorized=True)
```

`track_cuda`无torch参数/任意callback注入口。授权检查先于torch导入或root访问；之后导入真实torch、检查实际CUDA/普通enabled grad、确定性算法/TF32off/autocastoff，精确定位安装CEM代码，绑定冻结facade v2和normalized seam。参数/buffer须同一CUDA、全eval/frozen；输入须实际CUDA普通float32 Tensor且requires_grad=false，RAM伪对象不被当HOST证据。不暗中detach caller带梯度输入；该tracker不是本轮梯度probe。使用临时官方MATH SDPA context，finally自动恢复，不patch任何root/method/global。

`expected_state`必须是MAIN外层已认证的全部actual state_dict逐key `{shape,dtype,sha256}`，运行前精确等于实际值，结束再次hash；此参数本身不是认证来源，caller编个字典不获得历史配对证明。source前后闭包精确相等，包含旧facade/loader方程、CEM/utils、新三件套、相关torch源；不冒充binary/hardware全认证。实际loader/425资产/完整上游consume、context/target来源和模型proof仍由MAIN外层绑定，不能把本地value一致检查叫完整生产authority。

## 显式spec，不选择生产配置

所有字段必须提供，任何缺项/额外字段拒绝：

| 字段 | 含义/限制 |
|---|---|
| horizon | 完整H，1..16，每packet已normalized width10；没有padding/截断 |
| samples / iterations / topk | 样本2..64、迭代1..128、2<=topk<=samples；topk>=2保障原unbiased std有定义 |
| var_scale / seed | 正有限原分布尺度、显式整数种子；var是官方命名，实际按std使用 |
| segments | 正整数完整partition，sum=H；不重置、不重建history |
| initialization | 仅 `full_horizon_zero_no_witness`，显式全H zero mean；无actor接口 |
| action_domain | 仅 `unclipped_normalized_gaussian`；不暗含raw [-1,1]或clamp |
| objective | 仅 `identity192_terminal_meanMSE_float64`；先转float64再减/平方/192维均值 |
| predict_api_cap / predict_sample_cap | 独立上限；调用前分别检查，不相加，不padding；失败不退款 |
| array_bytes_cap / wall_seconds | array payload上限1KiB..256MiB、core协作wall上限900s |

MAIN尚未选择生产spec。tests内数字仅RAM fixture，不是默认生产预算/seed。目标固定terminal identity mean-MSE，无Q拟合，不把原Hi terminal **sum**成本冒成同metric。float64是本新共同objective的显式选择，不声称与旧float32 sum排序/舍入总相等。中间全部H predictions/context仍保存。没有模型梯度请求；AD primal/J/VJP/backward均0，不把开启grad模式的普通冻结forward冒充梯度实验。

## 冻结CEM数学与真正返回的mean

先完整读安装版 `.venv/lib/python3.12/site-packages/stable_worldmodel/solver/cem.py` 与 `utils.py`，SHA分别：

- CEM `a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b`
- utils `d30c6225a9c4c83782132180a94c94188b07e0381d2c938c658fe5a49aed48e1`

原循环：normal sample×std+mean；candidate0强制prevmean；最小cost topk；elite.mean更新mean、elite.std（sample correction=1）更新var。返回 `actions=mean.detach().cpu()`；`outputs.costs`则是最后elite costs均值，**不是returned mean的cost**。新HOST使用原 `CEMSolver` 全循环，不重写算法/排序/更新、不clamp，不调用旧low search的固定H2/zero-history入口。

RAM `track_fixture`仅用NumPy明确转写该数学顺序，RNG和reduction不声称CUDA相同。它标记 `ram_fixture_not_production`，不能传入host kernels改标生产。HOST Recorder逐迭代同CUDA精确复算topk/gather/mean/std/mean-candidate，并保存所有callback原数组；saved消费只把这些作为source-bound运行见证，不声称独立重放CUDA RNG/梯度或reduction。RAM侧则直接精确复算NumPy更新。

## inference边界：明确新整合，不修改旧模块

安装版 `solve` 带 `@torch.inference_mode()`，原可微facade拒绝此模式及inference输入。新 `Cost.get_cost` 在**唯一明确边界**执行 `inference_mode(False), enable_grad()`，clone实际CUDA候选为普通CUDA张量，保存source/delivered完整数组并逐位验证，之后调用同一Engine。输入context/target从原ordinary causal输入显式expand；同时核安装CEM实际传入的expanded info逐位相同。没有二次normalize、重建候选数值、detach模型/替换参数，复制/capture/sync计wall与bytes。

这是结果盲的新接口，不冒称未经适配官方policy原生复现。CEM info只含z/a/target三个Tensor，绝不让planner_level字符串进入len/slice批量路径。隐含B固定1，不为本slice增加无需求的多环境chunk框架。

## 完整实付mean whole / segmented replay

候选搜索结束后，仅取原返回actual mean，保存actual std与elite平均成本（明确不同字段）。mean复制回CUDA，逐位等于最后callback mean。将它作为S1 **独立执行两次**：whole H与明确partition segmented H，均从同一真实history3/a2/target开始。保全每步完整z/a历史、append outgoing→AE全history→last3 predictor→flatten proj→末帧append；保存全H预测、全部模块I/O、每cp与terminal identity mean-MSE。whole/seg每prediction/完整context/cost逐位相同才完整通过；不拿elite rollout充mean replay，也不省第二次forward。

设N=samples,I=iterations,H=horizon，完整计划：predict API=`H*(I+2)`，predictor samples=`H*(N*I+2)`；两者不相加。AE/proj API同predict API，AE tokens=`(N*I+2)*sum(3+i,i=0..H-1)`，predictor ctx tokens/proj rows=`3*H*(N*I+2)`。全部候选/两replay都实计；低层forward0次AD，不加造high/encoder成本。API或samples cap不足时前置拒绝具体dispatch，保留之前AE/已付模块/partial，不自动增cap或降H。授权后已预扣的失败调用不退款；debit不等于successful return。

## 保存、严格本地检查与边界

返回 `(report, arrays)`，数组是实际CPU capture（HOST从真实Tensor），不自动写文件/碰MAIN结果。`validate_saved`纯NumPy：全数组shape/dtype/hash/引用闭合、完整iterations/H/events、实际context/target/future交付、AE/predict/proj接缝、每cp完整history、returned mean的两次replay、独立float64 MSE、预算/失败门、CEM回调与模型/source前后守卫。它不重新调用模型或fullstrict。独立MSE采用fsum，对float64规约仅允许事前`8*192*eps64*max(abs(reference),tiny)`舍入界；跨replay数组和cost仍逐位，无parity容差。elite average相对同一float64舍入界核对，不能被当mean score。

array cap指**完整保存payload**，不是磁盘输出总cap：NPY headers/JSON和文件floor需MAIN外层排他writer另管。事件先记attempt，数组/模块失败保存partial，`completed=false`，local consumer拒绝partial而不把它删掉。非法初始ABI/授权/模型绑定在生产调用前直接拒绝，无模型动作。异常report不补数组/补步骤。core wall含CEM/同步/capture/两replay，不含入口source/hash和退出hash，额外记录entry-through-postguard wall；两个scope不能混称纯GPU时间或相加成总时。同步阻塞kernel不能被协作wall抢占。

生产外层仍需补：MAIN选择spec；实际constructor/export/parity receipt的完整authority join；共享budget cohort/真实三臂target来源；受限排他文件writer及process级资源/异常处理。**这是可调用core与HOST seam，不是三臂完整生产driver。**MAIN可在既有可信host模板调用它；worker不发起生产。

## 实际命令（无生产run）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_agent_hilewm_history3_target_tracker_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_history3_target_tracker_20261029_v1 verify-sources
# MAIN先提供实际spec JSON路径，不提供伪生产默认
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_history3_target_tracker_20261029_v1 validate-spec --spec "$HILEWM_TRACKER_SPEC"
```

`raw_inverse`无条件拒绝，不因public常数或caller-certified字典而放行；原stats仍unknown。没有实现arbitrary/learned/compiled三proposal生产，不宣称native可执行率/三臂gap收益/科学P2或独立fresh。

## 本地实际验证

联合本core、冻结normalized seam、冻结facade v2：**138 passed in 0.94s**。全部为RAM/符号callback或保存ABI测试，无torch神经/权重/physics/fullstrict执行。本core含完整mean两replay、四种合法partition、所有显式spec拒绝、分轴cap/失败不退款/partial保存、fullhistory/target/mean/模块接缝/MSE/预算/authority篡改、原CEM source pin及未授权入口不导入torch、CLI实际source/spec校验。已实际运行`verify-sources`通过；HOST真实循环尚待MAIN受审外层执行，RAM通过不声称CUDA集成已通过。
