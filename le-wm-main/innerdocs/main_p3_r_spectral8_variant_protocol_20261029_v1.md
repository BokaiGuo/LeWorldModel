# MAIN：原生 R 的 spectral8 探索性变体事前合同

本合同在本变体任何新模型/物理输出出现前固定。目标是补齐现有树在R请求处的实现覆盖，不宣称原proposal或旧compiler全等。旧源码/结果/阈值不改；P1/P3/C3/freshness/physical_release继续false。

## 生产 bank 的确定规则

每个parent使用该parent已付并完整认证的原raw-zero signal：h由原choose给出，只能1或3；J为192×(10h)，nominal192，Q=identity192。normalizer已在AD图内，不再缩放J。不得借root J或另一parent J。bank源须绑定完整signal mapping/原数组指针、parent/root/goal/Q/normalizer/action domain及源码SHA。

NumPy float64对Q@J执行一次full_matrices=False SVD；每个right vector按绝对值最大坐标（相等取最先索引）规范符号，使该坐标非负。取前两个right vector，两个半径固定sqrt(5)*0.25与sqrt(5)*0.75；顺序radius→mode0/1→sign(-1,+1)，共8个logits10h。float64 tanh→一次float32 cast→reshape(5h,2)作为原raw edge。无随机seed、无动作优化、无结果择方向；原始right vectors与singular values全部保存。

h=1时对应旧spectral_bank去除zero/root_dls_progress的八个方向，须逐位fixture核验。h=3是明确新增30维推广，不调用旧只接受2/10维函数假称支持；两h保持同一个总logit范数半径，不隐含sqrt(h)放大。原L2半径/每packet强度不同，不宣称跨h等作用。额外zero单独raw全零，独立付费；root-DLS不进入本R八候选。生成器不计算未使用的root-DLS或默默收费为已计算。

## 资格与树接线

沿已冻结R kernel合同，每条edge长5h，完整9项（zero+8）postcontext各自H1 nominal/J、原bounded_ridge damping=.002/cap=sqrt(5)*.75、float64 tanh→f32 suffix5 lookahead；suffix仅诊断、不拼进可执行edge。immediate>zero.immediate且reach<zero.reach为严格资格，无epsilon。全部8行保留；全体eligible edge交原modelQ beam规则，不能只挑最低reach当完整beam。kernel的(reach,index) winner只作诊断。无eligible时停该分支no_witness，不退回progress，不选择zero冒R。

后续orchestrator独占新schema，保留原choose/width/depth、144槽与原资源上限；若预算不足，先对整组九项和最终root whole/segmented验证reserve原子拒绝，不只运行部分有利候选。bank生成SVD单列真实费用，postcontext账与native账同一计算不重复相加。原nominal boundary直接恢复完整context，不额外重放；精度/模型/来源/加载bytes/完整IO/失败不退款按现有MAIN闭包。实际费用不足、资格为空或runtime失败不同类记录，不写成机制负结果。

本次批准的是生成器与接线实现，未批准worker生产CUDA；MAIN须审读/测试/prepare后才跑主机CUDA，绝不CPU神经fallback。新原生runner未实现前，不得从RAM fixture或本合同签全policy执行。
