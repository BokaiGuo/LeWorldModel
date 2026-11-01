# Selected physical RGB v2 独立只读审查

## 审查对象与结论

审查者只读producer、tests、doc及相关封存helper源码；未执行prepare/run、模型、CUDA、物理或pytest，
未读取construction witness/certificate/sourcepack。仅新增本文。使用code-review-and-quality技能检查
正确性、输入边界、架构、错误记账和资源限制；不另派代理。

冻结三文件SHA256已现场核对：

- experiments/spectral_reachability/p1_constructive_selected_rgb_cuda_20261027_v2.py：`e1448739bbae827a2e3b1f62bf50f0f27aa06d47aef60163483fa289e6eb432e`
- tests/test_p1_constructive_selected_rgb_cuda_20261027_v2.py：`5bb5a89d9f81b845e17b801dbb11b814b8a2560cda8d2252ef5eb6923fc6ef7f`
- innerdocs/p1_constructive_selected_rgb_cuda_20261027_v2.md：`95e3f36b3018a06def1b470984102fd8e77614df0d17b236bdc7dc586f4d31ad`

**未发现阻断MAIN对matched2branches／feedback4branches执行“真实RGB编码及actual-Q诊断”的确定性问题。**
这不是实际prepare通过、CUDA完成或科学准入确认。MAIN仍须批准明确physical/task路径及完整SHA，
检查prepare/verify及实际receipt。预测reference桥接未实现，详见下节；若交付要求包含feedback预测误差，
这一部分尚未完成，不能用实际Q代替。

## 必须保留的预测语义边界

producer第93–97、144–149、164–169行明确`predicted=None`，输出predicted_Q、predicted_vs_real_mse、
Q_improvement_sign_reversed均null、prediction_status=unknown_not_recomputed。
因此本版**不会**把旧open-loop预测拼成feedback新预测，亦不会把缺失误差填0；但它也没有提供可消费的
prefix/terminal预测reference或JSON pointer。已有compiler报告和完整artifact SHA被保留在bindings，
只是证据闭包，不等于已选择、解释和验证了预测参考。

后续NEW saved consumer若需误差，必须显式绑定compiler报告、artifact、pointer及task/selection：

1. matched GN：使用本次matched最终被选中完整10步轨迹的prefix和terminal预测，不能取旧GN对应值。
2. feedback step5：确认本次前5动作与哪条冻结prefix预测对应；只有命令、task、history一致才可作为参考。
3. feedback step10：使用本次observed-prefix context下重新编译且实际执行的suffix预测，不能取原open-loop
   terminal，也不能把两种context的预测简单拼成同一rollout。
4. 每项标明prediction origin、conditioning（imagined或observed）、horizon、原history/actual checkpoint身份。
   缺任何必要参考就保持unknown；不得为补误差新跑模型或拟合Q。

这是下游预测误差结论的必要条件，不阻止本版单独保存真实native192 latent。既有v1诊断器固定schema与
四臂join，不能直接把v2 receipt冒充v1交入；需要NEW consumer/bridge，不改旧冻结分析器。

## 按要求逐项核查

| 项目 | 源码证据与判断 |
| --- | --- |
| 任意固定arms | 第25–29行允许1–4个唯一安全名称；第48–50、121行要求root-major×arms与manifest一致，不按结果选臂。不是任意root数。 |
| matched2 / feedback4 | 一arm时两根×两checkpoint=4图；两arm时8图。第32–34、59行按分支数重算物理账本；第145–146行按arms计算实际编码预算。 |
| source_slot | 第65行task=tasks[root_slot]，不是tasks[branch]；第109–114行外部批准task严格slot0/1，blind metadata与固定批准源一致；physical.selection逐pointer核对compiler slot/arm/blind及commands。 |
| 原task/goal/Q | load_export验证外部SHA、task ID/contract、模型/normalizer和无target合同；第113行identity192；第83–86行actual attached和每条returned goal RGB与task目标像素SHA一致，目标latent直接取批准task而非重新编码。 |
| root身份 | 第78–81行复算root+goal float64 SHA，并对齐physical source_slots；不把hidden state送入encoder。未读取sourcepack重认证物理源科学有效性。 |
| physical链 | 第116–135行report外部SHA→execution binding→prepare→manifest→compiler pointer；journal全SHA/hashchain，run_started内prepare精确匹配，terminal counts/errors一致。 |
| 自身PNG | 第75–92行10条自身命令和fresh checkpoint/guard绑定；第136–138行限定同目录平面PNG，重算file/pixel SHA、RGB224，不做颜色转换；checkpoint核state/observation/snapshot一致。 |
| 无future替换 | 完整0/5/10 checkpoint、full10、无partial/早停替换；本轮只取实际新physical report自身PNG，不从旧future帧读取。 |
| CUDA接口 | 第201行以后MAIN-only cuda:0，既有exact image transform，MATH SDP／TF32关闭，float32 native192；callback仅接RGB，encode→existing projector。实际device/shape由guard记账约束。 |
| 禁用调用 | 继承enc.guard，封禁predict、predictor、action_encoder、pred_proj和AD API；每图顺序encoder/projector，未记账callback拒绝，无dedup/重试。仅源码审查，不是本轮运行计数。 |
| 小输出 | Store继承v1，NEW目录、总cap2MiB含partial/final，reserve200MiB；prepare<256KiB。未证明在所有其他进程并发写盘下绝对保持floor；实际逐次写前检查。 |

## 信任边界与非阻断限制

**FYI：compiler task身份不是通用pointer合同的直接核验项。** physical.selection的四类pointer核对
commands/source_slot/blind/arm，不读取compiler task_id/goal/context pointer。v2虽然验证外部批准task和
实际goal RGB，仍依赖MAIN批准“该compiler确实使用这份原task”的既有compiler/source审计。
仅凭blind相同并不能在一般情形证明goal latent、normalizer或history完全相同。
对本次固定原task、MAIN已审阅compiler的工作流，未发现实际错配证据；不能把此评价扩大为任意未知compiler
的完整goal/context审计。未来预测bridge必须增加相应task ID/SHA与context绑定。

**FYI：guard核验的范围有限。** 本工具验证selected step5/10 guard accepted与PNG/checkpoint身份，
加上物理journal计数，不独立重演所有逐步guard逻辑或重算official success。它是编码producer，不是
物理科学审计器；native success由独立physical saved审计确认，不从Q推断成功。

## Failure accounting审查

- run初始verify/Store发生于try之前：若源不符、目录已存在或disk预检拒绝，直接报错；无模型执行，亦不承诺生成partial report。
- 模型加载至编码主体受BaseException捕获；首执行异常保留，journal异常追加，不覆盖首错。
- module_attempt在实际调用前持久化，return在同步后写入；失败不继续下一图，已写NPY不删除，固定rows保留not_started/failed_partial。
- finally重验model tensor和source，任一错误使complete=false；complete还要求所有行encoded、全预算attempt/completed相等、无blocked调用。
- 若最终磁盘或I/O连report都不可写，只能保留已写证据并stdout报错，不能称“完整durable错误报告”。此边界与cap/floor一致，不是自动补假成功。

## 测试审阅与执行边界

测试源码覆盖1–4 arms、matched/feedback通用pointer build、严格分母/顺序、partial/checkpoint/slot/goal/selection/
guard/transition/snapshot/API count拒绝、原始NPY保存、编码记账、无dedup/重试及不读privileged源。
mock的两根主要复用同一基础数值，能测slot标签接线，不能独自证明真实两根未交换；实际prepare应检查两份
task SHA、source_slot和每根goal binding。本文只审阅，**未独立重跑作者声明的31项测试**。

MAIN实际prepare/run及receipt决定工程完成状态；即使真实编码完成，也只补充actual-Q/latent证据，
不改变四臂物理无官方success的既有事实，不建立非贪心、能力改变、独立确认或P1/P2通过。
