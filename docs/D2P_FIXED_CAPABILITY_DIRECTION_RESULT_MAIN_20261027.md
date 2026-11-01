# 固定探针方向预测：主机CUDA与完整任务轴复核

## 结果与解释边界

预声明`D2P_FIXED_CAPABILITY_DIRECTION_PROTOCOL_MAIN_20261027.md`已落实为实际主机CUDA：固定40图像编码、80次teacher/open H1预测，14.441633秒，冻结模型不变、errors=[]、无AD/训练/新物理/CPU神经fallback。全部320条模块输入输出、完整257token hidden/CLS/projector、完整三帧预测和真实PNG身份经另一个saved-only消费者核验通过。

**真实prefix替换模型自身prefix预测后，八个锚点的预测MSE全部下降。** 这支持前缀预测漂移是当前误差的重要来源，不支持“冻结模型完全没有短时响应能力”，也不是全部代码或训练合同都已排除。teacher仍有误差和排序反转；原生成功仍0/40。本批只有两个已知校准根，不是新根泛化或独立成功率。

## 证据与封存

路径以下相对`/home/gbk/code/LeWorldModel/le-wm-main/`。

| 产物 | SHA256 |
| --- | --- |
| `results/moving_capability_direction_cuda_MAIN_20261027_v1_prepare_MAIN01.json` | `c76a5c8b50aa47887cdb4e354f182a5700b64aa89068b1a7fcb9ba830ea41dab` |
| `results/moving_capability_direction_cuda_MAIN_20261027_v1_MAIN01/report.json` | `eac431fca369c977c57d38a32a600d1ef02ae2030b4112636d4c3df92bf143fe` |
| `results/consume_capability_direction_MAIN_20261027_v1_MAIN01.json` | `f564c6771c1940dc3f6925999647569670a9dbbb4381bb6da5a37a5636a0c5bd` |
| `results/audit_capability_native_goal_MAIN_20261027_v1_MAIN01.json` | `ee0fdadbd99de36681bd3dbf93a4cbcbd77b4e69dbe7b8a73b35a93d1bf39bc2` |

执行器source SHA `8fb7322b95832c78c672cbd87f33e547ce43f4a10a67cdb7800a88cdc879bb10`、tests `704ee88c456fbb0b68df14787c92774816de6709f7bfee285531387de24e9211`、innerdoc `f291f94daeda9b060d6e8fa324a57ecb3f627f2e31e842dc5292c43595af25df`。实际prepare与verify通过后才运行CUDA；三文件及协议/全部输入均已绑定，不再编辑。

原生模块实际完成：encoder/projector各40，action_encoder/predictor/pred_proj各80，合320条；每次预测batch=1，API80、预测samples80。原28/16图像编码为历史支付成本，复用不意味着全管线成本为零。新runner保存原PNG/pixels/预处理/全hidden及每个模块全部实际输入输出；无预测成本减免、去重或失败重试。全部预测封存后另行评分，无新动作选择。

消费者不导入模型或执行器，直接检查源文件SHA、artifact完整清单、320次attempt-return、40行row seal、每帧257hidden至CLS的精确桥接、AE输出至predictor输入、predictor输出至pred_proj输入及完整预测输出；再复算所有Q/MSE/cross项、固定pairwise和zero-relative比较。接口测试10项、消费者7项、完整任务轴5项，实际共22项通过。测试是工程证据，不替代真实CUDA或物理。

## 八锚点完整描述（不是独立统计样本）

每锚点5个固定探针，两两比较10对。Q为原生mean192目标距离；全部相等、反转与负数保留，没有事后新阈值。MSE对本锚点5个真实endpoint编码求均值。

| root | condition | arm | open MSE | teacher MSE | open排序一致/10 | teacher排序一致/10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | off | GN | .06628464 | .03309545 | 7 | 9 |
| 0 | off | D2P | .46254645 | .02647900 | 9 | 10 |
| 0 | on | GN | .03562096 | .01058929 | 8 | 9 |
| 0 | on | D2P | .32023852 | .01502058 | 7 | 10 |
| 1 | off | GN | .06401650 | .01716425 | 9 | 10 |
| 1 | off | D2P | .22862358 | .02083542 | 6 | 7 |
| 1 | on | GN | .03390591 | .00939906 | 9 | 9 |
| 1 | on | D2P | .10535648 | .02307441 | 7 | 8 |

汇总描述：open MSE均值.16457412922、teacher .01945718330；pairwise sign一致62/80→72/80；相对zero的四非zero探针一致24/32→29/32。80对比较共享端点，不能称80独立样本，更不能据此宣布显著控制优势。

## 纠正“物体方向没有改善”等于任务失败的解读

安装的PushT `env.py:347`原生成功要求`norm(goal[:4]-state[:4])<20`及原生angle差`<pi/9`；前四维同时包含agent和object。reward使用state7距离，success却不含速度。之前probe表的object-only方向诊断不是完整官方任务指标。

MAIN因此逐40分支按原goal与原始returned state另行复算联合位置、agent/object位置、角度、速度、原生与wrapped成功，并逐项核原terminated标志。结果：在32非zero-vs-zero比较上，真实Q改善方向与联合位置改善方向一致30/32；teacher预测与联合位置一致29/32，open24/32。0/40原生成功保持，不改标签或阈值。

例如root0/off D2P的+x与−y没有推动物体，但联合位置分别改善8.78555与14.03679px，真实Q亦下降，不能把它们叫作“物理毫无收益”。相反，root1/off −x改善物体方向却让联合位置误差增大6.16092px，说明单一物体轴也可能误导。所有五probe仍完整输出，没有挑最有利的方向执行。

原生raw-angle公式可能给出负angle term；本分析如实保留并另报wrapped诊断，不在本批修改环境成功定义。这是历史协议边界，不将其隐藏或用修正标签回填通过。

## 下一步与剩余闭环

可检验的修复方向是实际观测后的上下文刷新/反馈重编译，以及受控步幅下的任务一致选择，而不是只靠增强谱norm或改margin。此轮没有执行新优化动作，**不是反馈控制修复成功**。反馈方案需新协议、原强基线对照、实际任务收益及fresh确认；已有独立新根支线继续解析历史来源，P2/P3支线并行实现可放行队列。主线程继续审计和统一主机CUDA调度，原C1/C2/C3与三实验族未完成，goal保持active。
