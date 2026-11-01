# Teacher共同库RGB协议：实际拒绝原因与物理标签更正

补充已冻结`agent_p1_teacher_cross_library_rgb_protocol_20261028_v1.md`，不改旧协议、producer、结果、guard或阈值。后续RGB实现须同时绑定本更正。仅saved metadata核对，无新模型/物理或重复严格消费。

权威证据：`results/audit_teacher_cross_physical_MAIN_20261028_v1_MAIN02.json`，实际只读SHA256核对为`3dfc301c40471e0620caf8660baf4d8ad6534f25055e546aadc7466f61e79b78`。对应physical report SHA仍为`464656755bb83cec14b7acdd457641592b11be6d335f285166687df15368bc01`。

1. index12末次off incident `b12_suffix_05_0.png` **仍有8个canonical blue pixels**（边缘残留）。实际拒绝原因是agent_y=`514.7195833901377`超出512 canvas；不是蓝像素计数为0。后续missing字段应包含`guard_failure=canvas_out_of_bounds`、`canonical_blue_pixels=8`，而不能标注`agent_pixels_absent`。像素存在不能绕过canvas guard。原协议仅写canvas拒绝，并未声称蓝像素为零；任何此前口头“无蓝”描述不作为结论。
2. 40行均有最后suffix5真实控制返回，native success为**0/40**。这可作为全部40实际最后返回的native指标，但不代表完整40有效RGB或latent评分；index12缺final-on与checkpoint10仍须unknown/null，不用partial state填图。
3. **31/40行**root之后block pose发生改变，**800次**控制返回中**161次**`n_contacts`标签非零。因此不能声称全程无接触或物体静止。这里是返回标签计数，不是161个独立碰撞、接触持续时长或整轨迹连续接触证据；也不由物体位姿改变反推接触机制/任务成功。完整PushT成功含agent+object位置和角度，不以object-only变化代替成功。
4. 物理full40动作返回与RGB完成是两条账：39行`horizon_completed`、1行guard failure、总计2638实际render。index12同condition final-on不存在，禁止拿off incident、旧帧、预测或重新渲染替代。固定40分母、`full40_ready=false`、`full40_released=false`保持不变。若MAIN另行明确授权partial39编码，仍需40槽/null和独占新schema，不包装成完整40可发布。

此更正不改变原动作、库、根、科学阈值、神经预算或释放规则。known roots仅2、无独立新根认证，P1未准入。
