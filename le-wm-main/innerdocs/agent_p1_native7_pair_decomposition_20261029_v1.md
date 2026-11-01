# P1 固定80对 native7平方精确分解

固定原40槽、8 anchors×10 pairs；逐槽读取原已封存 evaluation 的 returned state7 与原 physical goal7，再与已封存 margin 以 index、task、anchor、job、plan、native return event 身份接合。两个输入均 pin 完整 SHA；继承之前 saved closure，不重新运行上游 strict、神经或物理。每个二进制浮点数转换精确 Fraction，计算七轴 `(state-goal)^2`、两臂 i−j 和 `xy+raw_angle²+agent_velocity²`，恒等式精确为零；并检查 native distance 的原返回与精确平方符号一致。原 native success 独立原样保留，不通过距离推断。

速度指相对原 goal7 后两维的 agent velocity 误差，不是速度幅度或 object 速度；不将原非零目标设为零。`velocity_necessary_for_existing_reversal` 仅对原观测latent→native严格反向计算：数学上移除速度平方项后是否不再严格反向；`velocity_alone_opposes_latent` 是速度项自身符号是否与 latent 排序相反。前者平局也算不再严格反向；其他情形 null。它们均是后验代数反事实，不是可实行干预、唯一原因或对原生 reward 的替换。latent缺失的4对仍有物理轴，但排序依赖项为 null。

完整两已见根和所有80对是相关设计，非80独立样本、非前瞻确认；原科学gate、目标、动作不变。只读分析及验证，不进行新模型、physics、训练或源 fullstrict。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_native7_pair_decomposition_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_native7_pair_decomposition_20261029_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_native7_pair_decomposition_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_native7_pair_decomposition_20261029_v1 verify --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_native7_pair_decomposition_20261029_v1_MAIN01.json
```

analyze排他新建、fsync；verify重算整个输出字节，包含三件套 source closure；不覆盖旧产物。fixture不构成科研证据。
