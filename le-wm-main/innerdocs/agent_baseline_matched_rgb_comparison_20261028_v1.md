# 同身份的beam/GN/D2P/matched CEM/nominal-MPPI saved分栏

仅新增本stem source/tests/本文，旧producer/physical/RGB及共享主表不改。MAIN审代码和运行；本模块只读saved，不调用模型/physics，不创建候选、训练、normalizer/Q拟合、调阈值、挑根或按beam优劣挑对手。

## 固定分母与实际来源

两原off moving known roots × 固定顺序五臂：`matched_domain_H2_beam20x6`、`line_search_GN`、`full10_conditional_D2P`、`CEM`、`kernel_faithful_mppi_nominal`，**全部10分母**。失败保留，不能删除任务失败或只报赢家；无跨论文作者值排名。

实际RGB报告在 `/home/gbk/code/LeWorldModel/le-wm-main/results/`：

|cohort|report相对路径|外部SHA|
|---|---|---|
|新beam|`agent_kinodynamic_matched_rgb_20261028_v1_cuda_MAIN01/report.json`|`52e6d7c025dcb836aed3ec28aefc9b249d449be2eb19618c8e6ebdae7ea4db49`|
|原GN/D2P|`p1_constructive_moving_history_rgb_cuda_20261027_v1_cuda_main01/report.json`|`7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71`|
|matched CEM/nominalMPPI|`moving_prediction_matched_sampling_rgb_cuda_20261027_v2_MAIN01/report.json`|`63c48c16b91d16b8f20c5f6ba08010d366544558f6e7eb9dd7a9542a5e50fa2f`|

不是旧不等预算sampling、旧beam16x4、zero-history、observed-feedback或overlay/on cohort。MAIN另已有beam strict receipt `results/consume_beam_rgb_MAIN_20261028_v1_MAIN01.json` SHA `9d4321544e5cdb1f4b83b83b20ad493820f25d3a41fdf03d32f8c1a8f2df487b`；该回执为额外MAIN证据，本模块仍直接consume原报告及其artifact，不以汇总回执替代原始闭包。

beam/sampling调用各自已冻结saved consumer、verify、全部encoding seal与NPY/PNG收据。GN/D2P旧adapter没有consume，复用其saved verify和已封存comparison v2的六事件RGB receipt验证：8NPY+binding/events共10artifact，逐项SHA、原prepare、实际8encoder/proj、全48事件和saved指标复算。额外核原GN物理全部88 fresh PNG像素/文件SHA。它的旧ledger没有保存fullhidden；不会谎称已像新版beam一样核完整hidden。这是保存证据强度差异，不补跑编码。

所有planner ownconsume用于saved原始选择/命令/资源验证，绝不运行其CUDA/search。实际raw10必须与原selected logits逐字tanh(float32)一致；beam从原bank+selected_path取logits，不用逆tanh推回球。

## 身份join与support边界

每row构造完整identity：sourcepack SHA、原source_slot、原task绝对路径/SHA/id、blind SHA、真实root+native goal7及root snapshot、原RGB3/goal pixelSHA、task完整3latent+两incoming normalized packets、原raw历史/normalizer、模型/config/tensorSHA、goal192、identityQ hash、H2/packet5/layout/with_targetFalse。不同identity不能联合：有效但不同的行保留在独立partition，单行partition `descriptive_joint_allowed=false`，不静默舍弃；无效/缺失原材料直接failclosed，无伪造null来减少10分母。

每partition只列其同身份臂，root-major/固定算法顺序，不按数值排序；`all_expected_same_identity`只有两partition且各五臂齐全才true。合法identity差异不是任务成败。原两known roots仍不是fresh独立确认。

共同数学admissible动作域为两个10维packet logit球、cap=.75*sqrt(5)，每packet五步二维raw，float32 tanh；normalizer沿原task一次subtract/divide，非零warmup前5[.1,0]/后5[0,.1]不变。**同admissible ball不等于相同搜索support/覆盖**：

- beam固定zero+19 seed solid-ball点、首20→beam6→120，有限动作库，非RRT。
- GN四次更新与固定line trials，局部AD引导。
- D2P固定10个conditional prefix/own-suffix候选及谱/AD账。
- CEM72完整H2、3×24、elite4、Gaussian更新。
- nominal-MPPI72完整H2、box参数扰动后packet投影，非完整官方MPPI。

beam源float64 logit点在tanh前转float32；GN/采样的projected proposals有float32 inward feasibility处理，D2P则原float64谱半径投影。原GN/D2P的`p1_conditional_task_admission_20261027_v1.check_prediction`沿封存`cap+1e-12`数值收据规则，beam/采样沿严格cap，**比较器不新加容差**。实际root0 D2P prefix norm为1.6770509831248435，cap1.6770509831248424，超出1.1102230246251565e-15；必须保留此差异，不能改原动作或说全部严格闭球内。每row保原norm/excess、validation_epsilon/来源、raw hash/full10；partition另列`all_selected_strictly_inside_closed_balls`及`identical_numeric_validation_bounds`。`same_admissible_packet_domain`仅指原声明数学域，不证明相同浮点验证域/搜索support。若数学动作合同不同，该字段false；所有差异分栏，不声称逐浮点support全等。

## 指标与资源

每row保0/5/10原PNG hash/pixelSHA、mean192原task-goal Q、相对原rootQ变化、原model终态Q及native7指标。prefix/terminal全部取现成真实编码；GN没有新root编码，step0明确标`original_task_root_not_new_RGB_encode`，仅在物理root PNG与原task observed10逐字身份闭合后沿原rootlatent使用；beam/sampling root取其actual saved RGB编码。不能冒充GN也额外编码root，不能把新编码goal默默替换原task目标。

native距离用完整state7（含velocity）norm；success沿原实现，仅agent+object前4维norm<20且原angle=min(abs(delta),2pi-abs(delta))<pi/9，无mod风险保留。object-only不达标不能替代完整task判断，velocity进入距离但无独立success阈值。初终态native距离/success与原physical报告核等，不能以Q改善判任务成功。

各row资源保留原key和实际attempted/completed，不改名字凑预算：beam/CEM/nominalMPPI每根144forward，GN78/D2P66并分别有4J/768数学VJP/4SVDsolve和11J/2112数学VJP/12SVD。GN/D2P旧report无单独predictor samples表时为null，不虚构新实测轴。`AD_API_resources`与数学J/VJP另列，不能相加为额外模型样本。原source8encoder/proj只继承一次；三个RGB cohort实际8/14/8与物理计账各自单列，不能把API和内部module相加，也不把这些不同采集批次说成同一成本。GN旧physical constructor/close未采用新meter，边界差异明确保留。

不做显著性、独立样本、等FLOPs/总算力或论文基线排名；全部负结果保留。输出是描述性saved分栏，`scientific_admission=false`、P1关闭，不产生新实验。

## CLI与验证

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。`build`只读核验并stdout摘要，不写结果；`run`只是保存JSON，不是CUDA任务。新报告<=2MiB、free reserve200MiB、独占NEW文件，不覆盖MAIN已有产物。`consume`依据外部实际SHA重建完整saved结果/闭包；不重新encode/预测/physics。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_baseline_matched_rgb_comparison_20261028_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_matched_rgb_comparison_20261028_v1 build
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_matched_rgb_comparison_20261028_v1 run --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baseline_matched_rgb_comparison_20261028_v1_MAIN01.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_baseline_matched_rgb_comparison_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_baseline_matched_rgb_comparison_20261028_v1_MAIN01.json --report-sha "$ACTUAL_COMPARISON_SHA"
```

本worker只RAM/mock与实际saved build smoke；不写MAIN输出，不运行神经/物理/网络。测试和实际build结果、三SHA随冻结交付报告；测试通过不是科学通过。缺/漂移任何绑定、错误source/task/PNG或不同动作见证都failclosed，不改封存求通过。

实际验证：**31 RAM tests passed in 0.34s**；实际saved `build`完整通过，739bindings、450199B canonical结果（只在内存，未写结果文件），10行保留，两个identity partitions各完整五臂，`all_expected_same_identity=true`。原生0成功/10失败全部保留；这是原记录汇合，不是新实验。共同声明数学动作球不意味着相同数值验证epsilon、相同搜索support或等AD/FLOPs。最终只需MAIN审读后运行上述saved run/consume，不需再运行任何神经或物理。
