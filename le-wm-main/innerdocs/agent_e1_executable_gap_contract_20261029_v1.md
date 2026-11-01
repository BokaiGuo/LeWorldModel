# E1 executable-gap：现有证据接线与三臂比较合同

本NEW source/tests/doc只做离线prepare/verify，无CUDA、physics、训练、上游fullstrict或后代。不是新实验runner，不修改原gate或旧文件。source内CATALOG固定14份已存在JSON的完整SHA；输出所有实际读取文件的path/SHA和JSON pointer+value SHA。只读原known roots，无reserved outcome。目录/作者声明不代替本地实测。

## 先排重：哪些已经完成

原E1 `agent_subgoal_gap_contract/runner/assets_20261028_v1`已有三臂评分、callback/meter和metadata打包；本次不重写它们的评分/runner。`agent_p2_subgoal_gap_protocol_20261029_v1`已有H15六分支/18cp协议，不能把旧H10结果换成H15结果。本工具固定识别旧E1 known2、H2/raw10、四实际dispatch、六科学槽，不增加根数。

真实common-controller report `results/agent_experiment1_common_controller_20261028_v1_cuda_MAIN01/report.json` SHA bc3cd74c85f878cdfc7b170d0b5b555827e0dc4a59cdaea6fb5781e71f03767f；完整receipt b4b72dea9d782c6d2e1b93f50523a1089e7f3d894e2e2c1171fc13f9b8cd425d。四工程分支已576predict/API/AE/pred_proj，共同代码同首轮draw；不是未实现controller。

实际physical MAIN02 report a75d97bfd5e6c19bf942ce5510ad7689ad3fd79f7bb3539b2ca96a6cda9fec6c，完整receipt 3926373a79e5c0eb9a6295dadd4ce1403dfca13d9baad51639cd7012c913c12b。实际RGB report 3f64f2b3f74be98fdd0f0eda36fb0c309990ae42d046a71ce4ca8d7289d654c9，完整receipt 4d5d8e9b5014487e7f4a932fe19a92a5d3a011c4db874cb9cd5ca14b47f04dc7。新工具完整连接selected target→raw10→原生执行→cp5/10 RGB连续tracking；不再重跑或重算该八图评分。

此外，不能再说“无冻结learned资产/无计算配对/无history3 tracker”：Hi high-FIRST→low实际receipt 40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0；catalog五组tensor join 1523183330336bd3e66884a15fa96e7a15161abd8a343d1f50dba7a0427494b8；known2 bounded计算parity receipt 3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480；history3 normalized tracker实际receipt 080fe1173895621ea4c0565671509a267ba456135fedfac313ddac7894f98138。它们都是不同scope的真实增量，不换绑为旧E1 learned两个slot。catalog原computation=false与后来parity=true按时间/证据范围分开，不覆盖旧catalog。

## 三臂与比较边界

|臂|已有旧E1 H10证据|不可推出|
|---|---|---|
|arbitrary_latent_waypoint|固定20点latent搜索选target，同144-predict controller，实际raw10/八图中的四图|不是任意latent优化全集；proposal无动作来源不等于不可达|
|existing_learned_prior|旧六槽中两条unknown；Hi另有真实FIRST集成与normalized tracker|不得拿goal/random/compiled目标代learned；不能拿Hi宏步当native cp5|
|dynamics_compiled_witness|原beam的prefix5模型postcondition有原raw10/context见证；共同controller另付费追踪|模型见证不保证原生到达；原beam动作不是共同controller的privileged warmstart|

每root逐字段比较task/root/完整history来源/goal、saved模型hash、encoder预处理来源、normalizer、Q、raw域、时域/反馈、controller源及预算、信息权限、native evaluator。只有字段value一致才输出match；learned缺失为unknown；两个已知值不同即mismatch，不能被第三臂缺失掩盖。当前输出特别注明这是**saved合同字段比较**，不是读取模型权重/全source闭包后的loaded-byte认证。history用原task_export及SHA链接，不用“都是3frame”代替身份；in-memory tensor hash不冒称权重加载字节；动作map描述不冒称Hi原训练scaler。

所有六行保留。learned target/witness/execution/tracking/native_success均null，不填0。四条已执行分支分别保留proposal action witness、controller selected-plan witness、native实际执行、已有RGB连续tracking及native_success；native success不能替代latent arrival。没有事前阈值，arrival保持null；不按已读误差拟合阈值，不把MSE>0写成不可实现证明，不把低Q或单次执行成功写成能力改变/非贪心收益。

## 剩余最小条件与真正下一次实验

旧E1 H10校准链已经闭合，无需再花神经/物理费用包装重复证据。当前缺的是**同任务同时间同坐标的learned target交付**，不是所有learned模型都不存在。SAGE保存metadata仅声明generator/object及hash、未下载权重；H10会final-goal fallback，不能缩tau伪造learned。Hi的高层macro FIRST和H3 normalized tracker已有，但其原raw inverse/scaler authority及H10真实deliver语义未认证；不训练mapper、不fit normalizer、不仅cast192维。

可实施的最小新slice是：MAIN先封存选择同H10可用learned prior的合法receipt/源/实际权重加载证据、完整target交付及权限/时间语义；若只能用Hi/H50，则三臂统一另协议、重编全部targets/controller，不能只迁learned一臂。随后用已有同controller内核，新third-arm每root144predict，单追加低层预算288predict/API/AE/pred_proj，仅是数学费用说明而非当前运行授权；prior高层/编码/重放/资源仍需预先独立封顶，未知不填零。若为消除跨job基线疑虑事前选择全三臂新run，则六dispatch低层864predict，不能在读新结果后补两臂。

若真能沿旧H10相同native recipe追加两条learned部署，无早停运输/执行上界为40public/408physics/44PNG，四端点新encoder/projector各4；不是现在已发生的费用或允许默认跑的预算。原四条80public/816physics/88PNG、RGB8+8继续另列；所有constructor/reset/goal/render/capture仍须新runner按真实调用计量，不能只列predict而隐去额外过程。换H50或Hi新时域时这些上界全部失效，需重新审定。

真实物理比较之前还必须封存：（1）原reset/warm10/full-history guard；（2）不向compiled泄漏见证动作的统一target-only controller ABI；（3）相同raw域、一次normalize、完整final动作delivery/replay账；（4）六分支cp5/10全部端点和早停/missing规则；（5）结果盲的tracking/arrival判据；（6）原native成功与含速度distance分栏。目标先seal、全部预测先seal后标签，所有失败保分母。proposal旧beam成本与新learned成本不能混成等总算力。新独立根需另获真实cohort，不能把known2×多臂当独立样本。P1与科学门保持false。

本工具没有在这些缺项未齐时伪造生产run CLI。现成可运行的是下列离线prepare/verify，完整未来实验应复用已审内核并由MAIN另授权最小接线；不是要求重新实现旧全pipeline。

## MAIN命令及信任边界

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q tests/test_agent_e1_executable_gap_contract_20261029_v1.py
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_e1_executable_gap_contract_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e1_executable_gap_contract_20261029_v1_prepare_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_e1_executable_gap_contract_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e1_executable_gap_contract_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_NEW_PREPARE_SHA
```

每JSON/新输出≤16MiB，无目录遍历、权重反序列化、网络、旧consumer import或调用。metadata文件缺失明确missing并保全六槽；hash/身份/次序/源链冲突立即拒绝，不当科学负结果。原全PNG/journal/NN IO认证继承MAIN给定完整receipt，不重fullstrict。verify完整重建本离线contract，再核自身output外部SHA。stdout仅路径/SHA或小型验证状态。worker测试使用真实保存ABI只读副本和临时盘，未在results生成生产artifact。

最终独占21tests通过；与旧E1 scorer及P2 protocol联合84passed/4.45s。覆盖真实saved ABI全部六槽链、目标/动作/父report/任务/时刻/model交叉错接、compiled prefix target见证、learned缺项与零填拒绝、临时盘完整prepare→verify、missing与错误SHA、重封假native-success拒绝、known mismatch不被unknown掩盖。采用experiment-code的固定输入/来源/持久日志要求；用户限定saved-only，不采用训练、调参、生成实验图或改进结果建议。
