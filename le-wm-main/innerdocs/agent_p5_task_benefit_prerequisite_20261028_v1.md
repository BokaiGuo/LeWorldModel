# P5 原生任务指标校准与下一决策收益硬前置

仅新增本stem三件套，纯saved，无环境/模型runner，不修改冻结资产。有限检索未找到同类P5任务收益实现；复用既有goal-scene与四词response strict分析，不再生成原始trajectory/RGB/pose差异报告。

## 已闭合的原生合同

`external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py:495–505`计算intersection_area/goal_area作为coverage，reward=clip(coverage/.95,0,1)，行为done=False；coverage达阈值的注释不构成done。源码SHA固定为`6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723`。

`p5_native_goal_scene_20261027_v1`已有原生独立goal资产，原生目标[256,256,pi/4]，不另造goal。既有四词physics/response strict完整验过goal、公域goalRGB来源与native reward合同。本模块重新调用已封存response analyze，完整重算MAIN96MiB薄桥strict和模型strict，再与实际保存analysis精确比较。原source/private source witness不传planner；本模块是离线评估器，不产生planner payload。

## 固定任务指标（不是事后挑选胜词）

固定xy/yx/x_minus_x/zero_zero顺序，全部4分母、frame11–20全部40行，原frame10基线。主指标为terminal coverage−own prefix coverage，越大越好；副指标保留terminal coverage、十帧coverage算术均值、十帧相对prefix均值、终点native clipped reward。所有词完整保留，不输出argmax、selected word或best-of-four。固定XY−YX与cancel−zero配对差；不按结果更换比较组。

coverage为目标几何面积分数，不是物体位移、agent惯性、contact数、RGB或feature MSE。reward饱和仍保留未饱和coverage。不给coverage增加研究成功门，不定义coverage>.95为done。task_success仍null；不同单位不相加。模型feature contrasts与所有资源轴只并列，非goal cost、非等FLOPs。

重要时间边界：本指标合同在已有结果之后实现，旧四词是**回顾性OOD校准**，不能追认成预注册实验。单构造OOD根上四个配对词不是四个独立根，不做独立样本推断。即便某词coverage增益为正、物体确实不同，也不证明C2或D2P决策收益。

## 真正下一任务收益试验仍缺的硬输入

1. MAIN在新trial outcome出现前封存的goal-conditioned决策规则，只能用原public context+goal；不得读private pose/contact/suffix选择动作。
2. 合法冻结模型goal-scoring原生接口，精确checkpoint/预处理/normalizer/action ABI配对。当前四词报告仅有pair feature MSE，不能把它当到goal的代价。本模块不另拟合readout、teacher、adapter或repair。
3. 预先固定比较策略及确定性tie规则，共同四word/action horizon，分别申明encoder/predictor/derivative/physics/wall等资源cap，不宣称等总算力。
4. MAIN在执行前批准的新trial声明。可以声明单已知OOD根校准限制，但不能把重复运行改称fresh，不得从旧结果选根、词或门槛。

输出engineering_ready只表示已保存task指标可严格重算；prospective_trial_ready=false、C2_established=false、scientific_admission=false固定，所有缺项逐项输出。没有虚构决策接口或空runner。当前可直接执行下面saved任务指标准备；新的科学trial执行受上述输入硬阻塞。

## MAIN调用

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，worker只RAM测试与--help，未实际prepare/verify。输出exclusive create，不覆盖旧文件。沿用封存response JSON reader/write_new的8MiB限额、拒绝重复键/非有限值/符号链接、fsync；大journal仅由MAIN原96MiB严格薄桥消费，无原module/global修改。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_task_benefit_prerequisite_20261028_v1.py

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/spectral_reachability/agent_p5_task_benefit_prerequisite_20261028_v1.py prepare \
  --analysis results/agent_p5_contact_word_response_analysis_20261028_v1_MAIN01.json \
  --analysis-sha256 f38b62550c347b66e5533885694a53db22ae1b58798d18de3dc11f6f1c3edf2c \
  --output results/agent_p5_task_benefit_prerequisite_20261028_v1_MAIN01.json
```

verify：同命令替换prepare为verify，并追加`--output-sha256`与MAIN实际输出FILE SHA。不预造输出hash。源/tests/doc SHA写入输出，末尾复查drift。RAM fixtures不是真实任务收益证据。
