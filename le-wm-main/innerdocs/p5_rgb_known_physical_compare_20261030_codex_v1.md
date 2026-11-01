# P5 已知面板离线对照合同

仅允许已开放的 `agent_observed_raw_coordinate_physical_20261028_v1_consume_MAIN01.json`（SHA `8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb`）及固定 panel。开放来源记录于 `p5_rgb_causal_panel_bridge_20261030_codex_v1.md`；该面板有两个已知 root、零 fresh root。用户本次请求允许对这些已开放历史结果作对照，不允许读取 holdout 或保留根。

读取物理 receipt 前必须校验 MAIN02 指定 accept SHA、监督记录、预测完整 journal、前后快照、生产者和 checkpoint 来源；失败即停止。仅 CPU 数值运算，不进行模型推理或物理计算。

固定 164 枝（每条件 82，含各自两个 baseline）及 80 对（每条件 40）。逐枝五个预测对应 plan step 6–10；按 checkpoint 已保存均值、标准差将归一化输出还原，真实状态映射为 `[agent_x, agent_y, block_x, block_y, sin(block_angle), cos(block_angle)]`，映射依据 `experiments/run_causal_visual_pusht_world_model.py` 的 `observable_pose_vector`。每 horizon、每条件分别报告六坐标逐枝绝对误差均值，以及固定 plus-minus 配对响应六坐标绝对误差均值。前四维是模拟器位置坐标，后两维无量纲；不跨坐标混合，不筛除样本、不填补缺失；不除以扰动幅度，不计算角度或单一综合分数。配对方向和 branch identity 来自固定 panel。

这是已见面板的描述性核验，不是新 root 复现、科学通过、因果证明或公平跨模型排名。off/on 输入分布不同；该 RGB 输入模型与接收额外位置速度的 JEPA 观察权限不同，不能将后者称为 RGB-only。模型原训练涉及物理状态监督，运行时输入为 RGB 和动作。
