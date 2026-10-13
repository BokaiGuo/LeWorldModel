# 数据字典：给第一次接触这个项目的合作者

这份文件解释每个结果文件“是什么、每一列/每个字段代表什么、它能支持哪句话”。这里的“数据”包括理论模拟输出和 learned-model 测量输出。

## 1. 理论模拟数据

理论部分不是从真实机器人采集的轨迹，而是一个完全可控的矩阵实验。我们先构造 latent covariance `Σ` 和 predictor spectrum `Λ`，再用旋转矩阵 `Rθ` 得到：

```text
Pθ = Rθ Λ Rθᵀ
```

改变 `θ` 时，`Pθ` 的特征值完全不变，改变的只有 predictor eigenspace 相对于 `Σ` 的方向。

关键量：

- `commutator`：`||ΣP - PΣ||` 的归一化版本。为 0 表示两个算子可在同一基下对角化；越大表示 covariance 和 predictor 的作用顺序越依赖方向关系。
- `top_band_lift`：预测后能量落在 latent covariance 的 top-k eigenspace 中的比例，相对于均匀随机方向的归一化值。
- `residual_growth`：长 horizon residual trace 相对短 horizon residual trace 的增长。
- `stable_rank`：谱平坦程度的摘要；越大通常表示谱越平，越小表示能量越集中。
- `null`：保持 predictor spectrum 不变、但改变 basis 的参考分布。它不是“另一个模型性能”，而是回答“只要 spectrum 一样，随机 basis 是否也会产生同样现象”。

## 2. LeWorldModel PushT frozen latent

这是已训练 LeWorldModel 的离线 latent 记录。它包含冻结 encoder 产生的 latent、相邻状态关系和 action-conditioned prediction 相关量。这里不重新训练模型，也不做在线控制；目的是测量真实 learned latent 中是否存在 covariance-relative structure。

主要字段/结果：

- `dynamics_lift`：真实相邻 latent dynamics 在 covariance top band 中的能量集中。
- `predicted_dynamics_lift`：模型预测 dynamics 的同类测量。
- `prediction_residual_lift`：预测残差的 top-band 集中。
- `planner_weighted_lift`：按 planner 权重加权后的 top-band 集中。
- `transition_overlap`：真实 transition displacement 与 covariance top eigenspace 的重叠。

这些量与 randomized control 比较。大于 1 或高于 random-subspace mean，表示观察到的 displacement 更集中在 covariance 的高方差方向。

## 3. I-JEPA feature measurements

每个 benchmark 取 30 张平衡采样图片。对每张图片做水平翻转：

```text
x  ->  T(x)
f(x)  ->  f(T(x))
displacement = f(T(x)) - f(x)
```

字段含义：

- `overlap`：displacement top-16 子空间与 source-feature covariance top-16 子空间的重叠。
- `random_subspace`：随机 16 维子空间的平均重叠，作为基准。
- `lift`：displacement 在 source covariance top-16 中的能量提升。
- `random_lift`：范数匹配的随机方向得到的提升，理想基准接近 1。
- `source_rank`、`displacement_rank`：source feature 和 displacement 的 stable rank。

因此，`overlap >> random_subspace` 且 `lift >> random_lift` 的含义是：view-induced feature change 不是各向同性噪声，而是相对于 source covariance 有结构的方向变化。

## 4. I-JEPA held-out operator rank sweep

这里进一步尝试从 source feature 拟合一个低秩 ridge operator，再在 held-out view pairs 上评估。

- `held_out_R2`：operator 对 held-out transformed feature 的预测能力。
- `commutator`：拟合 operator 与 source covariance 的 alignment-sensitive 关系。
- `null_mean`：对拟合 operator 做 spectrum-preserving orthogonal conjugation 后的平均 commutator。
- `predicted_lift`：拟合 operator 预测的 displacement 在 covariance top band 中的 lift。
- `shuffle_lift`：pair-shuffled displacement 的 lift。

这个表最重要的解读是：`held_out_R2` 很高，并不自动意味着 covariance-relative operator 已被识别。预测性和 operator-pair geometry 是两个不同 estimand。

## 5. JEPA-WM / DINO-WM temporal measurements

每个 temporal JSON 按 horizon 记录三类量：

- `latent_prediction_error`：模型预测 latent 与真实 latent 的 MSE。
- `covariance_operator_commutator`：用当前 latent covariance 和 fitted transition operator 计算的 commutator。
- `spectrum_preserving_null`：保持 fitted operator eigenvalues 不变、随机旋转 eigenspace 后的 null mean。

协议字段：

- `context_length`：送入模型的历史帧数。
- `frame_stride`：每个动作块对应的 simulator step 数。
- `horizons`：测量的多步预测长度，这里是 1、2、4。
- `holdout_episode_count`：实际用于评估的 episode 数量。
- `split_mode`：数据如何划分 train/holdout；`public_val_directory` 是目录直接提供的验证集，`code_reconstructed` 是根据发布 loader、seed 和数据顺序重建 membership。
- `normalization`：动作和 proprio 是否使用官方发布的全数据统计量。

跨模型结果的正确读法：DINO-WM PointMaze 在三个 horizon 都高于 null；第二个 DINO-WM PushT 只在 horizon 4 高于 null；JEPA-WM PointMaze 在 full-rank 和 rank-truncated 测量中都高于 null。这个组合说明测量协议可跨实现运行，同时 transfer response 会随环境和 horizon 改变。

## 6. 如何从字段回到论文句子

1. 先找论文中的对象：理论旋转、LeWM latent、I-JEPA view displacement，还是 temporal world-model rollout。
2. 再找对应结果目录，不要把不同目录的数字拼成一个表。
3. 读取 JSON 的 `protocol`，确认 context、horizon、split 和 normalization 一致。
4. 最后才把数字写进正文；任何新的数字都应同时更新来源说明和复现记录。
