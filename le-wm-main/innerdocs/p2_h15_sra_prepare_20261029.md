# H15 无组合谱候选：saved-only 准备

本入口只准备两条已知根，不执行模型或物理。输入是原 H15 physical prepare 和已有 strict receipt 的固定 SHA；逐槽核对任务、原生起点和目标、同根完整 latent/action 上下文及 warm10，再读取两份原任务 gzip，核对实际文件 SHA、task ID、完整上下文、目标和动作归一化参数。receipt 仅作为既有执行链的绑定，未重新执行 upstream fullstrict。两根均已观测结果，非预注册的新根或独立确认。

六槽 joint 本身已有 root-compiled 全 H15、无后缀组合的迭代对照；其资源和早停按原记录处理。这里提出的是不同的单次联合谱候选变体，并非声称历史 H2 SRA 与 H15 joint 完全相同，更不重复执行现有 joint。历史 H2 SRA 四根不同、目标不同且 10 步，不能充当本组 H15 配对。

数组内核固定 192×30 Jacobian、零候选、3 个 bounded-ridge scale、9 谱方向×2 半径×2 符号，共 40 个 15 步 raw 候选；单个 30 维 logit 球半径 `0.75*sqrt(15)`，用 H15 已有范围。仅属 H2 方法的 H15 适配设计，不是原 H2 参数的逐字复现；退化奇异值的 LAPACK 基底与运行环境需要记录。目标为同一最终 identity192 Q，按完整 H15 候选稳定排序；任何候选失败整槽不可交付，不补样。计划每根 nominal/Jacobian 6、40 个完整候选 120、整段及分段重放 6，共 132 predictor samples；真实 instrumentation、复算和异常计费尚待实现，这些数字不是实测或完整算力。

工作目录 `le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_sra_prepare_20261029 prepare --output results/p2_h15_sra_prepare_20261029_NEW.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_h15_sra_prepare_20261029 verify --output results/p2_h15_sra_prepare_20261029_MAIN02.json --sha256 f583f6ff1dc4db5614f20570a7f6498a77db6cc81a2123d6b29216b62914e556
```

输出独占创建，已存在即报错。MAIN01 是未加入 task 文件实际重读的早期产物，保留不覆盖；MAIN02 为当前版本。准备未建立新的时间证明，时间戳或内容 SHA 不代表研究者未见结果。

阻塞：尚无本变体的 H15 HOST CUDA runner、完整资源计量/可恢复失败账本、相同 native setup 的物理 adapter、独立 saved consumer；需要验证已绑定模型/normalizer 和环境完整 source closure、明确预算及独立授权。已有全 H15 joint 可先作为限定性的无组合对照，但两根、资源不等、FLOPs 未知及 root1 第14步停止均限制解释；不补第15步、不宣称公平或 P2 成立。`ready_to_execute=false` 始终保留。
