# P5 四词接触源 response：纯 saved 分析

本模块只消费 MAIN 已完成的四词新物理与四词模型产物，不执行模型、物理或网络，不生成新源、不挑词、不拟合 latent→physics。旧 push_x/zero 参考结果不代替本次四词结果。

## 固定合同与解释边界

固定顺序 xy、yx、x_minus_x、zero_zero；每词保留 prefix frame10 基线和全部 suffix frame11–20。输出4基线、40逐帧行、20配对行（XY−YX、X−X−zero 各10行），以及2个描述性摘要。中途 block 差异不因终点相同而丢弃。缺词、partial、失败、帧缺失、SHA或严格回执不匹配均拒绝，不补零。

逐帧保留 agent position/velocity 与 block position/velocity/原始 angle/angular_velocity，来自完整 native body doubles；不是用 state7 最后两维冒充 block velocity。保留 coverage/reward/done（绑定 native done=False），goal、原始 command、累计 raw u、实际 callback 事件与 body pair，以及原 uint8 RGB 差异。相减前转有符号类型；RGB MSE 是像素强度平方单位，不能与位置/速度/角度/模型 feature MSE 相加。原始角差和主值弧度差分别保留。

callback 数为实际 callback entries，points 为实际累计 points，均非独立接触事件数。原始 impulse/points 原样保留，不换算力/功。完整 RGB 差异的物体归因 unknown；agent 惯性/位置差不能证明 block 非交换机制。模型报告仅终点 feature contrasts 与 meter 并栏，不冒充每帧物理预测。物理 meter、operation_meter、wall_seconds 与模型各轴分别输出，不称等总算力/FLOPs。

单个构造 OOD source 的四个配对 word 不是四个独立 root；不做独立样本推断。coverage 与 done 分离，task_success=null，scientific_admission=false，object_noncommutativity_claim=false，physical_release=false。分析不开放 P5 科学准入。

## 实际权威与消费接线

固定物理 manifest SHA `46f8cf63d045123edfa61b257ad11bd94b3741b47dfe2244463c5c5096df0ab7`；report SHA `a07c2583c5d5dd553324d0e5c0b740ffbde3fa7a806798b3a8dcb0fe77ab6633`。

MAIN strict wrapper `experiments/spectral_reachability/consume_contact_words_MAIN_20261028_v1.py` SHA `8bb51bddb3b2836d43ce5a51e0162b05d3f98ad2aa7f6cb4de23739c8b4fe7d5`，ABI `consume(manifest_path, manifest_sha, report_path, report_sha)`。先检查已存在的实际 strict receipt（SHA `f6cd1f53d7537a6292c49647a3e20cfd61caf8a4b5e5e9d395223fc5845a39ee`），再导入并重新调用 wrapper，要求完整返回值与 receipt 精确相等。wrapper 使用封存 payload cap 范围内的96MiB有界 journal reader；原8MiB reader 阻断是工程限制，不是科学负结果。本模块不修改任何原 module/global。

随后验证物理 manifest authority 中实际模型 manifest/report/receipt 的 SHA 与纯 saved numeric consume 返回，要求同 public payload、四词完整 raw10 plan 精确一致。模型 report SHA `4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f`，receipt SHA `247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3`。模型与原物理/helper 源码 SHA 写入源码 PINS；只复用旧分析器纯 quantities/difference/rgb_difference，不调用旧 analyze 或借用旧结果。

输入 SHA、三个 helper/source SHA 及本三件套 SHA 写入输出，末尾再查 drift。JSON拒绝重复键、NaN、符号链接；NPY 禁止 pickle。输入JSON/输出各8MiB上限，输出独占 xb 写入并 fsync。大 journal 由 MAIN 严格 wrapper 的封存96MiB边界负责，不经本模块 JSON reader。

## MAIN 精确命令

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行；下列输出必须尚不存在。worker 未执行实际 analyze/verify，不声称已有分析结果。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/spectral_reachability/agent_p5_contact_word_response_analysis_20261028_v1.py analyze \
  --manifest results/agent_p5_contact_action_words_physical_20261028_v1_prepare_MAIN01.json \
  --manifest-sha256 46f8cf63d045123edfa61b257ad11bd94b3741b47dfe2244463c5c5096df0ab7 \
  --report results/agent_p5_contact_action_words_physical_20261028_v1_physics_MAIN01/report.json \
  --report-sha256 a07c2583c5d5dd553324d0e5c0b740ffbde3fa7a806798b3a8dcb0fe77ab6633 \
  --consumer experiments/spectral_reachability/consume_contact_words_MAIN_20261028_v1.py \
  --consumer-sha256 8bb51bddb3b2836d43ce5a51e0162b05d3f98ad2aa7f6cb4de23739c8b4fe7d5 \
  --receipt results/agent_p5_contact_action_words_physical_20261028_v1_consume_MAIN01.json \
  --receipt-sha256 f6cd1f53d7537a6292c49647a3e20cfd61caf8a4b5e5e9d395223fc5845a39ee \
  --output results/agent_p5_contact_word_response_analysis_20261028_v1_MAIN01.json
```

复算：同一命令将 `analyze` 替换为 `verify`，追加 `--output-sha256` 后接上述实际输出的 SHA；该 SHA 必须由 MAIN 实际生成，不预填。verify 全量重新消费并与已保存分析精确比较，不重跑神经/物理。

RAM 测试：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_contact_word_response_analysis_20261028_v1.py
```

测试是工程回归，不是科学证据。实际分析产物与实际 verify 尚待 MAIN 执行；本交付不改变任何既有负结果、科学门槛或封存产物。
