# Hi-LeWM：实际文本取件、低层绑定与可执行接口合同

2026-10-27。新增独占 `results/p1_hilewm_text_contract_20261027_v1/`，旧结果/源码/总账均未修改。
按 scientific-critical-thinking 区分源码声明、checkpoint本体认证与科学效果：本轮取得**真实归档内文本**，不声称已验证权重。

## 1. 本轮实际解除什么

官方 Zenodo record21353240 的 package.zip，经先前完整414项中央目录绑定，精确读取33个文本成员。
24个首批、6个正式矩阵/依赖补充、3个动作合同补充，**全部33个CRC/大小/路径校验通过**。
累计66个官方 HTTP Range 请求（每文件30-byte localheader，然后精确name/extra/compressed body）：
**64,090 bytes收到，226,727 bytes展开**；低于压缩含header2MiB/展开8MiB/单文件1MiB。
所有下载内容仅保存 `.source.txt`，压缩字节以base64放在JSON收据中；未执行下载代码、未读取pickle、未下载权重、未运行神经或物理。

每次必须206且Content-Range总长2,299,863,924精确一致；200在read前关闭，不自动跟redirect。
localheader签名/method/flags/name及非descriptor CRC/size与中央目录一致；使用中央目录压缩长度限制读取。
解压采用有输出上限的raw DEFLATE，只接受完整流、无尾随/未消费数据，核对解压大小、CRC32与UTF-8。
拒绝绝对路径、反斜杠、空/点/父目录段、NUL、symlink、权重后缀；写入扁平文本名，独占xb。
网络重试或新目标不会隐含扩大限额。累计所有请求bytes记录于followthrough.json。

中央目录SHA256仍为 `75e0036bcb4014b30c297583fb1d96983cdcc3dbfa09a77fcf7f058e73eda67b`；
原始尾部和旧report SHA也在新report中强绑定。本轮每个范围的原始base64、SHA256、headers/status/range均可离线复算。
CRC32证明取件内部一致，不等于签名或已验证整包MD5。

## 2. 对应低层已定位，但文件/tensor配对hash确实未发布在已读合同中

实际 `package/checkpoints/pusht/main/config.yaml:55`：

```yaml
pretrained_low_level:
  enabled: true
  source_policy: lewm
  checkpoint:
    search_dir: null
    path: data/stablewm/pusht/lewm_object.ckpt
    selection_mode: explicit_path
    epoch: 1
```

`epoch:1`在explicit_path模式下不是自动找epoch1的证据，resolver优先显式path。
实际 `h_le_wm/checkpoint_registry.yaml` 对应 `baseline/pusht/lewm -> pusht/lewm_object.ckpt`。
实际 `h_le_wm/checkpoints.py:17` 的 `HF_BASELINE_SOURCES` 将该项指向
**https://huggingface.co/quentinll/lewm-pusht**。
`checkpoints/README.md:34` 明言flat基线两个checkpoint不在本包重分发；不能继续把它们当zip里未找到的隐藏文件。

源码fetch路径明确是该repo的 `config.json` + `weights.pt`，converter生成 `<cache>/<run-name>_object.ckpt`。
converter的HF URL未指定revision时用main；registry/config均没有SHA256/tensor hash或immutable HF revision。
因此已解除“哪个低层来源不明”，**尚不能认证哪一个历史版本就是训练epoch15 high所用低层**。
不借用SAGE registry的object hash来填这个缺项，它是另一资产发布链。

本地候选重新只读计算：

| 文件（共同前缀 `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/`） | SHA256 |
|---|---|
| weights.pt | `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
| config.json | `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09` |

本地config是tiny ViT14/224、projected CLS192、predictor3frame、actionencoder输入10、projector/pred_proj192。
这些架构与Hi config名义相容；**既未认证tensor相同，也不能因容器hash不同判定不兼容**。
目前无需为“可能已在本地”的flat权重再下载72MB；先做下节的内嵌低层认证。

## 3. 113MB对象不是纯high网络：最小取件应调整

`models/jepa.py:7` 的 `HiJEPA.__init__` 同时持有：

```text
encoder / projector
low_predictor / action_encoder / low_pred_proj
high_predictor / latent_action_encoder / macro_to_condition / high_pred_proj
```

`eval/hierarchical.py:526` 只加载**一个** `swm.policy.AutoCostModel(cfg.policy)`，to指定设备、eval、requires_grad_(False)，
同一个model交给高低两层solver；eval入口并没有再调用 `load_pretrained_low_level_model` 加载第二份flat权重。
`checkpoints/README.md:30`明确paper-facing workflow用object checkpoint，不是training weights文件。

因此最小候选是：

```text
package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt
  compressed 113,104,429 ; uncompressed 122,424,517 bytes
```

同目录config已取，无需再取。相邻222MB的 `..._weights.ckpt` 是另一训练权重格式，不能当“必需的低层配套”。
源码强烈支持object为共同高低层实例；本轮没有读取object，故**不能保证实际pickle字段完全符合源码**。
下一次MAIN应先授权选择性获取这个object并记录实测SHA/CRC，可信加载后检查上述属性与冻结状态，
对内嵌encoder/projector/low_predictor/action_encoder/low_pred_proj逐tensor语义hash。
将这些模块用于共同backbone的flat-vs-hierarchical新比较，可避免重新证明外部flat文件是历史配对；
若还要复用当前本地LeWM预测/端点，则必须完成内嵌模块与本地候选的tensor/config/预处理等价证书。
没有证书时不复用旧endpoint，也不另训练mapper。

准确地说：现在可先取得并认证**自带共同坐标的候选object**，而不是先声称某外部低层已配对再盲目下两份checkpoint。
如果MAIN坚持权重下载前必须已有作者发布的高低配对SHA，则该门仍无法由现有registry解除，需要作者manifest。

## 4. 正式模型配置与真正learned-subgoal接点

main config的模型合同：image224/patch14/tiny；history3，embed192；macro latent32；5个waypoint、random_sorted、
min_stride1、max_span15、frameskip5；macro encoder 2层/4heads/MLP768/max_seq_len15；
high/low predictor名义6层/16heads/MLP2048；train_low_level=false。
encoder、low predictor、action encoder、projector、low pred_proj均声明冻结，high pred_proj不冻结。
这些是训练配置声明，不是对实际checkpoint张量的现场检查。

**Hi-LeWM与SAGE不是同一个接口**：

1. `HiJEPA._encode_pixels_sequence` 用encoder的CLS `last_hidden_state[:,0]`再projector，输出 `[B,T,192]`。
2. `get_cost_high` 对候选32维macro-actions进行 `rollout_high`，以最终高层latent与goal的平方L2评分。
3. `HierarchicalWorldModelPolicy._plan_high:793` 调high_solver，取receding部分、reshape macro序列，真正调用
   `model.rollout_high(z_init,macro_seq)`，随后**取 `pred[:,0,0,:]` 作为 z_subgoal**（不是final goal重命名）。
4. `rollout_high(z_init,latent_actions)` 接受 `[B,D]` 或 `[B,S,D]` 与 `[B,S,T,32]`，输出 `[B,S,T,192]`；
   内部 `predict_high` 是macro_to_condition→high_predictor→high_pred_proj。
5. `get_cost_low`/`rollout_low` 让相同低层网络朝z_subgoal规划；low候选输入为已标准化的grouped动作，最终逐primitive逆标准化发给env。

这是真正调用learned high predictor生成中间latent的**优化式learned-subgoal baseline**，不是无优化的goal-conditioned神经generator。
goal通过high CEM目标进入；没有SAGE的 `(goal_offset_steps,subgoal_offset_steps)`神经forward参数，也没有rem<=tau必然fallback分支。
把它接为Experiment1 comparator须标Hi-LeWM hierarchical adaptation，不改称SAGE prior或官方直接条件采样器。
如原Experiment1严格要求“直接goal-conditioned生成网络”，则Hi-LeWM不能单独替代SAGE该臂，应保留两个不同估计对象。

下一CUDA-only smoke应保存：原RGB与共同编码、high candidates/costs、选中macro与high rollout、`pred[:,0,0,:]`、
low候选/选中normalized与rawcommands；分别计actual high/low predictor、macro encoder、image encoder/projector、replay次数。
冻结权重前后hash与每次attempt/partial结果也必须保留。只调用rollout_high但不把其输出交给低层评分，不算完成learned-subgoal接入。

## 5. 正式horizon：不能把SAGE H50门槛套在本实现

实际 `h_le_wm/experiments/matrix/pusht/hierarchical_matrix.csv` 共23行：

| goal offset（原始环境步） | eval_budget | 行数 | high CEM | low CEM |
|---:|---:|---:|---|---|
| 25 | 50 | 7 | 900samples×20steps/topk10 | 300×30/topk150 |
| 50 | 100 | 8 | 1500×40/topk10 | 900×30/topk150 |
| 75 | 150 | 8 | 1200×60/topk10 | 1200×30/topk150 |

这些是归档正式first-class matrix配置，**不是本轮复现过的成功率**。
matrix seed42/num_eval50；不同high/low horizon、receding及hiStaged是预列多个配置，不能根据未来结果挑一个称官方最优。
low action_block=5，所以low horizon2/3/5分别预测10/15/25原始commands；high action_block=1，high horizon1/2/3计macro步，不能换算为固定环境步数。
macro的训练max_seq_len15是输入token数上限；在实际对象input_proj输入10确认后，每token为5个2维动作，至多75个原始动作的输入chunk。
这既不是每个macro固定75步，也不是模型所有推理horizon的硬上限；autoregressive rollout与实际执行预算分别定义。

base `config/eval/hi_pusht.yaml` 默认为goal25/budget50，但high topk30、low600samples/topk60等不同于正式CSV覆盖值。
直接跑默认yaml不能声称复现matrix某行；experiment runner传入的CSV覆盖必须随receipt封存。
shortest正式goal-offset为25，仍可真正走high rollout；当前任意placement H10是新域诊断，不是已有D25官方query。

## 6. 精确坐标/normalizer/history门

`eval/hierarchical.py:287` 的图像顺序是 ToImage→ToDtype(float32,scale=True)→ImageNet Normalize→Resize224。
`build_process_map:319` 对配置的HDF5 dataset（默认pusht_expert_train）各列非NaN行fit StandardScaler；
不是源码中硬编码两组mean/std。本轮没有dataset，不编造数值，也不将SAGE stats或当前本地公共normalizer无条件替入。

`policies.py` 中先标准化primitive2维动作，再按模型input_proj输入宽度拼grouped tokens；预期10维时是5×2。
latent-prior bounds默认用2048个chunk、chunk_len5、percentile5/95、margin.05、clamp±3；
维数/数量不足会fallback，不得静默当成成功校准。校准是额外神经计算，不是训练；仍需CUDA-only且明确计账。
`get_action` 最后inverse_transform回raw action；不增加/100或额外clip。

native policy每次只 `_encode_pixels_last`，low_info用单帧 `z_init.unsqueeze(1)` 与零a_hist；
这不同于当前P1真实三帧/两段history合同。比较可沿native共同历史规则重跑全部新臂，
或新适配器保留我们真实history，但后者不能称完全官方执行路径；不能用重复receipt掩盖实际输入差异。
高/低 native metric是同一projected CLS空间平方L2，不是当前full_packet conditional Q。
若原Experiment1要求同Q，须在独立新adapter中统一全部臂并明示偏离；不拟合新Q或改成功阈值。

## 7. Loader与依赖的具体风险，不是空泛缺项

- `train/pretrained.py:86`：torch.load(weights_only=False,map_location=cpu)，解包可选`.model`，要求encoder/predictor/action_encoder/projector/pred_proj。
  它是训练前flat模型读取器，不是epoch15 high的推理入口。这里本轮未调用。
- `eval/hierarchical.py`：触发baseline adapter注册 `_baseline_lewm_module` 后AutoCostModel读取高对象；CPU map-location wrapper只对device=cpu启用，
  MAIN必须新封存CUDA-only gate；官方matrix spec的 `eval_device:cpu` 也不能照搬成我们允许CPU神经的依据。
- downloaded `train/hierarchical.py` 实际也有eval Hydra入口、world.evaluate_from_dataset，没有HiJEPA组装/预训练低层导入；
  它还注册legacy `hi_jepa/hi_module/...` aliases，而eval入口没有这些同等aliases。
  **这是实际包内源码差异**；不能用这个“train”文件追溯构建完整epoch15训练闭包，也不能断言eval已兼容未知pickle class路径。
  是否导致真实加载失败仍待可信对象检查，未先假定失败。
- converter可以用本地 `--weights ... --config ...` 做strict state_dict加载并生成object，默认不需下载；
  但存在stable_worldmodel→third_party旧类的compat rewrite，`--allow-non-strict`不得用来伪造配对。
  本轮不执行converter，不实例化模型。序列化输出hash相等也不是预期目标，应比较语义tensor/config。
- `THIRD_PARTY_LEWM.md`仅给upstream URL和未填写的 `<PINNED_COMMIT>`；`.gitmodules`只有URL，没有commit；
  完整目录没有BASELINE_LOCK.md。`environment-gpu.yml`只有无版本 `stable-worldmodel[train,env]`，未锁stable-pretraining/torch等闭包。
  不能直接clone latest或升级依赖后冒称原发布环境；本地已有源码可先记录commit/dirty和类接口，再做MAIN review。
- 完整414项中无LICENSE/COPYING文件；README和THIRD_PARTY仅给出处。记录级CC-BY-4.0仍在，包内代码/模型独立许可需要作者或额外官方声明确认。

## 8. MAIN最小落地顺序（明确不在本轮执行）

1. 在新独占结果中封存目标main object的中央目录定位、预计字节、CRC和取件器；MAIN批准约113.1MB选择性取件后记录object实测SHA。
   无需整包，无需先下203.8MB training weights，也无需先重下本地flat候选。
2. 可信pickle审查与离线依赖类绑定后，仅做对象结构/配置/模块语义hash检查，不forward；确认32/192、low action input10和模块闭包。
   如需受信任外部flat，先对比本地candidate，不匹配则停止或共同采用embedded low，而不是无证替换。
3. 新CUDA-only诊断runner以固定saved observation/goal，保留原RGB和native vs我们的history差异；真实调用high solver和rollout_high，
   固定取第一waypoint并实际交给low cost。所有配置、latent bounds来源与总资源向量在新结果前封存。
4. 通过接口诊断后才另行授权新matched physics；以同embedded low的flat arm控制backbone差异。不得拿旧H10物理结果或论文D25成功率充当matched control。

本轮交付不是仅路径表：已给出真正网络调用链、模型归属、正式CSV预算、normalizer/time/history合同与具体source缺陷。
剩余核心不是“再查文件名”，而是**尚未授权获取的高对象所含低层参数的认证**。我们不能在不读权重时编造这组hash。

## 9. 交付、验证与SHA

结果目录包含 acquire.py/supplement.py/followthrough.py、三个实际请求report、33份不可导入text、
build_contract.py/contract.json、verify_saved.py。contract.json含每个源码成员SHA、正式23行矩阵、本地candidate hash和待MAIN门。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/p1_hilewm_text_contract_20261027_v1/verify_saved.py
```

实际 **8 tests通过**：路径/权重拒绝、200与错range不读body、CRC/size/路径/解压cap、所有saved range→文本离线复算、
正式CSV预算、源码AST接口（不执行）、train入口实际eval差异、缺license/hash声明。零tmp/神经/物理/网络测试。

| 结果 | SHA256 |
|---|---|
| report.json | `ae4fff5ba6c2eb89a097bdcfc7c128282781093ca7bbcd0451948f2cc201187d` |
| supplement.json | `088fd8bb66c0afe3abfed044498d4074b10eeb1963c453e2e4b62dfa3230fae4` |
| followthrough.json | `18a64288a9018bb3ee8fe68949df0d6e0389a0d8f78be77e36f0caa73c47f56a` |
| contract.json | `c25edf542f38faa9cf0150df4822198d28753e2cf77a0d243e990ffabe849ded` |

科学状态：source contract部分解除、weight pairing待认证、无真实neural/physics效能证据、Experiment1/P1/C1不准入。
