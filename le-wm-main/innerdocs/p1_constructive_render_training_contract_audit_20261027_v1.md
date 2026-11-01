# Constructive render / checkpoint training contract：独立只读审计

日期：2026-10-27。工作树：`/home/gbk/code/LeWorldModel/le-wm-main`；HEAD=`a9679090e6d2195bc761c67cef1300726d051c2d`。HEAD不是所有当前文件的历史真实性证明，下面逐文件SHA才标识本次读到的文本。仅新增本文；没有修改任何contract、源码、封存或共享文档，没有网络、模型加载/forward、物理执行、训练、下载或pickle反序列化。没有读取或重新hash `weights.pt` 内容。

## 结论先行

**发布checkpoint实际训练图像是否含target overlay：unknown，当前本地证据不能闭合。**

可以确认的三个不同层级不能混淆：

1. 本次constructive source明确`with_target=False`；当前安装的native PushT构造参数默认`True`，且`render_goal`默认1。这是实际渲染合同差异，不自动构成bug。
2. 本地`pusht_expert_train.h5`三个事先固定位置样本均可见绿色T形目标标记，支持这份文件这些帧包含overlay。不是仅从环境默认值猜测，也不是用精确单色像素计数推断。没有检查全库，不声称所有帧都有可见marker。
3. checkpoint architecture config / model card / 当前training代码没有给出该发布权重→特定训练run→特定数据revision/完整hash→渲染生成配置的绑定。故第2条不能升级为“此checkpoint已经证实以overlay图像训练”。

因此：**存在有依据的训练—部署渲染分布不一致风险，但尚未确认checkpoint层面的mismatch，更未确认它解释当前native success=0。** 不得以本审计推翻2/2可达source或把任何Q/pose结果追认为overlay根因。不修改旧`with_target`或旧sealed产物。

## 1. 渲染合同：已确认

- [constructive source](/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/p1_constructive_reachable_source_20261027_v1.py:54)：prepare明确`with_target=False`；同文件341行factory显式传False，不是遗漏参数。
- [安装版PushT](/home/gbk/code/LeWorldModel/le-wm-main/.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py:29)：29–39行构造函数默认`with_target=True`。196–197行variation的`render_goal`默认1；422–454行只有`render_goal && self.with_target`才画goal polygon；466–467行再画实际agent/block。157–162行goal默认颜色LightGreen。
- `False`移除的是额外goal-pose标记，不移除实际agent，不表示“goal RGB不含agent”。这是独立于identity192 Q、normalization、模型权重的输入图像语义。
- 当前native `_set_goal_state`仅赋`goal_state`（同env.py529–530行）；renderer读的是`goal_pose`（428行，setup中577行初始化）。因此即使将来证实training含overlay，也不能未经新协议就把sealed source切到True：必须另核derived goal与图上marker的真实一致性。这里不实施或建议事后更改旧任务。

## 2. checkpoint和loader：没有训练渲染权威

本地目录：`/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/`。

- [config.json](/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/config.json:1) 共49行：encoder/predictor/action_encoder/projector/pred_proj结构；224输入、192 latent、3帧context。无dataset revision/hash、collector kwargs、`with_target`、`render_goal`或训练run身份。结构可加载不等于训练图像合同已认证。
- [README](/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/README.md:1) 仅7行：声明official pretrained PushT，datasets为`quentinll/lewm-pusht`。仓库名称不是数据文件hash，更不是每次训练采样/渲染配置。
- 本地HF下载metadata：config与README的第1行均是`22b330c28c27ead4bfd1888615af1340e3fe9052`，第2行分别`2e534ee5c6286721c0ef967e24bddbfde482abf9`、`ff45e6fb2af057489eacebcb263f372061bdcab2`。这些是模型repo下载metadata，不是dataset版本或training-run manifest；未联网解释/验证这些ID。
- [compat loader](/home/gbk/code/LeWorldModel/le-wm-main/experiments/checkpoint_compat.py:73) 的81–95行只读模型config、instantiate、load/remap权重、eval/freeze；没有恢复训练collector渲染参数或移除/添加overlay的逻辑。这里只读其源码，**未调用loader**。
- [库model save/load](/home/gbk/code/LeWorldModel/le-wm-main/.venv/lib/python3.12/site-packages/stable_worldmodel/wm/utils.py:13) 的26–40行保存state_dict和给定config；47–93行加载时也是结构+权重。未执行这些函数。

## 3. 当前training/data源码：消费离线pixels，不能倒推出collector设置

- [train data config](/home/gbk/code/LeWorldModel/le-wm-main/config/train/data/pusht.yaml:1)：`frameskip: 5`、`name: pusht_expert_train.lance`，读取pixels/action/proprio/state；没有渲染选项。
- [train.py](/home/gbk/code/LeWorldModel/le-wm-main/train.py:58)：58–76行load_dataset并应用图像预处理及非图像normalizer；27行model.encode消费batch。它不通过在线PushT renderer生成训练pixels。不能因为当前env默认True就推断这些离线pixels生成时为True。
- [utils.py](/home/gbk/code/LeWorldModel/le-wm-main/utils.py:9)：9–13行预处理为ToImage(ImageNet stats)+Resize；没有明确删除goal overlay的步骤。train.py122–123行可存run config.yaml，但checkpoint目录本次列表仅见config.json/README/weights及HF metadata，未发现该发布模型的完整run config.yaml。不能把当前代码当发布权重原始训练源码证明。
- [training README](/home/gbk/code/LeWorldModel/le-wm-main/README.md:39)：39–52行描述HDF5及`pusht_expert_train.h5`；当前YAML却指`.lance`。54–71行给出训练和导出说明。该差异说明文档/当前config并非一个已绑定的历史run receipt，不证明任何格式曾被此权重实际使用。
- [data resolver](/home/gbk/code/LeWorldModel/le-wm-main/.venv/lib/python3.12/site-packages/stable_worldmodel/data/utils.py:35)：load_dataset解析已有离线数据格式；[HDF5 reader](/home/gbk/code/LeWorldModel/le-wm-main/.venv/lib/python3.12/site-packages/stable_worldmodel/data/formats/hdf5.py:111) 111–132行直接切片/frameskip/channel permute/transform，无依据当前env渲染重建pixels。上述库源码未import执行，避免resolver的网络/目录创建副作用。
- 当前[eval config](/home/gbk/code/LeWorldModel/le-wm-main/config/eval/pusht.yaml:6) 的world仅指定PushT且无with_target override（6–9行），这至多说明当前eval可能继承默认，**不是训练证据**。

## 4. 本地HDF5：元数据与固定三帧只读观察

路径：`/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5`；stat大小46,300,921,856 bytes，mtime=`2026-10-14 01:56:11.576170341 +0800`。没有对46GB文件做完整hash，文件名/mtime均不当作官方数据revision证明。

用本地h5py只读`r`检查：file attrs={}，所有列attrs={}。pixels=`uint8 (2336736,224,224,3)`，chunk=`(100,224,224,3)`，filter ID32001 blosc；ep_len/ep_offset均18685条；另有action/proprio/state/episode_idx/step_idx。无collector render kwargs、checkpoint hash或训练runID字段。

最初直接读pixel失败，原因是已安装HDF5未自动找到blosc plugin目录。随后仅import**已经安装**的hdf5plugin注册codec，成功读取固定索引`[0, N//2, N-1]`，不安装/下载任何依赖，不加载模型/环境。通过BytesIO生成内存图像并人工查看，不落盘PNG。三帧都有绿色T形背景标记及独立灰色物体、蓝色agent；这是有限帧视觉证据，不认证生成代码/with_target参数名。

| flat frame index | 原始decoded RGB bytes SHA256 | 精确LightGreen `[144,238,144]`像素数 |
|---|---|---:|
| 0 | `135fa1f958224f0f56e5a925a5e50b9ce5cfa219325228dd0996a50ac4d842e2` | 1 |
| 1168368 | `1201ee9f24edc6b63e7442484d7229e540eedb3184fad91b5cd2b9de5e10647e` | 5 |
| 2336735 | `d25b1195627cc693c41eda091a38453e0e4297e5f92b79909e7d1f9ba82cdeeb` | 7 |

颜色计数仅作可复验记录：图像颜色存在大量近似值，精确色少不表示marker缺失；不能用这三个数当overlay检测阈值。未读取数据state/action内容，未看测试物理结果来挑这三帧。

可复验核心（只读本地数据，无neural/physics/pickle）：

```python
import hdf5plugin, h5py, hashlib
with h5py.File('/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5', 'r') as f:
    d = f['pixels']
    for j in (0, len(d)//2, len(d)-1):
        a = d[j]
        print(j, hashlib.sha256(a.tobytes()).hexdigest())
```

## 5. 缺失证据及允许的结论边界

要把checkpoint训练overlay状态从unknown升级，需要至少一条可信的完整关联链：

1. 发布权重身份→原始训练run（发布者manifest/配置/日志，且对权重artifact内容hash绑定）。本次按禁止触碰模型payload要求，没有自行读取或重新hash权重。
2. 该run→实际读取的dataset revision/file hash、split、transform；不能只给相似文件名或同HF collection。
3. 该数据→collector代码commit/原始env版本及`with_target`/`render_goal`/目标绘制设置，或能与该数据身份绑定的充分采样/生成记录。若只证明图上marker存在，应把结论写为像素现象，不假造collector参数。

本地三帧可以作为后续已授权、预声明render distribution检查的出发证据，但本次不设计/运行新paired physics或neural试验，不选择阈值，不修改旧sealed渲染合同。已有实际Q改善但native未成功也不能仅凭此归咎overlay；空间几何、goal定义、度量/动力学等仍是独立未归因因素。P1/P2准入不变。

## 6. 本次文本证据SHA256

相对路径均相对`/home/gbk/code/LeWorldModel/le-wm-main`；`SITE`=`/home/gbk/code/LeWorldModel/le-wm-main/.venv/lib/python3.12/site-packages`；`CKPT`=`/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm`。行号以上面的绝对链接为准。本次仅写本文，以下文件均未改。

| 文件 | SHA256 |
|---|---|
| `experiments/spectral_reachability/p1_constructive_reachable_source_20261027_v1.py` | `c32b9f169adaed31d07ecef5c507ce409c2ecec5743f8c6fde4fab928b92a878` |
| `SITE/stable_worldmodel/envs/pusht/env.py` | `c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2` |
| `CKPT/config.json` | `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09` |
| `CKPT/README.md` | `660de4e79b109ea18602618d90c72c44630957978ebd8474910713d896340292` |
| `CKPT/.cache/huggingface/download/config.json.metadata` | `be56e5818d876e965d24d8ebf4c5443028c40b65421f4e593c182c867d66a3df` |
| `CKPT/.cache/huggingface/download/README.md.metadata` | `4d072a80de0b42ab9b3f385559594b9d27505ead4a3ab2f37d87088e308f1cc6` |
| `config/train/data/pusht.yaml` | `dcdd082ebdc628cdc8a88f15cb647fce6eec87343f9fe190271946ff5d52f757` |
| `config/train/lewm.yaml` | `fef2e609e3e1be367400e69e0929229ee0b6bd95cc4d4681c075ec9ab7d4cf89` |
| `config/eval/pusht.yaml` | `66eca39ebc1034d0cf746fef80e1c6309c07dc1fbd6fa067779ac97c446d2856` |
| `train.py` | `c5b4ced2295113186d2834671020f92104b56793839c7fce490ba0050bef90d4` |
| `utils.py` | `048701806dfc045f16d33a85475861a9b4c78b314b7dda84161f5d018f17e9d7` |
| `README.md` | `0e10255536bcd50bcb9e9638e5b9d46c7db77a914eef9099276269483414b693` |
| `experiments/checkpoint_compat.py` | `50b01e20864dc929943df7cb3566c271a771e69c06ea0fdbc9b598c59e6e9dff` |
| `SITE/stable_worldmodel/data/utils.py` | `a132734367c219c0fdf59cd9d3631ed83bfe58bc427aa41e85146e26ac79ab4c` |
| `SITE/stable_worldmodel/data/formats/hdf5.py` | `afe836ec706acc6c55450f3c936a5898b57ed65a999695af25b4b59cd14d10f3` |
| `SITE/stable_worldmodel/data/dataset.py` | `f0b0027cfc9c3341a93b15b957596c2c0bc62fc2b48a609a4814eeaf89a47982` |
| `SITE/stable_worldmodel/wm/utils.py` | `48b8ea9fad66887d4f4fecc18d7e47b3d6fa9e7c3791c9769137ef20ab128424` |
| `SITE/hdf5plugin/__init__.py` | `db6108dd35035d097d15644b0cf2215c86e805764f80e1a11ad764997ba4d34a` |
