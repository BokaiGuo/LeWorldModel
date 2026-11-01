# Hi-LeWM资产恢复交接（离线核对，2026-10-28）

## 结论与边界

**在本次限定的既有Hi-LeWM路径内，没有已认证完整的PushT高层checkpoint，也没有可直接安全续传的完整资产。** 已有完整config与33份归档文本、414项完整中央目录；两个失败目录只有展开前缀/压缩片段，不能加载、拼成完整对象或追认probe成功。这里不是断言全盘不存在其它副本：未扫描全盘、未检查未知镜像/缓存、未联网。

本轮只新增本文，不下载、不解pickle（也不做pickle opcode扫描）、不执行归档代码/模型/physics、不训练、不改旧文件。以下资产恢复成功也仅证明字节完整性，不认证模型配对、可安全反序列化、Hi-LeWM-C bank正确性或P1/freshness科学准入。

## 1. 已核对的来源与完整性依据

下表路径以 `/home/gbk/code/LeWorldModel/le-wm-main/` 为根，全部是已存在输入。

|来源路径|实际SHA256 / 用途|
|---|---|
|`results/p1_learned_subgoal_remote_inventory_20261027_v1/zenodo_api.json`|`2fc221128c12b609b86dfb7810eda46f2399bc796348f36c5d7e016ab25dd773`；官方record元数据缓存|
|同目录`report.json`|`7feaf480934462619149c72e293de7df09cf55835f1001a8f7d88f9194b26c66`；完整中央目录与历史请求|
|同目录`archive_tail.bin`|`43a0c87d4f1b652caccbc26ac5e1b1ed73c0c6cf97c8e00683ecc1bdd8c16aef`；65,557 bytes尾部|
|`results/p1_hilewm_text_contract_20261027_v1/contract.json`|`c25edf542f38faa9cf0150df4822198d28753e2cf77a0d243e990ffabe849ded`；文本/模型/低层来源合同|
|同目录`texts/package__checkpoints__pusht__main__config.yaml.source.txt`|`dff6ac0dc9e7c4cfb7d76b268ceb568f6fc0916eaa4a725e6e1a5b53587e9075`；已完整取得的原配置，无需再下|
|`results/p1_hilewm_bounded_object_fetch_20261027_v1_main01/report.json`|`5370e9884fe2552cc7bc0b4b57d1af6a6ae01065e9ca1b9f47b1992a718c230e`；v1失败保留|
|`results/p1_hilewm_bounded_object_fetch_20261027_v2_probe_main01/report.json`|`669bf41043dfae01a9bca183fe230d61baad29234c5cc73038c80409844e0b90`；v2失败保留|

固定官方来源为 `https://zenodo.org/api/records/21353240/files/package.zip/content`，record DOI `10.5281/zenodo.21353240`。缓存API列出：

- `package.zip`：**2,299,863,924 bytes**，官方提供的是整包 `md5:aaf78b8b4fd6f5d1f5e720927951152f`，不是成员SHA256。
- record为`open`、license `cc-by-4.0`。完整目录中无LICENSE/COPYING；不能把record许可升级为对每项第三方代码/模型均已有独立授权证明。恢复交接应保留出处、许可范围及MAIN许可审查，本文不认证“任意pickle合法安全”。
- 中央目录414项、51,130 bytes、offset 2,299,812,772，SHA256 `75e0036bcb4014b30c297583fb1d96983cdcc3dbfa09a77fcf7f058e73eda67b`。

唯一目标成员为 `package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt`：

|字段|固定值|依据|
|---|---:|---|
|local header offset|634,194,328|已存完整中央目录|
|payload offset|634,194,454|历史local header/name/extra实际请求|
|压缩长度|113,104,429 bytes|中央目录|
|展开长度|122,424,517 bytes|中央目录|
|CRC32|`d59f865b`|中央目录；不是密码学签名|
|method / flags|8 / 8|raw DEFLATE / data descriptor；header不能代替中央目录CRC/长度|

**尚无完整对象实测SHA256或作者成员SHA256。** 本地tail/目录SHA只绑定已观测版本，不能替代整包MD5或成员全量校验；历史响应没有ETag/Last-Modified，不能据此声称服务器immutable revision。

## 2. Truncated range精确定位（不是403，也不是已知权重损坏）

所有下列Range均针对上述同一个官方URL，端点含首尾；`Accept-Encoding: identity`。请求/响应细节来自各目录`events.jsonl`及report，未重发。

|执行|失败attempt与Range|响应头期望|实际接收 / 缺口|保存状态|
|---|---|---|---|---|
|v1/main01|attempt4，`bytes=634194454-638388757`|206；Content-Length 4,194,304；Content-Range `bytes 634194454-638388757/2299863924`|4,185,588 / **8,716** bytes|首错`ValueError: Truncated range`；展开前缀4,624,388 bytes；4 attempts/3 completed|
|v2/probe_main01|attempt5，`bytes=634456598-634718741`|206；Content-Length 262,144；Content-Range `bytes 634456598-634718741/2299863924`|229,608 / **32,536** bytes|首错`TransferError: Truncated range`；5 attempts/4 completed，extra_attempts=0|

v1在失败前完成tail65,557 + header30 + name/extra96，body总实收4,251,271，保守预留4,259,987。失败压缩body SHA为`8c7bac7a069cd35f079155a4e2fb508e94c808189f5dce7f05e4b2bf619e8eaa`，但**压缩body本身未单独保留**；`download.part`是已展开前缀，不是可从该offset续接的压缩文件。

v2完成相同三项控制请求，再完整取得首段 `bytes=634194454-634456597`。总实收557,435，预留589,971；两次历史合计实收**4,808,706 bytes**，不能忽略失败流量。故障仅证明正确长度头之后body提前EOF，**不能定位为代理/服务端某一方故障或认定checkpoint损坏**。

|现有文件（相对于各失败目录）|实际bytes|实测SHA256|能否恢复完整对象|
|---|---:|---|---|
|v1 `download.part`|4,624,388|`53774a5c1ce57eb2da4a57303b68ff7a9156955c8e4e832cfa7812a505737a4f`|否；展开流前缀，缺剩余数据/完整流状态/终态CRC|
|v2 `probe_0.bin`|262,144|`8732e30218c512cca6f5486effdceb0e097c8484b8d2785e74ef804cdd56ccbb`|否；只有首个完整压缩段|
|v2 `failed_range_0005.part`|229,608|`f055e854b6b96c9257ff27548b780ce77c63cf6cc360c4b1436435dc5313191f`|否；第二段截断，未认证完整段|

旧v1明确无resume；旧v2 full从头创建新DEFLATE流且要求**成功**probe。当前probe为`failed_preserved`，不能拿它的SHA运行full、换flag绕过或重复probe。本文不提供curl续传指令；不删除/覆盖/拼写这些partial。它们保留作失败证据。

## 3. checkpoint / config / macro bank现状

|项目|已存在的内容|仍未认证 / 最小缺口|
|---|---|---|
|PushT main高层对象|上述目录/范围元数据、partial；没有在本次已知路径找到正式完整对象|122,424,517-byte完整对象，及可信来源全量校验收据；不能用同目录training `_weights.ckpt`替代|
|完整config/实现文本|`results/p1_hilewm_text_contract_20261027_v1/texts/`的33份已校验文本，包括main config、registry、eval、policies、23行正式matrix|文本已完整，不需重新下载；不是完整可安装依赖环境|
|低层候选|`/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt`，72,290,721 bytes，SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`；同目录`config.json` SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`|官方代码指向HF `quentinll/lewm-pusht`的config.json+weights.pt，但未锁immutable revision/配对tensor hash。已有文件无需重下；不是epoch15高层历史配对证明|
|内嵌低层|保存源码`models/jepa.py`显示HiJEPA同时持有高低层模块，eval加载一个object|真实对象未取得/未可信加载，不能据源码断言实物字段/参数匹配。未来可认证共同embedded low，不能无证复用旧本地endpoint|
|Hi-LeWM-C empirical bank|414项完整目录未出现`macro_bank`/`macro-bank`/`.npz`/`.npy`/`.h5`/`.hdf5`成员；已存eval/policies文本给出动态生成路径|这是文件名/源码范围结论，不排除对象内字段。`eval/hierarchical.py:357–395`运行时调用`build_empirical_macro_action_bank`，无已认证可直接加载的bank文件/摘要；不能声称checkpoint自动带齐-C资产|
|normalizer/数据身份|`eval/hierarchical.py:310–330`从dataset建StandardScaler；默认`pusht_expert_train`|已知Hi-LeWM路径没有已认证该训练数据、统计digest、bank receipt。本轮不搜索其它项目dataset，不拟合/替换normalizer|

bank源码依据为`texts/package__code__h_le_wm__planning__policies.py.source.txt:143–246`：过滤NaN、维持episode/step连续片段、对action作原process.transform、按seed抽样并调用`model.encode_macro_actions`，返回actions/row_indices/goal_row_indices等。默认empirical开关false；默认4096序列/chunk_len5/seed42来自eval yaml，**不是每一正式matrix行或已生成bank的证据**。输出还依赖high horizon/action_block、原数据行/顺序、原normalizer与macro encoder参数。只补一个generic npz或换用我们proposal bank不合格。

许可/配对仍unknown的最小外部材料：MAIN认可的来源/许可范围说明；完整main对象及可信全量收据（下节任选路线）；如要认证-C，再提供与冻结对象/指定matrix行绑定的bank+数据/normalizer身份及生成参数/代码hash/行索引，或提供原数据与确切合同供MAIN另审CUDA生成。**不要求本轮训练，不在本轮生成bank。**

## 4. 一次性恢复交接：外部供件后离线验收，不重试网络

当前不能直接恢复成功；外部先以已有合法副本供件，不自动执行获取。任选其一：

1. **完整官方package.zip**：提供实际本地绝对路径。可由现存API整包size+MD5以及冻结tail校验，再流式验证唯一main成员与已存config。无需先假造作者成员SHA，也无需解压写盘；虽然需顺序读2.3GB，但本轮未取得或读取此整包。
2. **仅完整main object（最小体积）**：提供实际本地路径和经MAIN独立认可来源的expected SHA256；不得对收到的同一文件现算hash再把它冒充外部期望。下方同时校size+CRC，仍需供件者说明该SHA与record/member的绑定和许可来源；SHA字符串本身不是作者签名。

以下命令仅标准库，**不联网、不导入torch、不解pickle、不写文件**；失败立即退出，无重试。`HILEWM_ASSET_FILE`和`HILEWM_ASSET_KIND`由MAIN填真实值，kind为`archive`或`object`；object另需可信`HILEWM_EXPECTED_OBJECT_SHA256`，不得填未来hash。这里不设置/改写系统HOME等变量。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B - "$HILEWM_ASSET_KIND" "$HILEWM_ASSET_FILE" "${HILEWM_EXPECTED_OBJECT_SHA256:-}" <<'PY'
import hashlib, json, re, sys, zipfile, zlib
from pathlib import Path

def require(ok, message):
    if not ok:
        raise SystemExit(message)

def check_object(stream, expected=None):
    h, crc, size = hashlib.sha256(), 0, 0
    while True:
        b = stream.read(1024 * 1024)
        if not b:
            break
        size += len(b)
        require(size <= 122424517, 'object too large')
        h.update(b)
        crc = zlib.crc32(b, crc)
    require(size == 122424517, 'object incomplete')
    require(crc & 0xffffffff == 0xd59f865b, 'object CRC mismatch')
    sha = h.hexdigest()
    require(expected is None or sha == expected, 'object SHA mismatch')
    return {'bytes': size, 'crc32': f'{crc & 0xffffffff:08x}', 'sha256': sha}

kind, name, expected = sys.argv[1:]
p = Path(name)
require(p.is_absolute() and p.is_file() and not p.is_symlink(), 'need actual regular absolute file')
member = 'package/checkpoints/pusht/main/pusht_hi_lewm_epoch15_object.ckpt'
if kind == 'object':
    require(re.fullmatch('[0-9a-f]{64}', expected) is not None, 'need externally trusted object SHA')
    require(p.stat().st_size == 122424517, 'object size mismatch')
    with p.open('rb') as f:
        receipt = check_object(f, expected)
elif kind == 'archive':
    root = Path('results/p1_learned_subgoal_remote_inventory_20261027_v1')
    raw = (root / 'zenodo_api.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == '2fc221128c12b609b86dfb7810eda46f2399bc796348f36c5d7e016ab25dd773', 'API cache changed')
    meta = json.loads(raw)
    rows = [r for r in meta['files'] if r['key'] == 'package.zip']
    require(meta['id'] == 21353240 and len(rows) == 1, 'record/file ambiguity')
    row = rows[0]
    require(p.stat().st_size == row['size'] == 2299863924, 'archive size mismatch')
    md5, sha = hashlib.md5(), hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            md5.update(b)
            sha.update(b)
        f.seek(-65557, 2)
        tail = f.read()
    require('md5:' + md5.hexdigest() == row['checksum'] == 'md5:aaf78b8b4fd6f5d1f5e720927951152f', 'archive MD5 mismatch')
    require(hashlib.sha256(tail).hexdigest() == '43a0c87d4f1b652caccbc26ac5e1b1ed73c0c6cf97c8e00683ecc1bdd8c16aef', 'tail mismatch')
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) == 414, 'entry count/duplicates')
        i = z.getinfo(member)
        require((i.header_offset, i.compress_size, i.file_size, i.CRC, i.compress_type, i.flag_bits) == (634194328, 113104429, 122424517, 0xd59f865b, 8, 8), 'member spec mismatch')
        config = 'package/checkpoints/pusht/main/config.yaml'
        saved = Path('results/p1_hilewm_text_contract_20261027_v1/texts/package__checkpoints__pusht__main__config.yaml.source.txt').read_bytes()
        require(hashlib.sha256(saved).hexdigest() == 'dff6ac0dc9e7c4cfb7d76b268ceb568f6fc0916eaa4a725e6e1a5b53587e9075', 'config cache changed')
        require(z.getinfo(config).file_size == len(saved), 'config size mismatch')
        require(z.read(config) == saved, 'config mismatch')
        with z.open(i) as f:
            receipt = check_object(f)
    receipt['archive_sha256_measured'] = sha.hexdigest()
else:
    raise SystemExit('kind must be archive or object')
print(json.dumps({'status': 'bytes_verified_only', 'kind': kind, 'file': str(p),
                  'member': member, 'object': receipt, 'unpickled': False,
                  'pairing_verified': False, 'macro_bank_verified': False,
                  'scientific_admission': False}, sort_keys=True))
PY
```

完整归档路线的object流是解压ZIP成员原字节，**不是反序列化checkpoint**。脚本不调用`extract`/`testzip`，不读其它权重。整包MD5是发布方提供的传输完整性依据，不是抗恶意来源认证；实测SHA需保存于MAIN新的独占收据，不能回写旧失败目录。执行时须保持输入不可变；它不是防并发恶意换文件的安全沙箱。对象路线也不认证config嵌入内容或内层依赖。取得`bytes_verified_only`后仍由MAIN另行审查可信加载、类依赖、embedded low配对和CUDA权限；不自动进入模型运行。

## 5. 本轮实际验证及未完成项

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`，以下全部实际离线执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/p1_learned_subgoal_remote_inventory_20261027_v1/verify_saved.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B results/p1_hilewm_text_contract_20261027_v1/verify_saved.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_hilewm_bounded_object_fetch_20261027_v1.py tests/test_p1_hilewm_bounded_object_fetch_20261027_v2.py
```

结果：**5 + 8 + 57 tests通过**。另直接复算两个失败目录全部7个文件的bytes/SHA、指定本地flat候选SHA、414项目录相关bank文件名；本文交接代码仅语法/拒绝路径检查，不把mock或现存partial当成功完整对象。旧测试中的网络均为mock，不重新探针；不触发旧v1静态pickle扫描。

未完成：无完整main对象可运行正向字节验收；无已认证高低配对/normalizer/bank；许可细分与可信加载尚须MAIN确认；无Hi-LeWM本地神经/物理效果证据。最小下一步是**外部供件及可信校验来源**，不是再次失败探针、训练补齐、改URL或放宽P1。
