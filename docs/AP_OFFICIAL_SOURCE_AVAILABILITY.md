# Official Anchored Planning (AP) source: local availability

2026-11-03（再次核验）：本机目录、README 和 `gaussian_cem.py` 均存在，
后两者 SHA-256 与下方记录一致。此前“官方下载失败”是已过时的状态，
不应再次作为缺少官方源码的结论。完整源码树和来源认证范围仍以下文为准。

## 项目交接提示

官方 AP 源码已经下载并解压到本项目的
`external/anchored-planning-984327424ca7/`。此前的代理下载失败只是历史记录，
不能再作为当前源码缺失的依据；后续对话先检查此目录及下列哈希，不需要重复下载。
2026-11-03 再次核验了目录、CEM 源文件和 README，两个文件的 SHA-256
均与下文记录一致。这只核验了这两个文件，不代表整个源码树或版本来源已重新认证。

Status checked locally: 2026-11-03, Asia/Shanghai. **The official AP source
has already been downloaded.** Do not repeat the older statement that the
source is unavailable merely because the first proxy download attempt failed.
If in doubt, check the paths and hashes below before reporting its status.
On 2026-11-03 the extracted directory and the CEM file were checked again;
the CEM SHA-256 matched the value below. This check did not reverify the
complete extracted tree or its recorded revision.

- Authors' repository: `daybraeklaxry/Aim-Short-to-Reach-Far` (paper arXiv
  `2609.30036`).
- Fixed revision recorded in the original acquisition:
  `984327424ca7a2e52c75a8279197750d9d0b7c8e`; recorded tree:
  `95f9f45db171af9bb56a3a7e2a71dd97f0479182`.
- Local extracted source: `external/anchored-planning-984327424ca7/`.
  This is an extracted archive, **not a Git checkout**; the directory name
  alone does not independently authenticate every file.
- Official CEM source:
  `external/anchored-planning-984327424ca7/code/ap/anchored_planning/gaussian_cem.py`.
  Local SHA-256 checked on 2026-11-02:
  `cb4fe801edfadd7a8a039b0fd6154dca7db018d39864534196b02f180d95a4a4`.
- Local `README.md` SHA-256 checked on 2026-11-02:
  `c1547676c0015c557417965c17a56abe30a9afaf32340a72cf1760c9367841e8`.
- The acquisition record reports archive SHA-256
  `f22083b2aacc9abd4c8b4a230a24e4c0e227623ab2e54372df4538337ce295a4`
  and selected Git blob comparisons. The archive itself and the complete
  extracted tree were not reverified for this entry.

The CEM and memory retrieval parity fixtures have run against this local
source. Their results establish only the documented narrow kernel checks,
not full official AP controller, data/checkpoint, cost, physical trajectory,
or published-results reproduction. See
`docs/PC_AP_OFFICIAL_CEM_PARITY_20261102.md`,
`docs/PC_AP_OFFICIAL_KERNEL_CUDA_SMOKE_RESULT_20261102.md`, and
`docs/PC_AP_OFFICIAL_KERNEL_PHYSICAL_RESULT_20261102.md`.

The `external/` tree was untracked in the local Git status when this record
was written. **Local presence does not mean it has been pushed to GitHub**.
If transferring the project, separately preserve the extracted source and
its license, or reacquire the fixed revision and revalidate its contents.
