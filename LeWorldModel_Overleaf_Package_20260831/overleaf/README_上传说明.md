# Overleaf 上传说明

把本目录中的全部文件上传到 Overleaf 项目根目录，并保留 `figures/` 子目录。主文件选择 `main.tex`；补充材料可单独选择 `supplement.tex` 编译。

需要上传的文件包括：

- `main.tex`
- `supplement.tex`
- `references.bib`
- `iclr2027_conference.sty`
- `iclr2027_conference.bst`
- `fancyhdr.sty`
- `figures/*.pdf`、`figures/operator_alignment_summary.png`、`figures/hardened_future_action_ablation.png` 和 `figures/hardened_split_robustness.png`

`figure_sources/` 是 SVG/PPTX 编辑源，不参与编译。若要修改图，先在本地导出新的 PDF，再替换 `figures/` 中相同文件名的 PDF。
