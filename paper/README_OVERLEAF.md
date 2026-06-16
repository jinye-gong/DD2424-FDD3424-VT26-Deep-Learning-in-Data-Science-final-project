# Overleaf 编译说明（DD2424 Group 148）

## 中英文分包（推荐上传）

| Zip | 主文件 | 编译器 |
|-----|--------|--------|
| **`paper_overleaf_en.zip`**（仓库根目录） | `main.tex`（英文） | pdfLaTeX + BibTeX |
| **`paper_overleaf_zh.zip`**（仓库根目录） | `main.tex`（中文） | **XeLaTeX** + BibTeX |

本目录 `paper/` 为开发用源码：`main.tex`（英）、`main_zh.tex`（中）。

---

## 课程页数要求（Canvas）

| 模板 | 正文页数上限 | 参考文献 |
|------|-------------|----------|
| **NeurIPS 2021**（推荐，与本仓库一致） | **6 页** | **不计入** |
| CVPR 2021（双栏备选） | 4 页 | 不计入 |

- 正文 = 标题、摘要、正文、图表、致谢等；**参考文献单独排版，不占 6 页**。
- 附录、自评、复现细节若超出 6 页，请放到**单独 PDF**（如 `self_assessment.pdf`），或压缩进正文。
- `main.tex` 含完整**实验流程**（§2）、**13 条编号发现**、**RQ 结论与设计规则**；编译后请确认正文 ≤6 页（不含参考文献）。若超出，可删 `width_scaling` 或 `depth_scan` 图。

## 上传到 Overleaf

1. 上传 `paper/` 或 `paper_overleaf.zip`（仓库根目录）。
2. 主文件：**`main.tex`**；编译器：**pdfLaTeX + BibTeX**。
3. 样式：优先使用 [NeurIPS 2021 官方模板](https://neurips.cc/Conferences/2021/PaperInformation/StyleFiles) 中的 `neurips_2021.sty`，将 `\usepackage[final]{neurips_2026}` 改为 `\usepackage[final]{neurips_2021}`（版式与 2026 相近，以课程要求为准）。

## 编译后检查页数

在 PDF 阅读器中确认：**参考文献起始页之前 ≤ 6 页**。若超出，可删减 Discussion 或合并表格。

## 图表更新

```bash
python scripts/generate_report_assets.py
```

然后重新编译 `main.tex`。

## 文件对应

| 文件 | 用途 |
|------|------|
| `DefaultProject2.pdf` | 作业 E / C–D / A–B 要求 |
| `DD2424_Project.pdf` | Proposal |
| `REPORT_zh.md` | 完整中文草稿（可长于 6 页） |
| `paper/main.tex` | **提交用 6 页英文终稿**（含 **[E]/[D/C]/[B/A]** 对照） |
| `paper/main_zh.tex` | **提交用 6 页中文 LaTeX 终稿**（内容同 `REPORT_zh.md` 精简版） |
| `REPORT_zh.md` | 中文 Markdown 完整稿（可长于 6 页） |
