# LaTeX 编译说明

## 编译方式

推荐在项目根目录执行：

```powershell
cd paper/latex
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
Copy-Item build/main.pdf ../final_paper.pdf -Force
```

如果没有 `latexmk`，可使用：

```powershell
cd paper/latex
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
Copy-Item main.pdf ../final_paper.pdf -Force
```

## LaTeX 引擎

本文包含中文，使用 `ctexart` 文档类，优先使用 XeLaTeX 编译。

## 主要输入文件

- 主文件：`paper/latex/main.tex`
- 分章文件：`paper/latex/sections/*.tex`
- 参考文献：`paper/latex/references.bib`
- 图表来源：`figures/*.png`
- Markdown 原稿：`paper/full_paper_draft.md`

## 最终输出

- 最终 PDF：`paper/final_paper.pdf`

## 常见问题

1. 中文乱码：确认使用 XeLaTeX，而不是 pdfLaTeX。
2. 找不到图片：确认从 `paper/latex` 目录编译，或检查 `main.tex` 中的 `\graphicspath{{../../figures/}}`。
3. 表格过宽：优先减少列数，必要时使用 `\resizebox{\textwidth}{!}{...}`。
4. 参考文献问号：使用 `latexmk -xelatex` 或手动运行 BibTeX 后再运行两次 XeLaTeX。
5. 临时文件：`build/`、`*.aux`、`*.log`、`*.out`、`*.toc`、`*.fls`、`*.fdb_latexmk` 不作为最终提交材料。
