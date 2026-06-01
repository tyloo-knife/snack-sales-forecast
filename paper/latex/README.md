# LaTeX 论文编译说明

论文源文件位于 `paper/latex/`，主文件为 `main.tex`，分章文件位于 `sections/`，参考文献位于 `references.bib`。

## 推荐编译命令

```powershell
cd paper/latex
latexmk -xelatex -interaction=nonstopmode main.tex
Copy-Item main.pdf ..\..\submission\final_paper.pdf -Force
```

## 手动编译

如果没有 `latexmk`，可依次执行：

```powershell
cd paper/latex
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
Copy-Item main.pdf ..\..\submission\final_paper.pdf -Force
```

## 注意事项

- 本文使用 `ctexart` 文档类，应使用 XeLaTeX 编译。
- 图片路径由 `main.tex` 中的 `\graphicspath{{../../figures/}}` 指定。
- `*.aux`、`*.log`、`*.out`、`*.fls`、`*.fdb_latexmk`、`*.xdv`、`main.pdf` 等编译产物不纳入版本管理。
